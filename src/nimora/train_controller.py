from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from nimora.config import ControllerRunConfig, load_controller_config
from nimora.data import PackedTokenDataset
from nimora.model import NimoraController, build_optimizer_groups
from nimora.runtime import assert_training_runtime, require_torch, resolve_device
from nimora.tokenizer import load_tokenizer

torch = require_torch()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, config: ControllerRunConfig) -> float:
    optimizer = config.optimizer
    training = config.training
    if step < optimizer.warmup_steps:
        return optimizer.learning_rate * (step + 1) / max(1, optimizer.warmup_steps)
    progress = (step - optimizer.warmup_steps) / max(
        1, training.max_steps - optimizer.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return optimizer.minimum_learning_rate + cosine * (
        optimizer.learning_rate - optimizer.minimum_learning_rate
    )


def _autocast_context(device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    output_dir: Path,
    step: int,
    checkpoint_model,
    optimizer,
    scaler,
    config: ControllerRunConfig,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": checkpoint_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "metadata": checkpoint_model.checkpoint_metadata(),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    temporary = checkpoint_dir / "training-state.pt.tmp"
    final = checkpoint_dir / "training-state.pt"
    torch.save(payload, temporary)
    os.replace(temporary, final)
    (output_dir / "latest").write_text(checkpoint_dir.name + "\n", encoding="utf-8")
    return final


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for checkpoint in checkpoints[:-keep] if keep > 0 else []:
        state = checkpoint / "training-state.pt"
        if state.exists():
            state.unlink()
        checkpoint.rmdir()


def load_checkpoint(path: str | Path, model, optimizer, scaler, device) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    random.setstate(payload["random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if device.type == "cuda" and "cuda_random_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    return int(payload["step"])


@torch.no_grad()
def evaluate(model, loader, device, precision: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            loss = model(input_ids, labels=labels)["loss"]
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config_path: str | Path) -> None:
    config = load_controller_config(config_path)
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    assert_training_runtime(device)

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != config.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({actual_vocab}) does not match model.vocab_size "
            f"({config.model.vocab_size})"
        )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved-config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
        handle.write("\n")

    checkpoint_model = NimoraController(config.model).to(device)
    model = checkpoint_model
    if config.training.compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        build_optimizer_groups(model, config.optimizer.weight_decay),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.epsilon,
        fused=False,
    )
    scaler = _grad_scaler(config.training.precision == "fp16")
    start_step = 0
    if config.training.resume_from:
        start_step = load_checkpoint(
            config.training.resume_from, checkpoint_model, optimizer, scaler, device
        )

    examples_per_step = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
    )
    total_examples = config.training.max_steps * examples_per_step
    consumed_examples = start_step * examples_per_step
    full_train_dataset = PackedTokenDataset(
        config.data.train_dir,
        config.data.sequence_length,
        total_examples,
        config.training.seed,
    )
    remaining_train_dataset = torch.utils.data.Subset(
        full_train_dataset,
        range(consumed_examples, total_examples),
    )
    validation_dataset = PackedTokenDataset(
        config.data.validation_dir,
        config.data.sequence_length,
        config.data.validation_samples,
        config.training.seed + total_examples,
    )
    train_loader = torch.utils.data.DataLoader(
        remaining_train_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
    )

    batches = iter(train_loader)
    metrics_path = output_dir / "metrics.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, config.training.max_steps + 1):
        started = time.perf_counter()
        accumulated_loss = 0.0
        supervised_tokens = 0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            supervised_tokens += int(labels.ne(-100).sum())
            with _autocast_context(device, config.training.precision):
                loss = model(input_ids, labels=labels)["loss"]
                scaled_loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.max_grad_norm
        )
        current_lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        elapsed = time.perf_counter() - started
        metric = {
            "step": step,
            "loss": accumulated_loss / config.training.gradient_accumulation_steps,
            "learning_rate": current_lr,
            "grad_norm": float(grad_norm),
            "supervised_tokens": supervised_tokens,
            "seconds": elapsed,
            "supervised_tokens_per_second": supervised_tokens / max(elapsed, 1e-9),
        }

        if step % config.training.evaluate_every == 0:
            metric["validation_loss"] = evaluate(
                model, validation_loader, device, config.training.precision
            )
        if step % config.training.log_every == 0:
            print(json.dumps(metric), flush=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")

        if step % config.training.save_every == 0 or step == config.training.max_steps:
            save_checkpoint(
                output_dir,
                step,
                checkpoint_model,
                optimizer,
                scaler,
                config,
            )
            prune_checkpoints(output_dir, config.training.keep_last_checkpoints)
