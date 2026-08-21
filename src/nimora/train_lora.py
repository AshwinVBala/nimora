from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from nimora.runtime import assert_training_runtime, require_torch, resolve_device
from nimora.serialization import canonical_json


@dataclass(slots=True)
class LoraAdapterConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def validate(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty")


@dataclass(slots=True)
class LoraRunConfig:
    model_name: str = "Qwen/Qwen3-4B"
    train_files: list[str] = field(default_factory=lambda: ["data/train.jsonl"])
    validation_files: list[str] = field(default_factory=lambda: ["data/validation.jsonl"])
    output_dir: str = "runs/nimora-code-4b-lora"
    sequence_length: int = 2_048
    precision: str = "fp16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    epochs: float = 2.0
    logging_steps: int = 5
    evaluation_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 1337
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    device: str = "auto"
    resume_from_checkpoint: str | None = None
    adapter: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)

    def validate(self) -> None:
        self.adapter.validate()
        if self.precision not in {"fp16", "bf16"}:
            raise ValueError("LoRA precision must be fp16 or bf16")
        if self.sequence_length < 128:
            raise ValueError("sequence_length is unexpectedly small")
        if self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("LoRA batch sizes must be positive")
        if self.learning_rate <= 0.0 or self.epochs <= 0.0:
            raise ValueError("LoRA learning_rate and epochs must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers cannot be negative")
        if not self.train_files or not self.validation_files:
            raise ValueError("LoRA train_files and validation_files cannot be empty")


def load_lora_config(path: str | Path) -> LoraRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    adapter_values = raw.get("adapter", {})
    run_values = {key: value for key, value in raw.items() if key != "adapter"}
    adapter = LoraAdapterConfig(**adapter_values)
    config = LoraRunConfig(**run_values, adapter=adapter)
    config.validate()
    return config


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Nimora trajectories into portable system/user/assistant chat turns."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and "action" in message:
            content = canonical_json(message["action"])
            normalized.append({"role": "assistant", "content": content})
        elif role in {"tool", "observation"}:
            name = message.get("name", "environment")
            normalized.append(
                {
                    "role": "user",
                    "content": f"OBSERVATION[{name}]\n{message.get('content', '')}",
                }
            )
        elif role in {"system", "user", "assistant"}:
            content = str(message.get("content", ""))
            channel = message.get("channel")
            if role == "assistant" and channel:
                content = f"{str(channel).upper()}\n{content}"
            normalized.append({"role": str(role), "content": content})
        else:
            raise ValueError(f"Unsupported role for LoRA data: {role!r}")
    return normalized


def _fallback_assistant_mask(tokenizer, messages, full_ids: list[int]) -> list[int]:
    mask = [0] * len(full_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before_ids = tokenizer.apply_chat_template(
            messages[:index],
            tokenize=True,
            add_generation_prompt=True,
        )
        through_ids = tokenizer.apply_chat_template(
            messages[: index + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        start = min(len(before_ids), len(mask))
        stop = min(len(through_ids), len(mask))
        for position in range(start, stop):
            mask[position] = 1
    return mask


def tokenize_trajectory(tokenizer, messages, sequence_length: int) -> dict[str, list[int]]:
    messages = normalize_messages(messages)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=sequence_length,
        )
        input_ids = list(encoded["input_ids"])
        assistant_mask = encoded.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = encoded.get("assistant_tokens_mask")
    except (TypeError, ValueError):
        input_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=sequence_length,
            )
        )
        assistant_mask = None

    if assistant_mask is None or not any(assistant_mask):
        assistant_mask = _fallback_assistant_mask(tokenizer, messages, input_ids)
    assistant_mask = list(assistant_mask)[: len(input_ids)]
    assistant_mask.extend([0] * (len(input_ids) - len(assistant_mask)))
    labels = [token if learns else -100 for token, learns in zip(input_ids, assistant_mask)]
    if not any(label != -100 for label in labels):
        raise ValueError("Trajectory contains no trainable assistant tokens")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class LoraTrajectoryDataset:
    def __init__(self, files: list[str], tokenizer, sequence_length: int) -> None:
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.rows: list[tuple[str, int, int]] = []
        self._handles: dict[str, BinaryIO] = {}
        for file_value in files:
            path = Path(file_value).resolve()
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {path}:{line_number}: {error}"
                        ) from error
                    if not isinstance(record, dict) or not isinstance(
                        record.get("messages"), list
                    ):
                        raise ValueError(
                            f"LoRA record at {path}:{line_number} needs a messages array"
                        )
                    self.rows.append((str(path), offset, line_number))
        if not self.rows:
            raise ValueError("The LoRA dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        path, offset, line_number = self.rows[index]
        handle = self._handles.get(path)
        if handle is None:
            handle = Path(path).open("rb")
            self._handles[path] = handle
        handle.seek(offset)
        record = json.loads(handle.readline())
        try:
            return tokenize_trajectory(
                self.tokenizer,
                record["messages"],
                self.sequence_length,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid trajectory at {path}:{line_number}: {error}") from error

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._handles.values():
            handle.close()


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples):
        torch = require_torch()
        width = max(len(item["input_ids"]) for item in examples)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids, attention_mask, labels = [], [], []
        for item in examples:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_lora(config_path: str | Path) -> None:
    torch = require_torch()
    config = load_lora_config(config_path)
    device = resolve_device(config.device)
    assert_training_runtime(device)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError("Install Nimora with the 'lora' extra before LoRA training") from error

    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    peft_config = LoraConfig(
        r=config.adapter.rank,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        target_modules=config.adapter.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = LoraTrajectoryDataset(
        config.train_files, tokenizer, config.sequence_length
    )
    validation_dataset = LoraTrajectoryDataset(
        config.validation_files, tokenizer, config.sequence_length
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved-config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(config),
            handle,
            indent=2,
        )
        handle.write("\n")

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.evaluation_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final-adapter"))
    tokenizer.save_pretrained(str(output_dir / "final-adapter"))
