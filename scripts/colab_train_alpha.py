"""Train and evaluate the publishable Nimora v0.0.1-alpha QLoRA adapter."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

WHEEL = Path("/content/nimora-0.1.0-py3-none-any.whl")
DATA_ARCHIVE = Path("/content/nimora-alpha-data.tar.gz")
WORK = Path("/content/nimora-agent-alpha")
MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def install_dependencies() -> None:
    for path in (WHEEL, DATA_ARCHIVE):
        if not path.is_file():
            raise FileNotFoundError(f"Required Colab payload is missing: {path}")
    print("Installing Colab training dependencies...", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.51,<5",
            "peft>=0.15,<1",
            "accelerate>=1.6,<2",
            "bitsandbytes>=0.46,<1",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--force-reinstall",
            "--no-deps",
            str(WHEEL),
        ]
    )
    print("Installed the exact local Nimora wheel.", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def score_output(raw: str, expected_value: dict[str, Any]) -> dict[str, Any]:
    from nimora.agent.types import Decision

    expected = Decision.from_value(expected_value)
    try:
        actual = Decision.from_value(raw)
    except ValueError as error:
        return {
            "valid": False,
            "semantic": False,
            "arguments_exact": False,
            "error": str(error),
            "raw": raw,
        }

    if expected.action is not None:
        semantic = actual.action is not None and actual.action.name == expected.action.name
        arguments_exact = semantic and actual.action.arguments == expected.action.arguments
    else:
        semantic = actual.result is not None and bool(actual.result.strip())
        arguments_exact = semantic
    return {
        "valid": True,
        "semantic": bool(semantic),
        "arguments_exact": bool(arguments_exact),
        "error": None,
        "raw": raw,
        "actual": {
            "plan": actual.plan,
            "action": actual.action.to_dict() if actual.action else None,
            "result": actual.result,
        },
    }


def evaluate_model(model, tokenizer, cases, label: str) -> dict[str, Any]:
    import torch

    from nimora.train_lora import normalize_messages

    results = []
    for index, case in enumerate(cases, start=1):
        messages = normalize_messages(case["messages"])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        score = score_output(raw, case["expected"])
        score["family"] = case["metadata"]["family"]
        score["expected"] = case["expected"]
        results.append(score)
        if index % 10 == 0:
            print(f"{label}: evaluated {index}/{len(cases)} cases", flush=True)

    count = len(results)
    return {
        "count": count,
        "valid_rate": sum(item["valid"] for item in results) / count,
        "semantic_rate": sum(item["semantic"] for item in results) / count,
        "arguments_exact_rate": (
            sum(item["arguments_exact"] for item in results) / count
        ),
        "cases": results,
    }


def run() -> None:
    started = time.time()
    install_dependencies()

    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from nimora.train_lora import train_lora

    if not torch.cuda.is_available():
        raise RuntimeError("Nimora alpha training requires a GPU runtime")
    print(f"Training device: {torch.cuda.get_device_name(0)}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DATA_ARCHIVE, "r:gz") as archive:
        archive.extractall(WORK, filter="data")
    data_dir = WORK / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    eval_cases = load_jsonl(data_dir / "eval.jsonl")

    output_dir = WORK / "run"
    config_path = WORK / "training-config.yaml"
    config = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "train_files": [str(data_dir / "train.jsonl")],
        "validation_files": [str(data_dir / "validation.jsonl")],
        "output_dir": str(output_dir),
        "sequence_length": 2048,
        "precision": "fp16",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "epochs": 1.0,
        "max_steps": None,
        "logging_steps": 10,
        "evaluation_steps": 100,
        "save_steps": 100,
        "save_total_limit": 2,
        "seed": 1337,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "device": "cuda",
        "resume_from_checkpoint": None,
        "assistant_mask_scope": "last",
        "quantization": {
            "enabled": True,
            "quant_type": "nf4",
            "double_quantization": True,
        },
        "adapter": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(
        f"Starting QLoRA training on {manifest['counts']['train']} trajectories...",
        flush=True,
    )
    training_metrics = train_lora(config_path)
    print("Training complete; loading base model for held-out evaluation.", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    adapter_dir = output_dir / "final-adapter"
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        revision=MODEL_REVISION,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.eval()
    base_evaluation = evaluate_model(base, tokenizer, eval_cases, "base")
    adapter = PeftModel.from_pretrained(base, adapter_dir)
    adapter.eval()
    adapter_evaluation = evaluate_model(adapter, tokenizer, eval_cases, "adapter")

    gate = {
        "valid_rate_at_least_0_90": adapter_evaluation["valid_rate"] >= 0.90,
        "semantic_rate_at_least_0_80": adapter_evaluation["semantic_rate"] >= 0.80,
        "not_materially_worse_than_base": (
            adapter_evaluation["semantic_rate"]
            >= base_evaluation["semantic_rate"] - 0.05
        ),
    }
    passed = all(gate.values())
    report = {
        "status": "passed" if passed else "failed",
        "release": "v0.0.1-alpha",
        "base_model": MODEL_NAME,
        "base_model_revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "dataset": manifest,
        "training": training_metrics,
        "base_evaluation": base_evaluation,
        "adapter_evaluation": adapter_evaluation,
        "publication_gate": gate,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
        "wheel_sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
        "data_archive_sha256": hashlib.sha256(DATA_ARCHIVE.read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = Path("/content/nimora-agent-alpha-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config_output = Path("/content/nimora-agent-alpha-training-config.yaml")
    config_output.write_text(config_path.read_text(), encoding="utf-8")

    artifact_path = Path("/content/nimora-agent-alpha-adapter.tar.gz")
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(adapter_dir, arcname="adapter")
        archive.add(config_path, arcname="training-config.yaml")
        archive.add(data_dir / "manifest.json", arcname="dataset-manifest.json")

    summary = {
        "status": report["status"],
        "gpu": report["gpu"],
        "train_loss": training_metrics["train"].get("train_loss"),
        "eval_loss": training_metrics["evaluation"].get("eval_loss"),
        "base_valid_rate": base_evaluation["valid_rate"],
        "base_semantic_rate": base_evaluation["semantic_rate"],
        "adapter_valid_rate": adapter_evaluation["valid_rate"],
        "adapter_semantic_rate": adapter_evaluation["semantic_rate"],
        "adapter_arguments_exact_rate": adapter_evaluation["arguments_exact_rate"],
        "gate": gate,
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))
    print(f"NIMORA_ALPHA_REPORT={report_path}")
    print(f"NIMORA_ALPHA_ADAPTER={artifact_path}")
    if not passed:
        raise RuntimeError("Nimora alpha did not pass its publication gate")


if __name__ == "__main__":
    run()
