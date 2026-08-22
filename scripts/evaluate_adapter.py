"""Evaluate a Nimora PEFT adapter against held-out next-decision cases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if not isinstance(value.get("messages"), list):
                raise ValueError(f"Missing messages array at {path}:{line_number}")
            if not isinstance(value.get("expected"), dict):
                raise ValueError(f"Missing expected decision at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Evaluation file is empty: {path}")
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


def evaluate_model(model, tokenizer, cases, label: str, max_new_tokens: int):
    import torch

    from nimora.train_lora import normalize_messages

    results: list[dict[str, Any]] = []
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        result = score_output(raw, case["expected"])
        result["family"] = case.get("metadata", {}).get("family")
        result["expected"] = case["expected"]
        results.append(result)
        if index % 10 == 0 or index == len(cases):
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--eval-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--min-valid-rate", type=float, default=0.90)
    parser.add_argument("--min-semantic-rate", type=float, default=0.80)
    parser.add_argument("--base-tolerance", type=float, default=0.05)
    parser.add_argument("--quantization", choices=("none", "4bit"), default="4bit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()

    import accelerate
    import peft
    import torch
    import transformers
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU")
    adapter_path = args.adapter.resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = args.base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise ValueError("Base model is missing; pass --base-model")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model_kwargs: dict[str, Any] = {
        "device_map": {"": 0},
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    cases = load_jsonl(args.eval_file.resolve())
    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    base.eval()
    base_evaluation = evaluate_model(
        base, tokenizer, cases, "base", args.max_new_tokens
    )
    adapter = PeftModel.from_pretrained(base, adapter_path)
    adapter.eval()
    adapter_evaluation = evaluate_model(
        adapter, tokenizer, cases, "adapter", args.max_new_tokens
    )

    gate = {
        "valid_rate": adapter_evaluation["valid_rate"] >= args.min_valid_rate,
        "semantic_rate": (
            adapter_evaluation["semantic_rate"] >= args.min_semantic_rate
        ),
        "not_materially_worse_than_base": (
            adapter_evaluation["semantic_rate"]
            >= base_evaluation["semantic_rate"] - args.base_tolerance
        ),
    }
    passed = all(gate.values())
    report = {
        "status": "passed" if passed else "failed",
        "base_model": base_model,
        "base_model_revision": args.model_revision,
        "adapter": str(adapter_path),
        "eval_file": str(args.eval_file.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
        "base_evaluation": base_evaluation,
        "adapter_evaluation": adapter_evaluation,
        "publication_gate": gate,
        "thresholds": {
            "min_valid_rate": args.min_valid_rate,
            "min_semantic_rate": args.min_semantic_rate,
            "base_tolerance": args.base_tolerance,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "base_semantic_rate": base_evaluation["semantic_rate"],
                "adapter_valid_rate": adapter_evaluation["valid_rate"],
                "adapter_semantic_rate": adapter_evaluation["semantic_rate"],
                "adapter_arguments_exact_rate": adapter_evaluation[
                    "arguments_exact_rate"
                ],
                "publication_gate": gate,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
