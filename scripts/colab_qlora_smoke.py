"""Bounded Qwen3-4B QLoRA pipeline smoke test for a Colab T4.

This script is sent to a running Colab session with ``colab exec``. The local
Nimora wheel must first be uploaded to ``/content/nimora-0.1.0-py3-none-any.whl``.
It intentionally runs only five optimizer steps and writes downloadable
artifacts under ``/content``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WHEEL = Path("/content/nimora-0.1.0-py3-none-any.whl")
WORK = Path("/content/nimora-qlora-smoke")
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_STEPS = 10
RUNTIME_INSTRUCTION = """You are the Nimora coding runtime controller.
Choose one small, verifiable action at a time. Never claim success without evidence.
Use a provided tool or finish with a JSON object shaped as {"result":"..."}.
When returning JSON directly, an action is shaped as
{"plan":"brief reason","action":{"name":"tool.name","arguments":{...}}}.
Never invent tool output, repository state, checks, revisions, or approvals.
"""


def install_dependencies() -> None:
    if not WHEEL.is_file():
        raise FileNotFoundError(f"Upload the Nimora wheel before running: {WHEEL}")
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


def trajectory(task: str, messages: list[dict]) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": RUNTIME_INSTRUCTION,
            },
            {"role": "user", "content": task},
            *messages,
        ],
        "metadata": {"source": "synthetic-smoke", "reviewed": True},
    }


def smoke_records() -> tuple[list[dict], list[dict]]:
    train = [
        trajectory(
            "Find the parser implementation before making changes.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Locate the implementation with a narrow search.",
                        "action": {
                            "name": "workspace.search",
                            "arguments": {"query": "def parse", "path": "src"},
                        },
                    },
                }
            ],
        ),
        trajectory(
            "Inspect README.md before editing it.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Read the current file and capture its hash.",
                        "action": {
                            "name": "workspace.read",
                            "arguments": {"path": "README.md"},
                        },
                    },
                }
            ],
        ),
        trajectory(
            "The patch is applied. Verify the parser tests.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Run the smallest relevant test command.",
                        "action": {
                            "name": "shell.run",
                            "arguments": {
                                "argv": ["pytest", "-q", "tests/test_parser.py"]
                            },
                        },
                    },
                }
            ],
        ),
        trajectory(
            "Open a pull request for the already-pushed branch fix/parser.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Open a focused change against main.",
                        "action": {
                            "name": "provider.open_change",
                            "arguments": {
                                "title": "Fix parser regression",
                                "body": "Fixes empty-input parsing and adds coverage.",
                                "head_branch": "fix/parser",
                                "base_branch": "main",
                            },
                        },
                    },
                }
            ],
        ),
        trajectory(
            "Review pull request 42 before considering approval.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Bind the review to the current remote revision first.",
                        "action": {
                            "name": "provider.get_change",
                            "arguments": {"number": 42},
                        },
                    },
                }
            ],
        ),
        trajectory(
            "The relevant tests passed and the requested change is complete.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "result": (
                            "Implemented the requested change and verified the relevant "
                            "tests pass."
                        )
                    },
                }
            ],
        ),
    ]
    validation = [
        trajectory(
            "Check the current repository revision before creating a branch.",
            [
                {
                    "role": "assistant",
                    "decision": {
                        "plan": "Read HEAD so later mutation can be revision-bound.",
                        "action": {"name": "git.revision", "arguments": {}},
                    },
                }
            ],
        )
    ]
    return train, validation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run() -> None:
    started = time.time()
    install_dependencies()

    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from nimora.agent.types import Decision
    from nimora.train_lora import train_lora

    if not torch.cuda.is_available():
        raise RuntimeError("The Colab smoke test requires a GPU runtime")

    WORK.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = smoke_records()
    train_path = WORK / "train.jsonl"
    validation_path = WORK / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    output_dir = WORK / "run"
    config_path = WORK / "config.yaml"
    config = {
        "model_name": MODEL_NAME,
        "train_files": [str(train_path)],
        "validation_files": [str(validation_path)],
        "output_dir": str(output_dir),
        "sequence_length": 512,
        "precision": "fp16",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "epochs": 1.0,
        "max_steps": MAX_STEPS,
        "logging_steps": 1,
        "evaluation_steps": MAX_STEPS,
        "save_steps": MAX_STEPS,
        "save_total_limit": 1,
        "seed": 1337,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "device": "cuda",
        "resume_from_checkpoint": None,
        "quantization": {
            "enabled": True,
            "quant_type": "nf4",
            "double_quantization": True,
        },
        "adapter": {
            "rank": 8,
            "alpha": 16,
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

    train_lora(config_path)
    gc.collect()
    torch.cuda.empty_cache()

    adapter_dir = output_dir / "final-adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": RUNTIME_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    "Inspect pyproject.toml before proposing a dependency change. The "
                    "available tool is workspace.read, whose arguments object requires "
                    "the string field path. Return the next decision as JSON. No tool "
                    "observation has been received yet."
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    decision_valid = True
    decision_error = None
    parsed_decision = None
    try:
        parsed = Decision.from_value(output)
        parsed_decision = {
            "plan": parsed.plan,
            "action": parsed.action.to_dict() if parsed.action else None,
            "result": parsed.result,
        }
    except ValueError as error:
        decision_valid = False
        decision_error = str(error)

    wheel_sha256 = hashlib.sha256(WHEEL.read_bytes()).hexdigest()
    report = {
        "status": "passed" if decision_valid else "failed",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": MODEL_NAME,
        "quantization": "4-bit NF4 double-quantized",
        "optimizer_steps": MAX_STEPS,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "adapter_reload": True,
        "decision_valid": decision_valid,
        "decision": parsed_decision,
        "decision_error": decision_error,
        "raw_generation": output,
        "wheel_sha256": wheel_sha256,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = Path("/content/nimora-smoke-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    artifact_path = Path("/content/nimora-smoke-adapter.tar.gz")
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(adapter_dir, arcname="nimora-smoke-adapter")

    print(json.dumps(report, indent=2))
    print(f"NIMORA_SMOKE_REPORT={report_path}")
    print(f"NIMORA_SMOKE_ADAPTER={artifact_path}")
    if not decision_valid:
        raise RuntimeError("Adapter reloaded, but inference did not emit a valid Decision")


if __name__ == "__main__":
    run()
