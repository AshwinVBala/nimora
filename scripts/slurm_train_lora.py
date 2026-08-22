"""Resolve a reproducible SLURM LoRA run and invoke Nimora's trainer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def git_metadata(root: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if (
            suffix.isdigit()
            and path.is_dir()
            and (path / "trainer_state.json").is_file()
        ):
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    metadata = git_metadata(root)
    if metadata["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Refusing a non-reproducible dirty-tree run; commit changes or pass --allow-dirty"
        )

    config_path = args.config.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("LoRA configuration must be a YAML object")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "final-adapter").is_dir():
        raise RuntimeError(f"Run is already complete: {run_dir}")
    raw["output_dir"] = str(run_dir)

    if args.resume == "auto":
        checkpoint = latest_checkpoint(run_dir)
        raw["resume_from_checkpoint"] = str(checkpoint) if checkpoint else None
    elif args.resume == "none":
        raw["resume_from_checkpoint"] = None
    else:
        checkpoint = Path(args.resume).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        raw["resume_from_checkpoint"] = str(checkpoint)

    resolved_config = run_dir / "slurm-resolved-config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "git": metadata,
        "source_config": str(config_path),
        "resolved_config": str(resolved_config),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
                "SLURM_GPUS",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
            )
        },
    }
    atomic_json(run_dir / "provenance.json", provenance)

    from nimora.train_lora import train_lora

    metrics = train_lora(resolved_config)
    atomic_json(run_dir / "slurm-training-metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
