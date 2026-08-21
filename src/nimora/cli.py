from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nimora", description="Nimora model-training toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("runtime-info", help="Print environment metadata only")
    info.set_defaults(handler=_runtime_info)

    tokenizer = subparsers.add_parser("train-tokenizer", help="Train the controller tokenizer")
    tokenizer.add_argument("--input", action="append", required=True)
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=32_768)
    tokenizer.add_argument("--minimum-frequency", type=int, default=2)
    tokenizer.set_defaults(handler=_train_tokenizer)

    prepare = subparsers.add_parser("prepare", help="Pack JSONL training records")
    prepare.add_argument("--input", action="append", required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--shard-tokens", type=int, default=10_000_000)
    prepare.set_defaults(handler=_prepare)

    estimate = subparsers.add_parser("estimate", help="Estimate controller parameter memory")
    estimate.add_argument("--config", required=True)
    estimate.set_defaults(handler=_estimate)

    controller = subparsers.add_parser("train-controller", help="Pretrain Nimora Controller")
    controller.add_argument("--config", required=True)
    controller.set_defaults(handler=_train_controller)

    lora = subparsers.add_parser("train-lora", help="Post-train Qwen with LoRA")
    lora.add_argument("--config", required=True)
    lora.set_defaults(handler=_train_lora)
    return parser


def _runtime_info(_args) -> None:
    from dataclasses import asdict

    from nimora.runtime import runtime_info

    print(json.dumps(asdict(runtime_info()), indent=2))


def _train_tokenizer(args) -> None:
    from nimora.tokenizer import train_tokenizer

    output = train_tokenizer(
        args.input, args.output, args.vocab_size, args.minimum_frequency
    )
    print(output)


def _prepare(args) -> None:
    from nimora.data import prepare_dataset

    metadata = prepare_dataset(
        args.input, args.tokenizer, args.output_dir, args.shard_tokens
    )
    print(json.dumps(metadata, indent=2))


def _estimate(args) -> None:
    from nimora.config import load_controller_config
    from nimora.sizing import estimate_model_size

    config = load_controller_config(Path(args.config))
    print(json.dumps(estimate_model_size(config.model), indent=2))


def _train_controller(args) -> None:
    from nimora.train_controller import train

    train(args.config)


def _train_lora(args) -> None:
    from nimora.train_lora import train_lora

    train_lora(args.config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
