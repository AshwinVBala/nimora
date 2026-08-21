from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nimora.runtime import require_torch
from nimora.serialization import iter_jsonl, serialize_messages
from nimora.tokenizer import load_tokenizer


@dataclass(frozen=True, slots=True)
class ShardInfo:
    tokens_path: Path
    mask_path: Path
    token_count: int


class ShardWriter:
    def __init__(self, output_dir: Path, shard_tokens: int) -> None:
        if shard_tokens < 1:
            raise ValueError("shard_tokens must be positive")
        self.output_dir = output_dir
        self.shard_tokens = shard_tokens
        self.tokens: list[int] = []
        self.mask: list[int] = []
        self.shards: list[dict[str, Any]] = []
        self.index = 0

    def add(self, token_ids: list[int], loss_mask: list[int]) -> None:
        if len(token_ids) != len(loss_mask):
            raise ValueError("token_ids and loss_mask have different lengths")
        offset = 0
        while offset < len(token_ids):
            remaining = self.shard_tokens - len(self.tokens)
            take = min(remaining, len(token_ids) - offset)
            self.tokens.extend(token_ids[offset : offset + take])
            self.mask.extend(loss_mask[offset : offset + take])
            offset += take
            if len(self.tokens) >= self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if not self.tokens:
            return
        stem = f"shard-{self.index:05d}"
        tokens_name = f"{stem}.tokens.bin"
        mask_name = f"{stem}.mask.bin"
        np.asarray(self.tokens, dtype=np.uint16).tofile(self.output_dir / tokens_name)
        np.asarray(self.mask, dtype=np.uint8).tofile(self.output_dir / mask_name)
        self.shards.append(
            {
                "tokens": tokens_name,
                "mask": mask_name,
                "token_count": len(self.tokens),
            }
        )
        self.tokens.clear()
        self.mask.clear()
        self.index += 1


def _encode_record(tokenizer, record: dict[str, Any]) -> tuple[list[int], list[int]]:
    if "text" in record:
        text = f"<|bos|>\n{record['text']}\n<|eos|>"
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        return token_ids, [1] * len(token_ids)
    if "messages" not in record:
        raise ValueError("Each record must contain either 'text' or 'messages'")

    all_tokens: list[int] = []
    all_mask: list[int] = []
    for text, learns in serialize_messages(record["messages"]):
        segment = tokenizer.encode(text, add_special_tokens=False).ids
        all_tokens.extend(segment)
        all_mask.extend([int(learns)] * len(segment))
    return all_tokens, all_mask


def prepare_dataset(
    inputs: list[str | Path],
    tokenizer_path: str | Path,
    output_dir: str | Path,
    shard_tokens: int = 10_000_000,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() > 65_535:
        raise ValueError("Tokenizer vocabulary does not fit the uint16 packed-data format")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(destination, shard_tokens)
    records = 0
    assistant_tokens = 0
    total_tokens = 0
    for record in iter_jsonl(inputs):
        tokens, mask = _encode_record(tokenizer, record)
        if len(tokens) < 2:
            continue
        writer.add(tokens, mask)
        records += 1
        total_tokens += len(tokens)
        assistant_tokens += sum(mask)
    writer.flush()

    metadata = {
        "format_version": 1,
        "token_dtype": "uint16",
        "mask_dtype": "uint8",
        "tokenizer": str(Path(tokenizer_path).resolve()),
        "vocab_size": tokenizer.get_vocab_size(),
        "record_count": records,
        "token_count": total_tokens,
        "supervised_token_count": assistant_tokens,
        "shards": writer.shards,
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def load_shards(directory: str | Path) -> list[ShardInfo]:
    root = Path(directory)
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    shards = [
        ShardInfo(root / item["tokens"], root / item["mask"], int(item["token_count"]))
        for item in metadata["shards"]
    ]
    if not shards:
        raise ValueError(f"No packed shards found in {root}")
    return shards


class PackedTokenDataset:
    """Deterministic random windows over memory-mapped token shards."""

    def __init__(
        self,
        directory: str | Path,
        sequence_length: int,
        samples: int,
        seed: int,
    ) -> None:
        torch = require_torch()
        self._torch = torch
        self.sequence_length = sequence_length
        self.samples = samples
        self.seed = seed
        self.shards = load_shards(directory)
        self.tokens = [
            np.memmap(item.tokens_path, dtype=np.uint16, mode="r")
            for item in self.shards
        ]
        self.masks = [
            np.memmap(item.mask_path, dtype=np.uint8, mode="r")
            for item in self.shards
        ]
        self.usable = [
            max(0, item.token_count - sequence_length) for item in self.shards
        ]
        if sum(self.usable) <= 0:
            raise ValueError("Packed shards are shorter than the configured sequence length")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        choice = rng.randrange(sum(self.usable))
        shard_index = 0
        for shard_index, usable in enumerate(self.usable):
            if choice < usable:
                break
            choice -= usable
        start = choice
        stop = start + self.sequence_length + 1
        token_window = np.asarray(self.tokens[shard_index][start:stop], dtype=np.int64)
        mask_window = np.asarray(self.masks[shard_index][start:stop], dtype=np.bool_)
        input_ids = self._torch.from_numpy(token_window[:-1].copy())
        labels = self._torch.from_numpy(token_window[1:].copy())
        loss_mask = self._torch.from_numpy(mask_window[1:].copy())
        labels = labels.masked_fill(~loss_mask, -100)
        return {"input_ids": input_ids, "labels": labels}
