from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|plan|>",
    "<|action|>",
    "<|result|>",
    "<|turn_end|>",
]


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|observation|>",
    "observation": "<|observation|>",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def message_content(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "assistant" and "action" in message:
        return f"<|action|>\n{canonical_json(message['action'])}"
    if role in {"tool", "observation"}:
        name = message.get("name", "environment")
        content = message.get("content", "")
        return f"name={name}\n{content}"
    if role == "assistant" and message.get("channel") == "plan":
        return f"<|plan|>\n{message.get('content', '')}"
    if role == "assistant" and message.get("channel") == "result":
        return f"<|result|>\n{message.get('content', '')}"
    return str(message.get("content", ""))


def serialize_messages(messages: Iterable[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Serialize a trajectory into text segments and assistant-loss flags."""
    segments: list[tuple[str, bool]] = [("<|bos|>", False)]
    for message in messages:
        role = str(message.get("role", ""))
        if role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported trajectory role: {role!r}")
        text = f"{ROLE_TOKENS[role]}\n{message_content(message)}\n<|turn_end|>\n"
        segments.append((text, role == "assistant"))
    segments.append(("<|eos|>", False))
    return segments


def iter_jsonl(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}")
                yield record


def record_text(record: dict[str, Any]) -> str:
    if "text" in record:
        return str(record["text"])
    if "messages" in record:
        return "".join(text for text, _ in serialize_messages(record["messages"]))
    raise ValueError("Each record must contain either 'text' or 'messages'")


def iter_training_text(paths: Iterable[str | Path]) -> Iterator[str]:
    for record in iter_jsonl(paths):
        yield record_text(record)

