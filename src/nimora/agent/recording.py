from __future__ import annotations

import fcntl
import json
import re
from pathlib import Path
from typing import Any

from nimora.agent.types import Action, JsonObject, ToolResult


REDACTIONS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END[^-]+-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,255}\b"),
    re.compile(r"\b(?:hf_|sk-)[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
]


def _sanitize(value: Any, maximum_string: int = 50_000) -> Any:
    if isinstance(value, str):
        result = value[:maximum_string]
        for pattern in REDACTIONS:
            result = pattern.sub("<REDACTED>", result)
        return result
    if isinstance(value, dict):
        return {str(key): _sanitize(item, maximum_string) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, maximum_string) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class TrajectoryRecorder:
    """Opt-in JSONL recorder with secret-pattern redaction and append locking."""

    def __init__(self, output: str | Path | None) -> None:
        self.output = Path(output) if output is not None else None
        self.messages: list[JsonObject] = []

    def begin(self, system_prompt: str, task: str) -> None:
        self.messages = [
            {"role": "system", "content": _sanitize(system_prompt)},
            {"role": "user", "content": _sanitize(task)},
        ]

    def plan(self, content: str) -> None:
        self.messages.append(
            {"role": "assistant", "channel": "plan", "content": _sanitize(content)}
        )

    def action(self, action: Action) -> None:
        self.messages.append({"role": "assistant", "action": _sanitize(action.to_dict())})

    def observation(self, name: str, result: ToolResult) -> None:
        self.messages.append(
            {
                "role": "tool",
                "name": name,
                "content": json.dumps(_sanitize(result.to_dict()), sort_keys=True),
            }
        )

    def finish(self, result: str, status: str, steps: int) -> None:
        self.messages.append(
            {"role": "assistant", "channel": "result", "content": _sanitize(result)}
        )
        if self.output is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "messages": self.messages,
            "metadata": {"runtime": "nimora", "status": status, "steps": steps},
        }
        with self.output.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

