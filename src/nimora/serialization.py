from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from nimora.agent.types import Decision

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


def decision_dict(value: Any) -> dict[str, Any]:
    """Validate and return the canonical wire representation of a decision."""
    decision = Decision.from_value(value)
    result: dict[str, Any] = {}
    if decision.plan is not None:
        result["plan"] = decision.plan
    if decision.action is not None:
        result["action"] = decision.action.to_dict()
    else:
        result["result"] = decision.result
    return result


def canonicalize_trajectory_messages(
    messages: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coalesce legacy assistant records into one complete decision per turn.

    New trajectories store assistant decisions under ``decision``. The legacy
    plan/action/result representation remains accepted so existing authorized
    trajectories can be migrated without changing their meaning.
    """
    canonical: list[dict[str, Any]] = []
    pending_plan: str | None = None
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            raise ValueError("Trajectory messages must be objects")
        message = dict(raw_message)
        role = message.get("role")
        if role != "assistant":
            if pending_plan is not None:
                raise ValueError("Assistant plan must be followed by an action or result")
            canonical.append(message)
            continue

        channel = message.get("channel")
        if channel == "plan":
            if pending_plan is not None:
                raise ValueError("Consecutive assistant plans are not a complete decision")
            pending_plan = str(message.get("content", ""))
            continue

        if "decision" in message:
            if pending_plan is not None:
                raise ValueError("A canonical decision cannot follow a separate plan")
            value = decision_dict(message["decision"])
        elif "action" in message:
            value = {"action": message["action"]}
            if pending_plan is not None:
                value["plan"] = pending_plan
            value = decision_dict(value)
            pending_plan = None
        elif channel == "result":
            value = {"result": str(message.get("content", ""))}
            if pending_plan is not None:
                value["plan"] = pending_plan
            value = decision_dict(value)
            pending_plan = None
        else:
            content = str(message.get("content", ""))
            try:
                value = decision_dict(content)
            except ValueError:
                value = decision_dict({"result": content})
            if pending_plan is not None:
                if "plan" in value:
                    raise ValueError("Decision contains a plan after a separate plan")
                value["plan"] = pending_plan
                value = decision_dict(value)
                pending_plan = None
        canonical.append({"role": "assistant", "decision": value})

    if pending_plan is not None:
        raise ValueError("Trajectory ends with an incomplete assistant plan")
    return canonical


def message_content(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "assistant" and "decision" in message:
        decision = decision_dict(message["decision"])
        sections = []
        if "plan" in decision:
            sections.append(f"<|plan|>\n{decision['plan']}")
        if "action" in decision:
            sections.append(f"<|action|>\n{canonical_json(decision['action'])}")
        else:
            sections.append(f"<|result|>\n{decision['result']}")
        return "\n".join(sections)
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
    for message in canonicalize_trajectory_messages(messages):
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
