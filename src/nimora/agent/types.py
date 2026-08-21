from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


JsonObject = dict[str, Any]


class ProtocolError(ValueError):
    """Raised when a model decision violates the runtime protocol."""


class PolicyDenied(PermissionError):
    """Raised when an action is not authorized by runtime policy."""


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    arguments: JsonObject

    @classmethod
    def from_value(cls, value: Any) -> Action:
        if not isinstance(value, dict):
            raise ProtocolError("action must be an object")
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ProtocolError("action.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ProtocolError("action.arguments must be an object")
        return cls(name.strip(), arguments)

    def to_dict(self) -> JsonObject:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class Decision:
    plan: str | None = None
    action: Action | None = None
    result: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> Decision:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ProtocolError("model output must be a JSON object") from error
        if not isinstance(value, dict):
            raise ProtocolError("model decision must be an object")
        plan_value = value.get("plan")
        result_value = value.get("result")
        plan = None if plan_value is None else str(plan_value)
        result = None if result_value is None else str(result_value)
        action = Action.from_value(value["action"]) if "action" in value else None
        if (action is None) == (result is None):
            raise ProtocolError("decision must contain exactly one of action or result")
        return cls(plan=plan, action=action, result=result)


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> JsonObject:
        value: JsonObject = {"ok": self.ok}
        if self.output is not None:
            value["output"] = self.output
        if self.error is not None:
            value["error"] = self.error
        return value


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonObject

    @property
    def wire_name(self) -> str:
        return self.name.replace(".", "__")

    def to_openai_tool(self) -> JsonObject:
        return {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ModelBackend(Protocol):
    def decide(self, messages: list[JsonObject], tools: list[ToolSpec]) -> Decision: ...
