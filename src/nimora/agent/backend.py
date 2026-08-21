from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from nimora.agent.types import Decision, JsonObject, ProtocolError, ToolSpec


RUNTIME_INSTRUCTION = """You are the Nimora coding runtime controller.
Choose one small, verifiable action at a time. Never claim success without evidence.
Use a provided tool or finish with a JSON object shaped as {"result":"..."}.
When returning JSON directly, an action is shaped as
{"plan":"brief reason","action":{"name":"tool.name","arguments":{...}}}.
Never invent tool output, repository state, checks, revisions, or approvals.
"""


class ScriptedBackend:
    """Deterministic backend for tests and evaluation fixtures."""

    def __init__(self, decisions: Iterable[Decision | JsonObject]) -> None:
        self._decisions = iter(decisions)

    def decide(self, messages: list[JsonObject], tools: list[ToolSpec]) -> Decision:
        del messages, tools
        value = next(self._decisions)
        return value if isinstance(value, Decision) else Decision.from_value(value)


class OpenAICompatibleBackend:
    """Small HTTP backend compatible with vLLM, llama.cpp, and chat APIs."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 300,
        temperature: float = 0.0,
        max_tokens: int = 2_048,
    ) -> None:
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_environment(
        cls,
        endpoint: str,
        model: str,
        api_key_environment: str | None,
        **values: Any,
    ) -> OpenAICompatibleBackend:
        key = os.environ.get(api_key_environment) if api_key_environment else None
        if api_key_environment and not key:
            raise ValueError(f"API key environment variable is unset: {api_key_environment}")
        return cls(endpoint, model, key, **values)

    def decide(self, messages: list[JsonObject], tools: list[ToolSpec]) -> Decision:
        payload = {
            "model": self.model,
            "messages": self._portable_messages(messages),
            "tools": [tool.to_openai_tool() for tool in tools],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise RuntimeError("model endpoint response exceeded 10 MB")
                body = json.loads(raw)
        except urllib.error.HTTPError as error:
            detail = error.read(4_096).decode("utf-8", errors="replace")
            raise RuntimeError(f"model endpoint returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"model endpoint request failed: {error.reason}") from error
        wire_names = {tool.wire_name: tool.name for tool in tools}
        return self._parse_response(body, wire_names)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _portable_messages(messages: list[JsonObject]) -> list[JsonObject]:
        portable: list[JsonObject] = [{"role": "system", "content": RUNTIME_INSTRUCTION}]
        for message in messages:
            role = message.get("role")
            if role in {"system", "user", "assistant"}:
                portable.append({"role": role, "content": str(message.get("content", ""))})
            elif role == "tool":
                portable.append(
                    {
                        "role": "user",
                        "content": (
                            f"OBSERVATION[{message.get('name', 'tool')}]\n"
                            f"{message.get('content', '')}"
                        ),
                    }
                )
        return portable

    @staticmethod
    def _parse_response(body: Any, wire_names: dict[str, str]) -> Decision:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProtocolError("model endpoint response has no assistant message") from error
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) != 1:
                raise ProtocolError("runtime accepts exactly one tool call per turn")
            function = tool_calls[0].get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError as error:
                raise ProtocolError("tool-call arguments are not valid JSON") from error
            wire_name = function.get("name")
            name = wire_names.get(str(wire_name), str(wire_name))
            return Decision.from_value(
                {
                    "plan": message.get("content") or None,
                    "action": {"name": name, "arguments": arguments},
                }
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProtocolError("assistant returned neither a tool call nor content")
        try:
            decision = Decision.from_value(content)
            if decision.action and decision.action.name in wire_names:
                return Decision(
                    plan=decision.plan,
                    action=type(decision.action)(
                        wire_names[decision.action.name], decision.action.arguments
                    ),
                )
            return decision
        except ProtocolError:
            return Decision(result=content.strip())
