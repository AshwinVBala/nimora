"""Provider-neutral agent runtime for Nimora."""

from typing import TYPE_CHECKING, Any

from nimora.agent.policy import RuntimePolicy
from nimora.agent.types import Action, Decision, ToolResult, ToolSpec

if TYPE_CHECKING:
    from nimora.agent.loop import AgentRuntime, RunResult

__all__ = [
    "Action",
    "AgentRuntime",
    "Decision",
    "RunResult",
    "RuntimePolicy",
    "ToolResult",
    "ToolSpec",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentRuntime", "RunResult"}:
        from nimora.agent.loop import AgentRuntime, RunResult

        return {"AgentRuntime": AgentRuntime, "RunResult": RunResult}[name]
    raise AttributeError(name)
