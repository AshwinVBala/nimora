from __future__ import annotations

import json
from dataclasses import dataclass

from nimora.agent.recording import TrajectoryRecorder
from nimora.agent.tools import ToolRegistry
from nimora.agent.types import JsonObject, ModelBackend


DEFAULT_SYSTEM_PROMPT = """You are Nimora, a careful software-engineering partner.
Inspect evidence before editing. Make small changes, verify them, and report uncertainty.
Never claim a test, Git operation, remote check, review, approval, or merge occurred unless
its tool observation proves it. Respect every permission denial.
"""


@dataclass(frozen=True, slots=True)
class RunResult:
    status: str
    result: str
    steps: int
    messages: list[JsonObject]


class AgentRuntime:
    def __init__(
        self,
        backend: ModelBackend,
        tools: ToolRegistry,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.policy = tools.policy
        self.recorder = recorder or TrajectoryRecorder(None)

    def run(self, task: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> RunResult:
        messages: list[JsonObject] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        self.recorder.begin(system_prompt, task)
        observations = 0
        for step in range(1, self.policy.max_steps + 1):
            try:
                decision = self.backend.decide(
                    self._context(messages),
                    self.tools.specs,
                )
            except Exception as error:
                result = f"Backend failed: {type(error).__name__}: {error}"
                self.recorder.finish(result, "backend_error", step - 1)
                return RunResult("backend_error", result, step - 1, messages)

            if decision.plan:
                plan_message = {
                    "role": "assistant",
                    "content": json.dumps({"plan": decision.plan}, sort_keys=True),
                }
                messages.append(plan_message)
                self.recorder.plan(decision.plan)
            if decision.result is not None:
                if self.policy.require_evidence_before_completion and observations == 0:
                    result = "Completion rejected because no tool evidence was collected."
                    self.recorder.finish(result, "insufficient_evidence", step)
                    return RunResult("insufficient_evidence", result, step, messages)
                messages.append({"role": "assistant", "content": decision.result})
                self.recorder.finish(decision.result, "completed", step)
                return RunResult("completed", decision.result, step, messages)

            assert decision.action is not None
            action = decision.action
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"action": action.to_dict()}, sort_keys=True),
                }
            )
            self.recorder.action(action)
            observation = self.tools.execute(action)
            observations += int(observation.ok)
            observation_content = json.dumps(observation.to_dict(), sort_keys=True)
            if len(observation_content) > self.policy.max_observation_chars:
                observation_content = (
                    observation_content[: self.policy.max_observation_chars]
                    + "\n<observation truncated>"
                )
            messages.append(
                {"role": "tool", "name": action.name, "content": observation_content}
            )
            self.recorder.observation(action.name, observation)

        result = f"Stopped after reaching the {self.policy.max_steps}-step limit."
        self.recorder.finish(result, "step_limit", self.policy.max_steps)
        return RunResult("step_limit", result, self.policy.max_steps, messages)

    def _context(self, messages: list[JsonObject]) -> list[JsonObject]:
        if sum(len(str(message)) for message in messages) <= self.policy.max_context_chars:
            return messages
        fixed = messages[:2]
        remaining = self.policy.max_context_chars - sum(len(str(item)) for item in fixed)
        selected: list[JsonObject] = []
        for message in reversed(messages[2:]):
            size = len(str(message))
            if size > remaining:
                break
            selected.append(message)
            remaining -= size
        notice = {
            "role": "system",
            "content": "Earlier tool turns were dropped to fit the context budget.",
        }
        return [*fixed, notice, *reversed(selected)]
