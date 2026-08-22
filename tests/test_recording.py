import json

from nimora.agent.recording import TrajectoryRecorder
from nimora.agent.types import Action, ToolResult


def test_recorder_writes_complete_decisions(tmp_path):
    output = tmp_path / "trajectory.jsonl"
    recorder = TrajectoryRecorder(output)
    recorder.begin("System", "Task")
    recorder.plan("Inspect first.")
    recorder.action(Action("workspace.read", {"path": "README.md"}))
    recorder.observation("workspace.read", ToolResult(True, {"content": "hello"}))
    recorder.finish("Done.", "completed", 1)

    record = json.loads(output.read_text())
    assert record["messages"][2] == {
        "role": "assistant",
        "decision": {
            "plan": "Inspect first.",
            "action": {"name": "workspace.read", "arguments": {"path": "README.md"}},
        },
    }
    assert record["messages"][-1] == {
        "role": "assistant",
        "decision": {"result": "Done."},
    }
