import hashlib

import pytest

from nimora.agent.backend import ScriptedBackend
from nimora.agent.loop import AgentRuntime
from nimora.agent.policy import RuntimePolicy
from nimora.agent.tools import build_local_tools
from nimora.agent.types import Decision, ProtocolError
from nimora.agent.workspace import WorkspaceBoundary


def test_decision_requires_exactly_one_terminal_shape():
    with pytest.raises(ProtocolError, match="exactly one"):
        Decision.from_value({"plan": "nothing"})
    with pytest.raises(ProtocolError, match="exactly one"):
        Decision.from_value(
            {
                "action": {"name": "workspace.list", "arguments": {}},
                "result": "done",
            }
        )


def test_workspace_write_requires_current_hash(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("old\n", encoding="utf-8")
    workspace = WorkspaceBoundary(tmp_path)
    digest = hashlib.sha256(b"old\n").hexdigest()
    result = workspace.write_file("module.py", "new\n", digest)
    assert result["sha256"] == hashlib.sha256(b"new\n").hexdigest()
    with pytest.raises(ValueError, match="changed since"):
        workspace.write_file("module.py", "again\n", digest)


def test_workspace_replace_requires_exact_match_count(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("one\none\n", encoding="utf-8")
    workspace = WorkspaceBoundary(tmp_path)
    digest = hashlib.sha256(b"one\none\n").hexdigest()
    with pytest.raises(ValueError, match="found 2"):
        workspace.replace_text("module.py", "one", "two", digest)
    workspace.replace_text("module.py", "one", "two", digest, expected_occurrences=2)
    assert target.read_text(encoding="utf-8") == "two\ntwo\n"


def test_workspace_refuses_sensitive_reads(tmp_path):
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    workspace = WorkspaceBoundary(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        workspace.read_file(".env")
    assert workspace.search("secret")["matches"] == []


def test_runtime_collects_evidence_before_completion(tmp_path):
    (tmp_path / "README.md").write_text("evidence\n", encoding="utf-8")
    policy = RuntimePolicy(max_steps=2)
    tools = build_local_tools(WorkspaceBoundary(tmp_path), policy)
    backend = ScriptedBackend(
        [
            {
                "action": {
                    "name": "workspace.read",
                    "arguments": {"path": "README.md"},
                }
            },
            {"result": "Verified the requested file."},
        ]
    )
    result = AgentRuntime(backend, tools).run("Inspect the README.")
    assert result.status == "completed"
    assert result.steps == 2


def test_runtime_rejects_unsupported_completion_without_evidence(tmp_path):
    policy = RuntimePolicy(max_steps=1)
    tools = build_local_tools(WorkspaceBoundary(tmp_path), policy)
    result = AgentRuntime(ScriptedBackend([{"result": "Done."}]), tools).run("Fix it")
    assert result.status == "insufficient_evidence"


def test_tool_wire_names_are_api_safe(tmp_path):
    tools = build_local_tools(WorkspaceBoundary(tmp_path), RuntimePolicy())
    wire_names = [tool.to_openai_tool()["function"]["name"] for tool in tools.specs]
    assert "workspace__read" in wire_names
    assert all("." not in name for name in wire_names)
