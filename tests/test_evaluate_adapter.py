from __future__ import annotations

import importlib.util
from pathlib import Path


def load_evaluator():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_adapter.py"
    spec = importlib.util.spec_from_file_location("nimora_evaluate_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_output_accepts_exact_action() -> None:
    evaluator = load_evaluator()
    expected = {
        "plan": "Inspect the repository.",
        "action": {"name": "shell", "arguments": {"command": "git status"}},
    }
    actual = (
        '{"plan":"Check it.","action":{"name":"shell",'
        '"arguments":{"command":"git status"}}}'
    )

    score = evaluator.score_output(actual, expected)

    assert score["valid"] is True
    assert score["semantic"] is True
    assert score["arguments_exact"] is True


def test_score_output_distinguishes_tool_and_argument_errors() -> None:
    evaluator = load_evaluator()
    expected = {
        "action": {"name": "read_file", "arguments": {"path": "README.md"}}
    }

    wrong_arguments = evaluator.score_output(
        '{"action":{"name":"read_file","arguments":{"path":"LICENSE"}}}',
        expected,
    )
    wrong_tool = evaluator.score_output(
        '{"action":{"name":"shell","arguments":{"path":"README.md"}}}',
        expected,
    )

    assert wrong_arguments["semantic"] is True
    assert wrong_arguments["arguments_exact"] is False
    assert wrong_tool["semantic"] is False
    assert wrong_tool["arguments_exact"] is False


def test_score_output_handles_result_and_invalid_json() -> None:
    evaluator = load_evaluator()

    result = evaluator.score_output('{"result":"Done."}', {"result": "Complete."})
    invalid = evaluator.score_output("not json", {"result": "Complete."})

    assert result["valid"] is True
    assert result["semantic"] is True
    assert invalid["valid"] is False
    assert invalid["semantic"] is False
