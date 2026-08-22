import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from nimora.agent.types import Decision

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_alpha_trajectories.py"


def _module():
    spec = spec_from_file_location("build_alpha_trajectories", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alpha_dataset_is_valid_and_split_disjoint(tmp_path):
    module = _module()
    manifest = module.build_dataset(
        tmp_path,
        train_per_family=2,
        validation_per_family=1,
        eval_per_family=1,
    )
    family_count = len(module.FAMILIES)
    assert manifest["counts"] == {
        "train": family_count * 2,
        "validation": family_count,
        "eval": family_count,
    }

    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line)
        for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    evaluation = [
        json.loads(line) for line in (tmp_path / "eval.jsonl").read_text().splitlines()
    ]
    assert {row["metadata"]["index"] for row in train}.isdisjoint(
        {row["metadata"]["index"] for row in validation}
    )
    for row in [*train, *validation]:
        assert row["messages"][-1]["role"] == "assistant"
        Decision.from_value(row["messages"][-1]["decision"])
        assert row["metadata"]["supervision"] == "prefix_to_next_decision"
        assert 0 <= row["metadata"]["target_assistant_ordinal"] < (
            row["metadata"]["trajectory_assistant_count"]
        )
    for case in evaluation:
        Decision.from_value(case["expected"])
        assert case["messages"][-1]["role"] != "assistant" or (
            case["messages"][-1]["decision"] != case["expected"]
        )
