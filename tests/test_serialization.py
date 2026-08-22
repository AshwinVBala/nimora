import pytest

from nimora.serialization import (
    canonical_json,
    canonicalize_trajectory_messages,
    serialize_messages,
)
from nimora.train_lora import normalize_messages


def test_canonical_json_is_stable():
    assert canonical_json({"z": 1, "a": {"b": True}}) == '{"a":{"b":true},"z":1}'


def test_trajectory_masks_only_assistant_segments():
    messages = [
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "action": {"name": "search", "arguments": {}}},
        {"role": "tool", "name": "search", "content": "one result"},
        {"role": "assistant", "channel": "result", "content": "Done."},
    ]
    segments = serialize_messages(messages)
    trainable = [text for text, learns in segments if learns]
    frozen = [text for text, learns in segments if not learns]
    assert len(trainable) == 2
    assert "<|action|>" in trainable[0]
    assert "<|result|>" in trainable[1]
    assert any("<|user|>" in text for text in frozen)
    assert any("<|observation|>" in text for text in frozen)


def test_legacy_plan_and_action_become_one_canonical_decision():
    messages = [
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "channel": "plan", "content": "Inspect first."},
        {
            "role": "assistant",
            "action": {"name": "workspace.search", "arguments": {"query": "bug"}},
        },
    ]

    canonical = canonicalize_trajectory_messages(messages)
    assert canonical[1] == {
        "role": "assistant",
        "decision": {
            "plan": "Inspect first.",
            "action": {"name": "workspace.search", "arguments": {"query": "bug"}},
        },
    }
    normalized = normalize_messages(messages)
    assert normalized[1]["content"] == (
        '{"action":{"arguments":{"query":"bug"},"name":"workspace.search"},'
        '"plan":"Inspect first."}'
    )


def test_result_is_serialized_as_runtime_decision_json():
    normalized = normalize_messages(
        [{"role": "assistant", "channel": "result", "content": "Done."}]
    )
    assert normalized == [{"role": "assistant", "content": '{"result":"Done."}'}]


def test_incomplete_plan_is_rejected():
    with pytest.raises(ValueError, match="incomplete assistant plan"):
        canonicalize_trajectory_messages(
            [{"role": "assistant", "channel": "plan", "content": "Soon."}]
        )


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="Unsupported trajectory role"):
        serialize_messages([{"role": "admin", "content": "no"}])
