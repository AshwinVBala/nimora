import json

import pytest

from nimora.serialization import canonical_json, serialize_messages


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


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="Unsupported trajectory role"):
        serialize_messages([{"role": "admin", "content": "no"}])

