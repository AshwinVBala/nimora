from dataclasses import asdict

import pytest

from nimora.config import ControllerRunConfig, ModelConfig


def test_default_model_dimensions_are_valid():
    config = ControllerRunConfig()
    config.validate()
    assert config.model.hidden_size // config.model.num_attention_heads == 64
    assert asdict(config)["training"]["precision"] == "fp16"


def test_invalid_gqa_ratio_is_rejected():
    config = ModelConfig(num_attention_heads=12, num_key_value_heads=5)
    with pytest.raises(ValueError, match="num_attention_heads"):
        config.validate()


def test_uint16_vocabulary_limit_is_enforced():
    config = ModelConfig(vocab_size=100_000)
    with pytest.raises(ValueError, match="uint16"):
        config.validate()

