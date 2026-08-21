from nimora.config import ModelConfig
from nimora.sizing import estimate_model_size


def test_static_estimate_is_positive():
    estimate = estimate_model_size(ModelConfig())
    assert estimate["parameters"] > 0
    assert estimate["adamw_training_state_gib"] > estimate["fp16_weights_gib"]
