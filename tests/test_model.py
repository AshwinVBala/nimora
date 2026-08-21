import pytest

torch = pytest.importorskip("torch")

from nimora.config import ModelConfig
from nimora.model import NimoraController


def tiny_config():
    return ModelConfig(
        vocab_size=256,
        max_sequence_length=32,
        num_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        gradient_checkpointing=False,
    )


def test_forward_and_masked_loss_on_cpu():
    model = NimoraController(tiny_config())
    inputs = torch.randint(0, 256, (2, 16))
    labels = inputs.clone()
    labels[:, :8] = -100
    result = model(inputs, labels=labels)
    assert result["logits"].shape == (2, 16, 256)
    assert torch.isfinite(result["loss"])
