from __future__ import annotations

from nimora.config import ModelConfig


def estimate_model_size(config: ModelConfig) -> dict[str, float | int]:
    """Estimate parameter storage without importing PyTorch or allocating a model."""
    config.validate()
    head_dim = config.hidden_size // config.num_attention_heads
    query = config.hidden_size * config.num_attention_heads * head_dim
    key_value = 2 * config.hidden_size * config.num_key_value_heads * head_dim
    output = config.hidden_size * config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    per_layer = query + key_value + output + mlp + norms
    embeddings = config.vocab_size * config.hidden_size
    untied_head = 0 if config.tie_word_embeddings else embeddings
    total = embeddings + untied_head + (config.num_layers * per_layer) + config.hidden_size
    return {
        "parameters": total,
        "parameters_millions": round(total / 1_000_000, 2),
        "fp16_weights_gib": round(total * 2 / (1024**3), 3),
        "adamw_training_state_gib": round(total * 16 / (1024**3), 3),
    }
