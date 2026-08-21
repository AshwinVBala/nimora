from __future__ import annotations

from dataclasses import asdict

from nimora.config import ModelConfig
from nimora.runtime import require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional
checkpoint = __import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint


class RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.epsilon)
        return self.weight * normalized.to(input_dtype)


def rotate_half(values):
    left, right = values.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", embeddings.cos(), persistent=False)
        self.register_buffer("sine", embeddings.sin(), persistent=False)

    def forward(self, sequence_length: int, dtype, device):
        cosine = self.cosine[:sequence_length].to(device=device, dtype=dtype)[None, None, :, :]
        sine = self.sine[:sequence_length].to(device=device, dtype=dtype)[None, None, :, :]
        return cosine, sine


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.groups = self.num_heads // self.num_key_value_heads
        self.dropout = config.attention_dropout

        self.query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, tensor, heads: int):
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, cosine, sine):
        query = self._shape(self.query(hidden_states), self.num_heads)
        key = self._shape(self.key(hidden_states), self.num_key_value_heads)
        value = self._shape(self.value(hidden_states), self.num_key_value_heads)

        query = (query * cosine) + (rotate_half(query) * sine)
        key = (key * cosine) + (rotate_half(key) * sine)

        if self.groups > 1:
            key = key.repeat_interleave(self.groups, dim=1)
            value = value.repeat_interleave(self.groups, dim=1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, sequence, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states, cosine, sine):
        hidden_states = hidden_states + self.residual_dropout(
            self.attention(self.attention_norm(hidden_states), cosine, sine)
        )
        hidden_states = hidden_states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(hidden_states))
        )
        return hidden_states


class NimoraController(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        head_dim = config.hidden_size // config.num_attention_heads

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(
            head_dim, config.max_sequence_length, config.rope_theta
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def _initialize(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )

        hidden_states = self.token_embeddings(input_ids)
        cosine, sine = self.rotary(
            sequence_length, hidden_states.dtype, hidden_states.device
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    cosine,
                    sine,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, cosine, sine)

        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            flat_labels = labels.reshape(-1)
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                flat_labels,
                ignore_index=-100,
                reduction="none",
            )
            valid = flat_labels.ne(-100)
            loss = token_losses[valid].mean() if valid.any() else token_losses.sum() * 0.0
        return {"logits": logits, "loss": loss}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_metadata(self) -> dict:
        return {
            "architecture": "nimora_controller",
            "model_config": asdict(self.config),
            "parameter_count": self.parameter_count,
        }


def build_optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith("token_embeddings.weight"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
