from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 32_768
    max_sequence_length: int = 4_096
    num_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2_560
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    rope_theta: float = 1_000_000.0
    rms_norm_epsilon: float = 1e-6
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
        }
        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head size must be even for rotary embeddings")
        if self.vocab_size > 65_535:
            raise ValueError("The packed-data format uses uint16; vocab_size must be <= 65535")
        for name in ("attention_dropout", "residual_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass(slots=True)
class DataConfig:
    train_dir: str = "data/processed/train"
    validation_dir: str = "data/processed/validation"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    sequence_length: int = 2_048
    validation_samples: int = 256
    num_workers: int = 2


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    warmup_steps: int = 500
    max_grad_norm: float = 1.0


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "runs/controller-120m"
    seed: int = 1337
    device: str = "auto"
    precision: str = "fp16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = 50_000
    log_every: int = 10
    evaluate_every: int = 500
    save_every: int = 500
    keep_last_checkpoints: int = 3
    compile_model: bool = False
    resume_from: str | None = None


@dataclass(slots=True)
class ControllerRunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        if self.data.sequence_length < 1:
            raise ValueError("data.sequence_length must be positive")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("data.sequence_length exceeds model.max_sequence_length")
        if self.training.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be fp32, fp16, or bf16")
        if self.training.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.data.validation_samples < 1:
            raise ValueError("validation_samples must be positive")
        if self.data.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.training.max_steps < 1:
            raise ValueError("max_steps must be positive")
        for name in ("log_every", "evaluate_every", "save_every"):
            if getattr(self.training, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.training.keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be positive")
        if self.optimizer.warmup_steps > self.training.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    return cls(**(values or {}))


def load_controller_config(path: str | Path) -> ControllerRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ControllerRunConfig(
        model=_construct(ModelConfig, raw.get("model")),
        data=_construct(DataConfig, raw.get("data")),
        optimizer=_construct(OptimizerConfig, raw.get("optimizer")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
    config.validate()
    return config
