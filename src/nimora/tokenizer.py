from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from nimora.serialization import SPECIAL_TOKENS, iter_training_text


def _tokenizers():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as error:
        raise RuntimeError(
            "Install Nimora's base dependencies to use tokenizer commands"
        ) from error
    return Tokenizer, decoders, models, pre_tokenizers, trainers


def train_tokenizer(
    inputs: Iterable[str | Path],
    output: str | Path,
    vocab_size: int = 32_768,
    minimum_frequency: int = 2,
) -> Path:
    Tokenizer, decoders, models, pre_tokenizers, trainers = _tokenizers()
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=minimum_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_training_text(inputs), trainer=trainer)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return output_path


def load_tokenizer(path: str | Path):
    Tokenizer, _, _, _, _ = _tokenizers()
    return Tokenizer.from_file(str(path))
