from __future__ import annotations

import importlib.util
from pathlib import Path

from nimora.train_lora import load_lora_config

ROOT = Path(__file__).parents[1]


def load_slurm_wrapper():
    path = ROOT / "scripts" / "slurm_train_lora.py"
    spec = importlib.util.spec_from_file_location("nimora_slurm_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_checkpoint_selects_highest_complete_numeric_checkpoint(
    tmp_path: Path,
) -> None:
    wrapper = load_slurm_wrapper()
    for name in ("checkpoint-2", "checkpoint-100", "checkpoint-broken"):
        path = tmp_path / name
        path.mkdir()
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-200").mkdir()

    assert wrapper.latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_latest_checkpoint_returns_none_without_resumable_state(tmp_path: Path) -> None:
    wrapper = load_slurm_wrapper()
    (tmp_path / "checkpoint-25").mkdir()

    assert wrapper.latest_checkpoint(tmp_path) is None


def test_orcd_config_is_cuda_qlora_with_pinned_model() -> None:
    config = load_lora_config(ROOT / "configs" / "lora-qwen3-4b-orcd.yaml")

    assert config.device == "cuda"
    assert config.quantization.enabled is True
    assert config.model_revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.save_steps == 25
    assert config.assistant_mask_scope == "last"
