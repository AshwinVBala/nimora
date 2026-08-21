from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    python: str
    torch: str | None
    hip: str | None
    cuda_available: bool
    device_name: str | None


def require_torch():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is not installed. Install AMD's ROCm PyTorch build first; "
            "do not install a CUDA wheel on the Radeon workstation."
        )
    import torch

    return torch


def runtime_info() -> RuntimeInfo:
    if importlib.util.find_spec("torch") is None:
        return RuntimeInfo(platform.python_version(), None, None, False, None)
    import torch

    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else None
    return RuntimeInfo(
        python=platform.python_version(),
        torch=torch.__version__,
        hip=getattr(torch.version, "hip", None),
        cuda_available=available,
        device_name=name,
    )


def resolve_device(requested: str):
    torch = require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def assert_training_runtime(device) -> None:
    torch = require_torch()
    if device.type != "cuda":
        raise RuntimeError(
            "Training is intentionally GPU-only. Configure ROCm so "
            "torch.cuda.is_available() returns True (ROCm uses the torch.cuda API)."
        )
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError(
            "A non-ROCm PyTorch build was detected. Install AMD's ROCm PyTorch build."
        )
