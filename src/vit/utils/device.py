"""Device management utilities.

Centralizes all logic for choosing between CPU and GPU so that no other
module in the ``vit`` package ever calls ``.cuda()``, ``.to("cuda")``, or
checks ``torch.cuda.is_available()`` directly. Every component (Trainer,
Evaluator, Predictor) receives an already-resolved ``torch.device`` at
construction time.
"""

from __future__ import annotations

import platform
from typing import Dict

import torch

__all__ = ["resolve_device", "get_device_info"]


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference string into a concrete :class:`torch.device`.

    Args:
        preference: One of:
            * ``"auto"`` - use CUDA if available, otherwise CPU.
            * ``"cpu"`` - force CPU, even if CUDA is available.
            * ``"cuda"`` - use the default CUDA device (``cuda:0``).
            * ``"cuda:N"`` - use CUDA device index ``N`` explicitly.

    Returns:
        A resolved :class:`torch.device`.

    Raises:
        ValueError: If ``preference`` is not one of the supported forms.
        RuntimeError: If ``"cuda"`` or ``"cuda:N"`` is requested but CUDA is
            not available on this machine, or if ``N`` is out of range for
            the available device count.

    Example:
        >>> resolve_device("cpu")
        device(type='cpu')
    """
    preference = preference.strip().lower()

    if preference == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda" or preference.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device preference '{preference}' requested but CUDA is not available "
                "on this machine."
            )
        device = torch.device(preference)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {device.index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return device

    raise ValueError(
        f"Unsupported device preference '{preference}'. "
        "Expected 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )


def get_device_info(device: torch.device | None = None) -> Dict[str, str]:
    """Collect a human-readable summary of the compute environment.

    Useful for logging at the start of a training run so that experiment
    logs record exactly what hardware/software produced a given result.

    Args:
        device: If provided, GPU-specific fields are reported for this
            device. If ``None``, GPU fields are reported for the current
            default CUDA device (if any).

    Returns:
        A dictionary with keys ``"platform"``, ``"python_version"``,
        ``"torch_version"``, ``"cuda_available"``, and, when CUDA is
        available, ``"cuda_version"``, ``"gpu_name"``, and
        ``"gpu_count"``.
    """
    info: Dict[str, str] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        index = device.index if (device is not None and device.index is not None) else 0
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_count"] = str(torch.cuda.device_count())

    return info