"""Mixed precision (AMP) helper utilities.

Wraps ``torch.cuda.amp`` so that the rest of the codebase (in particular
``engine.trainer.Trainer``) can request mixed precision unconditionally via
configuration, without having to special-case CPU-only environments at every
call site. On CPU, both helpers safely degrade to no-ops (a disabled
``GradScaler`` and a null autocast context), so the exact same training code
path runs correctly, only without the speed/memory benefit.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

__all__ = ["build_grad_scaler", "autocast_context"]


def build_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    """Construct a :class:`torch.amp.GradScaler` targeting CUDA.

    Note that PyTorch itself force-disables the scaler (regardless of
    ``enabled``) when CUDA is not available on the current machine, emitting
    a ``UserWarning``. This is correct, intentional behaviour: mixed
    precision via ``GradScaler`` is a CUDA-only optimization, so training on
    CPU transparently and correctly falls back to full precision.

    Args:
        enabled: Whether the scaler should actually scale/unscale gradients.
            Should be set to ``config.training.mixed_precision and
            device.type == "cuda"`` by the caller (mixed precision has no
            effect, and ``GradScaler`` should be disabled, on CPU).

    Returns:
        A ``GradScaler`` instance. When disabled (either because ``enabled``
        is False or because CUDA is unavailable), all of its methods
        (``scale``, ``step``, ``update``, ``unscale_``) become identity
        operations, so calling code does not need an ``if`` branch around
        scaler usage.

    Example:
        >>> scaler = build_grad_scaler(enabled=False)
        >>> scaler.is_enabled()
        False
    """
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_context(enabled: bool, device_type: str = "cuda") -> AbstractContextManager:
    """Return an autocast context manager, or a no-op context if disabled.

    Args:
        enabled: Whether automatic mixed precision should be active for the
            wrapped forward pass. Should be set to
            ``config.training.mixed_precision and device.type == "cuda"``
            by the caller.
        device_type: The device type autocast should target (``"cuda"`` or
            ``"cpu"``). Only relevant when ``enabled=True``.

    Returns:
        ``torch.autocast(device_type, enabled=True)`` if ``enabled``,
        otherwise ``contextlib.nullcontext()`` — a context manager that does
        nothing, so call sites can always write
        ``with autocast_context(...):`` unconditionally.

    Example:
        >>> with autocast_context(enabled=False):
        ...     x = torch.zeros(1)
        >>> x.dtype
        torch.float32
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=True)