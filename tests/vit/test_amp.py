"""Unit tests for vit.utils.amp."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from vit.utils.amp import autocast_context, build_grad_scaler


class TestBuildGradScaler:
    def test_disabled_scaler_is_identity(self) -> None:
        scaler = build_grad_scaler(enabled=False)
        assert scaler.is_enabled() is False

        loss = torch.tensor(2.0, requires_grad=True)
        scaled = scaler.scale(loss)
        assert torch.equal(scaled, loss)

    def test_enabled_scaler_matches_cuda_availability(self) -> None:
        # torch.amp.GradScaler force-disables itself when CUDA is not
        # available, regardless of the `enabled` flag passed in - this is
        # correct, intentional CPU fallback behaviour, not a bug.
        scaler = build_grad_scaler(enabled=True)
        assert scaler.is_enabled() is torch.cuda.is_available()


class TestAutocastContext:
    def test_disabled_returns_nullcontext(self) -> None:
        ctx = autocast_context(enabled=False)
        assert isinstance(ctx, nullcontext)

    def test_disabled_context_does_not_change_dtype(self) -> None:
        with autocast_context(enabled=False):
            x = torch.zeros(2, 2)
        assert x.dtype == torch.float32

    def test_enabled_context_is_usable_on_cpu(self) -> None:
        # torch.autocast supports device_type="cpu"; this should not raise.
        with autocast_context(enabled=True, device_type="cpu"):
            x = torch.zeros(2, 2)
            y = x + 1
        assert y is not None