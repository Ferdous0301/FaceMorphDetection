"""Cross-cutting utilities for reproducibility, device management, and AMP.

Public API:

    from vit.utils import (
        set_global_seed,
        seed_worker,
        make_generator,
        resolve_device,
        get_device_info,
        build_grad_scaler,
        autocast_context,
    )
"""

from __future__ import annotations

from FaceMorphDetection.Src.vit.utils.amp import autocast_context, build_grad_scaler
from FaceMorphDetection.Src.vit.utils.device import get_device_info, resolve_device
from FaceMorphDetection.Src.vit.utils.seed import make_generator, seed_worker, set_global_seed

__all__ = [
    "set_global_seed",
    "seed_worker",
    "make_generator",
    "resolve_device",
    "get_device_info",
    "build_grad_scaler",
    "autocast_context",
]