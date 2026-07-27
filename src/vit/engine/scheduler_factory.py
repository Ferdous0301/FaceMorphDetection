"""
Learning-rate scheduler factory for the ViT morph-attack-detection module.

Supports:
    - "cosine":  linear warmup followed by cosine decay to `min_lr`,
                 stepped every training *step* (not epoch) via LambdaLR.
    - "step":    StepLR, decays by 0.1 every third of total_epochs.
    - "plateau": ReduceLROnPlateau, stepped every *epoch* with a monitored
                 metric (caller must call .step(metric) explicitly rather
                 than relying on Trainer's per-step scheduler.step()).
    - "none":    a no-op scheduler that never changes the LR.

Because "cosine" is stepped per-batch while "step"/"plateau" are stepped
per-epoch, `Trainer` must be aware of which stepping cadence a given
scheduler expects. To keep that logic simple and explicit, each scheduler
object returned here carries a `step_every` attribute ("batch" or "epoch")
that the Trainer inspects.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from src.vit.configs.schema import SchedulerConfig

_SUPPORTED_SCHEDULERS = ("cosine", "step", "plateau", "none")


class _NoOpScheduler:
    """Scheduler that never changes the learning rate. Mimics the minimal
    _LRScheduler interface (`step`, `get_last_lr`) used by Trainer."""

    step_every = "batch"

    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def step(self, *args, **kwargs) -> None:  # noqa: D401 - intentional no-op
        return None

    def get_last_lr(self):
        return [group["lr"] for group in self._optimizer.param_groups]

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        return None


def _cosine_with_warmup_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Returns a multiplicative factor applied to the base LR."""
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))

    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    # Interpolate between 1.0 (base lr) and min_lr_ratio at the end of decay.
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor


def _build_cosine(
    optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int
) -> LambdaLR:
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.total_epochs * steps_per_epoch

    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = config.min_lr / base_lr if base_lr > 0 else 0.0

    lr_lambda: Callable[[int], float] = lambda step: _cosine_with_warmup_lambda(
        step, warmup_steps, total_steps, min_lr_ratio
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.step_every = "batch"  # type: ignore[attr-defined]
    return scheduler


def _build_step(
    optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int
) -> StepLR:
    step_size = max(1, config.total_epochs // 3)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    scheduler.step_every = "epoch"  # type: ignore[attr-defined]
    return scheduler


def _build_plateau(
    optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int
) -> ReduceLROnPlateau:
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, min_lr=config.min_lr
    )
    scheduler.step_every = "epoch"  # type: ignore[attr-defined]
    return scheduler


def _build_none(
    optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int
) -> _NoOpScheduler:
    return _NoOpScheduler(optimizer)


_SCHEDULER_BUILDERS: Dict[str, Callable] = {
    "cosine": _build_cosine,
    "step": _build_step,
    "plateau": _build_plateau,
    "none": _build_none,
}


def build_scheduler(
    optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int
):
    """
    Build a learning-rate scheduler per `config`.

    Args:
        optimizer: the optimizer whose LR will be scheduled.
        config: SchedulerConfig specifying name/warmup_epochs/total_epochs/min_lr.
        steps_per_epoch: number of optimizer steps per training epoch
            (i.e. len(train_loader)), required to convert epoch-based
            warmup/total into per-step schedules for "cosine".

    Returns:
        A scheduler object exposing `.step()`, `.get_last_lr()`,
        `.state_dict()`, `.load_state_dict()`, and a `step_every`
        attribute ("batch" or "epoch") that Trainer uses to decide
        when to call `.step()`.

    Raises:
        ValueError: if config.name is not a supported scheduler name.
    """
    name = config.name.lower()
    builder = _SCHEDULER_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unsupported scheduler '{config.name}'. "
            f"Supported schedulers: {_SUPPORTED_SCHEDULERS}"
        )
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    return builder(optimizer, config, steps_per_epoch)