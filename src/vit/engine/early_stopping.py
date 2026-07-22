"""Early stopping utility for training loops.

``EarlyStopping`` tracks a single monitored metric across epochs and signals
when training should stop because the metric has not improved for
``patience`` consecutive epochs. Used by :class:`vit.engine.trainer.Trainer`
in conjunction with :class:`vit.checkpoint.checkpoint_manager.CheckpointManager`.
"""

from __future__ import annotations

__all__ = ["EarlyStopping"]

_VALID_MODES = ("min", "max")


class EarlyStopping:
    """Monitors a metric and signals when training should stop.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate
            before signalling a stop.
        mode: ``"min"`` if lower metric values are better (e.g. loss, EER),
            ``"max"`` if higher values are better (e.g. accuracy, AUC).
        min_delta: Minimum absolute change from the current best required to
            count as an improvement. Must be non-negative.

    Raises:
        ValueError: If ``patience`` is negative, ``mode`` is not one of
            ``"min"``/``"max"``, or ``min_delta`` is negative.

    Example:
        >>> stopper = EarlyStopping(patience=2, mode="min")
        >>> stopper.step(1.0)
        False
        >>> stopper.step(1.1)
        False
        >>> stopper.step(1.2)
        True
    """

    def __init__(self, patience: int, mode: str, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")

        self._patience = patience
        self._mode = mode
        self._min_delta = min_delta

        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0

    def step(self, metric_value: float) -> bool:
        """Record the latest metric value and check whether to stop.

        Args:
            metric_value: The value of the monitored metric for the current
                epoch.

        Returns:
            ``True`` if training should stop (no improvement for more than
            ``patience`` consecutive calls), ``False`` otherwise.
        """
        if self._is_improvement(metric_value):
            self._best_value = metric_value
            self._counter = 0
        else:
            self._counter += 1

        return self._counter > self._patience

    def _is_improvement(self, value: float) -> bool:
        if self._mode == "min":
            return value < (self._best_value - self._min_delta)
        return value > (self._best_value + self._min_delta)

    @property
    def best_value(self) -> float:
        """The best metric value observed so far."""
        return self._best_value

    @property
    def counter(self) -> int:
        """Number of consecutive epochs since the last improvement."""
        return self._counter

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self._patience}, mode={self._mode!r}, "
            f"best_value={self._best_value:.6f}, counter={self._counter})"
        )