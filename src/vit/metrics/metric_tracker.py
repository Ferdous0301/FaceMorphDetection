"""Running metric aggregation for training/validation/test loops.

``MetricTracker`` accumulates per-batch metric values (weighted by batch
size) and exposes their running weighted average at any point, so that both
:class:`vit.engine.trainer.Trainer` and :class:`vit.engine.evaluator.Evaluator`
can report a correct epoch-level average even when the final batch is a
different (typically smaller) size than the rest.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["MetricTracker"]


class MetricTracker:
    """Accumulates batch-weighted running averages for named metrics.

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update("loss", 1.0, batch_size=8)
        >>> tracker.update("loss", 0.5, batch_size=2)
        >>> round(tracker.average("loss"), 4)
        0.9
    """

    def __init__(self) -> None:
        """Initialize an empty tracker with no recorded metrics."""
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, name: str, value: float, batch_size: int = 1) -> None:
        """Record one batch's metric value.

        Args:
            name: Metric name (e.g. ``"loss"``, ``"accuracy"``).
            value: The metric value computed for this batch (assumed to
                already be a per-sample average over the batch, e.g. mean
                loss — it is weighted by ``batch_size`` when aggregated).
            batch_size: Number of samples ``value`` was computed over. Used
                as the weight in the running weighted average.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._sums[name] = self._sums.get(name, 0.0) + value * batch_size
        self._counts[name] = self._counts.get(name, 0) + batch_size

    def average(self, name: str) -> float:
        """Return the running weighted average for a metric.

        Args:
            name: Metric name previously passed to :meth:`update`.

        Returns:
            The batch-size-weighted average of all recorded values.

        Raises:
            KeyError: If ``name`` has never been recorded via :meth:`update`.
        """
        if name not in self._sums:
            raise KeyError(f"Metric '{name}' has not been recorded")
        return self._sums[name] / self._counts[name]

    def reset(self) -> None:
        """Clear all recorded metrics, typically called at the start of each epoch."""
        self._sums.clear()
        self._counts.clear()

    def as_dict(self) -> Dict[str, float]:
        """Return all tracked metrics' running averages as a plain dict.

        Returns:
            A mapping from metric name to its current weighted average.
            Empty if no metrics have been recorded.
        """
        return {name: self._sums[name] / self._counts[name] for name in self._sums}

    def __repr__(self) -> str:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in self.as_dict().items())
        return f"MetricTracker({metrics})"