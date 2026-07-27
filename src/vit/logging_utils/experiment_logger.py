"""Unified logging facade over CSV, TensorBoard, and console output.

:class:`Trainer` and :class:`Evaluator` depend only on ``ExperimentLogger``
and never touch :class:`~vit.logging_utils.csv_logger.CSVLogger` or
:class:`~vit.logging_utils.tb_logger.TensorBoardLogger` directly. This keeps
a single call site (:meth:`ExperimentLogger.log_epoch`) responsible for
writing an identical metric set to every sink, which prevents the CSV and
TensorBoard logs from drifting apart.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.vit.logging_utils.csv_logger import CSVLogger
from src.vit.logging_utils.tb_logger import TensorBoardLogger

__all__ = ["ExperimentLogger"]


class ExperimentLogger:
    """Facade combining CSV + TensorBoard + console logging.

    Args:
        csv_logger: Sink for per-epoch rows written to disk as CSV.
        tb_logger: Sink for per-epoch scalars written as TensorBoard events.
        name: Logger name, used both as the stdlib ``logging.Logger`` name
            and as a human-readable prefix in console output.
    """

    def __init__(self, csv_logger: CSVLogger, tb_logger: TensorBoardLogger, name: str) -> None:
        self._csv_logger = csv_logger
        self._tb_logger = tb_logger
        self._name = name
        self._console = logging.getLogger(name)
        if not self._console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            self._console.addHandler(handler)
            self._console.setLevel(logging.INFO)

    def log_epoch(self, split: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Log one epoch's metrics for a given split to every sink.

        Args:
            split: Data split the metrics were computed on, e.g.
                ``"train"``, ``"val"``, or ``"test"``.
            epoch: Epoch index (0-based) these metrics correspond to.
            metrics: Mapping of metric name to value, e.g.
                ``{"loss": 0.42, "accuracy": 0.91, "eer": 0.05}``.
        """
        row = {"epoch": epoch, "split": split, **metrics}
        self._csv_logger.log(row)
        self._tb_logger.log_scalars(split, metrics, step=epoch)

        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._console.info(f"epoch={epoch} split={split} {metrics_str}")

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        """Log the run's hyperparameters to the console.

        Args:
            hparams: Flat mapping of hyperparameter name to value, typically
                derived from a flattened :class:`~vit.config.schema.ExperimentConfig`.
        """
        hparams_str = ", ".join(f"{k}={v!r}" for k, v in hparams.items())
        self._console.info(f"hparams: {hparams_str}")

    def close(self) -> None:
        """Release resources held by the underlying sinks (e.g. TensorBoard writer)."""
        self._tb_logger.close()