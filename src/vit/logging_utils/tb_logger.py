"""TensorBoard logging wrapper.

Thin wrapper around ``torch.utils.tensorboard.SummaryWriter`` so that the
rest of the codebase (:class:`vit.logging_utils.experiment_logger.ExperimentLogger`,
:class:`vit.engine.trainer.Trainer`, :class:`vit.engine.evaluator.Evaluator`)
never imports ``SummaryWriter`` directly, which keeps it trivial to mock in
unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

__all__ = ["TensorBoardLogger"]


class TensorBoardLogger:
    """Writes scalars and images to a TensorBoard event file.

    Args:
        log_dir: Directory in which TensorBoard event files are written.
            Created if it does not exist.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def log_scalars(self, tag_prefix: str, values: Dict[str, float], step: int) -> None:
        """Log a batch of scalars under a shared tag prefix.

        Args:
            tag_prefix: Prefix prepended to each metric name, e.g.
                ``"train"`` or ``"val"``, producing tags like
                ``"train/loss"``.
            values: Mapping of metric name to scalar value.
            step: Global step (typically the epoch number) for the x-axis.
        """
        for name, value in values.items():
            self._writer.add_scalar(f"{tag_prefix}/{name}", value, global_step=step)

    def log_image(self, tag: str, image: Tensor, step: int) -> None:
        """Log a single image tensor.

        Args:
            tag: Tag under which the image is recorded.
            image: Image tensor in ``(C, H, W)`` format, values expected in
                ``[0, 1]`` (matching ``add_image`` conventions).
            step: Global step (typically the epoch number) for the x-axis.
        """
        self._writer.add_image(tag, image, global_step=step)

    def close(self) -> None:
        """Flush and close the underlying event file writer."""
        self._writer.flush()
        self._writer.close()

    @property
    def log_dir(self) -> Path:
        """The directory this logger writes TensorBoard event files to."""
        return self._log_dir