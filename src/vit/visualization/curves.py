"""Plot training/validation curves from a completed run's history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vit.engine.trainer import TrainingHistory

__all__ = ["plot_training_curves"]

_METRICS = ("loss", "accuracy", "eer", "auc")


def plot_training_curves(history: TrainingHistory, output_path: Path) -> None:
    """Render loss/accuracy/EER/AUC curves (train vs. val) to a single PNG.

    Args:
        history: The :class:`~vit.engine.trainer.TrainingHistory` returned
            by ``Trainer.fit``.
        output_path: Destination PNG path. Parent directories are created
            if needed.

    Raises:
        ValueError: If ``history`` contains no epochs.
    """
    if not history.train_results:
        raise ValueError("Cannot plot training curves: history has no epochs")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [r.epoch for r in history.train_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, _METRICS):
        train_values = [getattr(r, metric) for r in history.train_results]
        val_values = [getattr(r, metric) for r in history.val_results]
        ax.plot(epochs, train_values, label="train")
        ax.plot(epochs, val_values, label="val")
        if history.best_epoch >= 0:
            ax.axvline(history.best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)