"""Plotting utilities for training curves, confusion matrices, and attention maps.

Public API:

    from vit.visualization import (
        plot_training_curves,
        plot_confusion_matrix,
        plot_attention_rollout,
    )
"""

from __future__ import annotations

from src.vit.visualization.attention_maps import plot_attention_rollout
from src.vit.visualization.confusion_matrix import plot_confusion_matrix
from src.vit.visualization.curves import plot_training_curves

__all__ = ["plot_training_curves", "plot_confusion_matrix", "plot_attention_rollout"]