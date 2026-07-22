"""Plot a confusion matrix as a heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

__all__ = ["plot_confusion_matrix"]


def plot_confusion_matrix(cm: Tensor, class_names: List[str], output_path: Path) -> None:
    """Render a confusion matrix heatmap with per-cell counts to a PNG.

    Args:
        cm: Integer confusion-matrix tensor of shape
            ``(num_classes, num_classes)``, as returned by
            :func:`vit.metrics.classification_metrics.compute_confusion_matrix`.
        class_names: Display name for each class index, length must equal
            ``cm.shape[0]``.
        output_path: Destination PNG path. Parent directories are created
            if needed.

    Raises:
        ValueError: If ``len(class_names) != cm.shape[0]`` or ``cm`` is not
            square.
    """
    cm_np = cm.detach().cpu().numpy() if hasattr(cm, "detach") else np.asarray(cm)

    if cm_np.shape[0] != cm_np.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape {cm_np.shape}")
    if len(class_names) != cm_np.shape[0]:
        raise ValueError(
            f"len(class_names)={len(class_names)} does not match "
            f"cm.shape[0]={cm_np.shape[0]}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_np, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    max_val = cm_np.max() if cm_np.size > 0 else 0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            color = "white" if cm_np[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(int(cm_np[i, j])), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)