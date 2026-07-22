"""Classification metrics and running aggregation for the ViT stage.

Public API:

    from vit.metrics import (
        compute_accuracy,
        compute_confusion_matrix,
        compute_auc,
        compute_eer,
        compute_apcer_bpcer,
        MetricTracker,
    )
"""

from __future__ import annotations

from vit.metrics.classification_metrics import (
    compute_accuracy,
    compute_apcer_bpcer,
    compute_auc,
    compute_confusion_matrix,
    compute_eer,
)
from vit.metrics.metric_tracker import MetricTracker

__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_auc",
    "compute_eer",
    "compute_apcer_bpcer",
    "MetricTracker",
]