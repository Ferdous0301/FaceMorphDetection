"""Publication-quality report generation for Face Morph Attack Detection.

Builds human-readable (Markdown, plain-text) and machine-readable (JSON)
summaries from an :class:`~vit.evaluation.evaluator.EvaluationResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from vit.evaluation import metrics as metric_fns
from vit.evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentSummary",
    "classification_report",
    "confusion_matrix_summary",
    "metric_table",
    "build_experiment_summary",
    "markdown_report",
    "json_report",
]


@dataclass(frozen=True)
class ExperimentSummary:
    """A concise, self-contained summary of a single evaluation run.

    Attributes:
        experiment_name: Human-readable name identifying the experiment
            (e.g. model architecture + dataset + run id).
        timestamp: ISO-8601 UTC timestamp of when the summary was created.
        num_samples: Total number of samples evaluated.
        accuracy: Overall accuracy.
        f1: F1 score for the attack class.
        roc_auc: ROC AUC, if defined.
        eer: Equal Error Rate, if defined.
        acer: Average Classification Error Rate.
        notes: Optional free-form notes about the run (e.g. dataset split,
            hyperparameters).
    """

    experiment_name: str
    timestamp: str
    num_samples: int
    accuracy: float
    f1: float
    roc_auc: Optional[float]
    eer: Optional[float]
    acer: Optional[float]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty.")
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative.")

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "eer": self.eer,
            "acer": self.acer,
            "notes": self.notes,
        }


def classification_report(
    result: EvaluationResult, target_names: Optional[Dict[int, str]] = None
) -> str:
    """Generate a per-class precision/recall/F1/support text report.

    Args:
        result: The evaluation result to summarise.
        target_names: Optional mapping from label integer to display name.
            Defaults to ``{0: "bonafide", 1: "attack"}``.

    Returns:
        A formatted, fixed-width text table similar to
        ``sklearn.metrics.classification_report``.
    """
    names = target_names or {0: "bonafide", 1: "attack"}
    labels = result.labels
    predictions = result.predictions

    rows = []
    for label_value in (0, 1):
        support = int(np.sum(labels == label_value))
        if support == 0:
            precision_val = recall_val = f1_val = 0.0
        else:
            # Compute one-vs-rest by temporarily relabeling.
            binary_true = (labels == label_value).astype(int)
            binary_pred = (predictions == label_value).astype(int)
            precision_val = metric_fns.precision(binary_true, binary_pred)
            recall_val = metric_fns.recall(binary_true, binary_pred)
            f1_val = metric_fns.f1_score(binary_true, binary_pred)
        rows.append((names.get(label_value, str(label_value)), precision_val, recall_val, f1_val, support))

    total_support = int(len(labels))
    header = f"{'class':<12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [header, "-" * len(header)]
    for name, p, r, f1_val, support in rows:
        lines.append(f"{name:<12}{p:>10.4f}{r:>10.4f}{f1_val:>10.4f}{support:>10d}")
    lines.append("-" * len(header))
    lines.append(f"{'accuracy':<12}{'':>10}{'':>10}{result.accuracy:>10.4f}{total_support:>10d}")
    return "\n".join(lines)


def confusion_matrix_summary(result: EvaluationResult) -> str:
    """Generate a labelled, human-readable confusion matrix summary.

    Args:
        result: The evaluation result to summarise.

    Returns:
        A formatted text block showing TN, FP, FN, TP counts with labels.
    """
    cm = np.asarray(result.confusion_matrix)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    lines = [
        "Confusion Matrix (rows=true, cols=predicted)",
        "                 Pred: bonafide   Pred: attack",
        f"True: bonafide   {tn:>14d}   {fp:>12d}",
        f"True: attack     {fn:>14d}   {tp:>12d}",
    ]
    return "\n".join(lines)


def metric_table(result: EvaluationResult) -> str:
    """Generate a formatted table listing every scalar metric.

    Args:
        result: The evaluation result to summarise.

    Returns:
        A formatted, fixed-width text table of metric name/value pairs.
    """
    metric_items = [
        ("Loss", result.loss),
        ("Accuracy", result.accuracy),
        ("Precision", result.precision),
        ("Recall", result.recall),
        ("F1 Score", result.f1),
        ("ROC AUC", result.roc_auc),
        ("PR AUC", result.pr_auc),
        ("EER", result.eer),
        ("EER Threshold", result.eer_threshold),
        ("APCER", result.apcer),
        ("BPCER", result.bpcer),
        ("ACER", result.acer),
        ("FAR", result.far),
        ("FRR", result.frr),
    ]
    header = f"{'Metric':<20}{'Value':>12}"
    lines = [header, "-" * len(header)]
    for name, value in metric_items:
        display = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"{name:<20}{display:>12}")
    return "\n".join(lines)


def build_experiment_summary(
    result: EvaluationResult, experiment_name: str, notes: str = ""
) -> ExperimentSummary:
    """Build an :class:`ExperimentSummary` from an evaluation result.

    Args:
        result: The evaluation result to summarise.
        experiment_name: Human-readable identifier for this run.
        notes: Optional free-form notes.

    Returns:
        A populated :class:`ExperimentSummary`.
    """
    return ExperimentSummary(
        experiment_name=experiment_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=result.num_samples,
        accuracy=result.accuracy,
        f1=result.f1,
        roc_auc=result.roc_auc,
        eer=result.eer,
        acer=result.acer,
        notes=notes,
    )


def markdown_report(
    result: EvaluationResult, experiment_name: str, notes: str = ""
) -> str:
    """Render a complete Markdown report for an evaluation run.

    Args:
        result: The evaluation result to summarise.
        experiment_name: Human-readable identifier for this run.
        notes: Optional free-form notes appended to the report.

    Returns:
        A Markdown-formatted string suitable for writing directly to a
        ``.md`` file.
    """
    summary = build_experiment_summary(result, experiment_name, notes)
    sections = [
        f"# Evaluation Report: {experiment_name}",
        "",
        f"*Generated: {summary.timestamp}*",
        "",
        "## Summary",
        "",
        f"- **Samples evaluated:** {result.num_samples}",
        f"- **Accuracy:** {result.accuracy:.4f}",
        f"- **F1 Score:** {result.f1:.4f}",
        f"- **ROC AUC:** {result.roc_auc:.4f}" if result.roc_auc is not None else "- **ROC AUC:** N/A",
        f"- **EER:** {result.eer:.4f}" if result.eer is not None else "- **EER:** N/A",
        f"- **ACER:** {result.acer:.4f}" if result.acer is not None else "- **ACER:** N/A",
        "",
        "## Metrics",
        "",
        "```",
        metric_table(result),
        "```",
        "",
        "## Classification Report",
        "",
        "```",
        classification_report(result),
        "```",
        "",
        "## Confusion Matrix",
        "",
        "```",
        confusion_matrix_summary(result),
        "```",
        "",
    ]
    if notes:
        sections.extend(["## Notes", "", notes, ""])
    return "\n".join(sections)


def json_report(result: EvaluationResult, experiment_name: str, notes: str = "") -> Dict[str, Any]:
    """Build a JSON-serialisable dictionary report for an evaluation run.

    Args:
        result: The evaluation result to summarise.
        experiment_name: Human-readable identifier for this run.
        notes: Optional free-form notes included in the summary section.

    Returns:
        A nested dictionary containing the experiment summary and full
        metric values, ready for ``json.dump``.
    """
    summary = build_experiment_summary(result, experiment_name, notes)
    return {
        "summary": summary.as_dict(),
        "metrics": result.as_dict(),
    }