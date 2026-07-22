"""Classification metrics for morph-attack detection.

Convention used throughout this module (matching ISO/IEC 30107-3
terminology commonly used in presentation-attack-detection literature):

    * Label ``0`` = bona fide (genuine, non-morphed face image).
    * Label ``1`` = attack (morphed face image).
    * "Score" means a continuous value where *higher* indicates more
      confidence that a sample is an *attack* (label 1) — e.g. the softmax
      probability of class 1.

Beyond generic accuracy/AUC, this module implements the standard
presentation-attack-detection metrics **APCER** and **BPCER**, and the
**Equal Error Rate (EER)**, since these — not raw accuracy — are the metrics
a face-morph-attack-detection thesis is expected to report.

All functions are pure (no side effects, no hidden state) and accept either
``torch.Tensor`` or any NumPy-array-convertible input, making them trivially
unit-testable against hand-computed reference values.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor

__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_auc",
    "compute_eer",
    "compute_apcer_bpcer",
]

ArrayLike = Union[Tensor, np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a Tensor or array-like input to a detached, CPU NumPy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_accuracy(preds: ArrayLike, labels: ArrayLike) -> float:
    """Compute classification accuracy.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        labels: Ground-truth class indices, shape ``(N,)``.

    Returns:
        The fraction of samples where ``preds == labels``, as a Python float.

    Raises:
        ValueError: If ``preds`` and ``labels`` have different lengths, or
            either is empty.

    Example:
        >>> compute_accuracy(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]))
        0.6666666666666666
    """
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)

    if preds_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"preds and labels must have the same length, "
            f"got {preds_np.shape[0]} and {labels_np.shape[0]}"
        )
    if preds_np.shape[0] == 0:
        raise ValueError("Cannot compute accuracy on empty input")

    return float(np.mean(preds_np == labels_np))


def compute_confusion_matrix(preds: ArrayLike, labels: ArrayLike, num_classes: int) -> Tensor:
    """Compute a confusion matrix.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        labels: Ground-truth class indices, shape ``(N,)``.
        num_classes: Total number of classes (matrix size is
            ``num_classes x num_classes``).

    Returns:
        An integer tensor ``cm`` of shape ``(num_classes, num_classes)``
        where ``cm[i, j]`` is the number of samples with true label ``i``
        predicted as class ``j``.

    Raises:
        ValueError: If ``num_classes`` is not positive, if ``preds`` and
            ``labels`` differ in length, or if any label/pred value falls
            outside ``[0, num_classes)``.

    Example:
        >>> compute_confusion_matrix(
        ...     torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 0]), num_classes=2
        ... )
        tensor([[2, 0],
                [1, 1]])
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    preds_t = torch.as_tensor(_to_numpy(preds), dtype=torch.long)
    labels_t = torch.as_tensor(_to_numpy(labels), dtype=torch.long)

    if preds_t.shape[0] != labels_t.shape[0]:
        raise ValueError(
            f"preds and labels must have the same length, "
            f"got {preds_t.shape[0]} and {labels_t.shape[0]}"
        )

    if preds_t.numel() > 0:
        out_of_range = (
            (preds_t.min() < 0)
            or (preds_t.max() >= num_classes)
            or (labels_t.min() < 0)
            or (labels_t.max() >= num_classes)
        )
        if out_of_range:
            raise ValueError(
                f"preds/labels contain values outside [0, {num_classes}); "
                f"pred range=({int(preds_t.min())}, {int(preds_t.max())}), "
                f"label range=({int(labels_t.min())}, {int(labels_t.max())})"
            )

    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    indices = labels_t * num_classes + preds_t
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    cm = counts.reshape(num_classes, num_classes)
    return cm


def compute_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Compute the Area Under the ROC Curve (AUC) for binary attack detection.

    Args:
        scores: Continuous attack-confidence scores (higher = more likely
            attack/morph), shape ``(N,)``.
        labels: Binary ground-truth labels (``0``=bona fide, ``1``=attack),
            shape ``(N,)``.

    Returns:
        The AUC as a Python float in ``[0, 1]``.

    Raises:
        ValueError: If ``labels`` contains anything other than exactly the
            two classes ``{0, 1}`` (AUC is undefined with a single class
            present), or if lengths mismatch.

    Example:
        >>> round(compute_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 4)
        0.75
    """
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)

    if scores_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {scores_np.shape[0]} and {labels_np.shape[0]}"
        )
    unique_labels = set(np.unique(labels_np).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(
            f"compute_auc requires both classes {{0, 1}} to be present in labels, "
            f"got unique labels {sorted(unique_labels)}"
        )

    return float(roc_auc_score(labels_np, scores_np))


def compute_eer(scores: ArrayLike, labels: ArrayLike) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and its corresponding decision threshold.

    The EER is the point on the ROC curve where the False Positive Rate
    (bona fide misclassified as attack) equals the False Negative Rate
    (attack misclassified as bona fide). It is the standard single-number
    summary of biometric/attack-detection system performance.

    Args:
        scores: Continuous attack-confidence scores (higher = more likely
            attack/morph), shape ``(N,)``.
        labels: Binary ground-truth labels (``0``=bona fide, ``1``=attack),
            shape ``(N,)``.

    Returns:
        A tuple ``(eer, threshold)`` where ``eer`` is a float in ``[0, 1]``
        and ``threshold`` is the score threshold (predict attack if
        ``score >= threshold``) at which FPR and FNR are (approximately)
        equal.

    Raises:
        ValueError: If ``labels`` does not contain exactly the two classes
            ``{0, 1}``, or if lengths mismatch.

    Example:
        >>> eer, thr = compute_eer([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
        >>> 0.0 <= eer <= 1.0
        True
    """
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)

    if scores_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {scores_np.shape[0]} and {labels_np.shape[0]}"
        )
    unique_labels = set(np.unique(labels_np).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(
            f"compute_eer requires both classes {{0, 1}} to be present in labels, "
            f"got unique labels {sorted(unique_labels)}"
        )

    fpr, tpr, thresholds = roc_curve(labels_np, scores_np, pos_label=1)
    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_apcer_bpcer(
    scores: ArrayLike, labels: ArrayLike, threshold: float
) -> Tuple[float, float]:
    """Compute APCER and BPCER at a fixed decision threshold.

    * **APCER** (Attack Presentation Classification Error Rate): fraction of
      attack (morph) samples incorrectly classified as bona fide.
    * **BPCER** (Bona fide Presentation Classification Error Rate): fraction
      of bona fide samples incorrectly classified as attack.

    A sample is predicted as "attack" if ``score >= threshold``.

    Args:
        scores: Continuous attack-confidence scores, shape ``(N,)``.
        labels: Binary ground-truth labels (``0``=bona fide, ``1``=attack),
            shape ``(N,)``.
        threshold: Decision threshold; predict attack if ``score >= threshold``.

    Returns:
        A tuple ``(apcer, bpcer)``, each a float in ``[0, 1]``.

    Raises:
        ValueError: If lengths mismatch, or if either class (``0`` or ``1``)
            has zero samples in ``labels`` (the corresponding rate would be
            undefined).

    Example:
        >>> compute_apcer_bpcer([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], threshold=0.5)
        (0.5, 0.0)
    """
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)

    if scores_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {scores_np.shape[0]} and {labels_np.shape[0]}"
        )

    attack_mask = labels_np == 1
    bonafide_mask = labels_np == 0

    if attack_mask.sum() == 0:
        raise ValueError("No attack (label=1) samples present; APCER is undefined")
    if bonafide_mask.sum() == 0:
        raise ValueError("No bona fide (label=0) samples present; BPCER is undefined")

    predicted_attack = scores_np >= threshold

    apcer = float(np.mean(~predicted_attack[attack_mask]))
    bpcer = float(np.mean(predicted_attack[bonafide_mask]))
    return apcer, bpcer