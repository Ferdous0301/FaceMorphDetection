"""Pure, stateless metric functions for Face Morph Attack Detection.

All functions in this module operate purely on NumPy arrays (or objects that
can be coerced to NumPy arrays, such as Python lists or PyTorch tensors that
have already been moved to CPU). They have no side effects, hold no internal
state and are safe to call repeatedly with the same inputs to obtain the
same outputs (deterministic behaviour).

Class-label convention used throughout this module and the rest of the
``evaluation`` package:

* ``0`` -> bonafide (genuine, live) sample
* ``1`` -> attack (morphed) sample

This mirrors the standard convention used in the ISO/IEC 30107-3 biometric
presentation attack detection (PAD) standard, from which the APCER, BPCER
and ACER metrics are drawn.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

ArrayLike = Union[Sequence[float], np.ndarray]

# ``np.trapezoid`` replaces the deprecated ``np.trapz`` in NumPy >= 2.0.
# This shim keeps the module compatible with both NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "to_numpy",
    "validate_binary_labels",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "roc_curve",
    "roc_auc",
    "precision_recall_curve",
    "pr_auc",
    "equal_error_rate",
    "apcer",
    "bpcer",
    "acer",
    "far",
    "frr",
]


def to_numpy(values: ArrayLike) -> np.ndarray:
    """Coerce a sequence-like input into a 1-D contiguous NumPy array.

    Args:
        values: A list, tuple, NumPy array, or any object exposing
            ``__iter__`` / ``__array__`` (e.g. a detached PyTorch tensor).

    Returns:
        A 1-D ``np.ndarray`` of dtype ``float64`` or ``int64`` depending on
        input content.

    Raises:
        ValueError: If ``values`` cannot be converted to a 1-D array.
    """
    if hasattr(values, "detach"):
        # Support torch.Tensor without importing torch as a hard dependency.
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def validate_binary_labels(y_true: ArrayLike, name: str = "y_true") -> np.ndarray:
    """Validate and coerce an array of binary ground-truth labels.

    Args:
        y_true: Ground-truth binary labels (``0`` or ``1``).
        name: Name used in error messages for clarity.

    Returns:
        The validated labels as an ``np.ndarray`` of dtype ``int64``.

    Raises:
        ValueError: If the array is empty or contains values other than
            0/1.
    """
    array = to_numpy(y_true)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    unique_values = np.unique(array)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must contain only binary labels {{0, 1}}, "
            f"got unique values: {unique_values.tolist()}"
        )
    return array.astype(np.int64)


def _validate_equal_length(*arrays: np.ndarray) -> None:
    """Raise ``ValueError`` if arrays do not all share the same length."""
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) > 1:
        raise ValueError(f"All arrays must have the same length, got lengths: {lengths}")


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        The fraction of correctly classified samples, in ``[0, 1]``.

    Raises:
        ValueError: If inputs are empty or of mismatched length.
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_pred_arr = validate_binary_labels(y_pred, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr)
    return float(np.mean(y_true_arr == y_pred_arr))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Compute a 2x2 binary confusion matrix.

    The returned matrix follows the layout::

        [[TN, FP],
         [FN, TP]]

    where rows index the true label (0=bonafide, 1=attack) and columns
    index the predicted label.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        A ``(2, 2)`` ``np.ndarray`` of dtype ``int64``.

    Raises:
        ValueError: If inputs are empty or of mismatched length.
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_pred_arr = validate_binary_labels(y_pred, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr)

    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def precision(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """Compute precision for the positive (attack) class.

    Precision = TP / (TP + FP)

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        zero_division: Value returned when the denominator is zero.

    Returns:
        Precision score in ``[0, 1]``.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    denominator = tp + fp
    if denominator == 0:
        logger.warning("Precision undefined (no positive predictions); returning %s", zero_division)
        return float(zero_division)
    return float(tp / denominator)


def recall(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """Compute recall (a.k.a. sensitivity / true positive rate).

    Recall = TP / (TP + FN)

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        zero_division: Value returned when the denominator is zero.

    Returns:
        Recall score in ``[0, 1]``.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    denominator = tp + fn
    if denominator == 0:
        logger.warning("Recall undefined (no positive ground truth); returning %s", zero_division)
        return float(zero_division)
    return float(tp / denominator)


def f1_score(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """Compute the F1 score, the harmonic mean of precision and recall.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        zero_division: Value returned when precision and recall are both 0.

    Returns:
        F1 score in ``[0, 1]``.
    """
    p = precision(y_true, y_pred, zero_division=0.0)
    r = recall(y_true, y_pred, zero_division=0.0)
    if p + r == 0:
        logger.warning("F1 score undefined (precision + recall == 0); returning %s", zero_division)
        return float(zero_division)
    return float(2 * p * r / (p + r))


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores/probabilities for the positive (attack)
            class. Higher values indicate more confidence in the attack
            class.

    Returns:
        A tuple ``(fpr, tpr, thresholds)`` of 1-D arrays, sorted by
        decreasing threshold, suitable for plotting or integration.

    Raises:
        ValueError: If inputs are empty, mismatched length, or only one
            class is present in ``y_true`` (curve undefined).
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_score_arr = to_numpy(y_score).astype(np.float64)
    _validate_equal_length(y_true_arr, y_score_arr)

    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "ROC curve requires both classes to be present in y_true "
            f"(found {n_pos} positives and {n_neg} negatives)."
        )

    # Sort scores descending; thresholds are the unique score values.
    order = np.argsort(-y_score_arr, kind="mergesort")
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_score_sorted[0] + 1.0, y_score_sorted[threshold_idxs]]

    tpr = tps / n_pos
    fpr = fps / n_neg
    return fpr, tpr, thresholds


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute the Area Under the ROC Curve (ROC AUC) via the trapezoidal rule.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores/probabilities for the positive class.

    Returns:
        ROC AUC in ``[0, 1]``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(_trapezoid(tpr, fpr))


def precision_recall_curve(
    y_true: ArrayLike, y_score: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Precision-Recall curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores/probabilities for the positive class.

    Returns:
        A tuple ``(precision, recall, thresholds)``. ``precision`` and
        ``recall`` have one more element than ``thresholds`` (the final
        point (precision=1, recall=0) is appended, matching scikit-learn's
        convention).

    Raises:
        ValueError: If no positive samples are present in ``y_true``.
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_score_arr = to_numpy(y_score).astype(np.float64)
    _validate_equal_length(y_true_arr, y_score_arr)

    n_pos = int(np.sum(y_true_arr == 1))
    if n_pos == 0:
        raise ValueError("Precision-Recall curve requires at least one positive sample.")

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precisions = tps / (tps + fps)
    recalls = tps / n_pos
    thresholds = y_score_sorted[threshold_idxs]

    # Append the sentinel point (recall=0, precision=1) and reverse so the
    # curve is ordered by ascending recall, matching common conventions.
    precisions = np.r_[precisions[::-1], 1.0]
    recalls = np.r_[recalls[::-1], 0.0]
    thresholds = thresholds[::-1]
    return precisions, recalls, thresholds


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute the Area Under the Precision-Recall Curve (PR AUC).

    Uses the standard step-wise summation
    ``AUC = sum_n (R_n - R_{n-1}) * P_n``, evaluated over recall points in
    ascending order as returned by :func:`precision_recall_curve`. This
    avoids the ambiguity a generic trapezoidal rule would introduce when
    multiple points share the same recall value (which happens whenever
    several samples receive the same score, or once recall reaches 1.0
    and additional negatives are still swept over).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores/probabilities for the positive class.

    Returns:
        PR AUC in ``[0, 1]``.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns points in descending-recall order;
    # reverse to ascending recall for the step-wise summation below. This
    # is safe because the underlying sequence is already monotonic.
    precisions = precisions[::-1]
    recalls = recalls[::-1]
    recall_deltas = np.diff(recalls)
    return float(np.sum(recall_deltas * precisions[1:]))


def equal_error_rate(y_true: ArrayLike, y_score: ArrayLike) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and its corresponding threshold.

    The EER is the point on the ROC curve at which the False Acceptance
    Rate (FAR, equivalent to the FPR) equals the False Rejection Rate
    (FRR, equivalent to 1 - TPR). The value is obtained via linear
    interpolation between the two closest operating points, which is the
    standard approach for finite score sets.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores/probabilities for the positive
            (attack) class.

    Returns:
        A tuple ``(eer, threshold)`` where ``eer`` is in ``[0, 1]`` and
        ``threshold`` is the score threshold at which FAR == FRR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    # Locate the index where the sign of (fpr - fnr) changes (crossing
    # point). fpr is non-decreasing and fnr is non-increasing as thresholds
    # decrease along the arrays, so the difference is monotonic.
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))

    if diff[idx] == 0 or idx == 0 or idx == len(diff) - 1:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])
        return eer, threshold

    # Interpolate between idx and the neighbour on the other side of zero.
    neighbour = idx - 1 if diff[idx - 1] * diff[idx] < 0 else idx + 1
    x = [diff[idx], diff[neighbour]]
    y_fpr = [fpr[idx], fpr[neighbour]]
    y_fnr = [fnr[idx], fnr[neighbour]]
    y_thr = [thresholds[idx], thresholds[neighbour]]

    if x[0] == x[1]:
        eer = float((y_fpr[0] + y_fnr[0]) / 2.0)
        threshold = float(y_thr[0])
        return eer, threshold

    # Linear interpolation to find where diff == 0.
    t = x[0] / (x[0] - x[1])
    eer = float(y_fpr[0] + t * (y_fpr[1] - y_fpr[0]))
    threshold = float(y_thr[0] + t * (y_thr[1] - y_thr[0]))
    return eer, threshold


def apcer(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the Attack Presentation Classification Error Rate (APCER).

    APCER is the proportion of attack (morph, label=1) samples that are
    incorrectly classified as bonafide (label=0). Defined per ISO/IEC
    30107-3.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        APCER in ``[0, 1]``.

    Raises:
        ValueError: If no attack samples are present in ``y_true``.
    """
    cm = confusion_matrix(y_true, y_pred)
    fn, tp = cm[1, 0], cm[1, 1]
    n_attack = fn + tp
    if n_attack == 0:
        raise ValueError("APCER is undefined: no attack (label=1) samples present in y_true.")
    return float(fn / n_attack)


def bpcer(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the Bonafide Presentation Classification Error Rate (BPCER).

    BPCER is the proportion of bonafide (label=0) samples that are
    incorrectly classified as attack (label=1). Defined per ISO/IEC
    30107-3.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        BPCER in ``[0, 1]``.

    Raises:
        ValueError: If no bonafide samples are present in ``y_true``.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    n_bonafide = tn + fp
    if n_bonafide == 0:
        raise ValueError("BPCER is undefined: no bonafide (label=0) samples present in y_true.")
    return float(fp / n_bonafide)


def acer(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the Average Classification Error Rate (ACER).

    ACER = (APCER + BPCER) / 2

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        ACER in ``[0, 1]``.
    """
    return float((apcer(y_true, y_pred) + bpcer(y_true, y_pred)) / 2.0)


def far(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> float:
    """Compute the False Acceptance Rate (FAR) at a given score threshold.

    FAR is the proportion of attack (impostor) samples whose score is at
    or above ``threshold`` and are therefore incorrectly *accepted* as
    bonafide-equivalent (i.e., not rejected).

    Note the convention: ``y_score`` is the probability of the *attack*
    class. A sample is "accepted" (treated as bonafide) when
    ``score < threshold``.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted attack-class scores/probabilities.
        threshold: Decision threshold in ``[0, 1]``; scores ``>=
            threshold`` are classified as attack.

    Returns:
        FAR in ``[0, 1]``.

    Raises:
        ValueError: If no attack samples are present in ``y_true``.
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_score_arr = to_numpy(y_score).astype(np.float64)
    _validate_equal_length(y_true_arr, y_score_arr)

    attack_mask = y_true_arr == 1
    n_attack = int(np.sum(attack_mask))
    if n_attack == 0:
        raise ValueError("FAR is undefined: no attack (label=1) samples present in y_true.")

    accepted_as_bonafide = np.sum(attack_mask & (y_score_arr < threshold))
    return float(accepted_as_bonafide / n_attack)


def frr(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> float:
    """Compute the False Rejection Rate (FRR) at a given score threshold.

    FRR is the proportion of bonafide samples whose score is at or above
    ``threshold`` and are therefore incorrectly *rejected* (classified as
    attack).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted attack-class scores/probabilities.
        threshold: Decision threshold in ``[0, 1]``; scores ``>=
            threshold`` are classified as attack.

    Returns:
        FRR in ``[0, 1]``.

    Raises:
        ValueError: If no bonafide samples are present in ``y_true``.
    """
    y_true_arr = validate_binary_labels(y_true, "y_true")
    y_score_arr = to_numpy(y_score).astype(np.float64)
    _validate_equal_length(y_true_arr, y_score_arr)

    bonafide_mask = y_true_arr == 0
    n_bonafide = int(np.sum(bonafide_mask))
    if n_bonafide == 0:
        raise ValueError("FRR is undefined: no bonafide (label=0) samples present in y_true.")

    rejected = np.sum(bonafide_mask & (y_score_arr >= threshold))
    return float(rejected / n_bonafide)