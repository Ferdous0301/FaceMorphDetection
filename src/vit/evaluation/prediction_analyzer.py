"""Error analysis utilities for Face Morph Attack Detection predictions.

Given a flat list of per-sample predictions, :class:`PredictionAnalyzer`
identifies false positives, false negatives, the most confidently wrong
predictions, the least confident predictions overall, and "hard examples"
(samples whose predicted probability sits close to the decision boundary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PredictionRecord",
    "MisclassificationRecord",
    "PredictionAnalysis",
    "PredictionAnalyzer",
]


@dataclass(frozen=True)
class PredictionRecord:
    """A single sample's ground truth, prediction and confidence.

    Attributes:
        index: Index or identifier of the sample (e.g. its position in the
            evaluation dataset, or a dataset-provided sample id).
        true_label: Ground-truth label (``0``=bonafide, ``1``=attack).
        predicted_label: Predicted hard label (``0`` or ``1``).
        probability: Predicted probability of the attack (positive) class,
            in ``[0, 1]``.
        image_path: Optional path to the source image, used for
            downstream inspection/reporting.
    """

    index: int
    true_label: int
    predicted_label: int
    probability: float
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.true_label not in (0, 1):
            raise ValueError(f"true_label must be 0 or 1, got {self.true_label}.")
        if self.predicted_label not in (0, 1):
            raise ValueError(f"predicted_label must be 0 or 1, got {self.predicted_label}.")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be within [0, 1], got {self.probability}.")

    @property
    def is_correct(self) -> bool:
        """Whether the prediction matches the ground-truth label."""
        return self.true_label == self.predicted_label

    @property
    def confidence(self) -> float:
        """Confidence of the *predicted* class.

        This is ``probability`` when the predicted label is the attack
        class (1), and ``1 - probability`` when the predicted label is
        bonafide (0).
        """
        return self.probability if self.predicted_label == 1 else 1.0 - self.probability


@dataclass(frozen=True)
class MisclassificationRecord:
    """A record describing a single misclassified sample.

    Attributes:
        index: Index or identifier of the sample.
        true_label: Ground-truth label.
        predicted_label: Predicted hard label.
        probability: Predicted probability of the attack class.
        error_type: Either ``"false_positive"`` (bonafide misclassified as
            attack) or ``"false_negative"`` (attack misclassified as
            bonafide).
        image_path: Optional path to the source image.
    """

    index: int
    true_label: int
    predicted_label: int
    probability: float
    error_type: str
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.error_type not in ("false_positive", "false_negative"):
            raise ValueError(
                f"error_type must be 'false_positive' or 'false_negative', got {self.error_type!r}."
            )
        if self.true_label == self.predicted_label:
            raise ValueError("MisclassificationRecord requires true_label != predicted_label.")

    @classmethod
    def from_prediction_record(cls, record: PredictionRecord) -> "MisclassificationRecord":
        """Build a :class:`MisclassificationRecord` from a misclassified ``PredictionRecord``.

        Raises:
            ValueError: If ``record`` is not actually misclassified.
        """
        if record.is_correct:
            raise ValueError("Cannot build a MisclassificationRecord from a correct prediction.")
        error_type = "false_positive" if record.predicted_label == 1 else "false_negative"
        return cls(
            index=record.index,
            true_label=record.true_label,
            predicted_label=record.predicted_label,
            probability=record.probability,
            error_type=error_type,
            image_path=record.image_path,
        )


@dataclass(frozen=True)
class PredictionAnalysis:
    """Aggregated error-analysis results over a set of predictions.

    Attributes:
        false_positives: Bonafide samples misclassified as attack.
        false_negatives: Attack samples misclassified as bonafide.
        highest_confidence_errors: Misclassified samples sorted by
            descending prediction confidence (most "surprising" mistakes
            first).
        lowest_confidence_predictions: All predictions (correct and
            incorrect) sorted by ascending confidence (least certain
            predictions first).
        hard_examples: Predictions whose probability lies within
            ``hard_example_margin`` of the decision boundary (0.5 by
            default), regardless of correctness.
        total_samples: Total number of predictions analysed.
        total_errors: Total number of misclassified samples.
    """

    false_positives: List[MisclassificationRecord]
    false_negatives: List[MisclassificationRecord]
    highest_confidence_errors: List[MisclassificationRecord]
    lowest_confidence_predictions: List[PredictionRecord]
    hard_examples: List[PredictionRecord]
    total_samples: int
    total_errors: int

    @property
    def error_rate(self) -> float:
        """Overall error rate, ``total_errors / total_samples``."""
        if self.total_samples == 0:
            return 0.0
        return self.total_errors / self.total_samples


class PredictionAnalyzer:
    """Analyses a collection of :class:`PredictionRecord` for error patterns.

    Args:
        top_k: Number of records to retain in each of the "highest
            confidence errors" and "lowest confidence predictions" lists.
        hard_example_margin: Half-width of the probability band around the
            0.5 decision boundary used to flag "hard examples". Must be
            within ``(0, 0.5]``.
    """

    def __init__(self, top_k: int = 10, hard_example_margin: float = 0.1) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not (0.0 < hard_example_margin <= 0.5):
            raise ValueError("hard_example_margin must be within (0, 0.5].")
        self.top_k = top_k
        self.hard_example_margin = hard_example_margin

    def analyze(self, records: Sequence[PredictionRecord]) -> PredictionAnalysis:
        """Run full error analysis over ``records``.

        Args:
            records: A sequence of :class:`PredictionRecord`, typically
                built from an :class:`~vit.evaluation.evaluator.EvaluationResult`.

        Returns:
            A :class:`PredictionAnalysis` summarising errors and
            confidence patterns.
        """
        records = list(records)
        total_samples = len(records)

        misclassified = [r for r in records if not r.is_correct]
        misclassification_records = [
            MisclassificationRecord.from_prediction_record(r) for r in misclassified
        ]

        false_positives = [
            m for m in misclassification_records if m.error_type == "false_positive"
        ]
        false_negatives = [
            m for m in misclassification_records if m.error_type == "false_negative"
        ]

        misclassified_by_confidence = sorted(
            zip(misclassified, misclassification_records),
            key=lambda pair: pair[0].confidence,
            reverse=True,
        )
        highest_confidence_errors = [
            m for _, m in misclassified_by_confidence[: self.top_k]
        ]

        lowest_confidence_predictions = sorted(records, key=lambda r: r.confidence)[: self.top_k]

        hard_examples = [
            r for r in records if abs(r.probability - 0.5) <= self.hard_example_margin
        ]

        logger.info(
            "Analysed %d predictions: %d errors (%d FP, %d FN), %d hard examples.",
            total_samples,
            len(misclassified),
            len(false_positives),
            len(false_negatives),
            len(hard_examples),
        )

        return PredictionAnalysis(
            false_positives=false_positives,
            false_negatives=false_negatives,
            highest_confidence_errors=highest_confidence_errors,
            lowest_confidence_predictions=lowest_confidence_predictions,
            hard_examples=hard_examples,
            total_samples=total_samples,
            total_errors=len(misclassified),
        )

    @staticmethod
    def from_evaluation_result(
        labels: Sequence[int],
        predictions: Sequence[int],
        probabilities: Sequence[float],
        image_paths: Optional[Sequence[str]] = None,
    ) -> List[PredictionRecord]:
        """Build a list of :class:`PredictionRecord` from parallel arrays.

        This is a convenience constructor intended to bridge the flat
        NumPy arrays stored on an ``EvaluationResult`` and the
        record-oriented API used by :class:`PredictionAnalyzer`.

        Args:
            labels: Ground-truth labels.
            predictions: Predicted hard labels.
            probabilities: Predicted attack-class probabilities.
            image_paths: Optional parallel sequence of image paths.

        Returns:
            A list of :class:`PredictionRecord`, one per sample.

        Raises:
            ValueError: If input sequences have mismatched lengths.
        """
        n = len(labels)
        if len(predictions) != n or len(probabilities) != n:
            raise ValueError("labels, predictions and probabilities must have equal length.")
        if image_paths is not None and len(image_paths) != n:
            raise ValueError("image_paths must have the same length as labels, if provided.")

        return [
            PredictionRecord(
                index=i,
                true_label=int(labels[i]),
                predicted_label=int(predictions[i]),
                probability=float(probabilities[i]),
                image_path=image_paths[i] if image_paths is not None else None,
            )
            for i in range(n)
        ]