"""Evaluation stage for the Face Morph Attack Detection ViT pipeline.

This package consumes the outputs of the training stage (trained model,
logits, probabilities, predictions, labels, checkpoints) and provides:

* :mod:`vit.evaluation.metrics` -- pure, stateless metric functions.
* :mod:`vit.evaluation.evaluator` -- the :class:`Evaluator` inference
  harness and :class:`EvaluationResult` dataclass.
* :mod:`vit.evaluation.reports` -- publication-quality report generation.
* :mod:`vit.evaluation.results_exporter` -- JSON/CSV export utilities.
* :mod:`vit.evaluation.prediction_analyzer` -- error analysis utilities.

No training logic lives in this package.
"""

from src.vit.evaluation.evaluator import EvaluationResult, Evaluator
from src.vit.evaluation.prediction_analyzer import (
    MisclassificationRecord,
    PredictionAnalysis,
    PredictionAnalyzer,
    PredictionRecord,
)
from src.vit.evaluation.reports import ExperimentSummary
from src.vit.evaluation.results_exporter import ResultsExporter

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "ExperimentSummary",
    "ResultsExporter",
    "PredictionAnalyzer",
    "PredictionRecord",
    "MisclassificationRecord",
    "PredictionAnalysis",
]