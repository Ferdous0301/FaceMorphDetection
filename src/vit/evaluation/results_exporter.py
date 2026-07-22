"""Export evaluation results, predictions and error analyses to disk.

Writes the standard evaluation artefact set into a ``results/`` directory:

.. code-block:: text

    results/
        metrics.json
        classification_report.csv
        predictions.csv
        misclassified.csv
        experiment_summary.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from vit.evaluation import reports as report_fns
from vit.evaluation.evaluator import EvaluationResult
from vit.evaluation.prediction_analyzer import MisclassificationRecord, PredictionRecord

logger = logging.getLogger(__name__)

__all__ = ["ResultsExporter"]

PathLike = Union[str, Path]


class ResultsExporter:
    """Exports evaluation artefacts to a results directory.

    Args:
        output_dir: Directory into which artefacts are written. Created
            (including parents) if it does not already exist.

    Example:
        >>> exporter = ResultsExporter("results/run_001")
        >>> exporter.export_metrics_json(result)
        >>> exporter.export_predictions_csv(prediction_records)
    """

    def __init__(self, output_dir: PathLike) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ResultsExporter writing to %s", self.output_dir.resolve())

    def export_metrics_json(
        self, result: EvaluationResult, filename: str = "metrics.json"
    ) -> Path:
        """Write the scalar metrics of ``result`` to a JSON file.

        Args:
            result: The evaluation result to export.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, indent=2)
        logger.info("Wrote metrics JSON to %s", path)
        return path

    def export_classification_report_csv(
        self, result: EvaluationResult, filename: str = "classification_report.csv"
    ) -> Path:
        """Write a per-class classification report to a CSV file.

        Args:
            result: The evaluation result to export.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        labels = result.labels
        predictions = result.predictions
        names = {0: "bonafide", 1: "attack"}

        from vit.evaluation import metrics as metric_fns

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["class", "precision", "recall", "f1_score", "support"])
            for label_value in (0, 1):
                support = int(np.sum(labels == label_value))
                if support == 0:
                    precision_val = recall_val = f1_val = 0.0
                else:
                    binary_true = (labels == label_value).astype(int)
                    binary_pred = (predictions == label_value).astype(int)
                    precision_val = metric_fns.precision(binary_true, binary_pred)
                    recall_val = metric_fns.recall(binary_true, binary_pred)
                    f1_val = metric_fns.f1_score(binary_true, binary_pred)
                writer.writerow(
                    [names[label_value], f"{precision_val:.6f}", f"{recall_val:.6f}", f"{f1_val:.6f}", support]
                )
            writer.writerow(["accuracy", "", "", f"{result.accuracy:.6f}", int(len(labels))])
        logger.info("Wrote classification report CSV to %s", path)
        return path

    def export_predictions_csv(
        self, records: Sequence[PredictionRecord], filename: str = "predictions.csv"
    ) -> Path:
        """Write every prediction record to a CSV file.

        Args:
            records: Sequence of :class:`PredictionRecord` to export.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["index", "true_label", "predicted_label", "probability", "is_correct", "image_path"]
            )
            for record in records:
                writer.writerow(
                    [
                        record.index,
                        record.true_label,
                        record.predicted_label,
                        f"{record.probability:.6f}",
                        record.is_correct,
                        record.image_path or "",
                    ]
                )
        logger.info("Wrote %d predictions to %s", len(records), path)
        return path

    def export_misclassified_csv(
        self, records: Sequence[MisclassificationRecord], filename: str = "misclassified.csv"
    ) -> Path:
        """Write every misclassification record to a CSV file.

        Args:
            records: Sequence of :class:`MisclassificationRecord` to
                export.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["index", "true_label", "predicted_label", "probability", "error_type", "image_path"]
            )
            for record in records:
                writer.writerow(
                    [
                        record.index,
                        record.true_label,
                        record.predicted_label,
                        f"{record.probability:.6f}",
                        record.error_type,
                        record.image_path or "",
                    ]
                )
        logger.info("Wrote %d misclassified records to %s", len(records), path)
        return path

    def export_experiment_summary_json(
        self,
        result: EvaluationResult,
        experiment_name: str,
        notes: str = "",
        filename: str = "experiment_summary.json",
    ) -> Path:
        """Write an :class:`ExperimentSummary` to a JSON file.

        Args:
            result: The evaluation result to summarise.
            experiment_name: Human-readable identifier for this run.
            notes: Optional free-form notes.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        summary = report_fns.build_experiment_summary(result, experiment_name, notes)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary.as_dict(), fh, indent=2)
        logger.info("Wrote experiment summary JSON to %s", path)
        return path

    def export_markdown_report(
        self,
        result: EvaluationResult,
        experiment_name: str,
        notes: str = "",
        filename: str = "report.md",
    ) -> Path:
        """Write a full Markdown report to disk.

        Args:
            result: The evaluation result to summarise.
            experiment_name: Human-readable identifier for this run.
            notes: Optional free-form notes appended to the report.
            filename: Name of the output file, relative to ``output_dir``.

        Returns:
            The path to the written file.
        """
        path = self.output_dir / filename
        content = report_fns.markdown_report(result, experiment_name, notes)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote Markdown report to %s", path)
        return path

    def export_all(
        self,
        result: EvaluationResult,
        prediction_records: Sequence[PredictionRecord],
        misclassification_records: Sequence[MisclassificationRecord],
        experiment_name: str,
        notes: str = "",
    ) -> None:
        """Export the full standard artefact set in one call.

        Writes ``metrics.json``, ``classification_report.csv``,
        ``predictions.csv``, ``misclassified.csv`` and
        ``experiment_summary.json`` into ``output_dir``.

        Args:
            result: The evaluation result to export.
            prediction_records: All per-sample prediction records.
            misclassification_records: All misclassified sample records.
            experiment_name: Human-readable identifier for this run.
            notes: Optional free-form notes included in the summary.
        """
        self.export_metrics_json(result)
        self.export_classification_report_csv(result)
        self.export_predictions_csv(prediction_records)
        self.export_misclassified_csv(misclassification_records)
        self.export_experiment_summary_json(result, experiment_name, notes)
        logger.info("Exported full artefact set to %s", self.output_dir.resolve())