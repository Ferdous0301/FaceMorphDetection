"""Unit tests for vit.evaluation.results_exporter."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from vit.evaluation import metrics as metric_fns
from vit.evaluation.evaluator import EvaluationResult
from vit.evaluation.prediction_analyzer import (
    MisclassificationRecord,
    PredictionAnalyzer,
    PredictionRecord,
)
from vit.evaluation.results_exporter import ResultsExporter


@pytest.fixture()
def sample_result() -> EvaluationResult:
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.9])
    return EvaluationResult(
        loss=0.35,
        accuracy=metric_fns.accuracy(labels, predictions),
        precision=metric_fns.precision(labels, predictions),
        recall=metric_fns.recall(labels, predictions),
        f1=metric_fns.f1_score(labels, predictions),
        roc_auc=metric_fns.roc_auc(labels, probabilities),
        pr_auc=metric_fns.pr_auc(labels, probabilities),
        eer=metric_fns.equal_error_rate(labels, probabilities)[0],
        eer_threshold=metric_fns.equal_error_rate(labels, probabilities)[1],
        apcer=metric_fns.apcer(labels, predictions),
        bpcer=metric_fns.bpcer(labels, predictions),
        acer=metric_fns.acer(labels, predictions),
        far=metric_fns.far(labels, probabilities, 0.5),
        frr=metric_fns.frr(labels, probabilities, 0.5),
        decision_threshold=0.5,
        confusion_matrix=metric_fns.confusion_matrix(labels, predictions),
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        num_samples=6,
    )


@pytest.fixture()
def prediction_records(sample_result) -> list:
    return PredictionAnalyzer.from_evaluation_result(
        sample_result.labels, sample_result.predictions, sample_result.probabilities
    )


@pytest.fixture()
def misclassification_records(prediction_records) -> list:
    return [
        MisclassificationRecord.from_prediction_record(r)
        for r in prediction_records
        if not r.is_correct
    ]


class TestResultsExporterConstruction:
    def test_creates_output_directory(self, tmp_path) -> None:
        output_dir = tmp_path / "results" / "nested"
        ResultsExporter(output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestExportMetricsJson:
    def test_writes_valid_json(self, tmp_path, sample_result) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_metrics_json(sample_result)
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert data["accuracy"] == pytest.approx(sample_result.accuracy)
        assert data["num_samples"] == 6
        assert isinstance(data["confusion_matrix"], list)

    def test_custom_filename(self, tmp_path, sample_result) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_metrics_json(sample_result, filename="custom_metrics.json")
        assert path.name == "custom_metrics.json"
        assert path.exists()


class TestExportClassificationReportCsv:
    def test_writes_valid_csv(self, tmp_path, sample_result) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_classification_report_csv(sample_result)
        assert path.exists()
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        assert header == ["class", "precision", "recall", "f1_score", "support"]
        class_names = [row[0] for row in rows[1:]]
        assert "bonafide" in class_names
        assert "attack" in class_names
        assert "accuracy" in class_names


class TestExportPredictionsCsv:
    def test_writes_all_records(self, tmp_path, prediction_records) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_predictions_csv(prediction_records)
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["index", "true_label", "predicted_label", "probability", "is_correct", "image_path"]
        assert len(rows) - 1 == len(prediction_records)

    def test_empty_records_writes_header_only(self, tmp_path) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_predictions_csv([])
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 1


class TestExportMisclassifiedCsv:
    def test_writes_only_misclassified(self, tmp_path, misclassification_records) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_misclassified_csv(misclassification_records)
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["index", "true_label", "predicted_label", "probability", "error_type", "image_path"]
        assert len(rows) - 1 == len(misclassification_records)
        for row in rows[1:]:
            assert row[4] in ("false_positive", "false_negative")

    def test_perfect_classifier_produces_empty_misclassified(self, tmp_path) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_misclassified_csv([])
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 1  # header only


class TestExportExperimentSummaryJson:
    def test_writes_valid_json(self, tmp_path, sample_result) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_experiment_summary_json(sample_result, "test_run", notes="hello")
        with path.open() as fh:
            data = json.load(fh)
        assert data["experiment_name"] == "test_run"
        assert data["notes"] == "hello"


class TestExportMarkdownReport:
    def test_writes_markdown_file(self, tmp_path, sample_result) -> None:
        exporter = ResultsExporter(tmp_path)
        path = exporter.export_markdown_report(sample_result, "test_run_md")
        content = path.read_text()
        assert "# Evaluation Report: test_run_md" in content


class TestExportAll:
    def test_writes_full_artifact_set(
        self, tmp_path, sample_result, prediction_records, misclassification_records
    ) -> None:
        exporter = ResultsExporter(tmp_path)
        exporter.export_all(
            sample_result,
            prediction_records,
            misclassification_records,
            experiment_name="full_run",
            notes="full export test",
        )
        expected_files = {
            "metrics.json",
            "classification_report.csv",
            "predictions.csv",
            "misclassified.csv",
            "experiment_summary.json",
        }
        actual_files = {p.name for p in tmp_path.iterdir()}
        assert expected_files.issubset(actual_files)

        with (tmp_path / "metrics.json").open() as fh:
            metrics_data = json.load(fh)
        assert metrics_data["num_samples"] == 6

        with (tmp_path / "experiment_summary.json").open() as fh:
            summary_data = json.load(fh)
        assert summary_data["experiment_name"] == "full_run"