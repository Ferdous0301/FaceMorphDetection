"""Unit tests for vit.evaluation.reports."""

from __future__ import annotations

import numpy as np
import pytest

from vit.evaluation.evaluator import EvaluationResult
from vit.evaluation import reports


def build_result(
    labels,
    predictions,
    probabilities,
    loss=0.3,
    roc_auc=0.9,
    pr_auc=0.85,
    eer=0.1,
    eer_threshold=0.5,
    apcer=0.1,
    bpcer=0.1,
    acer=0.1,
    far=0.1,
    frr=0.1,
) -> EvaluationResult:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    from vit.evaluation import metrics as metric_fns

    return EvaluationResult(
        loss=loss,
        accuracy=metric_fns.accuracy(labels, predictions),
        precision=metric_fns.precision(labels, predictions),
        recall=metric_fns.recall(labels, predictions),
        f1=metric_fns.f1_score(labels, predictions),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        eer=eer,
        eer_threshold=eer_threshold,
        apcer=apcer,
        bpcer=bpcer,
        acer=acer,
        far=far,
        frr=frr,
        decision_threshold=0.5,
        confusion_matrix=metric_fns.confusion_matrix(labels, predictions),
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        num_samples=len(labels),
    )


@pytest.fixture()
def sample_result() -> EvaluationResult:
    return build_result(
        labels=[0, 0, 0, 1, 1, 1],
        predictions=[0, 0, 1, 1, 0, 1],
        probabilities=[0.1, 0.4, 0.6, 0.8, 0.3, 0.9],
    )


class TestClassificationReport:
    def test_contains_both_classes(self, sample_result) -> None:
        text = reports.classification_report(sample_result)
        assert "bonafide" in text
        assert "attack" in text
        assert "accuracy" in text

    def test_custom_target_names(self, sample_result) -> None:
        text = reports.classification_report(sample_result, target_names={0: "real", 1: "morph"})
        assert "real" in text
        assert "morph" in text

    def test_perfect_classifier_report(self) -> None:
        result = build_result(labels=[0, 0, 1, 1], predictions=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.8, 0.9])
        text = reports.classification_report(result)
        assert "1.0000" in text

    def test_one_class_only(self) -> None:
        result = build_result(labels=[0, 0, 0], predictions=[0, 0, 0], probabilities=[0.1, 0.2, 0.3])
        text = reports.classification_report(result)
        assert "bonafide" in text


class TestConfusionMatrixSummary:
    def test_contains_expected_counts(self, sample_result) -> None:
        text = reports.confusion_matrix_summary(sample_result)
        assert "True: bonafide" in text
        assert "True: attack" in text
        assert "2" in text  # TN=2 for the sample result


class TestMetricTable:
    def test_contains_all_metrics(self, sample_result) -> None:
        text = reports.metric_table(sample_result)
        for expected in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "EER", "APCER", "BPCER", "ACER", "FAR", "FRR"]:
            assert expected in text

    def test_handles_none_values(self) -> None:
        result = build_result(
            labels=[0, 0, 0],
            predictions=[0, 0, 0],
            probabilities=[0.1, 0.2, 0.3],
            roc_auc=None,
            pr_auc=None,
            eer=None,
            eer_threshold=None,
            far=None,
            frr=None,
        )
        text = reports.metric_table(result)
        assert "None" in text


class TestExperimentSummary:
    def test_build_experiment_summary(self, sample_result) -> None:
        summary = reports.build_experiment_summary(sample_result, "run_001", notes="test run")
        assert summary.experiment_name == "run_001"
        assert summary.notes == "test run"
        assert summary.num_samples == 6

    def test_empty_experiment_name_raises(self, sample_result) -> None:
        with pytest.raises(ValueError):
            reports.build_experiment_summary(sample_result, "", notes="")

    def test_negative_num_samples_raises(self) -> None:
        from vit.evaluation.reports import ExperimentSummary

        with pytest.raises(ValueError):
            ExperimentSummary(
                experiment_name="x",
                timestamp="2026-01-01T00:00:00Z",
                num_samples=-1,
                accuracy=0.5,
                f1=0.5,
                roc_auc=None,
                eer=None,
                acer=0.1,
            )

    def test_as_dict_json_serialisable(self, sample_result) -> None:
        import json

        summary = reports.build_experiment_summary(sample_result, "run_002")
        serialised = json.dumps(summary.as_dict())
        assert "run_002" in serialised


class TestMarkdownReport:
    def test_contains_sections(self, sample_result) -> None:
        text = reports.markdown_report(sample_result, "run_003", notes="Some notes")
        assert "# Evaluation Report: run_003" in text
        assert "## Summary" in text
        assert "## Metrics" in text
        assert "## Classification Report" in text
        assert "## Confusion Matrix" in text
        assert "## Notes" in text
        assert "Some notes" in text

    def test_without_notes_omits_notes_section(self, sample_result) -> None:
        text = reports.markdown_report(sample_result, "run_004")
        assert "## Notes" not in text

    def test_none_roc_auc_reported_as_na(self) -> None:
        result = build_result(
            labels=[0, 0, 0],
            predictions=[0, 0, 0],
            probabilities=[0.1, 0.2, 0.3],
            roc_auc=None,
            eer=None,
        )
        text = reports.markdown_report(result, "run_005")
        assert "N/A" in text


class TestJsonReport:
    def test_structure(self, sample_result) -> None:
        payload = reports.json_report(sample_result, "run_006", notes="notes here")
        assert "summary" in payload
        assert "metrics" in payload
        assert payload["summary"]["experiment_name"] == "run_006"
        assert payload["metrics"]["accuracy"] == pytest.approx(sample_result.accuracy)

    def test_json_serialisable(self, sample_result) -> None:
        import json

        payload = reports.json_report(sample_result, "run_007")
        serialised = json.dumps(payload)
        assert "run_007" in serialised