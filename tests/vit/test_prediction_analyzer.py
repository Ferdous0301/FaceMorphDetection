"""Unit tests for vit.evaluation.prediction_analyzer."""

from __future__ import annotations

import pytest

from vit.evaluation.prediction_analyzer import (
    MisclassificationRecord,
    PredictionAnalyzer,
    PredictionRecord,
)


def make_record(index, true_label, predicted_label, probability, image_path=None) -> PredictionRecord:
    return PredictionRecord(
        index=index,
        true_label=true_label,
        predicted_label=predicted_label,
        probability=probability,
        image_path=image_path,
    )


class TestPredictionRecordValidation:
    def test_invalid_true_label_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionRecord(index=0, true_label=2, predicted_label=0, probability=0.5)

    def test_invalid_predicted_label_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionRecord(index=0, true_label=0, predicted_label=5, probability=0.5)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionRecord(index=0, true_label=0, predicted_label=0, probability=1.5)

    def test_is_correct(self) -> None:
        assert make_record(0, 1, 1, 0.9).is_correct is True
        assert make_record(0, 1, 0, 0.9).is_correct is False

    def test_confidence_for_positive_prediction(self) -> None:
        record = make_record(0, 1, 1, 0.8)
        assert record.confidence == pytest.approx(0.8)

    def test_confidence_for_negative_prediction(self) -> None:
        record = make_record(0, 0, 0, 0.2)
        assert record.confidence == pytest.approx(0.8)


class TestMisclassificationRecord:
    def test_from_prediction_record_false_positive(self) -> None:
        record = make_record(3, 0, 1, 0.9)
        misclass = MisclassificationRecord.from_prediction_record(record)
        assert misclass.error_type == "false_positive"
        assert misclass.index == 3

    def test_from_prediction_record_false_negative(self) -> None:
        record = make_record(4, 1, 0, 0.1)
        misclass = MisclassificationRecord.from_prediction_record(record)
        assert misclass.error_type == "false_negative"

    def test_from_correct_prediction_raises(self) -> None:
        record = make_record(0, 1, 1, 0.9)
        with pytest.raises(ValueError):
            MisclassificationRecord.from_prediction_record(record)

    def test_invalid_error_type_raises(self) -> None:
        with pytest.raises(ValueError):
            MisclassificationRecord(
                index=0, true_label=0, predicted_label=1, probability=0.5, error_type="bogus"
            )

    def test_matching_labels_raises(self) -> None:
        with pytest.raises(ValueError):
            MisclassificationRecord(
                index=0,
                true_label=1,
                predicted_label=1,
                probability=0.5,
                error_type="false_negative",
            )


class TestPredictionAnalyzerConstruction:
    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionAnalyzer(top_k=0)

    def test_invalid_margin_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionAnalyzer(hard_example_margin=0.6)
        with pytest.raises(ValueError):
            PredictionAnalyzer(hard_example_margin=0.0)


class TestPredictionAnalyzerAnalysis:
    def test_identifies_false_positives_and_negatives(self) -> None:
        records = [
            make_record(0, 0, 0, 0.1),  # correct bonafide
            make_record(1, 0, 1, 0.9),  # false positive
            make_record(2, 1, 1, 0.8),  # correct attack
            make_record(3, 1, 0, 0.2),  # false negative
        ]
        analysis = PredictionAnalyzer(top_k=10).analyze(records)
        assert len(analysis.false_positives) == 1
        assert analysis.false_positives[0].index == 1
        assert len(analysis.false_negatives) == 1
        assert analysis.false_negatives[0].index == 3
        assert analysis.total_errors == 2
        assert analysis.total_samples == 4
        assert analysis.error_rate == pytest.approx(0.5)

    def test_highest_confidence_errors_sorted_descending(self) -> None:
        records = [
            make_record(0, 0, 1, 0.55),  # FP, low confidence (0.55)
            make_record(1, 0, 1, 0.99),  # FP, high confidence (0.99)
            make_record(2, 1, 0, 0.01),  # FN, high confidence (1-0.01=0.99)
        ]
        analysis = PredictionAnalyzer(top_k=10).analyze(records)
        confidences = [r.probability if r.predicted_label == 1 else 1 - r.probability
                       for r in analysis.highest_confidence_errors]
        assert confidences == sorted(confidences, reverse=True)
        assert analysis.highest_confidence_errors[0].index in (1, 2)

    def test_top_k_limits_result_size(self) -> None:
        records = [make_record(i, 0, 1, 0.9) for i in range(20)]
        analysis = PredictionAnalyzer(top_k=5).analyze(records)
        assert len(analysis.highest_confidence_errors) == 5
        assert len(analysis.lowest_confidence_predictions) == 5

    def test_lowest_confidence_predictions_sorted_ascending(self) -> None:
        records = [
            make_record(0, 0, 0, 0.5),   # confidence 0.5
            make_record(1, 1, 1, 0.99),  # confidence 0.99
            make_record(2, 0, 0, 0.05),  # confidence 0.95
        ]
        analysis = PredictionAnalyzer(top_k=10).analyze(records)
        confidences = [r.confidence for r in analysis.lowest_confidence_predictions]
        assert confidences == sorted(confidences)

    def test_hard_examples_within_margin(self) -> None:
        records = [
            make_record(0, 0, 0, 0.48),  # within 0.1 margin of 0.5
            make_record(1, 1, 1, 0.95),  # not hard
            make_record(2, 1, 1, 0.55),  # within margin
        ]
        analysis = PredictionAnalyzer(top_k=10, hard_example_margin=0.1).analyze(records)
        hard_indices = {r.index for r in analysis.hard_examples}
        assert hard_indices == {0, 2}

    def test_empty_records(self) -> None:
        analysis = PredictionAnalyzer(top_k=10).analyze([])
        assert analysis.total_samples == 0
        assert analysis.total_errors == 0
        assert analysis.false_positives == []
        assert analysis.false_negatives == []
        assert analysis.error_rate == 0.0

    def test_perfect_classifier_no_errors(self) -> None:
        records = [make_record(i, i % 2, i % 2, 0.9 if i % 2 else 0.1) for i in range(6)]
        analysis = PredictionAnalyzer(top_k=10).analyze(records)
        assert analysis.total_errors == 0
        assert analysis.false_positives == []
        assert analysis.false_negatives == []

    def test_completely_wrong_classifier_all_errors(self) -> None:
        records = [make_record(i, 0, 1, 0.9) for i in range(4)] + [
            make_record(i + 4, 1, 0, 0.1) for i in range(4)
        ]
        analysis = PredictionAnalyzer(top_k=10).analyze(records)
        assert analysis.total_errors == 8
        assert len(analysis.false_positives) == 4
        assert len(analysis.false_negatives) == 4

    def test_probability_ties(self) -> None:
        records = [make_record(i, 0, 1, 0.5) for i in range(3)]
        analysis = PredictionAnalyzer(top_k=10, hard_example_margin=0.05).analyze(records)
        assert len(analysis.hard_examples) == 3
        assert len(analysis.false_positives) == 3


class TestFromEvaluationResult:
    def test_builds_matching_records(self) -> None:
        labels = [0, 1, 0, 1]
        predictions = [0, 1, 1, 0]
        probabilities = [0.1, 0.9, 0.6, 0.4]
        records = PredictionAnalyzer.from_evaluation_result(labels, predictions, probabilities)
        assert len(records) == 4
        assert records[2].true_label == 0
        assert records[2].predicted_label == 1
        assert records[2].probability == pytest.approx(0.6)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionAnalyzer.from_evaluation_result([0, 1], [0], [0.1, 0.2])

    def test_with_image_paths(self) -> None:
        records = PredictionAnalyzer.from_evaluation_result(
            [0], [0], [0.2], image_paths=["/path/img.png"]
        )
        assert records[0].image_path == "/path/img.png"

    def test_image_paths_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictionAnalyzer.from_evaluation_result([0, 1], [0, 1], [0.1, 0.2], image_paths=["a"])