"""Unit tests for vit.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vit.evaluation import metrics as m


class TestValidation:
    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError):
            m.validate_binary_labels([])

    def test_non_binary_labels_raises(self) -> None:
        with pytest.raises(ValueError):
            m.validate_binary_labels([0, 1, 2])

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError):
            m.accuracy([0, 1], [0, 1, 1])


class TestAccuracy:
    def test_perfect_classifier(self) -> None:
        assert m.accuracy([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_completely_wrong_classifier(self) -> None:
        assert m.accuracy([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_partial_accuracy(self) -> None:
        assert m.accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75

    def test_one_class_predictions(self) -> None:
        # All predictions are the bonafide class.
        assert m.accuracy([0, 0, 1, 1], [0, 0, 0, 0]) == 0.5


class TestConfusionMatrix:
    def test_known_values(self) -> None:
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        cm = m.confusion_matrix(y_true, y_pred)
        # TN=2, FP=1, FN=1, TP=2
        assert cm.tolist() == [[2, 1], [1, 2]]

    def test_perfect_classifier_matrix(self) -> None:
        cm = m.confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])
        assert cm.tolist() == [[2, 0], [0, 2]]

    def test_completely_wrong_matrix(self) -> None:
        cm = m.confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
        assert cm.tolist() == [[0, 2], [2, 0]]

    def test_all_bonafide_ground_truth(self) -> None:
        cm = m.confusion_matrix([0, 0, 0], [0, 1, 0])
        assert cm.tolist() == [[2, 1], [0, 0]]


class TestPrecisionRecallF1:
    def test_precision_recall_f1_known_values(self) -> None:
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        assert m.precision(y_true, y_pred) == pytest.approx(2 / 3)
        assert m.recall(y_true, y_pred) == pytest.approx(2 / 3)
        assert m.f1_score(y_true, y_pred) == pytest.approx(2 / 3)

    def test_precision_zero_division(self) -> None:
        # No positive predictions at all.
        assert m.precision([0, 1], [0, 0], zero_division=0.0) == 0.0

    def test_recall_zero_division(self) -> None:
        # No positive ground truth at all.
        assert m.recall([0, 0], [0, 1], zero_division=0.0) == 0.0

    def test_f1_zero_division(self) -> None:
        assert m.f1_score([0, 0], [0, 0], zero_division=0.0) == 0.0

    def test_perfect_classifier_scores(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        assert m.precision(y_true, y_pred) == 1.0
        assert m.recall(y_true, y_pred) == 1.0
        assert m.f1_score(y_true, y_pred) == 1.0


class TestROCAUC:
    def test_perfect_separation_gives_auc_one(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert m.roc_auc(y_true, y_score) == pytest.approx(1.0)

    def test_inverted_separation_gives_auc_zero(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.9, 0.8, 0.2, 0.1]
        assert m.roc_auc(y_true, y_score) == pytest.approx(0.0)

    def test_random_scores_give_auc_half(self) -> None:
        y_true = [0, 1, 0, 1]
        y_score = [0.5, 0.5, 0.5, 0.5]
        assert m.roc_auc(y_true, y_score) == pytest.approx(0.5)

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError):
            m.roc_auc([0, 0, 0], [0.1, 0.2, 0.3])

    def test_probability_ties(self) -> None:
        y_true = [0, 1, 0, 1]
        y_score = [0.3, 0.3, 0.7, 0.7]
        auc = m.roc_auc(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestPRAUC:
    def test_perfect_separation(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert m.pr_auc(y_true, y_score) == pytest.approx(1.0)

    def test_no_positive_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            m.pr_auc([0, 0, 0], [0.1, 0.2, 0.3])

    def test_bounded_between_zero_and_one(self) -> None:
        y_true = [0, 1, 0, 1, 1]
        y_score = [0.2, 0.9, 0.4, 0.3, 0.6]
        auc = m.pr_auc(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestEqualErrorRate:
    def test_perfect_separation_gives_zero_eer(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        eer, threshold = m.equal_error_rate(y_true, y_score)
        assert eer == pytest.approx(0.0, abs=1e-9)

    def test_worst_case_separation_gives_high_eer(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.9, 0.8, 0.2, 0.1]
        eer, _ = m.equal_error_rate(y_true, y_score)
        assert eer == pytest.approx(1.0, abs=1e-9)

    def test_eer_is_bounded(self) -> None:
        y_true = [0, 1, 0, 1, 1, 0]
        y_score = [0.2, 0.9, 0.4, 0.3, 0.6, 0.55]
        eer, threshold = m.equal_error_rate(y_true, y_score)
        assert 0.0 <= eer <= 1.0
        assert isinstance(threshold, float)

    def test_eer_matches_far_frr_crossing(self) -> None:
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
        eer, threshold = m.equal_error_rate(y_true, y_score)
        far_at_threshold = m.far(y_true, y_score, threshold)
        frr_at_threshold = m.frr(y_true, y_score, threshold)
        # At the EER threshold, FAR and FRR should be reasonably close.
        assert abs(far_at_threshold - frr_at_threshold) <= 0.5


class TestAPCERBPCERACER:
    def test_apcer_known_value(self) -> None:
        # 3 attacks, 1 misclassified as bonafide -> APCER = 1/3
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 0, 0]
        assert m.apcer(y_true, y_pred) == pytest.approx(1 / 3)

    def test_bpcer_known_value(self) -> None:
        # 4 bonafide, 1 misclassified as attack -> BPCER = 1/4
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 1, 1]
        assert m.bpcer(y_true, y_pred) == pytest.approx(0.25)

    def test_acer_is_average(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1, 0]
        expected = (m.apcer(y_true, y_pred) + m.bpcer(y_true, y_pred)) / 2
        assert m.acer(y_true, y_pred) == pytest.approx(expected)

    def test_apcer_perfect_classifier(self) -> None:
        assert m.apcer([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0

    def test_bpcer_perfect_classifier(self) -> None:
        assert m.bpcer([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0

    def test_apcer_completely_wrong(self) -> None:
        assert m.apcer([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0

    def test_bpcer_completely_wrong(self) -> None:
        assert m.bpcer([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0

    def test_apcer_no_attack_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            m.apcer([0, 0, 0], [0, 1, 0])

    def test_bpcer_no_bonafide_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            m.bpcer([1, 1, 1], [1, 0, 1])


class TestFARFRR:
    def test_far_known_value(self) -> None:
        y_true = [1, 1, 1, 0, 0]
        y_score = [0.9, 0.3, 0.8, 0.1, 0.2]
        # threshold=0.5: attack scores >= 0.5 classified attack.
        # attacks: 0.9(correct), 0.3(accepted as bonafide -> FAR error), 0.8(correct)
        assert m.far(y_true, y_score, 0.5) == pytest.approx(1 / 3)

    def test_frr_known_value(self) -> None:
        y_true = [0, 0, 0, 1]
        y_score = [0.1, 0.6, 0.2, 0.9]
        # bonafide: 0.1(accepted, correct), 0.6(rejected -> FRR error), 0.2(accepted, correct)
        assert m.frr(y_true, y_score, 0.5) == pytest.approx(1 / 3)

    def test_far_no_attacks_raises(self) -> None:
        with pytest.raises(ValueError):
            m.far([0, 0, 0], [0.1, 0.2, 0.3], 0.5)

    def test_frr_no_bonafide_raises(self) -> None:
        with pytest.raises(ValueError):
            m.frr([1, 1, 1], [0.1, 0.2, 0.3], 0.5)

    def test_far_zero_when_threshold_low(self) -> None:
        # threshold=0.0: nothing is "accepted" as bonafide, so FAR should be 0.
        assert m.far([1, 1], [0.4, 0.6], 0.0) == 0.0

    def test_frr_zero_when_threshold_high(self) -> None:
        # threshold=1.1: nothing is rejected (score never >= 1.1)
        assert m.frr([0, 0], [0.4, 0.6], 1.1) == 0.0