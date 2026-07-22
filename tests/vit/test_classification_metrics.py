"""Unit tests for vit.metrics.classification_metrics."""

from __future__ import annotations

import pytest
import torch

from vit.metrics.classification_metrics import (
    compute_accuracy,
    compute_apcer_bpcer,
    compute_auc,
    compute_confusion_matrix,
    compute_eer,
)


class TestComputeAccuracy:
    def test_all_correct(self) -> None:
        assert compute_accuracy(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 1])) == 1.0

    def test_all_wrong(self) -> None:
        assert compute_accuracy(torch.tensor([0, 0]), torch.tensor([1, 1])) == 0.0

    def test_partial(self) -> None:
        acc = compute_accuracy(torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 0]))
        assert acc == pytest.approx(0.75)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_accuracy(torch.tensor([0, 1]), torch.tensor([0]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_accuracy(torch.tensor([]), torch.tensor([]))


class TestComputeConfusionMatrix:
    def test_known_matrix(self) -> None:
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0])
        cm = compute_confusion_matrix(preds, labels, num_classes=2)
        expected = torch.tensor([[2, 1], [0, 1]])
        assert torch.equal(cm, expected)

    def test_diagonal_when_all_correct(self) -> None:
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        cm = compute_confusion_matrix(preds, labels, num_classes=3)
        assert torch.equal(cm, torch.eye(3, dtype=torch.long))

    def test_non_positive_num_classes_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_confusion_matrix(torch.tensor([0]), torch.tensor([0]), num_classes=0)

    def test_out_of_range_value_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_confusion_matrix(torch.tensor([2]), torch.tensor([0]), num_classes=2)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_confusion_matrix(torch.tensor([0, 1]), torch.tensor([0]), num_classes=2)


class TestComputeAuc:
    def test_perfect_separation(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        assert compute_auc(scores, labels) == pytest.approx(1.0)

    def test_worst_case_separation(self) -> None:
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [0, 0, 1, 1]
        assert compute_auc(scores, labels) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # Hand-computable case: 2 negatives, 2 positives, one tie region.
        scores = [0.1, 0.4, 0.35, 0.8]
        labels = [0, 0, 1, 1]
        assert compute_auc(scores, labels) == pytest.approx(0.75)

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_auc([0.1, 0.2, 0.3], [0, 0, 0])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_auc([0.1, 0.2], [0, 1, 1])


class TestComputeEer:
    def test_perfect_separation_gives_zero_eer(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        eer, threshold = compute_eer(scores, labels)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_eer_in_valid_range(self) -> None:
        scores = [0.1, 0.6, 0.4, 0.9, 0.3, 0.7]
        labels = [0, 1, 0, 1, 1, 0]
        eer, threshold = compute_eer(scores, labels)
        assert 0.0 <= eer <= 1.0

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_eer([0.1, 0.2], [1, 1])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_eer([0.1, 0.2, 0.3], [0, 1])


class TestComputeApcerBpcer:
    def test_known_values(self) -> None:
        # scores/labels: 2 bona fide (0), 2 attack (1); threshold=0.5
        scores = [0.1, 0.4, 0.35, 0.8]
        labels = [0, 0, 1, 1]
        # predicted attack: [F, F, F, T] (0.35 < 0.5!)
        # attacks (idx 2,3): predicted [F, T] -> 1 of 2 misclassified as bona fide -> APCER=0.5
        # bonafide (idx 0,1): predicted [F, F] -> 0 misclassified as attack -> BPCER=0.0
        apcer, bpcer = compute_apcer_bpcer(scores, labels, threshold=0.5)
        assert apcer == pytest.approx(0.5)
        assert bpcer == pytest.approx(0.0)

    def test_perfect_classification_gives_zero_rates(self) -> None:
        scores = [0.0, 0.1, 0.9, 1.0]
        labels = [0, 0, 1, 1]
        apcer, bpcer = compute_apcer_bpcer(scores, labels, threshold=0.5)
        assert apcer == pytest.approx(0.0)
        assert bpcer == pytest.approx(0.0)

    def test_no_attack_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_apcer_bpcer([0.1, 0.2], [0, 0], threshold=0.5)

    def test_no_bonafide_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_apcer_bpcer([0.8, 0.9], [1, 1], threshold=0.5)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_apcer_bpcer([0.1, 0.2], [0, 1, 1], threshold=0.5)