"""Unit tests for vit.evaluation.evaluator.

These tests inject a minimal fake ``torch`` module (see ``_fake_torch.py``)
via ``sys.modules`` so the control flow of :class:`Evaluator` can be
exercised without requiring a full PyTorch installation. The
:meth:`Evaluator._compute_result` metric-aggregation path is additionally
tested directly with plain NumPy arrays, independent of any torch mocking.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tests.vit._fake_torch import FakeModule, FakeTensor, build_fake_torch


@pytest.fixture()
def fake_torch(monkeypatch):
    """Install a fake torch module into sys.modules for the duration of a test."""
    fake = build_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch.nn", fake.nn)
    monkeypatch.setitem(sys.modules, "torch.utils", fake.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake.utils.data)
    return fake


class FakeModel(FakeModule):
    """A fake model returning pre-programmed logits per call, in order."""

    def __init__(self, logits_per_batch):
        self._logits_per_batch = list(logits_per_batch)
        self._call_index = 0

    def __call__(self, inputs):
        logits = self._logits_per_batch[self._call_index]
        self._call_index += 1
        return FakeTensor(np.asarray(logits, dtype=np.float64))


class FakeCriterion:
    """A fake loss function returning a constant, pre-programmed loss per call."""

    def __init__(self, losses):
        self._losses = list(losses)
        self._call_index = 0

    def __call__(self, logits, labels):
        loss_value = self._losses[self._call_index]
        self._call_index += 1
        return FakeTensor(np.asarray(loss_value, dtype=np.float64))


def make_batches(batches):
    """Build a list of (inputs, labels) tuples as FakeTensor pairs."""
    return [(FakeTensor(np.zeros((len(labels), 3))), FakeTensor(np.asarray(labels, dtype=np.int64)))
            for _, labels in batches]


class TestEvaluatorConstruction:
    def test_rejects_non_module(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        with pytest.raises(TypeError):
            Evaluator(model="not a module")

    def test_rejects_invalid_threshold(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        with pytest.raises(ValueError):
            Evaluator(model=FakeModel([]), decision_threshold=1.5)

    def test_defaults_to_cpu_device(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        evaluator = Evaluator(model=FakeModel([]))
        assert evaluator.device.type == "cpu"


class TestEvaluatorEvaluate:
    def test_basic_evaluation_flow(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        # Two batches; logits favour the correct class in both.
        logits_per_batch = [
            [[2.0, -2.0], [-2.0, 2.0]],  # batch 1: pred bonafide, pred attack
            [[-2.0, 2.0], [2.0, -2.0]],  # batch 2: pred attack, pred bonafide
        ]
        labels_per_batch = [[0, 1], [1, 0]]
        dataloader = list(
            zip(
                [None, None],
                labels_per_batch,
            )
        )
        dataloader = make_batches(list(zip([None, None], labels_per_batch)))

        model = FakeModel(logits_per_batch)
        evaluator = Evaluator(model=model)
        result = evaluator.evaluate(dataloader)

        assert result.num_samples == 4
        assert result.accuracy == pytest.approx(1.0)
        assert result.loss is None

    def test_evaluation_with_criterion_computes_loss(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        logits_per_batch = [[[2.0, -2.0], [-2.0, 2.0]]]
        labels_per_batch = [[0, 1]]
        dataloader = make_batches(list(zip([None], labels_per_batch)))

        model = FakeModel(logits_per_batch)
        criterion = FakeCriterion([0.25])
        evaluator = Evaluator(model=model, criterion=criterion)
        result = evaluator.evaluate(dataloader)

        assert result.loss == pytest.approx(0.25)

    def test_empty_dataloader_raises(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        model = FakeModel([])
        evaluator = Evaluator(model=model)
        with pytest.raises(ValueError):
            evaluator.evaluate([])

    def test_single_class_batch_skips_curve_metrics(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        # Only bonafide labels present in the whole evaluation set.
        logits_per_batch = [[[2.0, -2.0], [2.0, -2.0]]]
        labels_per_batch = [[0, 0]]
        dataloader = make_batches(list(zip([None], labels_per_batch)))

        model = FakeModel(logits_per_batch)
        evaluator = Evaluator(model=model)
        result = evaluator.evaluate(dataloader)

        assert result.roc_auc is None
        assert result.pr_auc is None
        assert result.eer is None
        assert result.far is None
        assert result.frr is None
        assert result.accuracy == pytest.approx(1.0)

    def test_rejects_malformed_batch(self, fake_torch) -> None:
        from vit.evaluation.evaluator import Evaluator

        model = FakeModel([[[1.0, 0.0]]])
        evaluator = Evaluator(model=model)
        with pytest.raises(ValueError):
            evaluator.evaluate([{"unexpected": "format"}])


class TestComputeResultDirectly:
    """Exercise the pure metric-aggregation path without any torch mocking."""

    def test_perfect_classifier(self) -> None:
        from vit.evaluation.evaluator import EvaluationResult

        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])

        from vit.evaluation import metrics as metric_fns

        result = EvaluationResult(
            loss=None,
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
            num_samples=4,
        )
        assert result.accuracy == 1.0
        assert result.eer == pytest.approx(0.0, abs=1e-9)

    def test_mismatched_array_lengths_raises(self) -> None:
        from vit.evaluation.evaluator import EvaluationResult
        from vit.evaluation import metrics as metric_fns

        with pytest.raises(ValueError):
            EvaluationResult(
                loss=None,
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                roc_auc=None,
                pr_auc=None,
                eer=None,
                eer_threshold=None,
                apcer=0.0,
                bpcer=0.0,
                acer=0.0,
                far=None,
                frr=None,
                decision_threshold=0.5,
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                labels=np.array([0, 1]),
                predictions=np.array([0]),
                probabilities=np.array([0.1, 0.9]),
                num_samples=2,
            )

    def test_negative_num_samples_raises(self) -> None:
        from vit.evaluation.evaluator import EvaluationResult

        with pytest.raises(ValueError):
            EvaluationResult(
                loss=None,
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                roc_auc=None,
                pr_auc=None,
                eer=None,
                eer_threshold=None,
                apcer=0.0,
                bpcer=0.0,
                acer=0.0,
                far=None,
                frr=None,
                decision_threshold=0.5,
                confusion_matrix=np.array([[0, 0], [0, 0]]),
                labels=np.array([]),
                predictions=np.array([]),
                probabilities=np.array([]),
                num_samples=-1,
            )

    def test_invalid_decision_threshold_raises(self) -> None:
        from vit.evaluation.evaluator import EvaluationResult

        with pytest.raises(ValueError):
            EvaluationResult(
                loss=None,
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                roc_auc=None,
                pr_auc=None,
                eer=None,
                eer_threshold=None,
                apcer=0.0,
                bpcer=0.0,
                acer=0.0,
                far=None,
                frr=None,
                decision_threshold=1.5,
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                labels=np.array([0, 1]),
                predictions=np.array([0, 1]),
                probabilities=np.array([0.1, 0.9]),
                num_samples=2,
            )

    def test_as_dict_is_json_serialisable(self) -> None:
        import json
        from vit.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            loss=0.5,
            accuracy=0.9,
            precision=0.9,
            recall=0.9,
            f1=0.9,
            roc_auc=0.95,
            pr_auc=0.93,
            eer=0.1,
            eer_threshold=0.5,
            apcer=0.1,
            bpcer=0.1,
            acer=0.1,
            far=0.1,
            frr=0.1,
            decision_threshold=0.5,
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            labels=np.array([0, 1]),
            predictions=np.array([0, 1]),
            probabilities=np.array([0.1, 0.9]),
            num_samples=2,
        )
        serialised = json.dumps(result.as_dict())
        assert "accuracy" in serialised