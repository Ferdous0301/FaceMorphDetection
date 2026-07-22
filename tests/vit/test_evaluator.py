"""Unit tests for vit.engine.evaluator.Evaluator.

Uses a tiny stub classifier (global-average-pool + linear) rather than the
real ViT backbone, since Evaluator's behaviour is entirely model-agnostic
and a stub keeps these tests fast.
"""

from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn

from vit.configs.schema import DataConfig
from vit.data.datamodule import ViTDataModule
from vit.engine.evaluator import Evaluator
from vit.engine.loss import build_loss


class _StubClassifier(nn.Module):
    """Global-average-pool + linear classifier; ignores spatial detail entirely.

    Trained briefly (a handful of gradient steps) inside fixtures so that it
    produces non-degenerate scores, since our synthetic bona-fide/morph
    images differ only by solid fill color (red vs green channel dominance),
    which this stub can trivially learn to separate.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


def _make_datamodule(synthetic_dataset: Dict[str, object], batch_size: int = 2) -> ViTDataModule:
    config = DataConfig(
        train_csv=synthetic_dataset["train_csv"],
        val_csv=synthetic_dataset["val_csv"],
        test_csv=synthetic_dataset["test_csv"],
        image_root=synthetic_dataset["image_root"],
        image_size=16,
        batch_size=batch_size,
        num_workers=0,
    )
    dm = ViTDataModule(config, seed=0)
    dm.setup()
    return dm


def _train_stub_briefly(model: _StubClassifier, dm: ViTDataModule, steps: int = 30) -> None:
    """Run a few plain SGD steps so the stub actually separates the two classes."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    loader = dm.train_dataloader()
    step_count = 0
    model.train()
    while step_count < steps:
        for images, labels, _ids in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break
    model.eval()


class TestEvaluate:
    def test_returns_all_expected_fields(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        model = _StubClassifier()
        _train_stub_briefly(model, dm)

        evaluator = Evaluator(model=model, device=torch.device("cpu"), loss_fn=build_loss("cross_entropy"))
        result = evaluator.evaluate(dm.val_dataloader())

        assert result.loss is not None
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.eer <= 1.0
        assert 0.0 <= result.auc <= 1.0
        assert 0.0 <= result.apcer <= 1.0
        assert 0.0 <= result.bpcer <= 1.0
        assert result.confusion_matrix.shape == (2, 2)
        assert len(result.image_ids) == len(synthetic_dataset["labels"]["val"])
        assert result.predictions.shape == result.labels.shape

    def test_perfectly_separable_data_yields_high_accuracy(
        self, synthetic_dataset: Dict[str, object]
    ) -> None:
        # Our synthetic images are solid-color and trivially separable by a
        # global-average-pool linear probe, so a well-trained stub should
        # nail this.
        dm = _make_datamodule(synthetic_dataset)
        model = _StubClassifier()
        _train_stub_briefly(model, dm, steps=100)

        evaluator = Evaluator(model=model, device=torch.device("cpu"))
        result = evaluator.evaluate(dm.test_dataloader())
        assert result.accuracy == pytest.approx(1.0)
        assert result.loss is None  # no loss_fn supplied

    def test_evaluate_empty_loader_raises(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        model = _StubClassifier()

        class _EmptyLoader:
            def __iter__(self):
                return iter([])

        evaluator = Evaluator(model=model, device=torch.device("cpu"))
        with pytest.raises(ValueError):
            evaluator.evaluate(_EmptyLoader())

    def test_predict_logits_shapes(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        model = _StubClassifier()
        evaluator = Evaluator(model=model, device=torch.device("cpu"))

        logits, labels, ids = evaluator.predict_logits(dm.test_dataloader())
        assert logits.shape == (len(synthetic_dataset["labels"]["test"]), 2)
        assert labels.shape == (len(synthetic_dataset["labels"]["test"]),)
        assert len(ids) == len(synthetic_dataset["labels"]["test"])

    def test_predict_logits_empty_loader_raises(self) -> None:
        model = _StubClassifier()
        evaluator = Evaluator(model=model, device=torch.device("cpu"))

        class _EmptyLoader:
            def __iter__(self):
                return iter([])

        with pytest.raises(ValueError):
            evaluator.predict_logits(_EmptyLoader())

    def test_model_left_in_eval_mode(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        model = _StubClassifier()
        model.train()
        evaluator = Evaluator(model=model, device=torch.device("cpu"))
        evaluator.evaluate(dm.val_dataloader())
        assert model.training is False