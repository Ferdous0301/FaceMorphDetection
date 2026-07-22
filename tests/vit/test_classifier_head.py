"""Unit tests for vit.models.classifier_head.ClassifierHead."""

from __future__ import annotations

import pytest
import torch

from vit.models.classifier_head import ClassifierHead


class TestConstruction:
    def test_rejects_non_positive_in_features(self) -> None:
        with pytest.raises(ValueError):
            ClassifierHead(in_features=0, num_classes=2)

    def test_rejects_non_positive_num_classes(self) -> None:
        with pytest.raises(ValueError):
            ClassifierHead(in_features=768, num_classes=0)

    @pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
    def test_rejects_invalid_dropout(self, dropout: float) -> None:
        with pytest.raises(ValueError):
            ClassifierHead(in_features=768, num_classes=2, dropout=dropout)

    def test_valid_construction(self) -> None:
        head = ClassifierHead(in_features=768, num_classes=2, dropout=0.1)
        assert head.in_features == 768
        assert head.num_classes == 2


class TestForward:
    def test_output_shape(self) -> None:
        head = ClassifierHead(in_features=768, num_classes=2)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 2)

    def test_multi_class_output_shape(self) -> None:
        head = ClassifierHead(in_features=128, num_classes=5)
        x = torch.randn(3, 128)
        out = head(x)
        assert out.shape == (3, 5)

    def test_gradients_flow(self) -> None:
        head = ClassifierHead(in_features=16, num_classes=2)
        x = torch.randn(2, 16, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert head.fc.weight.grad is not None

    def test_zero_dropout_is_deterministic_in_eval(self) -> None:
        head = ClassifierHead(in_features=16, num_classes=2, dropout=0.5)
        head.eval()
        x = torch.randn(2, 16)
        a = head(x)
        b = head(x)
        assert torch.equal(a, b)