"""Unit tests for vit.engine.optimizer_factory.build_optimizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW

from vit.configs.schema import OptimizerConfig
from vit.engine.optimizer_factory import build_optimizer


@pytest.fixture
def tiny_model() -> nn.Module:
    return nn.Linear(4, 2)


class TestBuildOptimizer:
    def test_adamw(self, tiny_model: nn.Module) -> None:
        config = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01)
        opt = build_optimizer(tiny_model, config)
        assert isinstance(opt, AdamW)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)

    def test_adam(self, tiny_model: nn.Module) -> None:
        config = OptimizerConfig(name="adam", lr=5e-4)
        opt = build_optimizer(tiny_model, config)
        assert isinstance(opt, Adam)

    def test_sgd(self, tiny_model: nn.Module) -> None:
        config = OptimizerConfig(name="sgd", lr=0.01)
        opt = build_optimizer(tiny_model, config)
        assert isinstance(opt, SGD)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.9)

    def test_only_trainable_params_included(self, tiny_model: nn.Module) -> None:
        for p in tiny_model.parameters():
            p.requires_grad = False
        # re-enable just the bias
        tiny_model.bias.requires_grad = True

        config = OptimizerConfig(name="adamw")
        opt = build_optimizer(tiny_model, config)
        included_params = list(opt.param_groups[0]["params"])
        assert len(included_params) == 1
        assert included_params[0] is tiny_model.bias

    def test_fully_frozen_model_raises(self, tiny_model: nn.Module) -> None:
        for p in tiny_model.parameters():
            p.requires_grad = False
        config = OptimizerConfig(name="adamw")
        with pytest.raises(ValueError):
            build_optimizer(tiny_model, config)

    def test_optimizer_step_updates_weights(self, tiny_model: nn.Module) -> None:
        config = OptimizerConfig(name="sgd", lr=0.1)
        opt = build_optimizer(tiny_model, config)
        before = tiny_model.weight.clone()

        x = torch.randn(3, 4)
        loss = tiny_model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(before, tiny_model.weight)