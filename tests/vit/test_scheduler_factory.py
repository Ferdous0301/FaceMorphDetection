"""Unit tests for vit.engine.scheduler_factory.build_scheduler."""

from __future__ import annotations

import pytest
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR

from vit.configs.schema import SchedulerConfig
from vit.engine.scheduler_factory import _NoOpScheduler, build_scheduler


@pytest.fixture
def optimizer() -> SGD:
    model = nn.Linear(4, 2)
    return SGD(model.parameters(), lr=0.1)


class TestBuildScheduler:
    def test_cosine_returns_lambda_lr_with_batch_stepping(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="cosine", warmup_epochs=1, total_epochs=5, min_lr=1e-5)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        assert isinstance(scheduler, LambdaLR)
        assert scheduler.step_every == "batch"

    def test_step_returns_step_lr_with_epoch_stepping(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="step", total_epochs=9, warmup_epochs=0)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_every == "epoch"
        assert scheduler.step_size == 3  # total_epochs // 3

    def test_plateau_returns_reduce_lr_on_plateau(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="plateau", total_epochs=5, warmup_epochs=0, min_lr=1e-6)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.step_every == "epoch"

    def test_none_returns_noop_scheduler(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="none", total_epochs=5, warmup_epochs=0)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        assert isinstance(scheduler, _NoOpScheduler)

    def test_noop_scheduler_never_changes_lr(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="none", total_epochs=5, warmup_epochs=0)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        before = optimizer.param_groups[0]["lr"]
        for _ in range(20):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == before

    def test_non_positive_steps_per_epoch_raises(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="cosine", total_epochs=5, warmup_epochs=1)
        with pytest.raises(ValueError):
            build_scheduler(optimizer, config, steps_per_epoch=0)

    def test_cosine_lr_increases_during_warmup(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="cosine", warmup_epochs=2, total_epochs=10, min_lr=1e-6)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=5)
        lrs = []
        for _ in range(10):  # 2 warmup epochs * 5 steps/epoch
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should be non-decreasing throughout warmup.
        assert all(b >= a - 1e-12 for a, b in zip(lrs, lrs[1:]))

    def test_cosine_lr_decreases_after_warmup(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="cosine", warmup_epochs=1, total_epochs=10, min_lr=1e-6)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=5)
        # Step past warmup (5 steps) into decay phase.
        for _ in range(5):
            scheduler.step()
        lr_at_decay_start = optimizer.param_groups[0]["lr"]
        for _ in range(20):
            scheduler.step()
        lr_later = optimizer.param_groups[0]["lr"]
        assert lr_later < lr_at_decay_start

    def test_cosine_lr_approaches_min_lr_at_end(self, optimizer: SGD) -> None:
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr = base_lr * 0.01
        config = SchedulerConfig(name="cosine", warmup_epochs=0, total_epochs=2, min_lr=min_lr)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        for _ in range(20):  # full schedule length
            scheduler.step()
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr == pytest.approx(min_lr, rel=0.05)

    def test_scheduler_state_dict_round_trip(self, optimizer: SGD) -> None:
        config = SchedulerConfig(name="step", total_epochs=9, warmup_epochs=0)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
        scheduler.step()
        state = scheduler.state_dict()

        scheduler2 = build_scheduler(optimizer, config, steps_per_epoch=10)
        scheduler2.load_state_dict(state)
        assert scheduler2.state_dict()["last_epoch"] == state["last_epoch"]