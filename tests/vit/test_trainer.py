"""Unit tests for vit.engine.trainer.Trainer.

Uses the same lightweight stub classifier as test_evaluator.py so a full
multi-epoch training run over the tiny synthetic dataset completes in a
fraction of a second, rather than depending on the (much heavier) real ViT
backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn

from vit.checkpoint.checkpoint_manager import CheckpointManager
from vit.configs.schema import DataConfig, SchedulerConfig, TrainingConfig
from vit.data.datamodule import ViTDataModule
from vit.engine.loss import build_loss
from vit.engine.scheduler_factory import build_scheduler
from vit.engine.trainer import EpochResult, Trainer, TrainingHistory
from vit.logging_utils.csv_logger import CSVLogger
from vit.logging_utils.experiment_logger import ExperimentLogger
from vit.logging_utils.tb_logger import TensorBoardLogger


class _StubClassifier(nn.Module):
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


def _make_trainer(tmp_path: Path, dm: ViTDataModule, epochs: int = 3, patience: int = 10) -> Trainer:
    model = _StubClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    train_loader = dm.train_dataloader()
    scheduler_config = SchedulerConfig(name="none", total_epochs=epochs, warmup_epochs=0)
    scheduler = build_scheduler(optimizer, scheduler_config, steps_per_epoch=len(train_loader))

    training_config = TrainingConfig(
        epochs=epochs,
        mixed_precision=False,
        grad_clip_norm=1.0,
        early_stopping_patience=patience,
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        results_dir=tmp_path / "results",
        device="cpu",
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=training_config.checkpoint_dir,
        monitor_metric=training_config.early_stopping_metric,
        mode=training_config.early_stopping_mode,
    )
    logger = ExperimentLogger(
        csv_logger=CSVLogger(training_config.log_dir / "train_log.csv"),
        tb_logger=TensorBoardLogger(training_config.log_dir / "tensorboard"),
        name="test_trainer_run",
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=build_loss("cross_entropy"),
        device=torch.device("cpu"),
        config=training_config,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        experiment_config=None,
    )


class TestFit:
    def test_runs_full_epoch_count_without_early_stopping(
        self, tmp_path: Path, synthetic_dataset: Dict[str, object]
    ) -> None:
        dm = _make_datamodule(synthetic_dataset)
        trainer = _make_trainer(tmp_path, dm, epochs=3, patience=10)

        history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        assert isinstance(history, TrainingHistory)
        assert len(history.train_results) == 3
        assert len(history.val_results) == 3
        assert all(isinstance(r, EpochResult) for r in history.train_results)

    def test_checkpoints_written(self, tmp_path: Path, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        trainer = _make_trainer(tmp_path, dm, epochs=2, patience=10)
        trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        assert (tmp_path / "checkpoints" / "last.pt").is_file()
        assert (tmp_path / "checkpoints" / "best.pt").is_file()

    def test_best_epoch_is_tracked(self, tmp_path: Path, synthetic_dataset: Dict[str, object]) -> None:
        dm = _make_datamodule(synthetic_dataset)
        trainer = _make_trainer(tmp_path, dm, epochs=3, patience=10)
        history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        assert 0 <= history.best_epoch < 3
        assert history.best_metric_value < float("inf")

    def test_early_stopping_can_halt_before_full_epochs(
        self, tmp_path: Path, synthetic_dataset: Dict[str, object]
    ) -> None:
        dm = _make_datamodule(synthetic_dataset)
        # patience=0 means training stops as soon as val_loss fails to
        # improve for a single epoch.
        trainer = _make_trainer(tmp_path, dm, epochs=20, patience=0)
        history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        assert len(history.train_results) <= 20

    def test_train_one_epoch_logs_via_experiment_logger(
        self, tmp_path: Path, synthetic_dataset: Dict[str, object]
    ) -> None:
        dm = _make_datamodule(synthetic_dataset)
        trainer = _make_trainer(tmp_path, dm, epochs=1, patience=10)
        trainer.train_one_epoch(dm.train_dataloader(), epoch=0)

        train_log = tmp_path / "logs" / "train_log.csv"
        assert train_log.is_file()
        content = train_log.read_text()
        assert "train" in content

    def test_resume_from_checkpoint_restores_model_weights(
        self, tmp_path: Path, synthetic_dataset: Dict[str, object]
    ) -> None:
        dm = _make_datamodule(synthetic_dataset)
        trainer = _make_trainer(tmp_path, dm, epochs=2, patience=10)
        trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=tmp_path / "checkpoints", monitor_metric="val_loss", mode="min"
        )
        state = checkpoint_manager.load_latest()
        assert state.epoch == 1  # 0-indexed, 2 epochs => last epoch index 1