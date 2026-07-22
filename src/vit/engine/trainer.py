"""Training loop orchestration.

``Trainer`` owns the full train/validation loop: forward/backward passes,
AMP scaling, gradient clipping, scheduler stepping, early stopping,
checkpoint saving, and logging dispatch. It delegates metric computation on
the validation split to :class:`vit.engine.evaluator.Evaluator` so the exact
same code path is used for per-epoch validation and standalone test
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from vit.checkpoint.checkpoint_manager import CheckpointManager, CheckpointState
from vit.configs.schema import ExperimentConfig, TrainingConfig
from vit.engine.early_stopping import EarlyStopping
from vit.engine.evaluator import Evaluator
from vit.logging_utils.experiment_logger import ExperimentLogger
from vit.metrics.classification_metrics import compute_accuracy, compute_auc, compute_eer
from vit.metrics.metric_tracker import MetricTracker
from vit.utils.amp import autocast_context, build_grad_scaler

__all__ = ["Trainer", "EpochResult", "TrainingHistory"]

_ATTACK_CLASS_INDEX = 1


@dataclass
class EpochResult:
    """Aggregate metrics for a single epoch (either train or val split)."""

    epoch: int
    loss: float
    accuracy: float
    eer: float
    auc: float
    learning_rate: float


@dataclass
class TrainingHistory:
    """Full per-epoch history accumulated over a call to :meth:`Trainer.fit`."""

    train_results: List[EpochResult] = field(default_factory=list)
    val_results: List[EpochResult] = field(default_factory=list)
    best_epoch: int = -1
    best_metric_value: float = float("inf")


class Trainer:
    """Owns the training/validation loop for a single experiment run.

    Args:
        model: Model to train. Must already be moved to ``device``.
        optimizer: Optimizer bound to ``model``'s parameters.
        scheduler: Learning-rate scheduler bound to ``optimizer``.
        loss_fn: Loss function used for both training and validation.
        device: Device to train on.
        config: Training hyperparameters (epochs, AMP, grad clipping,
            early-stopping settings, etc.).
        checkpoint_manager: Handles saving/rotating checkpoints each epoch.
        logger: Unified logging facade for console/CSV/TensorBoard output.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_fn: nn.Module,
        device: torch.device,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager,
        logger: ExperimentLogger,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_fn = loss_fn
        self._device = device
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._logger = logger
        self._experiment_config = experiment_config

        self._amp_enabled = config.mixed_precision and device.type == "cuda"
        self._scaler = build_grad_scaler(enabled=self._amp_enabled)
        self._evaluator = Evaluator(model=model, device=device, loss_fn=loss_fn)
        self._early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.early_stopping_mode,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        """Run the full training loop for up to ``config.epochs`` epochs.

        Each epoch trains one pass over ``train_loader``, validates over
        ``val_loader``, logs both splits, saves a checkpoint, and checks the
        early-stopping criterion (monitoring ``config.early_stopping_metric``
        from the validation results). Training stops early if the criterion
        fires, otherwise after ``config.epochs`` epochs.

        Args:
            train_loader: Training-split dataloader.
            val_loader: Validation-split dataloader.

        Returns:
            A :class:`TrainingHistory` with every epoch's train/val results
            and a pointer to the best epoch/metric value observed.
        """
        history = TrainingHistory()
        history.best_metric_value = (
            float("inf") if self._config.early_stopping_mode == "min" else float("-inf")
        )

        for epoch in range(self._config.epochs):
            train_result = self.train_one_epoch(train_loader, epoch)
            val_result = self.validate_one_epoch(val_loader, epoch)

            history.train_results.append(train_result)
            history.val_results.append(val_result)

            monitored_value = self._extract_monitored_value(val_result)

            if getattr(self._scheduler, "step_every", "batch") == "epoch":
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(monitored_value)
                else:
                    self._scheduler.step()

            is_best = self._is_better(monitored_value, history.best_metric_value, history)
            if is_best:
                history.best_metric_value = monitored_value
                history.best_epoch = epoch

            self._checkpoint_manager.save(self._build_checkpoint_state(epoch), is_best=is_best)

            if self._early_stopping.step(monitored_value):
                break

        return history

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> EpochResult:
        """Run one training pass over ``loader``.

        Performs forward/backward with optional AMP autocast + gradient
        scaling, optional gradient-norm clipping, an optimizer step, and a
        scheduler step per batch. Logs the epoch's aggregate metrics via
        ``self._logger`` under the ``"train"`` split.

        Args:
            loader: Training-split dataloader.
            epoch: Current epoch index (0-based), used for logging and the
                returned :class:`EpochResult`.

        Returns:
            The epoch's aggregate :class:`EpochResult`.
        """
        self._model.train()
        tracker = MetricTracker()

        all_scores: List[Tensor] = []
        all_preds: List[Tensor] = []
        all_labels: List[Tensor] = []

        for images, labels, _image_ids in loader:
            images = images.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            batch_size = images.size(0)

            self._optimizer.zero_grad(set_to_none=True)

            with autocast_context(enabled=self._amp_enabled, device_type=self._device.type):
                logits = self._model(images)
                loss = self._loss_fn(logits, labels)

            self._scaler.scale(loss).backward()

            if self._config.grad_clip_norm is not None:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip_norm)

            self._scaler.step(self._optimizer)
            self._scaler.update()
            if getattr(self._scheduler, "step_every", "batch") == "batch":
                self._scheduler.step()

            tracker.update("loss", float(loss.item()), batch_size=batch_size)

            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=1)
                all_scores.append(probs[:, _ATTACK_CLASS_INDEX].cpu())
                all_preds.append(torch.argmax(logits.detach(), dim=1).cpu())
                all_labels.append(labels.detach().cpu())

        result = self._build_epoch_result(epoch, tracker, all_scores, all_preds, all_labels)
        self._logger.log_epoch(
            split="train",
            epoch=epoch,
            metrics={
                "loss": result.loss,
                "accuracy": result.accuracy,
                "eer": result.eer,
                "auc": result.auc,
                "learning_rate": result.learning_rate,
            },
        )
        return result

    def validate_one_epoch(self, loader: DataLoader, epoch: int) -> EpochResult:
        """Run one validation pass over ``loader`` via :class:`Evaluator`.

        Args:
            loader: Validation-split dataloader.
            epoch: Current epoch index (0-based), used for logging and the
                returned :class:`EpochResult`.

        Returns:
            The epoch's aggregate :class:`EpochResult`.
        """
        eval_result = self._evaluator.evaluate(loader)

        result = EpochResult(
            epoch=epoch,
            loss=eval_result.loss if eval_result.loss is not None else float("nan"),
            accuracy=eval_result.accuracy,
            eer=eval_result.eer,
            auc=eval_result.auc,
            learning_rate=self._current_lr(),
        )
        self._logger.log_epoch(
            split="val",
            epoch=epoch,
            metrics={
                "loss": result.loss,
                "accuracy": result.accuracy,
                "eer": result.eer,
                "auc": result.auc,
                "learning_rate": result.learning_rate,
            },
        )
        return result

    def _build_epoch_result(
        self,
        epoch: int,
        tracker: MetricTracker,
        all_scores: List[Tensor],
        all_preds: List[Tensor],
        all_labels: List[Tensor],
    ) -> EpochResult:
        scores = torch.cat(all_scores, dim=0)
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        accuracy = compute_accuracy(preds, labels)
        eer, _threshold = compute_eer(scores, labels)
        auc = compute_auc(scores, labels)

        return EpochResult(
            epoch=epoch,
            loss=tracker.average("loss"),
            accuracy=accuracy,
            eer=eer,
            auc=auc,
            learning_rate=self._current_lr(),
        )

    def _current_lr(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])

    def _extract_monitored_value(self, val_result: EpochResult) -> float:
        metric_name = self._config.early_stopping_metric.removeprefix("val_")
        try:
            return float(getattr(val_result, metric_name))
        except AttributeError as exc:
            raise ValueError(
                f"early_stopping_metric '{self._config.early_stopping_metric}' does not "
                f"correspond to a field on EpochResult"
            ) from exc

    def _is_better(self, candidate: float, current_best: float, history: TrainingHistory) -> bool:
        if not history.train_results or len(history.val_results) == 1:
            return True
        if self._config.early_stopping_mode == "min":
            return candidate < current_best
        return candidate > current_best

    def _build_checkpoint_state(self, epoch: int) -> CheckpointState:
        return CheckpointState(
            epoch=epoch,
            model_state_dict=self._model.state_dict(),
            optimizer_state_dict=self._optimizer.state_dict(),
            scheduler_state_dict=self._scheduler.state_dict(),
            scaler_state_dict=self._scaler.state_dict() if self._scaler.is_enabled() else None,
            best_metric_value=self._early_stopping.best_value,
            config=self._experiment_config,
            rng_state=self._capture_rng_state(),
        )

    @staticmethod
    def _capture_rng_state() -> dict:
        import random

        import numpy as np

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state