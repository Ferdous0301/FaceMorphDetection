"""Training-engine building blocks: loss, optimizer/scheduler factories, trainer, evaluator, early stopping.

Public API:

    from vit.engine import (
        build_loss,
        FocalLoss,
        build_optimizer,
        build_scheduler,
        EarlyStopping,
        Evaluator,
        EvaluationResult,
        Trainer,
        EpochResult,
        TrainingHistory,
    )
"""

from __future__ import annotations

from src.vit.engine.early_stopping import EarlyStopping
from src.vit.engine.evaluator import EvaluationResult, Evaluator
from src.vit.engine.loss import FocalLoss, build_loss
from src.vit.engine.optimizer_factory import build_optimizer
from src.vit.engine.scheduler_factory import build_scheduler
from src.vit.engine.trainer import EpochResult, Trainer, TrainingHistory

__all__ = [
    "build_loss",
    "FocalLoss",
    "build_optimizer",
    "build_scheduler",
    "EarlyStopping",
    "Evaluator",
    "EvaluationResult",
    "Trainer",
    "EpochResult",
    "TrainingHistory",
]