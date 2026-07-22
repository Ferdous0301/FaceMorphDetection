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

from vit.engine.early_stopping import EarlyStopping
from vit.engine.evaluator import EvaluationResult, Evaluator
from vit.engine.loss import FocalLoss, build_loss
from vit.engine.optimizer_factory import build_optimizer
from vit.engine.scheduler_factory import build_scheduler
from vit.engine.trainer import EpochResult, Trainer, TrainingHistory

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