"""Model evaluation over a full dataloader.

``Evaluator`` runs a model in inference mode over a dataloader and produces
a full suite of morph-attack-detection metrics (accuracy, EER, AUC, APCER,
BPCER, confusion matrix) plus per-sample predictions for later error
analysis. It is used both by :class:`vit.engine.trainer.Trainer` for
per-epoch validation and standalone for final test-set evaluation via
``evaluate_cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from vit.metrics.classification_metrics import (
    compute_accuracy,
    compute_apcer_bpcer,
    compute_auc,
    compute_confusion_matrix,
    compute_eer,
)
from vit.metrics.metric_tracker import MetricTracker

__all__ = ["Evaluator", "EvaluationResult"]

# Index of the "attack" (morphed) class within the model's logits, matching
# the label convention documented in vit.metrics.classification_metrics.
_ATTACK_CLASS_INDEX = 1


@dataclass
class EvaluationResult:
    """Full set of results produced by :meth:`Evaluator.evaluate`.

    Attributes:
        loss: Mean loss over the dataset (``None`` if no ``loss_fn`` was
            supplied to the :class:`Evaluator`).
        accuracy: Fraction of correctly classified samples.
        eer: Equal Error Rate, see :func:`vit.metrics.classification_metrics.compute_eer`.
        eer_threshold: The score threshold at which EER was computed.
        auc: Area under the ROC curve.
        apcer: Attack Presentation Classification Error Rate at ``eer_threshold``.
        bpcer: Bona fide Presentation Classification Error Rate at ``eer_threshold``.
        confusion_matrix: ``(num_classes, num_classes)`` integer tensor.
        image_ids: Sample identifiers, aligned with ``predictions``/``labels``/``scores``.
        predictions: Predicted class index per sample.
        labels: Ground-truth class index per sample.
        scores: Attack-class probability per sample (used for EER/AUC/APCER/BPCER).
    """

    loss: Optional[float]
    accuracy: float
    eer: float
    eer_threshold: float
    auc: float
    apcer: float
    bpcer: float
    confusion_matrix: Tensor
    image_ids: List[str] = field(default_factory=list)
    predictions: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    scores: Tensor = field(default_factory=lambda: torch.empty(0))


class Evaluator:
    """Runs inference over a dataloader and computes evaluation metrics.

    Args:
        model: The model to evaluate. Must already be on ``device``.
        device: Device to run inference on.
        loss_fn: Optional loss function. If provided, ``EvaluationResult.loss``
            is populated with the batch-size-weighted mean loss; if omitted,
            ``loss`` is ``None`` (e.g. for pure inference with no labels
            available in the strict sense).
    """

    def __init__(self, model: nn.Module, device: torch.device, loss_fn: Optional[nn.Module] = None) -> None:
        self._model = model
        self._device = device
        self._loss_fn = loss_fn

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvaluationResult:
        """Run the model over every batch in ``loader`` and compute metrics.

        Args:
            loader: Dataloader yielding ``(image, label, image_id)`` batches,
                matching :class:`vit.data.dataset.MorphDataset`'s output.

        Returns:
            An :class:`EvaluationResult` populated with aggregate metrics and
            per-sample predictions/labels/scores/ids.

        Raises:
            ValueError: If ``loader`` yields no batches.
        """
        self._model.eval()
        tracker = MetricTracker()

        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_ids: List[str] = []

        for images, labels, image_ids in loader:
            images = images.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)

            logits = self._model(images)

            if self._loss_fn is not None:
                loss = self._loss_fn(logits, labels)
                tracker.update("loss", float(loss.item()), batch_size=images.size(0))

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_ids.extend(image_ids)

        if not all_logits:
            raise ValueError("Evaluator.evaluate received an empty dataloader")

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probs = F.softmax(logits, dim=1)
        scores = probs[:, _ATTACK_CLASS_INDEX]
        preds = torch.argmax(logits, dim=1)

        accuracy = compute_accuracy(preds, labels)
        eer, eer_threshold = compute_eer(scores, labels)
        auc = compute_auc(scores, labels)
        apcer, bpcer = compute_apcer_bpcer(scores, labels, threshold=eer_threshold)
        cm = compute_confusion_matrix(preds, labels, num_classes=logits.shape[1])

        return EvaluationResult(
            loss=tracker.average("loss") if self._loss_fn is not None else None,
            accuracy=accuracy,
            eer=eer,
            eer_threshold=eer_threshold,
            auc=auc,
            apcer=apcer,
            bpcer=bpcer,
            confusion_matrix=cm,
            image_ids=all_ids,
            predictions=preds,
            labels=labels,
            scores=scores,
        )

    @torch.no_grad()
    def predict_logits(self, loader: DataLoader) -> Tuple[Tensor, Tensor, List[str]]:
        """Run the model over ``loader`` and return raw logits (no metrics).

        Useful when only the raw outputs are needed (e.g. for downstream
        threshold sweeps or ensembling), avoiding the metric computation
        overhead of :meth:`evaluate`.

        Args:
            loader: Dataloader yielding ``(image, label, image_id)`` batches.

        Returns:
            A tuple ``(logits, labels, image_ids)`` where ``logits`` has
            shape ``(N, num_classes)`` and ``labels`` has shape ``(N,)``,
            both concatenated across all batches in dataloader order.

        Raises:
            ValueError: If ``loader`` yields no batches.
        """
        self._model.eval()

        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_ids: List[str] = []

        for images, labels, image_ids in loader:
            images = images.to(self._device, non_blocking=True)
            logits = self._model(images)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_ids.extend(image_ids)

        if not all_logits:
            raise ValueError("Evaluator.predict_logits received an empty dataloader")

        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), all_ids