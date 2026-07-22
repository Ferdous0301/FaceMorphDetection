"""Reusable evaluation harness for trained Vision Transformer morph detectors.

This module implements the :class:`Evaluator`, which consumes an already
trained model (produced by the training stage) together with a PyTorch
``DataLoader`` and computes a comprehensive set of evaluation metrics. It
deliberately contains **no training logic** -- it only performs forward
passes in inference mode and aggregates the outputs already produced by the
training module (logits, probabilities, predictions, labels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vit.evaluation import metrics as metric_fns

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard torch dependency
    from torch import nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

__all__ = ["EvaluationResult", "Evaluator"]


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable container holding the full result of a model evaluation.

    Attributes:
        loss: Average loss over the evaluated dataset (``None`` if no
            criterion was supplied to the :class:`Evaluator`).
        accuracy: Overall classification accuracy.
        precision: Precision for the attack (positive) class.
        recall: Recall for the attack (positive) class.
        f1: F1 score for the attack (positive) class.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the Precision-Recall curve.
        eer: Equal Error Rate.
        eer_threshold: Score threshold at which EER occurs.
        apcer: Attack Presentation Classification Error Rate.
        bpcer: Bonafide Presentation Classification Error Rate.
        acer: Average Classification Error Rate.
        far: False Acceptance Rate evaluated at ``decision_threshold``.
        frr: False Rejection Rate evaluated at ``decision_threshold``.
        decision_threshold: The threshold used to binarize scores into
            hard predictions and to compute FAR/FRR.
        confusion_matrix: A ``(2, 2)`` array ``[[TN, FP], [FN, TP]]``.
        labels: Ground-truth labels for every evaluated sample.
        predictions: Hard predicted labels for every evaluated sample.
        probabilities: Predicted attack-class probabilities for every
            evaluated sample.
        num_samples: Total number of samples evaluated.
    """

    loss: Optional[float]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    eer: Optional[float]
    eer_threshold: Optional[float]
    apcer: Optional[float]
    bpcer: Optional[float]
    acer: Optional[float]
    far: Optional[float]
    frr: Optional[float]
    decision_threshold: float
    confusion_matrix: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    num_samples: int

    def __post_init__(self) -> None:
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError("decision_threshold must be within [0, 1].")
        if len(self.labels) != len(self.predictions) or len(self.labels) != len(self.probabilities):
            raise ValueError("labels, predictions and probabilities must have equal length.")

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dictionary view of the scalar metrics.

        Array-valued fields (``confusion_matrix``, ``labels``,
        ``predictions``, ``probabilities``) are converted to nested Python
        lists so the result can be passed directly to ``json.dump``.
        """
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "apcer": self.apcer,
            "bpcer": self.bpcer,
            "acer": self.acer,
            "far": self.far,
            "frr": self.frr,
            "decision_threshold": self.decision_threshold,
            "confusion_matrix": np.asarray(self.confusion_matrix).tolist(),
            "num_samples": self.num_samples,
        }


class Evaluator:
    """Evaluates a trained Vision Transformer morph-attack-detection model.

    The evaluator performs a single inference pass over the supplied
    ``DataLoader``, collecting logits, probabilities and predictions, then
    computes every metric implemented in :mod:`vit.evaluation.metrics`.

    Args:
        model: A trained ``torch.nn.Module`` returning raw logits of shape
            ``(batch_size, 2)`` for a binary (bonafide / attack)
            classification problem.
        device: The device to run inference on. Defaults to ``"cuda"`` if
            available, otherwise ``"cpu"``.
        criterion: Optional loss function (e.g. ``nn.CrossEntropyLoss()``)
            used to additionally report average loss. If ``None``, the
            resulting ``loss`` field will be ``None``.
        use_amp: Whether to run the forward pass under automatic mixed
            precision (``torch.autocast``). Only has an effect on CUDA
            devices.
        decision_threshold: The probability threshold at which the attack
            (positive) class is chosen when a model does not already emit
            hard predictions. Must be within ``[0, 1]``.

    Example:
        >>> evaluator = Evaluator(model, device="cuda", criterion=nn.CrossEntropyLoss())
        >>> result = evaluator.evaluate(test_loader)
        >>> print(result.accuracy, result.eer)
    """

    def __init__(
        self,
        model: "nn.Module",
        device: Optional[str] = None,
        criterion: Optional["nn.Module"] = None,
        use_amp: bool = False,
        decision_threshold: float = 0.5,
    ) -> None:
        import torch
        from torch import nn as torch_nn

        if not isinstance(model, torch_nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module.")
        if not (0.0 <= decision_threshold <= 1.0):
            raise ValueError("decision_threshold must be within [0, 1].")

        self._torch = torch
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.criterion = criterion
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.decision_threshold = float(decision_threshold)

        logger.info(
            "Evaluator initialised on device=%s, amp=%s, decision_threshold=%.3f",
            self.device,
            self.use_amp,
            self.decision_threshold,
        )

    def evaluate(self, dataloader: "DataLoader") -> EvaluationResult:
        """Run inference over ``dataloader`` and compute all metrics.

        Args:
            dataloader: Yields ``(inputs, labels)`` batches. ``inputs`` is
                passed directly to the model; ``labels`` must be integer
                tensors of shape ``(batch_size,)`` with values in
                ``{0, 1}``.

        Returns:
            An :class:`EvaluationResult` populated with every computed
            metric.

        Raises:
            ValueError: If ``dataloader`` yields no batches at all.
        """
        torch = self._torch
        self.model.eval()

        all_labels: list = []
        all_predictions: list = []
        all_probabilities: list = []
        total_loss = 0.0
        total_samples = 0
        loss_computed = False

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = self._unpack_batch(batch)
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(inputs)
                    if self.criterion is not None:
                        batch_loss = self.criterion(logits, labels)
                        total_loss += float(batch_loss.item()) * labels.size(0)
                        loss_computed = True

                probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
                predictions = (probabilities >= self.decision_threshold).long()

                all_labels.append(labels.detach().cpu().numpy())
                all_predictions.append(predictions.detach().cpu().numpy())
                all_probabilities.append(probabilities.detach().cpu().numpy())
                total_samples += labels.size(0)

        if total_samples == 0:
            raise ValueError("Cannot evaluate: dataloader produced zero samples.")

        labels_arr = np.concatenate(all_labels).astype(np.int64)
        predictions_arr = np.concatenate(all_predictions).astype(np.int64)
        probabilities_arr = np.concatenate(all_probabilities).astype(np.float64)

        avg_loss = (total_loss / total_samples) if loss_computed else None

        logger.info("Evaluated %d samples.", total_samples)

        return self._compute_result(
            labels_arr, predictions_arr, probabilities_arr, avg_loss, total_samples
        )

    def _compute_result(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        loss: Optional[float],
        num_samples: int,
    ) -> EvaluationResult:
        """Compute every metric from raw arrays and assemble the result.

        Extracted as a separate method so tests can exercise metric
        aggregation without requiring an actual model or dataloader.
        """
        acc = metric_fns.accuracy(labels, predictions)
        prec = metric_fns.precision(labels, predictions)
        rec = metric_fns.recall(labels, predictions)
        f1 = metric_fns.f1_score(labels, predictions)
        cm = metric_fns.confusion_matrix(labels, predictions)

        n_attack = int(np.sum(labels == 1))
        n_bonafide = int(np.sum(labels == 0))

        apcer_val = metric_fns.apcer(labels, predictions) if n_attack > 0 else None
        bpcer_val = metric_fns.bpcer(labels, predictions) if n_bonafide > 0 else None
        if apcer_val is not None and bpcer_val is not None:
            acer_val = (apcer_val + bpcer_val) / 2.0
        else:
            acer_val = None
            logger.warning(
                "ACER undefined: evaluation set does not contain both bonafide and attack samples."
            )

        n_classes = len(np.unique(labels))
        if n_classes < 2:
            logger.warning(
                "Only one class present in labels; ROC AUC, PR AUC, EER, FAR and FRR are undefined."
            )
            roc_auc_val = None
            pr_auc_val = None
            eer_val = None
            eer_threshold = None
            far_val = None
            frr_val = None
        else:
            roc_auc_val = metric_fns.roc_auc(labels, probabilities)
            pr_auc_val = metric_fns.pr_auc(labels, probabilities)
            eer_val, eer_threshold = metric_fns.equal_error_rate(labels, probabilities)
            far_val = metric_fns.far(labels, probabilities, self.decision_threshold)
            frr_val = metric_fns.frr(labels, probabilities, self.decision_threshold)

        return EvaluationResult(
            loss=loss,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc_val,
            pr_auc=pr_auc_val,
            eer=eer_val,
            eer_threshold=eer_threshold,
            apcer=apcer_val,
            bpcer=bpcer_val,
            acer=acer_val,
            far=far_val,
            frr=frr_val,
            decision_threshold=self.decision_threshold,
            confusion_matrix=cm,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            num_samples=num_samples,
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> Any:
        """Unpack a dataloader batch into ``(inputs, labels)``.

        Supports both ``(inputs, labels)`` tuples/lists and dict-style
        batches with ``"image"``/``"input"`` and ``"label"`` keys, to
        remain compatible with the augmentation and dataset-split modules
        upstream.

        Raises:
            ValueError: If the batch format is not recognised.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            inputs_key = "image" if "image" in batch else "input"
            if inputs_key in batch and "label" in batch:
                return batch[inputs_key], batch["label"]
        raise ValueError(
            "Unsupported batch format from dataloader; expected a 2-tuple "
            "(inputs, labels) or a dict with 'image'/'input' and 'label' keys."
        )