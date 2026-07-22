"""
Loss function factory for the ViT morph-attack-detection module.

Supports:
    - "cross_entropy": standard (optionally class-weighted) cross entropy.
    - "focal_loss":     multi-class focal loss, useful for imbalanced
                        bona fide / morph datasets.

Both losses expect raw logits of shape (N, num_classes) and integer
labels of shape (N,), matching the output of ViTMorphClassifier.forward().
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_SUPPORTED_LOSSES = ("cross_entropy", "focal_loss")


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter. gamma=0 reduces to (weighted) cross entropy.
        class_weights: optional per-class alpha weighting, shape (num_classes,).
        reduction: "mean" | "sum" | "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = gamma
        self.reduction = reduction
        # Registered as buffer so it moves with .to(device) and is included
        # in state_dict-less modules cleanly (not a learnable parameter).
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        labels = labels.view(-1, 1)
        log_pt = log_probs.gather(1, labels).squeeze(1)
        pt = probs.gather(1, labels).squeeze(1)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.class_weights is not None:
            alpha_t = self.class_weights.gather(0, labels.squeeze(1))
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    name: str,
    class_weights: Optional[Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Build a loss module by name.

    Args:
        name: one of "cross_entropy" or "focal_loss".
        class_weights: optional 1D tensor of per-class weights
            (length == num_classes). For "cross_entropy" this is passed
            as the standard `weight` argument; for "focal_loss" it is
            used as the alpha_t term.
        focal_gamma: focusing parameter, only used when name == "focal_loss".

    Returns:
        An nn.Module whose forward(logits, labels) -> scalar loss Tensor.

    Raises:
        ValueError: if `name` is not a supported loss name.
    """
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal_loss":
        return FocalLoss(gamma=focal_gamma, class_weights=class_weights)
    raise ValueError(
        f"Unsupported loss '{name}'. Supported losses: {_SUPPORTED_LOSSES}"
    )