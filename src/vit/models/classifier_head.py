"""Classification head attached on top of a ViT backbone embedding."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["ClassifierHead"]


class ClassifierHead(nn.Module):
    """A dropout + linear classification head.

    Deliberately kept as a separate, minimal module (rather than inlined
    into :class:`vit.models.vit_model.ViTMorphClassifier`) so that it can be
    swapped for a deeper head (e.g. an MLP) independently of backbone choice,
    and so that it is trivially unit-testable in isolation.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1) -> None:
        """Construct the head.

        Args:
            in_features: Dimensionality of the input embedding (must match
                the backbone's ``hidden_dim``).
            num_classes: Number of output classes.
            dropout: Dropout probability applied to the input embedding
                before the linear projection.

        Raises:
            ValueError: If ``in_features`` or ``num_classes`` is not
                positive, or ``dropout`` is outside ``[0, 1)``.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Map an embedding batch to class logits.

        Args:
            x: Embedding tensor of shape ``(B, in_features)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.fc(self.dropout(x))