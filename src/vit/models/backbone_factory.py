"""Registry-based factory for constructing torchvision Vision Transformer backbones.

Adding support for a new ViT variant requires exactly one new entry in
``_BACKBONE_REGISTRY`` below — no other module needs to change (open/closed
principle). This is what lets the rest of the codebase (``ModelConfig``,
``ViTMorphClassifier``) stay agnostic to which specific ViT variant is in use.

Each registered backbone, once built, has its classification head replaced
with ``nn.Identity()`` so that ``build_backbone`` always returns a pure
feature extractor: calling it on a batch of images yields the pooled
class-token embedding of dimension ``get_hidden_dim(backbone)``, not
class logits. Attaching a task-specific head (see
:class:`vit.models.classifier_head.ClassifierHead`) is the caller's
responsibility, which keeps the backbone reusable across tasks beyond
binary morph-detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch.nn as nn
import torchvision.models as tv_models

__all__ = ["SUPPORTED_BACKBONES", "build_backbone", "get_hidden_dim"]


@dataclass(frozen=True)
class _BackboneEntry:
    """Registry entry describing how to construct one ViT variant."""

    constructor: Callable[..., nn.Module]
    weights_enum: type


_BACKBONE_REGISTRY: Dict[str, _BackboneEntry] = {
    "vit_b_16": _BackboneEntry(tv_models.vit_b_16, tv_models.ViT_B_16_Weights),
    "vit_b_32": _BackboneEntry(tv_models.vit_b_32, tv_models.ViT_B_32_Weights),
    "vit_l_16": _BackboneEntry(tv_models.vit_l_16, tv_models.ViT_L_16_Weights),
    "vit_l_32": _BackboneEntry(tv_models.vit_l_32, tv_models.ViT_L_32_Weights),
}

#: Names accepted by :func:`build_backbone`, kept in sync with
#: ``vit.config.schema.SUPPORTED_BACKBONES`` (duplicated there to avoid a
#: config -> model import cycle; see the note in ``schema.py``).
SUPPORTED_BACKBONES: Tuple[str, ...] = tuple(_BACKBONE_REGISTRY.keys())


def build_backbone(name: str, pretrained: bool) -> nn.Module:
    """Construct a torchvision ViT backbone with its head stripped off.

    Args:
        name: Backbone identifier, must be a key of ``_BACKBONE_REGISTRY``
            (see :data:`SUPPORTED_BACKBONES`).
        pretrained: If True, initializes the backbone from the default
            ImageNet-pretrained weights for that variant (downloaded via
            ``torchvision`` on first use, and cached locally thereafter). If
            False, weights are randomly initialized.

    Returns:
        An ``nn.Module`` whose ``forward`` maps a batch of images
        ``(B, 3, H, W)`` to pooled class-token embeddings
        ``(B, hidden_dim)``. Its original classification head has been
        replaced with ``nn.Identity()``.

    Raises:
        ValueError: If ``name`` is not a supported backbone identifier.

    Example:
        >>> backbone = build_backbone("vit_b_16", pretrained=False)
        >>> get_hidden_dim(backbone)
        768
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone '{name}' is not supported. Supported values: {SUPPORTED_BACKBONES}"
        )

    entry = _BACKBONE_REGISTRY[name]
    weights = entry.weights_enum.DEFAULT if pretrained else None
    model = entry.constructor(weights=weights)

    # Replace the classification head with Identity so the backbone is a
    # pure feature extractor; task-specific heads are attached externally.
    model.heads = nn.Identity()
    return model


def get_hidden_dim(backbone: nn.Module) -> int:
    """Return the embedding dimensionality produced by a backbone built via :func:`build_backbone`.

    Args:
        backbone: A model returned by :func:`build_backbone`.

    Returns:
        The hidden/embedding dimension (e.g. ``768`` for ViT-B/16,
        ``1024`` for ViT-L/16).

    Raises:
        AttributeError: If ``backbone`` does not expose a ``hidden_dim``
            attribute (i.e. it was not constructed via :func:`build_backbone`
            from this registry).
    """
    if not hasattr(backbone, "hidden_dim"):
        raise AttributeError(
            "backbone does not expose a 'hidden_dim' attribute; "
            "was it constructed via vit.models.backbone_factory.build_backbone?"
        )
    return int(backbone.hidden_dim)