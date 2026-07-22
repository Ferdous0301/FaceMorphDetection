"""ViT backbone construction, classification head, and the full model wrapper.

Public API:

    from vit.models import (
        build_backbone,
        get_hidden_dim,
        SUPPORTED_BACKBONES,
        ClassifierHead,
        ViTMorphClassifier,
    )
"""

from __future__ import annotations

from vit.models.backbone_factory import SUPPORTED_BACKBONES, build_backbone, get_hidden_dim
from vit.models.classifier_head import ClassifierHead
from vit.models.vit_model import ViTMorphClassifier

__all__ = [
    "build_backbone",
    "get_hidden_dim",
    "SUPPORTED_BACKBONES",
    "ClassifierHead",
    "ViTMorphClassifier",
]