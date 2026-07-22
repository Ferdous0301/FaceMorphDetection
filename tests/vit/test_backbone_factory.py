"""Unit tests for vit.models.backbone_factory.

These tests deliberately always use pretrained=False: downloading real
ImageNet weights requires network access to download.pytorch.org, which is
not part of this sandbox's allowed egress domains. The registry logic,
architecture selection, and hidden_dim reporting are fully exercised without
needing pretrained weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vit.models.backbone_factory import SUPPORTED_BACKBONES, build_backbone, get_hidden_dim


class TestSupportedBackbones:
    def test_contains_expected_variants(self) -> None:
        assert set(SUPPORTED_BACKBONES) == {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"}


class TestBuildBackbone:
    def test_unsupported_name_raises(self) -> None:
        with pytest.raises(ValueError):
            build_backbone("resnet50", pretrained=False)

    def test_vit_b_16_hidden_dim(self) -> None:
        backbone = build_backbone("vit_b_16", pretrained=False)
        assert get_hidden_dim(backbone) == 768

    def test_vit_b_32_hidden_dim(self) -> None:
        backbone = build_backbone("vit_b_32", pretrained=False)
        assert get_hidden_dim(backbone) == 768

    def test_vit_l_16_hidden_dim(self) -> None:
        backbone = build_backbone("vit_l_16", pretrained=False)
        assert get_hidden_dim(backbone) == 1024

    def test_head_is_identity(self) -> None:
        backbone = build_backbone("vit_b_16", pretrained=False)
        assert isinstance(backbone.heads, nn.Identity)

    def test_forward_pass_shape(self) -> None:
        backbone = build_backbone("vit_b_16", pretrained=False)
        backbone.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (2, 768)


class TestGetHiddenDim:
    def test_raises_on_non_vit_module(self) -> None:
        with pytest.raises(AttributeError):
            get_hidden_dim(nn.Linear(10, 10))