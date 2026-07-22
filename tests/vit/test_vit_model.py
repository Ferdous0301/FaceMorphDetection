"""Unit tests for vit.models.vit_model.ViTMorphClassifier.

Uses pretrained=False throughout (no network access to download weights in
this sandbox); architecture correctness does not depend on pretrained
weights.
"""

from __future__ import annotations

import pytest
import torch

from vit.config.schema import ModelConfig
from vit.models.vit_model import ViTMorphClassifier


@pytest.fixture(scope="module")
def small_model() -> ViTMorphClassifier:
    config = ModelConfig(backbone="vit_b_16", pretrained=False, num_classes=2, dropout=0.1)
    model = ViTMorphClassifier(config)
    model.eval()
    return model


class TestForward:
    def test_output_shape(self, small_model: ViTMorphClassifier) -> None:
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = small_model(x)
        assert logits.shape == (2, 2)

    def test_gradients_flow_through_head_and_backbone(self) -> None:
        config = ModelConfig(backbone="vit_b_16", pretrained=False, num_classes=2)
        model = ViTMorphClassifier(config)
        model.train()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        logits.sum().backward()
        assert model.head.fc.weight.grad is not None
        backbone_grad_exists = any(
            p.grad is not None for p in model.backbone.parameters() if p.requires_grad
        )
        assert backbone_grad_exists


class TestFreezeUnfreeze:
    def test_freeze_backbone_via_config(self) -> None:
        config = ModelConfig(backbone="vit_b_16", pretrained=False, freeze_backbone=True)
        model = ViTMorphClassifier(config)
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

    def test_freeze_and_unfreeze_toggle(self, small_model: ViTMorphClassifier) -> None:
        small_model.freeze_backbone()
        assert all(not p.requires_grad for p in small_model.backbone.parameters())

        small_model.unfreeze_backbone()
        assert all(p.requires_grad for p in small_model.backbone.parameters())

    def test_frozen_backbone_receives_no_gradient(self) -> None:
        config = ModelConfig(backbone="vit_b_16", pretrained=False, freeze_backbone=True)
        model = ViTMorphClassifier(config)
        model.train()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        logits.sum().backward()
        assert all(p.grad is None for p in model.backbone.parameters())


class TestAttentionMaps:
    def test_output_shape(self, small_model: ViTMorphClassifier) -> None:
        x = torch.randn(1, 3, 224, 224)
        attn = small_model.get_attention_maps(x)
        # ViT-B/16 @ 224: 14x14 patches + 1 class token = 197 tokens
        assert attn.shape == (1, 197, 197)

    def test_attention_rows_approximately_sum_to_one(self, small_model: ViTMorphClassifier) -> None:
        x = torch.randn(1, 3, 224, 224)
        attn = small_model.get_attention_maps(x)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

    def test_restores_original_forward_methods(self, small_model: ViTMorphClassifier) -> None:
        mha_modules = [m for m in small_model.backbone.modules() if isinstance(m, torch.nn.MultiheadAttention)]
        original_forwards = [m.forward for m in mha_modules]

        x = torch.randn(1, 3, 224, 224)
        small_model.get_attention_maps(x)

        restored_forwards = [m.forward for m in mha_modules]
        assert original_forwards == restored_forwards

    def test_restores_training_mode(self) -> None:
        config = ModelConfig(backbone="vit_b_16", pretrained=False)
        model = ViTMorphClassifier(config)
        model.train()
        x = torch.randn(1, 3, 224, 224)
        model.get_attention_maps(x)
        assert model.training is True

        model.eval()
        model.get_attention_maps(x)
        assert model.training is False