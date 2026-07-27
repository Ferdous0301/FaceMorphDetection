"""Full ViT-based morph-detection model: backbone + classification head wrapper."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from src.vit.configs.schema import ModelConfig
from src.vit.models.backbone_factory import build_backbone, get_hidden_dim
from src.vit.models.classifier_head import ClassifierHead

__all__ = ["ViTMorphClassifier"]


class ViTMorphClassifier(nn.Module):
    """End-to-end ViT morph-attack classifier: backbone + :class:`ClassifierHead`.

    Attributes:
        config: The :class:`~vit.config.schema.ModelConfig` this model was
            built from.
        backbone: The feature-extractor backbone (see
            :func:`vit.models.backbone_factory.build_backbone`).
        head: The classification head mapping embeddings to class logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Construct the model from a validated :class:`ModelConfig`.

        Args:
            config: Model configuration specifying backbone variant,
                pretraining, number of classes, freezing, and head dropout.
        """
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained)
        hidden_dim = get_hidden_dim(self.backbone)
        self.head = ClassifierHead(
            in_features=hidden_dim, num_classes=config.num_classes, dropout=config.dropout
        )

        if config.freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Image batch of shape ``(B, 3, H, W)``, preprocessed as
                expected by the backbone (see
                :mod:`vit.data.transforms`).

        Returns:
            Logit tensor of shape ``(B, config.num_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (``requires_grad = False``).

        The classification head remains trainable. Useful for linear-probe
        style training or when fine-tuning data is very limited.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (``requires_grad = True``)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def get_attention_maps(self, x: Tensor) -> Tensor:
        """Compute layer-averaged self-attention maps for a batch of images.

        Runs a forward pass through the backbone with attention-weight
        capture enabled on every encoder block's
        ``nn.MultiheadAttention`` module, then averages the (head-averaged)
        attention weights across all encoder layers. This is a simplified
        variant of "attention rollout": it captures where the model attends
        on average across depth, without propagating attention through
        residual connections layer-by-layer. It is intended for qualitative
        visualization (see :mod:`vit.visualization.attention_maps`), not as
        a precise measure of information flow.

        Args:
            x: Image batch of shape ``(B, 3, H, W)``.

        Returns:
            A tensor of shape ``(B, N, N)`` where ``N`` is the number of
            tokens (patches + class token), containing attention weights
            averaged over attention heads and encoder layers. Row ``i``,
            column ``j`` is the attention paid by token ``i`` to token ``j``.

        Note:
            Temporarily monkey-patches the ``forward`` method of each
            ``nn.MultiheadAttention`` submodule for the duration of this
            call (forcing ``need_weights=True``) and always restores the
            original method afterwards, even if the forward pass raises.
            The model is temporarily switched to eval mode for the duration
            of the call and restored to its original mode afterwards.
        """
        mha_modules: List[nn.MultiheadAttention] = [
            m for m in self.backbone.modules() if isinstance(m, nn.MultiheadAttention)
        ]
        if not mha_modules:
            raise RuntimeError(
                "No nn.MultiheadAttention modules found in the backbone; "
                "cannot compute attention maps."
            )

        captured_weights: List[Tensor] = []
        originals = {}

        def _make_wrapper(original_forward):
            def _wrapped(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                output = original_forward(*args, **kwargs)
                captured_weights.append(output[1].detach())
                return output

            return _wrapped

        for module in mha_modules:
            originals[module] = module.forward
            module.forward = _make_wrapper(module.forward)

        was_training = self.training
        self.eval()
        try:
            self.backbone(x)
        finally:
            for module, original_forward in originals.items():
                module.forward = original_forward
            self.train(was_training)

        # captured_weights: list of (B, N, N) tensors, one per encoder layer.
        stacked = torch.stack(captured_weights, dim=0)  # (num_layers, B, N, N)
        return stacked.mean(dim=0)