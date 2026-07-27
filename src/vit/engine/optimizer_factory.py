"""
Optimizer factory for the ViT morph-attack-detection module.

Builds a torch.optim.Optimizer from an OptimizerConfig (see
vit/config/schema.py). Adding a new optimizer only requires adding an
entry to `_OPTIMIZER_BUILDERS`.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from src.vit.configs.schema import OptimizerConfig

_SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd")


def _build_adamw(params: Iterable[nn.Parameter], config: OptimizerConfig) -> Optimizer:
    return AdamW(
        params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def _build_adam(params: Iterable[nn.Parameter], config: OptimizerConfig) -> Optimizer:
    return Adam(
        params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def _build_sgd(params: Iterable[nn.Parameter], config: OptimizerConfig) -> Optimizer:
    return SGD(
        params,
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )


_OPTIMIZER_BUILDERS: Dict[str, Callable[[Iterable[nn.Parameter], OptimizerConfig], Optimizer]] = {
    "adamw": _build_adamw,
    "adam": _build_adam,
    "sgd": _build_sgd,
}


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """
    Build an optimizer for `model`'s trainable parameters, per `config`.

    Only parameters with requires_grad=True are passed to the optimizer,
    so a frozen backbone (see ViTMorphClassifier.freeze_backbone) is
    correctly excluded without any special-casing here.

    Args:
        model: the model whose parameters will be optimized.
        config: OptimizerConfig specifying name/lr/weight_decay/betas.

    Returns:
        A constructed torch.optim.Optimizer instance.

    Raises:
        ValueError: if config.name is not a supported optimizer.
    """
    name = config.name.lower()
    builder = _OPTIMIZER_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unsupported optimizer '{config.name}'. "
            f"Supported optimizers: {_SUPPORTED_OPTIMIZERS}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "No trainable parameters found on model (is the entire model frozen?)."
        )
    return builder(trainable_params, config)