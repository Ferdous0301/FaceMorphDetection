"""Augmentation operators for the FMAD Data Augmentation stage.

This package provides every concrete augmentation operator used by
the Data Augmentation stage, all built on top of the shared
:class:`~src.augmentation.operators.base.BaseAugmentation` foundation.

For convenience, every operator class is re-exported directly from
this package, so callers can write::

    from src.augmentation.operators import BrightnessOperator

instead of reaching into each operator's own submodule.

The package also exposes a single, pre-populated
:class:`~src.augmentation.operators.base.OperatorRegistry` instance,
:data:`OPERATOR_REGISTRY`, with every operator below registered under
its default ``operator_name``. This gives callers (such as the
augmentation pipeline) a single, discoverable source of truth for
"which operators exist and what are they called", without having to
import each operator module individually.

Registered operators
---------------------
* ``"brightness"`` -> :class:`BrightnessOperator`
* ``"contrast"`` -> :class:`ContrastOperator`
* ``"gaussian_noise"`` -> :class:`GaussianNoiseOperator`
* ``"gaussian_blur"`` -> :class:`GaussianBlurOperator`
* ``"jpeg_compression"`` -> :class:`JPEGCompressionOperator`
* ``"gamma"`` -> :class:`GammaOperator`
* ``"sharpen"`` -> :class:`SharpenOperator`
* ``"horizontal_flip"`` -> :class:`HorizontalFlipOperator`
"""

from __future__ import annotations

from src.augmentation.operators.base import (
    AugmentationError,
    AugmentationResult,
    BaseAugmentation,
    InvalidImageError,
    InvalidOperatorConfigError,
    OperatorNotFoundError,
    OperatorRegistrationError,
    OperatorRegistry,
)
from src.augmentation.operators.brightness import BrightnessOperator
from src.augmentation.operators.contrast import ContrastOperator
from src.augmentation.operators.gamma import GammaOperator
from src.augmentation.operators.gaussian_blur import GaussianBlurOperator
from src.augmentation.operators.gaussian_noise import GaussianNoiseOperator
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator
from src.augmentation.operators.jpeg_compression import JPEGCompressionOperator
from src.augmentation.operators.sharpen import SharpenOperator

__all__ = [
    "AugmentationError",
    "AugmentationResult",
    "BaseAugmentation",
    "InvalidImageError",
    "InvalidOperatorConfigError",
    "OperatorNotFoundError",
    "OperatorRegistrationError",
    "OperatorRegistry",
    "BrightnessOperator",
    "ContrastOperator",
    "GammaOperator",
    "GaussianBlurOperator",
    "GaussianNoiseOperator",
    "HorizontalFlipOperator",
    "JPEGCompressionOperator",
    "SharpenOperator",
    "OPERATOR_REGISTRY",
    "build_default_registry",
]


def build_default_registry() -> OperatorRegistry:
    """Build a fresh :class:`OperatorRegistry` with every operator registered.

    Each operator class is registered under the ``operator_name`` it
    uses by default (e.g. ``BrightnessOperator`` is registered as
    ``"brightness"``). Callers who need an isolated registry (for
    example, in tests) should call this factory rather than mutating
    the shared :data:`OPERATOR_REGISTRY` instance.

    Returns
    -------
    OperatorRegistry
        A new registry instance with all eight operators registered.
    """
    registry = OperatorRegistry()
    registry.register("brightness", BrightnessOperator)
    registry.register("contrast", ContrastOperator)
    registry.register("gaussian_noise", GaussianNoiseOperator)
    registry.register("gaussian_blur", GaussianBlurOperator)
    registry.register("jpeg_compression", JPEGCompressionOperator)
    registry.register("gamma", GammaOperator)
    registry.register("sharpen", SharpenOperator)
    registry.register("horizontal_flip", HorizontalFlipOperator)
    return registry


OPERATOR_REGISTRY: OperatorRegistry = build_default_registry()