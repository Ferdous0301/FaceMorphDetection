"""Public API for the FMAD Data Augmentation stage."""

from __future__ import annotations

# Constants
DEFAULT_FILENAME_TEMPLATE = "{identity}_{index:04d}.jpg"
DEFAULT_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif")

# Framework primitives
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

# Concrete operators
from src.augmentation.operators.brightness import BrightnessOperator
from src.augmentation.operators.contrast import ContrastOperator
from src.augmentation.operators.gamma import GammaOperator
from src.augmentation.operators.gaussian_blur import GaussianBlurOperator
from src.augmentation.operators.gaussian_noise import GaussianNoiseOperator
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator
from src.augmentation.operators.jpeg_compression import JPEGCompressionOperator
from src.augmentation.operators.sharpen import SharpenOperator

__all__ = [
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "DEFAULT_FILENAME_TEMPLATE",
    "BaseAugmentation",
    "AugmentationResult",
    "OperatorRegistry",
    "AugmentationError",
    "InvalidImageError",
    "InvalidOperatorConfigError",
    "OperatorRegistrationError",
    "OperatorNotFoundError",
    "BrightnessOperator",
    "ContrastOperator",
    "GammaOperator",
    "GaussianBlurOperator",
    "GaussianNoiseOperator",
    "HorizontalFlipOperator",
    "JPEGCompressionOperator",
    "SharpenOperator",
]
