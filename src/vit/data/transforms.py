"""Image preprocessing transforms for the ViT stage.

Design rationale:
    The heavy lifting of data augmentation (identity-preserving geometric and
    photometric perturbations) was already performed deterministically by the
    upstream **Data Augmentation** pipeline stage, and its outputs are what
    the CSV manifests point to. Consequently, the transforms defined here are
    intentionally minimal: they resize and normalize images to whatever the
    selected torchvision ViT backbone expects, and nothing more, by default.

    An optional, *disabled-by-default* light augmentation path
    (``extra_augmentation=True`` in :func:`build_train_transforms`) is
    provided for experimentation (e.g. ablations on whether additional
    on-the-fly flipping helps), but should not be relied upon as the primary
    augmentation mechanism for the thesis, since it would reintroduce
    stochasticity that is harder to audit than the upstream stage's
    deterministic augmentation.

    Normalization statistics are the standard ImageNet mean/std, matching
    what torchvision's pretrained ViT weights (``ViT_B_16_Weights``, etc.)
    were trained with. Using any other statistics with pretrained weights
    would silently degrade transfer-learning performance.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torchvision.transforms as T

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_train_transforms",
    "build_eval_transforms",
]

#: Standard ImageNet normalization statistics, matching torchvision's
#: pretrained ViT weights.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_eval_transforms(image_size: int) -> Callable:
    """Build the deterministic preprocessing pipeline used for val/test/inference.

    Applies, in order: resize to ``(image_size, image_size)``, conversion to
    a float tensor in ``[0, 1]``, and ImageNet normalization. Contains no
    randomness, so repeated calls on the same image always yield identical
    tensors.

    Args:
        image_size: Target square side length in pixels.

    Returns:
        A callable ``PIL.Image -> torch.Tensor`` transform.

    Raises:
        ValueError: If ``image_size`` is not positive.

    Example:
        >>> transform = build_eval_transforms(224)
        >>> from PIL import Image
        >>> img = Image.new("RGB", (300, 300))
        >>> transform(img).shape
        torch.Size([3, 224, 224])
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transforms(image_size: int, extra_augmentation: bool = False) -> Callable:
    """Build the preprocessing pipeline used for training.

    By default this is identical to :func:`build_eval_transforms`: resize,
    tensor conversion, and normalization, with no additional randomness,
    since augmentation is assumed to already have been applied upstream.

    Args:
        image_size: Target square side length in pixels.
        extra_augmentation: If True, additionally applies a light,
            stochastic ``RandomHorizontalFlip(p=0.5)`` before normalization.
            Disabled by default; enable only for deliberate ablation
            experiments, and note that doing so makes this transform
            non-deterministic per call (though still reproducible across
            *runs* as long as the global seed and DataLoader worker seeding
            from :mod:`vit.utils.seed` are used).

    Returns:
        A callable ``PIL.Image -> torch.Tensor`` transform.

    Raises:
        ValueError: If ``image_size`` is not positive.

    Example:
        >>> transform = build_train_transforms(224)
        >>> from PIL import Image
        >>> img = Image.new("RGB", (300, 300))
        >>> transform(img).shape
        torch.Size([3, 224, 224])
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    ops = [T.Resize((image_size, image_size))]
    if extra_augmentation:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return T.Compose(ops)