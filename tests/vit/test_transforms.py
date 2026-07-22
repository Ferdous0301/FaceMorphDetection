"""Unit tests for vit.data.transforms."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vit.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transforms,
    build_train_transforms,
)


class TestBuildEvalTransforms:
    def test_output_shape(self) -> None:
        transform = build_eval_transforms(224)
        img = Image.new("RGB", (300, 150), color=(10, 20, 30))
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_output_is_float_tensor(self) -> None:
        transform = build_eval_transforms(96)
        img = Image.new("RGB", (50, 50))
        out = transform(img)
        assert out.dtype == torch.float32

    def test_deterministic_across_calls(self) -> None:
        transform = build_eval_transforms(64)
        img = Image.new("RGB", (100, 100), color=(123, 45, 67))
        a = transform(img)
        b = transform(img)
        assert torch.equal(a, b)

    def test_rejects_non_positive_image_size(self) -> None:
        with pytest.raises(ValueError):
            build_eval_transforms(0)
        with pytest.raises(ValueError):
            build_eval_transforms(-10)

    def test_normalization_applied(self) -> None:
        # A pure-black image, after ToTensor (all zeros) and Normalize,
        # should equal -mean / std at every pixel/channel.
        transform = build_eval_transforms(8)
        img = Image.new("RGB", (8, 8), color=(0, 0, 0))
        out = transform(img)
        expected = torch.tensor(
            [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
        ).view(3, 1, 1)
        assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


class TestBuildTrainTransforms:
    def test_output_shape(self) -> None:
        transform = build_train_transforms(224)
        img = Image.new("RGB", (400, 200))
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_default_is_deterministic(self) -> None:
        # Without extra_augmentation, train transform must be exactly as
        # deterministic as eval transform (no upstream-duplicated randomness).
        transform = build_train_transforms(64)
        img = Image.new("RGB", (64, 64), color=(9, 9, 9))
        a = transform(img)
        b = transform(img)
        assert torch.equal(a, b)

    def test_extra_augmentation_still_returns_valid_tensor(self) -> None:
        transform = build_train_transforms(64, extra_augmentation=True)
        img = Image.new("RGB", (64, 64), color=(9, 9, 9))
        out = transform(img)
        assert out.shape == (3, 64, 64)
        assert out.dtype == torch.float32

    def test_rejects_non_positive_image_size(self) -> None:
        with pytest.raises(ValueError):
            build_train_transforms(0)