"""Unit tests for :mod:`src.augmentation.operators.geometric`.

These tests cover parameter validation, deterministic sampling, shape
and dtype preservation, and the inherited BaseAugmentation behaviour
(enabled/probability gating, AugmentationResult contents) for each of
the four geometric operators: rotation, translation, scaling, and
horizontal flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.augmentation.operators.base import (
    AugmentationResult,
    InvalidImageError,
    InvalidOperatorConfigError,
)
from src.augmentation.operators.geometric import (
    HorizontalFlipAugmentation,
    RotationAugmentation,
    ScalingAugmentation,
    TranslationAugmentation,
)

ALL_OPERATOR_CLASSES = (
    RotationAugmentation,
    TranslationAugmentation,
    ScalingAugmentation,
    HorizontalFlipAugmentation,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(height: int = 32, width: int = 32) -> np.ndarray:
    """Construct a deterministic, non-uniform uint8 RGB image for tests.

    A gradient pattern (rather than a flat fill) is used so that
    geometric transforms like rotation, translation, and flipping
    produce visibly different pixel arrangements that tests can
    meaningfully assert on.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _make_asymmetric_image(height: int = 16, width: int = 24) -> np.ndarray:
    """Construct an image with a distinct left half and right half.

    Useful for verifying horizontal-flip correctness: the left and
    right halves are filled with different constant values.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = 10
    image[:, width // 2 :] = 200
    return image


# ---------------------------------------------------------------------------
# RotationAugmentation
# ---------------------------------------------------------------------------


class TestRotationAugmentation:
    """Tests for RotationAugmentation."""

    def test_angle_within_configured_range(self) -> None:
        """The sampled angle always falls within [min_angle, max_angle]."""
        image = _make_rgb_image()
        operator = RotationAugmentation(
            min_angle=-10.0, max_angle=10.0, probability=1.0, seed=42
        )

        for _ in range(50):
            result = operator.apply(image)
            assert -10.0 <= result.parameters["angle"] <= 10.0

    def test_image_shape_and_dtype_preserved(self) -> None:
        """Rotation preserves the input image's shape and dtype."""
        image = _make_rgb_image(height=40, width=25)
        operator = RotationAugmentation(probability=1.0, seed=1)

        result = operator.apply(image)

        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype

    def test_deterministic_output_with_same_seed(self) -> None:
        """Two operators with the same seed produce identical angles and images."""
        image = _make_rgb_image()
        operator_a = RotationAugmentation(probability=1.0, seed=7)
        operator_b = RotationAugmentation(probability=1.0, seed=7)

        result_a = operator_a.apply(image)
        result_b = operator_b.apply(image)

        assert result_a.parameters["angle"] == result_b.parameters["angle"]
        assert np.array_equal(result_a.image, result_b.image)

    def test_invalid_angle_range_raises(self) -> None:
        """A min_angle greater than max_angle raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="angle"):
            RotationAugmentation(min_angle=10.0, max_angle=-10.0)

    def test_parameters_recorded_in_result(self) -> None:
        """The AugmentationResult.parameters dict contains the 'angle' key."""
        image = _make_rgb_image()
        operator = RotationAugmentation(probability=1.0, seed=3)

        result = operator.apply(image)

        assert "angle" in result.parameters
        assert isinstance(result.parameters["angle"], float)

    def test_zero_angle_range_produces_unchanged_shape(self) -> None:
        """A degenerate angle range (min == max == 0) still preserves image shape."""
        image = _make_rgb_image()
        operator = RotationAugmentation(
            min_angle=0.0, max_angle=0.0, probability=1.0, seed=1
        )

        result = operator.apply(image)

        assert result.parameters["angle"] == 0.0
        assert result.image.shape == image.shape


# ---------------------------------------------------------------------------
# TranslationAugmentation
# ---------------------------------------------------------------------------


class TestTranslationAugmentation:
    """Tests for TranslationAugmentation."""

    def test_image_dimensions_preserved(self) -> None:
        """Translation preserves the input image's shape and dtype."""
        image = _make_rgb_image(height=30, width=45)
        operator = TranslationAugmentation(probability=1.0, seed=1)

        result = operator.apply(image)

        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype

    def test_deterministic_output_with_same_seed(self) -> None:
        """Two operators with the same seed produce identical offsets and images."""
        image = _make_rgb_image()
        operator_a = TranslationAugmentation(probability=1.0, seed=11)
        operator_b = TranslationAugmentation(probability=1.0, seed=11)

        result_a = operator_a.apply(image)
        result_b = operator_b.apply(image)

        assert result_a.parameters == result_b.parameters
        assert np.array_equal(result_a.image, result_b.image)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_tx_percent": 0.5, "max_tx_percent": -0.5},
            {"min_ty_percent": 0.5, "max_ty_percent": -0.5},
        ],
    )
    def test_invalid_range_raises(self, kwargs: dict[str, float]) -> None:
        """A min percentage greater than its max percentage raises an error."""
        with pytest.raises(InvalidOperatorConfigError):
            TranslationAugmentation(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_tx_percent": -1.5},
            {"max_tx_percent": 1.5},
            {"min_ty_percent": -2.0},
            {"max_ty_percent": 2.0},
        ],
    )
    def test_invalid_percentage_out_of_bounds_raises(
        self, kwargs: dict[str, float]
    ) -> None:
        """A percentage outside [-1.0, 1.0] raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            TranslationAugmentation(**kwargs)

    def test_parameters_recorded_in_result(self) -> None:
        """The result records tx/ty percentages and their pixel equivalents."""
        image = _make_rgb_image()
        operator = TranslationAugmentation(probability=1.0, seed=5)

        result = operator.apply(image)

        assert set(result.parameters) == {
            "tx_percent",
            "ty_percent",
            "tx_pixels",
            "ty_pixels",
        }

    def test_zero_translation_range_leaves_image_unchanged(self) -> None:
        """A degenerate translation range (all zero) leaves the image unchanged."""
        image = _make_rgb_image()
        operator = TranslationAugmentation(
            min_tx_percent=0.0,
            max_tx_percent=0.0,
            min_ty_percent=0.0,
            max_ty_percent=0.0,
            probability=1.0,
            seed=1,
        )

        result = operator.apply(image)

        assert np.array_equal(result.image, image)


# ---------------------------------------------------------------------------
# ScalingAugmentation
# ---------------------------------------------------------------------------


class TestScalingAugmentation:
    """Tests for ScalingAugmentation."""

    def test_zoom_in_preserves_output_size(self) -> None:
        """A scale range entirely above 1.0 (zoom in) still yields the original shape."""
        image = _make_rgb_image(height=28, width=28)
        operator = ScalingAugmentation(
            min_scale=1.2, max_scale=1.5, probability=1.0, seed=1
        )

        result = operator.apply(image)

        assert result.parameters["scale"] >= 1.2
        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype

    def test_zoom_out_preserves_output_size(self) -> None:
        """A scale range entirely below 1.0 (zoom out) still yields the original shape."""
        image = _make_rgb_image(height=28, width=28)
        operator = ScalingAugmentation(
            min_scale=0.5, max_scale=0.8, probability=1.0, seed=1
        )

        result = operator.apply(image)

        assert result.parameters["scale"] <= 0.8
        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype

    def test_deterministic_output_with_same_seed(self) -> None:
        """Two operators with the same seed produce identical scale factors and images."""
        image = _make_rgb_image()
        operator_a = ScalingAugmentation(
            min_scale=0.8, max_scale=1.3, probability=1.0, seed=99
        )
        operator_b = ScalingAugmentation(
            min_scale=0.8, max_scale=1.3, probability=1.0, seed=99
        )

        result_a = operator_a.apply(image)
        result_b = operator_b.apply(image)

        assert result_a.parameters["scale"] == result_b.parameters["scale"]
        assert np.array_equal(result_a.image, result_b.image)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_scale": 1.5, "max_scale": 0.5},
            {"min_scale": -0.5, "max_scale": 1.0},
            {"min_scale": 1.0, "max_scale": -0.5},
            {"min_scale": 0.0, "max_scale": 1.0},
        ],
    )
    def test_invalid_scale_range_raises(self, kwargs: dict[str, float]) -> None:
        """Invalid or non-positive scale ranges raise InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            ScalingAugmentation(**kwargs)

    def test_output_size_preserved_for_non_square_image(self) -> None:
        """Scaling preserves shape even for a non-square input image."""
        image = _make_rgb_image(height=17, width=53)
        operator = ScalingAugmentation(
            min_scale=0.6, max_scale=1.4, probability=1.0, seed=2
        )

        result = operator.apply(image)

        assert result.image.shape == image.shape

    def test_identity_scale_leaves_shape_and_dtype_intact(self) -> None:
        """A degenerate scale range (min == max == 1.0) still preserves shape/dtype."""
        image = _make_rgb_image()
        operator = ScalingAugmentation(
            min_scale=1.0, max_scale=1.0, probability=1.0, seed=1
        )

        result = operator.apply(image)

        assert result.parameters["scale"] == 1.0
        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype


# ---------------------------------------------------------------------------
# HorizontalFlipAugmentation
# ---------------------------------------------------------------------------


class TestHorizontalFlipAugmentation:
    """Tests for HorizontalFlipAugmentation."""

    def test_image_flipped_correctly(self) -> None:
        """The left and right halves of the image are swapped left-right."""
        image = _make_asymmetric_image()
        operator = HorizontalFlipAugmentation(probability=1.0, seed=1)

        result = operator.apply(image)

        assert np.array_equal(result.image, np.flip(image, axis=1))
        # Sanity: left half of flipped image matches original right half.
        half_width = image.shape[1] // 2
        assert np.all(result.image[:, :half_width] == 200)
        assert np.all(result.image[:, half_width:] == 10)

    def test_no_vertical_flip_occurs(self) -> None:
        """Horizontal flip never flips the image vertically (top/bottom order intact)."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:5, :] = 1
        image[5:, :] = 250

        operator = HorizontalFlipAugmentation(probability=1.0, seed=1)
        result = operator.apply(image)

        assert np.all(result.image[:5, :] == 1)
        assert np.all(result.image[5:, :] == 250)

    def test_deterministic_output_with_same_seed(self) -> None:
        """Two operators with the same seed produce identical gating and images."""
        image = _make_rgb_image()
        operator_a = HorizontalFlipAugmentation(probability=0.5, seed=13)
        operator_b = HorizontalFlipAugmentation(probability=0.5, seed=13)

        results_a = [operator_a.apply(image) for _ in range(10)]
        results_b = [operator_b.apply(image) for _ in range(10)]

        for result_a, result_b in zip(results_a, results_b):
            assert result_a.applied == result_b.applied
            assert np.array_equal(result_a.image, result_b.image)

    def test_parameters_recorded_in_result(self) -> None:
        """The result records a 'flipped' flag set to True."""
        image = _make_rgb_image()
        operator = HorizontalFlipAugmentation(probability=1.0, seed=1)

        result = operator.apply(image)

        assert result.parameters == {"flipped": True}


# ---------------------------------------------------------------------------
# General: inherited BaseAugmentation behaviour, parametrized across operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator_cls", ALL_OPERATOR_CLASSES)
class TestInheritedBaseAugmentationBehaviour:
    """Tests verifying every geometric operator honours BaseAugmentation's contract."""

    def test_returns_augmentation_result(self, operator_cls: type) -> None:
        """apply() always returns an AugmentationResult instance."""
        image = _make_rgb_image()
        operator = operator_cls(probability=1.0, seed=1)

        result = operator.apply(image)

        assert isinstance(result, AugmentationResult)

    def test_disabled_operator_skips_augmentation(self, operator_cls: type) -> None:
        """A disabled operator returns an unapplied result with the original image."""
        image = _make_rgb_image()
        operator = operator_cls(probability=1.0, enabled=False, seed=1)

        result = operator.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert np.array_equal(result.image, image)

    def test_probability_zero_never_applies(self, operator_cls: type) -> None:
        """A probability of 0.0 means the operator never applies."""
        image = _make_rgb_image()
        operator = operator_cls(probability=0.0, seed=1)

        for _ in range(10):
            result = operator.apply(image)
            assert result.applied is False
            assert result.success is True

    def test_probability_one_always_applies(self, operator_cls: type) -> None:
        """A probability of 1.0 means the operator always applies."""
        image = _make_rgb_image()
        operator = operator_cls(probability=1.0, seed=1)

        for _ in range(10):
            result = operator.apply(image)
            assert result.applied is True
            assert result.success is True

    def test_multiple_runs_same_seed_produce_identical_outputs(
        self, operator_cls: type
    ) -> None:
        """Two freshly constructed operators with the same seed always agree."""
        image = _make_rgb_image()
        operator_a = operator_cls(probability=0.7, seed=2024)
        operator_b = operator_cls(probability=0.7, seed=2024)

        for _ in range(15):
            result_a = operator_a.apply(image)
            result_b = operator_b.apply(image)
            assert result_a.applied == result_b.applied
            assert result_a.parameters == result_b.parameters
            assert np.array_equal(result_a.image, result_b.image)

    def test_output_dtype_preserved(self, operator_cls: type) -> None:
        """The output image dtype always matches the input image dtype (uint8)."""
        image = _make_rgb_image()
        operator = operator_cls(probability=1.0, seed=1)

        result = operator.apply(image)

        assert result.image.dtype == np.uint8

    def test_output_dimensions_preserved(self, operator_cls: type) -> None:
        """The output image shape always matches the input image shape."""
        image = _make_rgb_image(height=37, width=41)
        operator = operator_cls(probability=1.0, seed=1)

        result = operator.apply(image)

        assert result.image.shape == image.shape

    def test_invalid_image_raises(self, operator_cls: type) -> None:
        """Passing an invalid image (wrong dtype) raises InvalidImageError."""
        operator = operator_cls(probability=1.0, seed=1)
        invalid_image = np.zeros((10, 10, 3), dtype=np.float32)

        with pytest.raises(InvalidImageError):
            operator.apply(invalid_image)

    def test_operator_name_defaults_are_non_empty(self, operator_cls: type) -> None:
        """Each operator class exposes a sensible, non-empty default operator_name."""
        operator = operator_cls()

        assert isinstance(operator.operator_name, str)
        assert operator.operator_name != ""