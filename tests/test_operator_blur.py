"""Unit tests for the blur augmentation operators.

Covers ``GaussianBlurAugmentation`` and ``MotionBlurAugmentation`` from
``src.augmentation.operators.blur``, as well as their inherited
:class:`BaseAugmentation` behaviour (enabled/probability gating,
deterministic execution, and :class:`AugmentationResult` contents).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.augmentation.operators.base import (
    AugmentationResult,
    InvalidOperatorConfigError,
)
from src.augmentation.operators.blur import (
    GaussianBlurAugmentation,
    MotionBlurAugmentation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial RGB test image with sharp edges."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    # Add a hard-edged block to make blurring visually/numerically detectable.
    base[20:40, 30:60] = 255
    base[0:10, 0:10] = 0
    return base


# ---------------------------------------------------------------------------
# GaussianBlurAugmentation
# ---------------------------------------------------------------------------


class TestGaussianBlurAugmentation:
    """Tests for :class:`GaussianBlurAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default parameters."""
        op = GaussianBlurAugmentation()

        assert op.min_kernel_size == 3
        assert op.max_kernel_size == 9
        assert op.min_sigma == 0.0
        assert op.max_sigma == 0.0
        assert op.operator_name == "gaussian_blur"

    def test_invalid_kernel_range_raises(self) -> None:
        """min_kernel_size greater than max_kernel_size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="kernel_size"):
            GaussianBlurAugmentation(min_kernel_size=9, max_kernel_size=3)

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_non_positive_kernel_size_raises(self, field_name: str) -> None:
        """A non-positive kernel size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            GaussianBlurAugmentation(**{field_name: 0})
        with pytest.raises(InvalidOperatorConfigError):
            GaussianBlurAugmentation(**{field_name: -3})

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_even_kernel_size_raises(self, field_name: str) -> None:
        """An even kernel size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="odd"):
            GaussianBlurAugmentation(**{field_name: 4})

    def test_invalid_sigma_range_raises(self) -> None:
        """min_sigma greater than max_sigma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="sigma"):
            GaussianBlurAugmentation(min_sigma=5.0, max_sigma=1.0)

    @pytest.mark.parametrize("field_name", ["min_sigma", "max_sigma"])
    def test_negative_sigma_raises(self, field_name: str) -> None:
        """A negative sigma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            GaussianBlurAugmentation(**{field_name: -1.0})

    def test_equal_kernel_bounds_allowed(self) -> None:
        """A degenerate kernel range where min == max is accepted."""
        op = GaussianBlurAugmentation(min_kernel_size=5, max_kernel_size=5, seed=0)

        assert op.min_kernel_size == op.max_kernel_size == 5

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image dimensions and uint8 dtype."""
        op = GaussianBlurAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_kernel_size_within_range(self, image: np.ndarray) -> None:
        """The sampled kernel size falls within the configured range."""
        op = GaussianBlurAugmentation(min_kernel_size=3, max_kernel_size=11, seed=1)
        _, params = op._apply(image)

        assert 3 <= params["kernel_size"] <= 11

    def test_apply_kernel_size_is_always_odd(self, image: np.ndarray) -> None:
        """The sampled kernel size is always odd across repeated sampling."""
        op = GaussianBlurAugmentation(min_kernel_size=3, max_kernel_size=21, seed=1)

        for _ in range(30):
            _, params = op._apply(image)
            assert params["kernel_size"] % 2 == 1

    def test_apply_sigma_within_range(self, image: np.ndarray) -> None:
        """The sampled sigma falls within the configured range."""
        op = GaussianBlurAugmentation(min_sigma=1.0, max_sigma=5.0, seed=2)
        _, params = op._apply(image)

        assert 1.0 <= params["sigma"] <= 5.0

    def test_zero_sigma_range_produces_auto_sigma(self, image: np.ndarray) -> None:
        """Default (0.0, 0.0) sigma range yields a sigma of exactly 0.0."""
        op = GaussianBlurAugmentation(seed=0)
        _, params = op._apply(image)

        assert params["sigma"] == 0.0

    def test_output_differs_from_input(self, image: np.ndarray) -> None:
        """Applying blur to a sharp-edged image changes pixel values."""
        op = GaussianBlurAugmentation(min_kernel_size=7, max_kernel_size=7, seed=0)
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)

    def test_blurring_reduces_local_variance(self, image: np.ndarray) -> None:
        """Blurring reduces the overall variance of a sharp-edged image."""
        op = GaussianBlurAugmentation(min_kernel_size=9, max_kernel_size=9, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).var() < image.astype(np.float64).var()

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = GaussianBlurAugmentation(seed=123)
        op2 = GaussianBlurAugmentation(seed=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_outputs(self, image: np.ndarray) -> None:
        """Different seeds are expected to yield different sampled parameters."""
        op1 = GaussianBlurAugmentation(min_kernel_size=3, max_kernel_size=15, seed=1)
        op2 = GaussianBlurAugmentation(min_kernel_size=3, max_kernel_size=15, seed=2)

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1 != params2

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records kernel_size and sigma in the result."""
        op = GaussianBlurAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True
        assert "kernel_size" in result.parameters
        assert "sigma" in result.parameters


# ---------------------------------------------------------------------------
# MotionBlurAugmentation
# ---------------------------------------------------------------------------


class TestMotionBlurAugmentation:
    """Tests for :class:`MotionBlurAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default parameters."""
        op = MotionBlurAugmentation()

        assert op.min_kernel_size == 3
        assert op.max_kernel_size == 15
        assert op.min_angle == 0.0
        assert op.max_angle == 180.0
        assert op.operator_name == "motion_blur"

    def test_invalid_kernel_range_raises(self) -> None:
        """min_kernel_size greater than max_kernel_size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="kernel_size"):
            MotionBlurAugmentation(min_kernel_size=15, max_kernel_size=3)

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_non_positive_kernel_size_raises(self, field_name: str) -> None:
        """A non-positive kernel size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            MotionBlurAugmentation(**{field_name: 0})
        with pytest.raises(InvalidOperatorConfigError):
            MotionBlurAugmentation(**{field_name: -5})

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_even_kernel_size_raises(self, field_name: str) -> None:
        """An even kernel size raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="odd"):
            MotionBlurAugmentation(**{field_name: 6})

    def test_invalid_angle_range_raises(self) -> None:
        """min_angle greater than max_angle raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="angle"):
            MotionBlurAugmentation(min_angle=90.0, max_angle=-90.0)

    def test_equal_angle_bounds_allowed(self) -> None:
        """A degenerate angle range where min == max is accepted."""
        op = MotionBlurAugmentation(min_angle=45.0, max_angle=45.0, seed=0)

        assert op.min_angle == op.max_angle == 45.0

    def test_negative_angle_bounds_allowed(self) -> None:
        """Negative angle bounds are valid, since angles are unrestricted in sign."""
        op = MotionBlurAugmentation(min_angle=-45.0, max_angle=45.0, seed=0)

        assert op.min_angle == -45.0
        assert op.max_angle == 45.0

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image dimensions and uint8 dtype."""
        op = MotionBlurAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_kernel_size_within_range(self, image: np.ndarray) -> None:
        """The sampled kernel size falls within the configured range."""
        op = MotionBlurAugmentation(min_kernel_size=5, max_kernel_size=13, seed=1)
        _, params = op._apply(image)

        assert 5 <= params["kernel_size"] <= 13

    def test_apply_kernel_size_is_always_odd(self, image: np.ndarray) -> None:
        """The sampled kernel size is always odd across repeated sampling."""
        op = MotionBlurAugmentation(min_kernel_size=3, max_kernel_size=21, seed=1)

        for _ in range(30):
            _, params = op._apply(image)
            assert params["kernel_size"] % 2 == 1

    def test_apply_angle_within_range(self, image: np.ndarray) -> None:
        """The sampled angle falls within the configured range."""
        op = MotionBlurAugmentation(min_angle=10.0, max_angle=80.0, seed=2)
        _, params = op._apply(image)

        assert 10.0 <= params["angle"] <= 80.0

    def test_kernel_is_normalized(self) -> None:
        """The internal kernel builder always produces a kernel summing to 1.0."""
        for kernel_size in (3, 5, 9, 15):
            for angle in (0.0, 37.5, 90.0, 180.0, 275.0):
                kernel = MotionBlurAugmentation._build_motion_kernel(
                    kernel_size, angle
                )
                assert kernel.shape == (kernel_size, kernel_size)
                assert kernel.sum() == pytest.approx(1.0, abs=1e-6)

    def test_kernel_is_non_negative(self) -> None:
        """The motion blur kernel never contains negative weights."""
        kernel = MotionBlurAugmentation._build_motion_kernel(9, 45.0)

        assert np.all(kernel >= 0.0)

    def test_output_differs_from_input(self, image: np.ndarray) -> None:
        """Applying motion blur to a sharp-edged image changes pixel values."""
        op = MotionBlurAugmentation(
            min_kernel_size=9, max_kernel_size=9, min_angle=30.0, max_angle=30.0, seed=0
        )
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = MotionBlurAugmentation(seed=321)
        op2 = MotionBlurAugmentation(seed=321)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_outputs(self, image: np.ndarray) -> None:
        """Different seeds are expected to yield different sampled parameters."""
        op1 = MotionBlurAugmentation(seed=1)
        op2 = MotionBlurAugmentation(seed=2)

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1 != params2

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records kernel_size and angle in the result."""
        op = MotionBlurAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True
        assert "kernel_size" in result.parameters
        assert "angle" in result.parameters


# ---------------------------------------------------------------------------
# General / inherited BaseAugmentation behaviour
# ---------------------------------------------------------------------------


class TestInheritedBaseAugmentationBehaviour:
    """Tests confirming blur operators correctly inherit base behaviour."""

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_disabled_operator_skips_augmentation(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = operator_cls(probability=1.0, enabled=False, seed=0)
        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_probability_gate_zero_never_applies(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """A probability of 0.0 means the operator is never applied."""
        op = operator_cls(probability=0.0, seed=123)

        for _ in range(10):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_probability_gate_one_always_applies(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = operator_cls(probability=1.0, seed=123)

        for _ in range(10):
            result = op.apply(image)
            assert result.applied is True
            assert result.success is True

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_augmentation_result_contents(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """apply() returns a well-formed AugmentationResult on success."""
        op = operator_cls(probability=1.0, seed=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == op.operator_name
        assert result.success is True
        assert result.applied is True
        assert result.error_message is None
        assert result.image.shape == image.shape
        assert result.image.dtype == image.dtype

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_repeated_runs_with_same_seed_are_identical(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """Repeated runs of two identically-seeded operators produce identical results."""
        op1 = operator_cls(probability=0.7, seed=2024)
        op2 = operator_cls(probability=0.7, seed=2024)

        for _ in range(15):
            result1 = op1.apply(image)
            result2 = op2.apply(image)

            assert result1.applied == result2.applied
            assert result1.parameters == result2.parameters
            np.testing.assert_array_equal(result1.image, result2.image)

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_output_dtype_remains_uint8(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """Output image dtype is always uint8 after applying the operator."""
        op = operator_cls(probability=1.0, seed=9)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianBlurAugmentation, MotionBlurAugmentation],
    )
    def test_output_dimensions_preserved(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """Output image dimensions match the input image dimensions."""
        op = operator_cls(probability=1.0, seed=9)
        result = op.apply(image)

        assert result.image.shape == image.shape