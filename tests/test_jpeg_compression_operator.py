"""Unit tests for :mod:`src.augmentation.operators.jpeg_compression`.

Covers ``JPEGCompressionOperator``, including construction-time
validation, deterministic quality sampling, the JPEG encode/decode
round trip, probability/enabled gating, the contents of the returned
:class:`AugmentationResult`, and compatibility with
:class:`OperatorRegistry`.

Only synthetic NumPy arrays are used as test images; no real images
are read from disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.augmentation.operators.base import (
    AugmentationResult,
    BaseAugmentation,
    OperatorRegistry,
)
from src.augmentation.operators.jpeg_compression import JPEGCompressionOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial synthetic RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for JPEGCompressionOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default quality range."""
        op = JPEGCompressionOperator()

        assert op.min_quality == 30
        assert op.max_quality == 90
        assert op.operator_name == "jpeg_compression"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = JPEGCompressionOperator(
            min_quality=10,
            max_quality=50,
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_jpeg",
        )

        assert op.min_quality == 10
        assert op.max_quality == 50
        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_jpeg"

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_quality == max_quality is accepted."""
        op = JPEGCompressionOperator(min_quality=50, max_quality=50, random_state=0)

        assert op.min_quality == op.max_quality == 50

    def test_boundary_qualities_allowed(self) -> None:
        """Boundary quality values 1 and 100 are accepted."""
        op = JPEGCompressionOperator(min_quality=1, max_quality=100, random_state=0)

        assert op.min_quality == 1
        assert op.max_quality == 100

    def test_is_subclass_of_base_augmentation(self) -> None:
        """JPEGCompressionOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(JPEGCompressionOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfiguration:
    """Tests for construction-time validation failures."""

    @pytest.mark.parametrize("quality", [0, -1, 101, 1000])
    def test_out_of_bounds_min_quality_raises(self, quality: int) -> None:
        """A min_quality outside [1, 100] raises ValueError."""
        with pytest.raises(ValueError, match="min_quality"):
            JPEGCompressionOperator(min_quality=quality)

    @pytest.mark.parametrize("quality", [0, -1, 101, 1000])
    def test_out_of_bounds_max_quality_raises(self, quality: int) -> None:
        """A max_quality outside [1, 100] raises ValueError."""
        with pytest.raises(ValueError, match="max_quality"):
            JPEGCompressionOperator(max_quality=quality)

    def test_min_greater_than_max_raises(self) -> None:
        """min_quality greater than max_quality raises ValueError."""
        with pytest.raises(ValueError, match="min_quality"):
            JPEGCompressionOperator(min_quality=90, max_quality=30)

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            JPEGCompressionOperator(probability=1.5)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven quality sampling."""

    def test_same_random_state_produces_identical_results(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = JPEGCompressionOperator(random_state=123)
        op2 = JPEGCompressionOperator(random_state=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_reproducibility_with_identical_random_state(
        self, image: np.ndarray
    ) -> None:
        """Repeated apply() calls on identically-seeded operators stay in lockstep."""
        op1 = JPEGCompressionOperator(random_state=55)
        op2 = JPEGCompressionOperator(random_state=55)

        for _ in range(5):
            result1 = op1.apply(image)
            result2 = op2.apply(image)

            assert result1.applied == result2.applied
            assert result1.parameters == result2.parameters
            np.testing.assert_array_equal(result1.image, result2.image)

    def test_different_seeds_produce_different_quality(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values are expected to yield different quality."""
        op1 = JPEGCompressionOperator(
            min_quality=1, max_quality=100, random_state=1
        )
        op2 = JPEGCompressionOperator(
            min_quality=1, max_quality=100, random_state=2
        )

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1["quality"] != params2["quality"]

    def test_quality_within_configured_range(self, image: np.ndarray) -> None:
        """The sampled quality always falls within [min_quality, max_quality]."""
        op = JPEGCompressionOperator(min_quality=20, max_quality=60, random_state=5)

        for _ in range(25):
            _, params = op._apply(image)
            assert 20 <= params["quality"] <= 60


# ---------------------------------------------------------------------------
# Successful augmentation
# ---------------------------------------------------------------------------


class TestSuccessfulAugmentation:
    """Tests for the JPEG encode/decode round trip effect."""

    def test_low_quality_changes_image(self, image: np.ndarray) -> None:
        """A low JPEG quality introduces visible compression artifacts."""
        op = JPEGCompressionOperator(
            min_quality=5, max_quality=5, random_state=0
        )
        result, params = op._apply(image)

        assert params["quality"] == 5
        assert not np.array_equal(result, image)

    def test_high_quality_stays_closer_than_low_quality(
        self, image: np.ndarray
    ) -> None:
        """A high JPEG quality introduces less distortion than a low quality.

        Uniform random noise is worst-case input for JPEG (no smooth
        regions to exploit), so this asserts the relative ordering of
        distortion rather than an absolute error bound.
        """
        low_quality_op = JPEGCompressionOperator(
            min_quality=5, max_quality=5, random_state=0
        )
        high_quality_op = JPEGCompressionOperator(
            min_quality=100, max_quality=100, random_state=0
        )

        low_quality_result, _ = low_quality_op._apply(image)
        high_quality_result, _ = high_quality_op._apply(image)

        low_quality_diff = np.abs(
            low_quality_result.astype(np.float64) - image.astype(np.float64)
        ).mean()
        high_quality_diff = np.abs(
            high_quality_result.astype(np.float64) - image.astype(np.float64)
        ).mean()

        assert high_quality_diff < low_quality_diff

    def test_high_quality_stays_close_to_smooth_original(self) -> None:
        """A high JPEG quality keeps a smooth, photo-like image close to the original."""
        gradient = np.linspace(0, 255, num=64, dtype=np.uint8)
        smooth_image = np.tile(gradient, (48, 1))
        smooth_image = np.stack([smooth_image] * 3, axis=-1).astype(np.uint8)

        op = JPEGCompressionOperator(min_quality=100, max_quality=100, random_state=0)
        result, params = op._apply(smooth_image)

        assert params["quality"] == 100
        mean_abs_diff = np.abs(
            result.astype(np.float64) - smooth_image.astype(np.float64)
        ).mean()
        assert mean_abs_diff < 10.0


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = JPEGCompressionOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = JPEGCompressionOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(8, 8), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = JPEGCompressionOperator(probability=1.0, random_state=0)

        result = op.apply(synthetic_image)

        assert result.image.shape == synthetic_image.shape
        assert result.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Probability and enabled gating
# ---------------------------------------------------------------------------


class TestGating:
    """Tests for probability=0 and enabled=False gating behaviour."""

    def test_probability_zero_never_applies(self, image: np.ndarray) -> None:
        """A probability of 0.0 means the operator is never applied."""
        op = JPEGCompressionOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = JPEGCompressionOperator(
            probability=1.0, enabled=False, random_state=123
        )

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = JPEGCompressionOperator(probability=1.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is True
            assert result.success is True


# ---------------------------------------------------------------------------
# AugmentationResult contents
# ---------------------------------------------------------------------------


class TestAugmentationResultContents:
    """Tests for the contents of the returned AugmentationResult."""

    def test_successful_result_contents(self, image: np.ndarray) -> None:
        """A successful application populates all expected result fields."""
        op = JPEGCompressionOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "jpeg_compression"
        assert result.applied is True
        assert result.success is True
        assert "quality" in result.parameters
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = JPEGCompressionOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "jpeg_compression"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = JPEGCompressionOperator(
            probability=1.0, random_state=1, operator_name="my_jpeg"
        )
        result = op.apply(image)

        assert result.operator_name == "my_jpeg"


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming JPEGCompressionOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """JPEGCompressionOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("jpeg_compression", JPEGCompressionOperator)

        assert registry.get("jpeg_compression") is JPEGCompressionOperator
        assert "jpeg_compression" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("jpeg_compression", JPEGCompressionOperator)

        operator_cls = registry.get("jpeg_compression")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True