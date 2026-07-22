"""Unit tests for :mod:`src.augmentation.operators.gaussian_blur`.

Covers ``GaussianBlurOperator``, including construction-time
validation, deterministic kernel-size sampling, blur effect on the
image, probability/enabled gating, the contents of the returned
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
from src.augmentation.operators.gaussian_blur import GaussianBlurOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial synthetic RGB image with sharp edges."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    # Add hard-edged blocks to make blurring numerically detectable.
    base[10:30, 15:45] = 255
    base[0:8, 0:8] = 0
    return base


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for GaussianBlurOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default parameters."""
        op = GaussianBlurOperator()

        assert op.min_kernel_size == 3
        assert op.max_kernel_size == 11
        assert op.sigma == 0.0
        assert op.operator_name == "gaussian_blur"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = GaussianBlurOperator(
            min_kernel_size=5,
            max_kernel_size=21,
            sigma=2.5,
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_gaussian_blur",
        )

        assert op.min_kernel_size == 5
        assert op.max_kernel_size == 21
        assert op.sigma == 2.5
        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_gaussian_blur"

    def test_equal_kernel_bounds_allowed(self) -> None:
        """A degenerate kernel range where min == max is accepted."""
        op = GaussianBlurOperator(
            min_kernel_size=7, max_kernel_size=7, random_state=0
        )

        assert op.min_kernel_size == op.max_kernel_size == 7

    def test_exposes_random_state_generator(self) -> None:
        """The operator exposes a numpy.random.Generator via random_state."""
        op = GaussianBlurOperator(random_state=1)

        assert isinstance(op.random_state, np.random.Generator)

    def test_is_subclass_of_base_augmentation(self) -> None:
        """GaussianBlurOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(GaussianBlurOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid kernel sizes
# ---------------------------------------------------------------------------


class TestInvalidKernelSizes:
    """Tests for construction-time kernel size validation failures."""

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_non_positive_kernel_size_raises(self, field_name: str) -> None:
        """A non-positive kernel size raises ValueError."""
        with pytest.raises(ValueError, match=field_name):
            GaussianBlurOperator(**{field_name: 0})
        with pytest.raises(ValueError, match=field_name):
            GaussianBlurOperator(**{field_name: -3})

    @pytest.mark.parametrize("field_name", ["min_kernel_size", "max_kernel_size"])
    def test_even_kernel_size_raises(self, field_name: str) -> None:
        """An even kernel size raises ValueError mentioning oddness."""
        with pytest.raises(ValueError, match="odd"):
            GaussianBlurOperator(**{field_name: 4})

    def test_min_greater_than_max_raises(self) -> None:
        """min_kernel_size greater than max_kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="min_kernel_size"):
            GaussianBlurOperator(min_kernel_size=11, max_kernel_size=3)


# ---------------------------------------------------------------------------
# Invalid sigma
# ---------------------------------------------------------------------------


class TestInvalidSigma:
    """Tests for construction-time sigma validation failures."""

    def test_negative_sigma_raises(self) -> None:
        """A negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            GaussianBlurOperator(sigma=-0.1)
        with pytest.raises(ValueError, match="sigma"):
            GaussianBlurOperator(sigma=-5.0)

    def test_zero_sigma_is_valid(self) -> None:
        """A sigma of exactly 0.0 is a valid configuration (auto-derived)."""
        op = GaussianBlurOperator(sigma=0.0, random_state=0)

        assert op.sigma == 0.0

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            GaussianBlurOperator(probability=1.5)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven kernel-size sampling."""

    def test_same_random_state_produces_identical_results(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = GaussianBlurOperator(random_state=123)
        op2 = GaussianBlurOperator(random_state=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_same_random_state_produces_identical_gate_decisions(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state make identical gate decisions."""
        op1 = GaussianBlurOperator(probability=0.5, random_state=99)
        op2 = GaussianBlurOperator(probability=0.5, random_state=99)

        results1 = [op1.apply(image) for _ in range(10)]
        results2 = [op2.apply(image) for _ in range(10)]

        for result1, result2 in zip(results1, results2):
            assert result1.applied == result2.applied

    def test_different_seeds_produce_different_kernel_sizes(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values are expected to yield different parameters."""
        op1 = GaussianBlurOperator(
            min_kernel_size=3, max_kernel_size=21, random_state=1
        )
        op2 = GaussianBlurOperator(
            min_kernel_size=3, max_kernel_size=21, random_state=2
        )

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1 != params2

    def test_sampled_kernel_is_always_odd(self, image: np.ndarray) -> None:
        """The sampled kernel size is always odd across repeated sampling."""
        op = GaussianBlurOperator(
            min_kernel_size=3, max_kernel_size=21, random_state=1
        )

        for _ in range(50):
            _, params = op._apply(image)
            assert params["kernel_size"] % 2 == 1

    def test_sampled_kernel_within_range(self, image: np.ndarray) -> None:
        """The sampled kernel size always falls within the configured range."""
        op = GaussianBlurOperator(
            min_kernel_size=5, max_kernel_size=15, random_state=2
        )

        for _ in range(50):
            _, params = op._apply(image)
            assert 5 <= params["kernel_size"] <= 15

    def test_fixed_kernel_range_always_yields_that_value(
        self, image: np.ndarray
    ) -> None:
        """A single-valued kernel range always samples that exact kernel size."""
        op = GaussianBlurOperator(
            min_kernel_size=7, max_kernel_size=7, random_state=3
        )

        for _ in range(10):
            _, params = op._apply(image)
            assert params["kernel_size"] == 7


# ---------------------------------------------------------------------------
# Blur effect
# ---------------------------------------------------------------------------


class TestBlurEffect:
    """Tests for the actual blurring effect on the image."""

    def test_blur_modifies_image(self, image: np.ndarray) -> None:
        """Applying blur to a sharp-edged image changes pixel values."""
        op = GaussianBlurOperator(
            min_kernel_size=9, max_kernel_size=9, random_state=0
        )
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)

    def test_blur_reduces_variance(self, image: np.ndarray) -> None:
        """Blurring reduces the overall variance of a sharp-edged image."""
        op = GaussianBlurOperator(
            min_kernel_size=11, max_kernel_size=11, random_state=0
        )
        result, _ = op._apply(image)

        assert result.astype(np.float64).var() < image.astype(np.float64).var()

    def test_sigma_parameter_is_used(self, image: np.ndarray) -> None:
        """A configured sigma is passed through to the OpenCV call and recorded."""
        op = GaussianBlurOperator(
            min_kernel_size=5, max_kernel_size=5, sigma=3.0, random_state=0
        )
        _, params = op._apply(image)

        assert params["sigma"] == 3.0


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = GaussianBlurOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = GaussianBlurOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(9, 9), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = GaussianBlurOperator(probability=1.0, random_state=0)

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
        op = GaussianBlurOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = GaussianBlurOperator(probability=1.0, enabled=False, random_state=123)

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = GaussianBlurOperator(probability=1.0, random_state=123)

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
        op = GaussianBlurOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "gaussian_blur"
        assert result.applied is True
        assert result.success is True
        assert "kernel_size" in result.parameters
        assert "sigma" in result.parameters
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = GaussianBlurOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "gaussian_blur"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = GaussianBlurOperator(
            probability=1.0, random_state=1, operator_name="my_gaussian_blur"
        )
        result = op.apply(image)

        assert result.operator_name == "my_gaussian_blur"


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming GaussianBlurOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """GaussianBlurOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("gaussian_blur", GaussianBlurOperator)

        assert registry.get("gaussian_blur") is GaussianBlurOperator
        assert "gaussian_blur" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("gaussian_blur", GaussianBlurOperator)

        operator_cls = registry.get("gaussian_blur")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True