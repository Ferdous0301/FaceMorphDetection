"""Unit tests for :mod:`src.augmentation.operators.contrast`.

Covers ``ContrastOperator``, including construction-time validation,
deterministic parameter sampling, contrast increase/decrease
behaviour, clipping, probability/enabled gating, the contents of the
returned :class:`AugmentationResult`, and compatibility with
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
from src.augmentation.operators.contrast import ContrastOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial synthetic RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def varied_image() -> np.ndarray:
    """A synthetic image with a clear split of dark and bright regions.

    Useful for asserting that increasing/decreasing contrast widens or
    narrows the spread of pixel values around the mean.
    """
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    image[:10, :, :] = 40
    image[10:, :, :] = 216
    return image


@pytest.fixture
def bright_image() -> np.ndarray:
    """A near-white synthetic image, useful for exercising clipping at 255."""
    return np.full((16, 16, 3), 250, dtype=np.uint8)


@pytest.fixture
def dark_image() -> np.ndarray:
    """A near-black synthetic image, useful for exercising clipping at 0."""
    return np.full((16, 16, 3), 5, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for ContrastOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default factor range."""
        op = ContrastOperator()

        assert op.min_factor == 0.7
        assert op.max_factor == 1.3
        assert op.operator_name == "contrast"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = ContrastOperator(
            min_factor=0.4,
            max_factor=2.0,
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_contrast",
        )

        assert op.min_factor == 0.4
        assert op.max_factor == 2.0
        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_contrast"

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_factor == max_factor is accepted."""
        op = ContrastOperator(min_factor=1.1, max_factor=1.1, random_state=0)

        assert op.min_factor == op.max_factor == 1.1

    def test_exposes_random_state_generator(self) -> None:
        """The operator exposes a numpy.random.Generator via random_state."""
        op = ContrastOperator(random_state=1)

        assert isinstance(op.random_state, np.random.Generator)

    def test_is_subclass_of_base_augmentation(self) -> None:
        """ContrastOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(ContrastOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfiguration:
    """Tests for construction-time validation failures."""

    def test_non_positive_min_factor_raises(self) -> None:
        """A non-positive min_factor raises ValueError."""
        with pytest.raises(ValueError, match="min_factor"):
            ContrastOperator(min_factor=0.0)
        with pytest.raises(ValueError, match="min_factor"):
            ContrastOperator(min_factor=-0.5)

    def test_non_positive_max_factor_raises(self) -> None:
        """A non-positive max_factor raises ValueError."""
        with pytest.raises(ValueError, match="max_factor"):
            ContrastOperator(max_factor=0.0)
        with pytest.raises(ValueError, match="max_factor"):
            ContrastOperator(max_factor=-1.0)

    def test_min_greater_than_max_raises(self) -> None:
        """min_factor greater than max_factor raises ValueError."""
        with pytest.raises(ValueError, match="min_factor"):
            ContrastOperator(min_factor=1.5, max_factor=0.5)

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            ContrastOperator(probability=1.5)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven parameter sampling."""

    def test_same_seed_produces_identical_output(self, image: np.ndarray) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = ContrastOperator(random_state=123)
        op2 = ContrastOperator(random_state=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1["factor"] == params2["factor"]
        np.testing.assert_array_equal(result1, result2)

    def test_same_seed_produces_identical_gate_decisions(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state make identical gate decisions."""
        op1 = ContrastOperator(probability=0.5, random_state=99)
        op2 = ContrastOperator(probability=0.5, random_state=99)

        results1 = [op1.apply(image) for _ in range(10)]
        results2 = [op2.apply(image) for _ in range(10)]

        for result1, result2 in zip(results1, results2):
            assert result1.applied == result2.applied

    def test_different_seeds_produce_different_factors(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values yield different sampled factors."""
        op1 = ContrastOperator(random_state=1)
        op2 = ContrastOperator(random_state=2)

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1["factor"] != params2["factor"]

    def test_factor_within_configured_range(self, image: np.ndarray) -> None:
        """The sampled factor always falls within [min_factor, max_factor]."""
        op = ContrastOperator(min_factor=0.6, max_factor=1.4, random_state=5)

        for _ in range(25):
            _, params = op._apply(image)
            assert 0.6 <= params["factor"] <= 1.4


# ---------------------------------------------------------------------------
# Contrast increase / decrease
# ---------------------------------------------------------------------------


class TestContrastAdjustment:
    """Tests for the directional effect of contrast adjustment."""

    def test_increased_contrast(self, varied_image: np.ndarray) -> None:
        """A factor above 1.0 increases the standard deviation of pixel values."""
        op = ContrastOperator(min_factor=1.5, max_factor=1.5, random_state=0)
        result, params = op._apply(varied_image)

        assert params["factor"] == pytest.approx(1.5)
        assert result.astype(np.float64).std() > varied_image.astype(np.float64).std()

    def test_decreased_contrast(self, varied_image: np.ndarray) -> None:
        """A factor below 1.0 decreases the standard deviation of pixel values."""
        op = ContrastOperator(min_factor=0.3, max_factor=0.3, random_state=0)
        result, params = op._apply(varied_image)

        assert params["factor"] == pytest.approx(0.3)
        assert result.astype(np.float64).std() < varied_image.astype(np.float64).std()

    def test_unchanged_factor_is_identity(self, image: np.ndarray) -> None:
        """A factor of exactly 1.0 leaves the image effectively unchanged."""
        op = ContrastOperator(min_factor=1.0, max_factor=1.0, random_state=0)
        result, params = op._apply(image)

        assert params["factor"] == 1.0
        np.testing.assert_allclose(
            result.astype(np.int16), image.astype(np.int16), atol=1
        )

    def test_image_changes_when_factor_not_one(self, varied_image: np.ndarray) -> None:
        """The output image differs from the input whenever factor != 1.0."""
        op = ContrastOperator(min_factor=1.4, max_factor=1.4, random_state=0)
        result, params = op._apply(varied_image)

        assert params["factor"] != 1.0
        assert not np.array_equal(result, varied_image)

    def test_mean_gray_level_is_approximately_preserved(
        self, varied_image: np.ndarray
    ) -> None:
        """Contrast adjustment about the mean roughly preserves overall brightness."""
        op = ContrastOperator(min_factor=1.5, max_factor=1.5, random_state=0)
        result, _ = op._apply(varied_image)

        original_mean = varied_image.astype(np.float64).mean()
        adjusted_mean = result.astype(np.float64).mean()

        assert adjusted_mean == pytest.approx(original_mean, abs=5.0)


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


class TestClipping:
    """Tests for pixel value clipping into the valid [0, 255] range."""

    def test_clipping_at_high_end(self, bright_image: np.ndarray) -> None:
        """A large contrast factor clips pixel values at 255, not overflow."""
        op = ContrastOperator(min_factor=4.0, max_factor=4.0, random_state=0)
        result, params = op._apply(bright_image)

        assert params["factor"] == 4.0
        assert result.max() <= 255
        assert result.dtype == np.uint8

    def test_clipping_at_low_end(self, dark_image: np.ndarray) -> None:
        """A large contrast factor keeps pixel values from underflowing below 0."""
        op = ContrastOperator(min_factor=4.0, max_factor=4.0, random_state=0)
        result, _ = op._apply(dark_image)

        assert result.min() >= 0
        assert result.dtype == np.uint8

    def test_clipping_never_produces_values_outside_valid_range(
        self, varied_image: np.ndarray
    ) -> None:
        """Output pixel values always remain within [0, 255] regardless of factor."""
        op = ContrastOperator(min_factor=0.1, max_factor=5.0, random_state=3)

        for _ in range(20):
            result, _ = op._apply(varied_image)
            assert result.min() >= 0
            assert result.max() <= 255


# ---------------------------------------------------------------------------
# Probability and enabled gating
# ---------------------------------------------------------------------------


class TestGating:
    """Tests for probability=0 and enabled=False gating behaviour."""

    def test_probability_zero_never_applies(self, image: np.ndarray) -> None:
        """A probability of 0.0 means the operator is never applied."""
        op = ContrastOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = ContrastOperator(probability=1.0, enabled=False, random_state=123)

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = ContrastOperator(probability=1.0, random_state=123)

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
        op = ContrastOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "contrast"
        assert result.applied is True
        assert result.success is True
        assert "factor" in result.parameters
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = ContrastOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "contrast"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = ContrastOperator(
            probability=1.0, random_state=1, operator_name="my_contrast"
        )
        result = op.apply(image)

        assert result.operator_name == "my_contrast"


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = ContrastOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = ContrastOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(1, 1), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = ContrastOperator(probability=1.0, random_state=0)

        result = op.apply(synthetic_image)

        assert result.image.shape == synthetic_image.shape
        assert result.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming ContrastOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """ContrastOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("contrast", ContrastOperator)

        assert registry.get("contrast") is ContrastOperator
        assert "contrast" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("contrast", ContrastOperator)

        operator_cls = registry.get("contrast")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True