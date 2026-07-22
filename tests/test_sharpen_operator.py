"""Unit tests for :mod:`src.augmentation.operators.sharpen`.

Covers ``SharpenOperator``, including construction-time validation,
deterministic strength sampling, the unsharp-mask sharpening effect,
probability/enabled gating, the contents of the returned
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
from src.augmentation.operators.sharpen import SharpenOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial synthetic RGB image with soft gradients."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    # Pre-blur slightly so sharpening has clear edges to amplify.
    return base


@pytest.fixture
def edged_image() -> np.ndarray:
    """A synthetic image with a hard-edged block, useful for sharpening checks."""
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    image[10:22, 10:22] = 200
    return image


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for SharpenOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default strength range."""
        op = SharpenOperator()

        assert op.min_strength == 0.5
        assert op.max_strength == 2.0
        assert op.operator_name == "sharpen"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = SharpenOperator(
            min_strength=0.1,
            max_strength=3.0,
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_sharpen",
        )

        assert op.min_strength == 0.1
        assert op.max_strength == 3.0
        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_sharpen"

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_strength == max_strength is accepted."""
        op = SharpenOperator(min_strength=1.0, max_strength=1.0, random_state=0)

        assert op.min_strength == op.max_strength == 1.0

    def test_zero_strength_bounds_allowed(self) -> None:
        """A strength range of exactly (0.0, 0.0) is a valid configuration."""
        op = SharpenOperator(min_strength=0.0, max_strength=0.0, random_state=0)

        assert op.min_strength == 0.0
        assert op.max_strength == 0.0

    def test_is_subclass_of_base_augmentation(self) -> None:
        """SharpenOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(SharpenOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfiguration:
    """Tests for construction-time validation failures."""

    def test_negative_min_strength_raises(self) -> None:
        """A negative min_strength raises ValueError."""
        with pytest.raises(ValueError, match="min_strength"):
            SharpenOperator(min_strength=-0.1)

    def test_negative_max_strength_raises(self) -> None:
        """A negative max_strength raises ValueError."""
        with pytest.raises(ValueError, match="max_strength"):
            SharpenOperator(max_strength=-1.0)

    def test_min_greater_than_max_raises(self) -> None:
        """min_strength greater than max_strength raises ValueError."""
        with pytest.raises(ValueError, match="min_strength"):
            SharpenOperator(min_strength=2.0, max_strength=0.5)

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            SharpenOperator(probability=1.5)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven strength sampling."""

    def test_same_random_state_produces_identical_results(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = SharpenOperator(random_state=123)
        op2 = SharpenOperator(random_state=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_reproducibility_with_identical_random_state(
        self, image: np.ndarray
    ) -> None:
        """Repeated apply() calls on identically-seeded operators stay in lockstep."""
        op1 = SharpenOperator(random_state=55)
        op2 = SharpenOperator(random_state=55)

        for _ in range(5):
            result1 = op1.apply(image)
            result2 = op2.apply(image)

            assert result1.applied == result2.applied
            assert result1.parameters == result2.parameters
            np.testing.assert_array_equal(result1.image, result2.image)

    def test_different_seeds_produce_different_strength(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values are expected to yield different strength."""
        op1 = SharpenOperator(min_strength=0.1, max_strength=5.0, random_state=1)
        op2 = SharpenOperator(min_strength=0.1, max_strength=5.0, random_state=2)

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1["strength"] != params2["strength"]

    def test_strength_within_configured_range(self, image: np.ndarray) -> None:
        """The sampled strength always falls within [min_strength, max_strength]."""
        op = SharpenOperator(min_strength=0.3, max_strength=1.7, random_state=5)

        for _ in range(25):
            _, params = op._apply(image)
            assert 0.3 <= params["strength"] <= 1.7


# ---------------------------------------------------------------------------
# Successful augmentation
# ---------------------------------------------------------------------------


class TestSuccessfulAugmentation:
    """Tests for the unsharp-mask sharpening effect."""

    def test_zero_strength_is_identity(self, edged_image: np.ndarray) -> None:
        """A strength of exactly 0.0 leaves the image effectively unchanged."""
        op = SharpenOperator(min_strength=0.0, max_strength=0.0, random_state=0)
        result, params = op._apply(edged_image)

        assert params["strength"] == 0.0
        np.testing.assert_allclose(
            result.astype(np.int16), edged_image.astype(np.int16), atol=1
        )

    def test_positive_strength_changes_image(self, edged_image: np.ndarray) -> None:
        """A strictly positive strength changes pixel values near edges."""
        op = SharpenOperator(min_strength=1.5, max_strength=1.5, random_state=0)
        result, params = op._apply(edged_image)

        assert params["strength"] == 1.5
        assert not np.array_equal(result, edged_image)

    def test_sharpening_increases_edge_contrast(self, edged_image: np.ndarray) -> None:
        """Sharpening increases the contrast (variance) around a hard edge."""
        op = SharpenOperator(min_strength=1.5, max_strength=1.5, random_state=0)
        result, _ = op._apply(edged_image)

        assert (
            result.astype(np.float64).var() >= edged_image.astype(np.float64).var()
        )


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = SharpenOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = SharpenOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(9, 9), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = SharpenOperator(probability=1.0, random_state=0)

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
        op = SharpenOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = SharpenOperator(probability=1.0, enabled=False, random_state=123)

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = SharpenOperator(probability=1.0, random_state=123)

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
        op = SharpenOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "sharpen"
        assert result.applied is True
        assert result.success is True
        assert "strength" in result.parameters
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = SharpenOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "sharpen"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = SharpenOperator(
            probability=1.0, random_state=1, operator_name="my_sharpen"
        )
        result = op.apply(image)

        assert result.operator_name == "my_sharpen"


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming SharpenOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """SharpenOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("sharpen", SharpenOperator)

        assert registry.get("sharpen") is SharpenOperator
        assert "sharpen" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("sharpen", SharpenOperator)

        operator_cls = registry.get("sharpen")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True