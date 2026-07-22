"""Unit tests for :mod:`src.augmentation.operators.horizontal_flip`.

Covers ``HorizontalFlipOperator``, including construction-time
validation, deterministic probability-gate behaviour, the flip
transform itself, probability/enabled gating, the contents of the
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
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator


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
    """Tests for HorizontalFlipOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default parameters."""
        op = HorizontalFlipOperator()

        assert op.operator_name == "horizontal_flip"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = HorizontalFlipOperator(
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_flip",
        )

        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_flip"

    def test_exposes_random_state_generator(self) -> None:
        """The operator exposes a numpy.random.Generator via random_state."""
        op = HorizontalFlipOperator(random_state=1)

        assert isinstance(op.random_state, np.random.Generator)

    def test_is_subclass_of_base_augmentation(self) -> None:
        """HorizontalFlipOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(HorizontalFlipOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfiguration:
    """Tests for construction-time validation failures."""

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            HorizontalFlipOperator(probability=1.5)
        with pytest.raises(Exception):
            HorizontalFlipOperator(probability=-0.1)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven probability-gate behaviour."""

    def test_same_random_state_produces_identical_results(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = HorizontalFlipOperator(random_state=123)
        op2 = HorizontalFlipOperator(random_state=123)

        result1 = op1.apply(image)
        result2 = op2.apply(image)

        assert result1.applied == result2.applied
        np.testing.assert_array_equal(result1.image, result2.image)

    def test_reproducibility_with_identical_random_state(
        self, image: np.ndarray
    ) -> None:
        """Repeated apply() calls on identically-seeded operators stay in lockstep."""
        op1 = HorizontalFlipOperator(probability=0.5, random_state=55)
        op2 = HorizontalFlipOperator(probability=0.5, random_state=55)

        for _ in range(10):
            result1 = op1.apply(image)
            result2 = op2.apply(image)

            assert result1.applied == result2.applied
            assert result1.parameters == result2.parameters
            np.testing.assert_array_equal(result1.image, result2.image)

    def test_different_seeds_can_produce_different_gate_decisions(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values can yield different gate decisions."""
        outcomes = set()
        for seed in range(20):
            op = HorizontalFlipOperator(probability=0.5, random_state=seed)
            outcomes.add(op.apply(image).applied)

        assert outcomes == {True, False}


# ---------------------------------------------------------------------------
# Successful augmentation
# ---------------------------------------------------------------------------


class TestSuccessfulAugmentation:
    """Tests for the horizontal flip transform itself."""

    def test_apply_flips_horizontally(self, image: np.ndarray) -> None:
        """_apply mirrors the image left-right."""
        op = HorizontalFlipOperator(random_state=0)
        result, params = op._apply(image)

        np.testing.assert_array_equal(result, image[:, ::-1, :])
        assert params == {"flipped": True}

    def test_double_flip_returns_original(self, image: np.ndarray) -> None:
        """Flipping twice returns the original image."""
        op = HorizontalFlipOperator(random_state=0)
        once, _ = op._apply(image)
        twice, _ = op._apply(once)

        np.testing.assert_array_equal(twice, image)

    def test_does_not_flip_vertically(self, image: np.ndarray) -> None:
        """The flip is strictly horizontal; it never mirrors vertically."""
        op = HorizontalFlipOperator(random_state=0)
        result, _ = op._apply(image)

        assert not np.array_equal(result, image[::-1, :, :])

    def test_flip_changes_asymmetric_image(self, image: np.ndarray) -> None:
        """Flipping a horizontally asymmetric image changes pixel values."""
        op = HorizontalFlipOperator(random_state=0)
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = HorizontalFlipOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = HorizontalFlipOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(1, 1), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = HorizontalFlipOperator(probability=1.0, random_state=0)

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
        op = HorizontalFlipOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = HorizontalFlipOperator(probability=1.0, enabled=False, random_state=123)

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = HorizontalFlipOperator(probability=1.0, random_state=123)

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
        op = HorizontalFlipOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "horizontal_flip"
        assert result.applied is True
        assert result.success is True
        assert result.parameters == {"flipped": True}
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = HorizontalFlipOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "horizontal_flip"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = HorizontalFlipOperator(
            probability=1.0, random_state=1, operator_name="my_flip"
        )
        result = op.apply(image)

        assert result.operator_name == "my_flip"


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming HorizontalFlipOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """HorizontalFlipOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("horizontal_flip", HorizontalFlipOperator)

        assert registry.get("horizontal_flip") is HorizontalFlipOperator
        assert "horizontal_flip" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("horizontal_flip", HorizontalFlipOperator)

        operator_cls = registry.get("horizontal_flip")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True