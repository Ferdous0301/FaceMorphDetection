"""Unit tests for :mod:`src.augmentation.operators.gamma`.

Covers ``GammaOperator``, including construction-time validation,
deterministic gamma sampling, the LUT-based gamma correction effect,
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
from src.augmentation.operators.gamma import GammaOperator


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
    """Tests for GammaOperator construction and exposed properties."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default gamma range."""
        op = GammaOperator()

        assert op.min_gamma == 0.7
        assert op.max_gamma == 1.5
        assert op.operator_name == "gamma"
        assert op.probability == 0.5
        assert op.enabled is True

    def test_custom_construction(self) -> None:
        """Custom constructor arguments are stored and exposed correctly."""
        op = GammaOperator(
            min_gamma=0.4,
            max_gamma=2.2,
            probability=0.9,
            enabled=False,
            random_state=7,
            operator_name="custom_gamma",
        )

        assert op.min_gamma == 0.4
        assert op.max_gamma == 2.2
        assert op.probability == 0.9
        assert op.enabled is False
        assert op.operator_name == "custom_gamma"

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_gamma == max_gamma is accepted."""
        op = GammaOperator(min_gamma=1.2, max_gamma=1.2, random_state=0)

        assert op.min_gamma == op.max_gamma == 1.2

    def test_is_subclass_of_base_augmentation(self) -> None:
        """GammaOperator is a proper subclass of BaseAugmentation."""
        assert issubclass(GammaOperator, BaseAugmentation)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfiguration:
    """Tests for construction-time validation failures."""

    def test_non_positive_min_gamma_raises(self) -> None:
        """A non-positive min_gamma raises ValueError."""
        with pytest.raises(ValueError, match="min_gamma"):
            GammaOperator(min_gamma=0.0)
        with pytest.raises(ValueError, match="min_gamma"):
            GammaOperator(min_gamma=-0.5)

    def test_non_positive_max_gamma_raises(self) -> None:
        """A non-positive max_gamma raises ValueError."""
        with pytest.raises(ValueError, match="max_gamma"):
            GammaOperator(max_gamma=0.0)
        with pytest.raises(ValueError, match="max_gamma"):
            GammaOperator(max_gamma=-1.0)

    def test_min_greater_than_max_raises(self) -> None:
        """min_gamma greater than max_gamma raises ValueError."""
        with pytest.raises(ValueError, match="min_gamma"):
            GammaOperator(min_gamma=2.0, max_gamma=0.5)

    def test_invalid_probability_raises(self) -> None:
        """An out-of-range probability raises an error via BaseAugmentation."""
        with pytest.raises(Exception):
            GammaOperator(probability=1.5)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------


class TestDeterministicRandomness:
    """Tests for deterministic, seed-driven gamma sampling."""

    def test_same_random_state_produces_identical_results(
        self, image: np.ndarray
    ) -> None:
        """Two operators with the same random_state produce identical output."""
        op1 = GammaOperator(random_state=123)
        op2 = GammaOperator(random_state=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_reproducibility_with_identical_random_state(
        self, image: np.ndarray
    ) -> None:
        """Repeated apply() calls on identically-seeded operators stay in lockstep."""
        op1 = GammaOperator(random_state=55)
        op2 = GammaOperator(random_state=55)

        for _ in range(5):
            result1 = op1.apply(image)
            result2 = op2.apply(image)

            assert result1.applied == result2.applied
            assert result1.parameters == result2.parameters
            np.testing.assert_array_equal(result1.image, result2.image)

    def test_different_seeds_produce_different_gamma(
        self, image: np.ndarray
    ) -> None:
        """Different random_state values are expected to yield different gamma."""
        op1 = GammaOperator(min_gamma=0.3, max_gamma=3.0, random_state=1)
        op2 = GammaOperator(min_gamma=0.3, max_gamma=3.0, random_state=2)

        _, params1 = op1._apply(image)
        _, params2 = op2._apply(image)

        assert params1["gamma"] != params2["gamma"]

    def test_gamma_within_configured_range(self, image: np.ndarray) -> None:
        """The sampled gamma always falls within [min_gamma, max_gamma]."""
        op = GammaOperator(min_gamma=0.5, max_gamma=1.8, random_state=5)

        for _ in range(25):
            _, params = op._apply(image)
            assert 0.5 <= params["gamma"] <= 1.8


# ---------------------------------------------------------------------------
# Successful augmentation
# ---------------------------------------------------------------------------


class TestSuccessfulAugmentation:
    """Tests for the gamma correction effect via the LUT implementation."""

    def test_identity_gamma_is_close_to_original(self, image: np.ndarray) -> None:
        """A gamma of 1.0 leaves pixel values effectively unchanged."""
        op = GammaOperator(min_gamma=1.0, max_gamma=1.0, random_state=0)
        result, params = op._apply(image)

        assert params["gamma"] == 1.0
        np.testing.assert_allclose(
            result.astype(np.int16), image.astype(np.int16), atol=1
        )

    def test_gamma_greater_than_one_darkens_image(self, image: np.ndarray) -> None:
        """A gamma greater than 1.0 darkens the image on average."""
        op = GammaOperator(min_gamma=2.0, max_gamma=2.0, random_state=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).mean() < image.astype(np.float64).mean()

    def test_gamma_less_than_one_brightens_image(self, image: np.ndarray) -> None:
        """A gamma less than 1.0 brightens the image on average."""
        op = GammaOperator(min_gamma=0.4, max_gamma=0.4, random_state=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).mean() > image.astype(np.float64).mean()

    def test_lookup_table_produces_valid_output(self) -> None:
        """The internal LUT builder produces a 256-entry, valid uint8 table."""
        lookup_table = GammaOperator._build_lookup_table(1.5)

        assert lookup_table.shape == (256,)
        assert lookup_table.dtype == np.uint8
        assert lookup_table.min() >= 0
        assert lookup_table.max() <= 255
        assert np.all(np.diff(lookup_table.astype(np.int16)) >= 0)


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------


class TestShapeAndDtypePreservation:
    """Tests confirming output shape and dtype match the input."""

    def test_output_shape_preserved(self, image: np.ndarray) -> None:
        """Output image shape matches the input image shape."""
        op = GammaOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.shape == image.shape

    def test_output_dtype_preserved(self, image: np.ndarray) -> None:
        """Output image dtype remains uint8."""
        op = GammaOperator(probability=1.0, random_state=2)
        result = op.apply(image)

        assert result.image.dtype == np.uint8

    @pytest.mark.parametrize("height,width", [(1, 1), (10, 20), (100, 50)])
    def test_various_shapes_preserved(self, height: int, width: int) -> None:
        """Output shape is preserved across a variety of synthetic image sizes."""
        rng = np.random.default_rng(0)
        synthetic_image = rng.integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        op = GammaOperator(probability=1.0, random_state=0)

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
        op = GammaOperator(probability=0.0, random_state=123)

        for _ in range(20):
            result = op.apply(image)
            assert result.applied is False
            assert result.success is True
            np.testing.assert_array_equal(result.image, image)

    def test_enabled_false_never_applies(self, image: np.ndarray) -> None:
        """A disabled operator never applies, regardless of probability."""
        op = GammaOperator(probability=1.0, enabled=False, random_state=123)

        result = op.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        np.testing.assert_array_equal(result.image, image)

    def test_probability_one_always_applies(self, image: np.ndarray) -> None:
        """A probability of 1.0 means the operator always applies."""
        op = GammaOperator(probability=1.0, random_state=123)

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
        op = GammaOperator(probability=1.0, random_state=1)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.operator_name == "gamma"
        assert result.applied is True
        assert result.success is True
        assert "gamma" in result.parameters
        assert result.error_message is None

    def test_skipped_result_contents(self, image: np.ndarray) -> None:
        """A skipped (disabled) application populates result fields correctly."""
        op = GammaOperator(enabled=False, random_state=1)
        result = op.apply(image)

        assert result.operator_name == "gamma"
        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert result.error_message is None

    def test_custom_operator_name_is_reflected_in_result(
        self, image: np.ndarray
    ) -> None:
        """A custom operator_name is reflected in the returned result."""
        op = GammaOperator(probability=1.0, random_state=1, operator_name="my_gamma")
        result = op.apply(image)

        assert result.operator_name == "my_gamma"


# ---------------------------------------------------------------------------
# Registry compatibility
# ---------------------------------------------------------------------------


class TestRegistryCompatibility:
    """Tests confirming GammaOperator integrates with OperatorRegistry."""

    def test_register_and_retrieve(self) -> None:
        """GammaOperator can be registered and retrieved by name."""
        registry = OperatorRegistry()

        registry.register("gamma", GammaOperator)

        assert registry.get("gamma") is GammaOperator
        assert "gamma" in registry

    def test_instantiate_from_registry(self, image: np.ndarray) -> None:
        """An operator class retrieved from the registry can be instantiated and applied."""
        registry = OperatorRegistry()
        registry.register("gamma", GammaOperator)

        operator_cls = registry.get("gamma")
        op = operator_cls(probability=1.0, random_state=42)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True