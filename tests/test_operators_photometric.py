"""Unit tests for the photometric augmentation operators.

Covers ``BrightnessAugmentation``, ``ContrastAugmentation``, and
``GammaAugmentation`` from ``src.augmentation.operators.photometric``,
as well as their inherited :class:`BaseAugmentation` behaviour
(enabled/probability gating, deterministic execution, and
:class:`AugmentationResult` contents).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.augmentation.operators.base import (
    AugmentationResult,
    InvalidOperatorConfigError,
)
from src.augmentation.operators.photometric import (
    BrightnessAugmentation,
    ContrastAugmentation,
    GammaAugmentation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic, non-trivial RGB test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


@pytest.fixture
def dark_image() -> np.ndarray:
    """A near-black image, useful for exercising clipping at the low end."""
    return np.full((32, 32, 3), 5, dtype=np.uint8)


@pytest.fixture
def bright_image() -> np.ndarray:
    """A near-white image, useful for exercising clipping at the high end."""
    return np.full((32, 32, 3), 250, dtype=np.uint8)


# ---------------------------------------------------------------------------
# BrightnessAugmentation
# ---------------------------------------------------------------------------


class TestBrightnessAugmentation:
    """Tests for :class:`BrightnessAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default factor range."""
        op = BrightnessAugmentation()

        assert op.min_factor == 0.7
        assert op.max_factor == 1.3
        assert op.operator_name == "brightness"

    def test_invalid_range_raises(self) -> None:
        """min_factor greater than max_factor raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="factor"):
            BrightnessAugmentation(min_factor=1.5, max_factor=0.5)

    @pytest.mark.parametrize("field_name", ["min_factor", "max_factor"])
    def test_non_positive_factor_raises(self, field_name: str) -> None:
        """A non-positive min_factor or max_factor raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            BrightnessAugmentation(**{field_name: 0.0})
        with pytest.raises(InvalidOperatorConfigError):
            BrightnessAugmentation(**{field_name: -1.0})

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_factor == max_factor is accepted."""
        op = BrightnessAugmentation(min_factor=1.2, max_factor=1.2, seed=0)

        assert op.min_factor == op.max_factor == 1.2

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image shape and uint8 dtype."""
        op = BrightnessAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_factor_within_range(self, image: np.ndarray) -> None:
        """The sampled brightness factor falls within the configured range."""
        op = BrightnessAugmentation(min_factor=0.5, max_factor=1.5, seed=1)
        _, params = op._apply(image)

        assert 0.5 <= params["factor"] <= 1.5

    def test_identity_factor_is_close_to_original(self, image: np.ndarray) -> None:
        """A factor of 1.0 leaves pixel values effectively unchanged."""
        op = BrightnessAugmentation(min_factor=1.0, max_factor=1.0, seed=0)
        result, params = op._apply(image)

        assert params["factor"] == 1.0
        np.testing.assert_array_equal(result, image)

    def test_clipping_at_high_end(self, bright_image: np.ndarray) -> None:
        """A large brightness factor clips pixel values to 255, not overflow."""
        op = BrightnessAugmentation(min_factor=3.0, max_factor=3.0, seed=0)
        result, params = op._apply(bright_image)

        assert params["factor"] == 3.0
        assert result.max() == 255
        assert result.dtype == np.uint8

    def test_clipping_at_low_end(self, dark_image: np.ndarray) -> None:
        """A small brightness factor keeps pixel values within [0, 255]."""
        op = BrightnessAugmentation(min_factor=0.1, max_factor=0.1, seed=0)
        result, params = op._apply(dark_image)

        assert params["factor"] == pytest.approx(0.1)
        assert result.min() >= 0
        assert result.dtype == np.uint8

    def test_darkening_reduces_mean_intensity(self, image: np.ndarray) -> None:
        """A factor below 1.0 reduces the average pixel intensity."""
        op = BrightnessAugmentation(min_factor=0.5, max_factor=0.5, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).mean() < image.astype(np.float64).mean()

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = BrightnessAugmentation(seed=123)
        op2 = BrightnessAugmentation(seed=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1["factor"] == params2["factor"]
        np.testing.assert_array_equal(result1, result2)

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records the sampled factor in the result."""
        op = BrightnessAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True
        assert "factor" in result.parameters


# ---------------------------------------------------------------------------
# ContrastAugmentation
# ---------------------------------------------------------------------------


class TestContrastAugmentation:
    """Tests for :class:`ContrastAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default factor range."""
        op = ContrastAugmentation()

        assert op.min_factor == 0.7
        assert op.max_factor == 1.3
        assert op.operator_name == "contrast"

    def test_invalid_range_raises(self) -> None:
        """min_factor greater than max_factor raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="factor"):
            ContrastAugmentation(min_factor=1.5, max_factor=0.5)

    @pytest.mark.parametrize("field_name", ["min_factor", "max_factor"])
    def test_non_positive_factor_raises(self, field_name: str) -> None:
        """A non-positive min_factor or max_factor raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            ContrastAugmentation(**{field_name: 0.0})
        with pytest.raises(InvalidOperatorConfigError):
            ContrastAugmentation(**{field_name: -2.0})

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image dimensions and uint8 dtype."""
        op = ContrastAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_factor_within_range(self, image: np.ndarray) -> None:
        """The sampled contrast factor falls within the configured range."""
        op = ContrastAugmentation(min_factor=0.6, max_factor=1.4, seed=2)
        _, params = op._apply(image)

        assert 0.6 <= params["factor"] <= 1.4

    def test_identity_factor_is_close_to_original(self, image: np.ndarray) -> None:
        """A factor of 1.0 leaves pixel values effectively unchanged."""
        op = ContrastAugmentation(min_factor=1.0, max_factor=1.0, seed=0)
        result, params = op._apply(image)

        assert params["factor"] == 1.0
        np.testing.assert_allclose(
            result.astype(np.int16), image.astype(np.int16), atol=1
        )

    def test_increasing_contrast_increases_spread(self, image: np.ndarray) -> None:
        """A factor above 1.0 increases the standard deviation of pixel values."""
        op = ContrastAugmentation(min_factor=1.5, max_factor=1.5, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).std() > image.astype(np.float64).std()

    def test_decreasing_contrast_decreases_spread(self, image: np.ndarray) -> None:
        """A factor below 1.0 decreases the standard deviation of pixel values."""
        op = ContrastAugmentation(min_factor=0.3, max_factor=0.3, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).std() < image.astype(np.float64).std()

    def test_clipping_does_not_overflow(self, bright_image: np.ndarray) -> None:
        """A large contrast factor keeps output within valid uint8 bounds."""
        op = ContrastAugmentation(min_factor=4.0, max_factor=4.0, seed=0)
        result, _ = op._apply(bright_image)

        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = ContrastAugmentation(seed=77)
        op2 = ContrastAugmentation(seed=77)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1["factor"] == params2["factor"]
        np.testing.assert_array_equal(result1, result2)

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records the sampled factor in the result."""
        op = ContrastAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert result.applied is True
        assert "factor" in result.parameters


# ---------------------------------------------------------------------------
# GammaAugmentation
# ---------------------------------------------------------------------------


class TestGammaAugmentation:
    """Tests for :class:`GammaAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default gamma range."""
        op = GammaAugmentation()

        assert op.min_gamma == 0.7
        assert op.max_gamma == 1.3
        assert op.operator_name == "gamma"

    def test_invalid_range_raises(self) -> None:
        """min_gamma greater than max_gamma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="gamma"):
            GammaAugmentation(min_gamma=1.5, max_gamma=0.5)

    @pytest.mark.parametrize("field_name", ["min_gamma", "max_gamma"])
    def test_non_positive_gamma_raises(self, field_name: str) -> None:
        """A non-positive min_gamma or max_gamma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            GammaAugmentation(**{field_name: 0.0})
        with pytest.raises(InvalidOperatorConfigError):
            GammaAugmentation(**{field_name: -0.5})

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image size and uint8 dtype."""
        op = GammaAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_gamma_within_range(self, image: np.ndarray) -> None:
        """The sampled gamma value falls within the configured range."""
        op = GammaAugmentation(min_gamma=0.5, max_gamma=1.8, seed=3)
        _, params = op._apply(image)

        assert 0.5 <= params["gamma"] <= 1.8

    def test_identity_gamma_is_close_to_original(self, image: np.ndarray) -> None:
        """A gamma of 1.0 leaves pixel values effectively unchanged."""
        op = GammaAugmentation(min_gamma=1.0, max_gamma=1.0, seed=0)
        result, params = op._apply(image)

        assert params["gamma"] == 1.0
        np.testing.assert_allclose(
            result.astype(np.int16), image.astype(np.int16), atol=1
        )

    def test_gamma_greater_than_one_darkens_image(self, image: np.ndarray) -> None:
        """A gamma greater than 1.0 darkens the image on average."""
        op = GammaAugmentation(min_gamma=2.0, max_gamma=2.0, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).mean() < image.astype(np.float64).mean()

    def test_gamma_less_than_one_brightens_image(self, image: np.ndarray) -> None:
        """A gamma less than 1.0 brightens the image on average."""
        op = GammaAugmentation(min_gamma=0.4, max_gamma=0.4, seed=0)
        result, _ = op._apply(image)

        assert result.astype(np.float64).mean() > image.astype(np.float64).mean()

    def test_lookup_table_produces_valid_output(self) -> None:
        """The internal LUT builder produces a 256-entry, valid uint8 table."""
        lookup_table = GammaAugmentation._build_lookup_table(1.5)

        assert lookup_table.shape == (256,)
        assert lookup_table.dtype == np.uint8
        assert lookup_table.min() >= 0
        assert lookup_table.max() <= 255
        # Gamma correction should be monotonically non-decreasing.
        assert np.all(np.diff(lookup_table.astype(np.int16)) >= 0)

    def test_lookup_table_endpoints_are_preserved(self) -> None:
        """Regardless of gamma, LUT entries 0 and 255 map to themselves."""
        for gamma in (0.2, 1.0, 3.0):
            lookup_table = GammaAugmentation._build_lookup_table(gamma)
            assert lookup_table[0] == 0
            assert lookup_table[255] == 255

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = GammaAugmentation(seed=55)
        op2 = GammaAugmentation(seed=55)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1["gamma"] == params2["gamma"]
        np.testing.assert_array_equal(result1, result2)

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records the sampled gamma in the result."""
        op = GammaAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert result.applied is True
        assert "gamma" in result.parameters


# ---------------------------------------------------------------------------
# General / inherited BaseAugmentation behaviour
# ---------------------------------------------------------------------------


class TestInheritedBaseAugmentationBehaviour:
    """Tests confirming photometric operators correctly inherit base behaviour."""

    @pytest.mark.parametrize(
        "operator_cls",
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
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
        [BrightnessAugmentation, ContrastAugmentation, GammaAugmentation],
    )
    def test_output_dimensions_preserved(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """Output image dimensions match the input image dimensions."""
        op = operator_cls(probability=1.0, seed=9)
        result = op.apply(image)

        assert result.image.shape == image.shape