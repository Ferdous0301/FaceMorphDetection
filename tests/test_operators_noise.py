"""Unit tests for the noise augmentation operators.

Covers ``GaussianNoiseAugmentation`` and ``SaltPepperNoiseAugmentation``
from ``src.augmentation.operators.noise``, as well as their inherited
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
from src.augmentation.operators.noise import (
    GaussianNoiseAugmentation,
    SaltPepperNoiseAugmentation,
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
# GaussianNoiseAugmentation
# ---------------------------------------------------------------------------


class TestGaussianNoiseAugmentation:
    """Tests for :class:`GaussianNoiseAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default sigma range."""
        op = GaussianNoiseAugmentation()

        assert op.min_sigma == 2.0
        assert op.max_sigma == 15.0
        assert op.mean == 0.0
        assert op.operator_name == "gaussian_noise"

    def test_invalid_range_raises(self) -> None:
        """min_sigma greater than max_sigma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="sigma"):
            GaussianNoiseAugmentation(min_sigma=10.0, max_sigma=1.0)

    @pytest.mark.parametrize("field_name", ["min_sigma", "max_sigma"])
    def test_non_positive_sigma_raises(self, field_name: str) -> None:
        """A non-positive min_sigma or max_sigma raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            GaussianNoiseAugmentation(**{field_name: 0.0})
        with pytest.raises(InvalidOperatorConfigError):
            GaussianNoiseAugmentation(**{field_name: -3.0})

    def test_equal_bounds_allowed(self) -> None:
        """A degenerate range where min_sigma == max_sigma is accepted."""
        op = GaussianNoiseAugmentation(min_sigma=5.0, max_sigma=5.0, seed=0)

        assert op.min_sigma == op.max_sigma == 5.0

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image dimensions and uint8 dtype."""
        op = GaussianNoiseAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_sigma_within_range(self, image: np.ndarray) -> None:
        """The sampled sigma falls within the configured range."""
        op = GaussianNoiseAugmentation(min_sigma=3.0, max_sigma=10.0, seed=1)
        _, params = op._apply(image)

        assert 3.0 <= params["sigma"] <= 10.0

    def test_mean_is_recorded_in_parameters(self, image: np.ndarray) -> None:
        """The configured mean is recorded in the returned parameters."""
        op = GaussianNoiseAugmentation(mean=5.0, seed=0)
        _, params = op._apply(image)

        assert params["mean"] == 5.0

    def test_output_differs_from_input(self, image: np.ndarray) -> None:
        """Adding noise with a non-trivial sigma changes pixel values."""
        op = GaussianNoiseAugmentation(min_sigma=10.0, max_sigma=10.0, seed=0)
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)

    def test_clipping_at_high_end(self, bright_image: np.ndarray) -> None:
        """Large positive noise is clipped to 255, not overflowed."""
        op = GaussianNoiseAugmentation(
            min_sigma=50.0, max_sigma=50.0, mean=100.0, seed=0
        )
        result, _ = op._apply(bright_image)

        assert result.max() == 255
        assert result.dtype == np.uint8

    def test_clipping_at_low_end(self, dark_image: np.ndarray) -> None:
        """Large negative noise is clipped to 0, not underflowed."""
        op = GaussianNoiseAugmentation(
            min_sigma=50.0, max_sigma=50.0, mean=-100.0, seed=0
        )
        result, _ = op._apply(dark_image)

        assert result.min() == 0
        assert result.dtype == np.uint8

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = GaussianNoiseAugmentation(seed=123)
        op2 = GaussianNoiseAugmentation(seed=123)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_outputs(self, image: np.ndarray) -> None:
        """Different seeds are expected to yield different noisy images."""
        op1 = GaussianNoiseAugmentation(min_sigma=10.0, max_sigma=10.0, seed=1)
        op2 = GaussianNoiseAugmentation(min_sigma=10.0, max_sigma=10.0, seed=2)

        result1, _ = op1._apply(image)
        result2, _ = op2._apply(image)

        assert not np.array_equal(result1, result2)

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records sigma and mean in the result."""
        op = GaussianNoiseAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True
        assert "sigma" in result.parameters
        assert "mean" in result.parameters


# ---------------------------------------------------------------------------
# SaltPepperNoiseAugmentation
# ---------------------------------------------------------------------------


class TestSaltPepperNoiseAugmentation:
    """Tests for :class:`SaltPepperNoiseAugmentation`."""

    def test_default_construction(self) -> None:
        """Default construction exposes the documented default parameters."""
        op = SaltPepperNoiseAugmentation()

        assert op.min_probability == 0.01
        assert op.max_probability == 0.05
        assert op.salt_vs_pepper == 0.5
        assert op.operator_name == "salt_pepper_noise"

    def test_invalid_probability_range_raises(self) -> None:
        """min_probability greater than max_probability raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="noise_probability"):
            SaltPepperNoiseAugmentation(min_probability=0.5, max_probability=0.1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_probability": -0.1},
            {"min_probability": 1.5},
            {"max_probability": -0.1},
            {"max_probability": 1.5},
        ],
    )
    def test_out_of_bounds_probability_raises(self, kwargs: dict) -> None:
        """A probability outside [0.0, 1.0] raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError):
            SaltPepperNoiseAugmentation(**kwargs)

    @pytest.mark.parametrize("ratio", [-0.1, 1.1, 5.0])
    def test_invalid_salt_vs_pepper_raises(self, ratio: float) -> None:
        """A salt_vs_pepper ratio outside [0.0, 1.0] raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="salt_vs_pepper"):
            SaltPepperNoiseAugmentation(salt_vs_pepper=ratio)

    @pytest.mark.parametrize("bound", [0.0, 1.0])
    def test_boundary_probabilities_allowed(self, bound: float) -> None:
        """Boundary probability values of 0.0 and 1.0 are accepted."""
        op = SaltPepperNoiseAugmentation(min_probability=bound, max_probability=bound)

        assert op.min_probability == bound
        assert op.max_probability == bound

    def test_apply_preserves_shape_and_dtype(self, image: np.ndarray) -> None:
        """_apply preserves image dimensions and uint8 dtype."""
        op = SaltPepperNoiseAugmentation(seed=0)
        result, _ = op._apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_probability_within_range(self, image: np.ndarray) -> None:
        """The sampled noise probability falls within the configured range."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.02, max_probability=0.2, seed=2
        )
        _, params = op._apply(image)

        assert 0.02 <= params["noise_probability"] <= 0.2

    def test_salt_vs_pepper_recorded_in_parameters(self, image: np.ndarray) -> None:
        """The configured salt_vs_pepper ratio is recorded in the returned parameters."""
        op = SaltPepperNoiseAugmentation(salt_vs_pepper=0.75, seed=0)
        _, params = op._apply(image)

        assert params["salt_vs_pepper"] == 0.75

    def test_zero_probability_is_identity(self, image: np.ndarray) -> None:
        """A noise probability of exactly 0.0 corrupts no pixels."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.0, max_probability=0.0, seed=0
        )
        result, params = op._apply(image)

        assert params["noise_probability"] == 0.0
        np.testing.assert_array_equal(result, image)

    def test_output_differs_from_input_with_high_probability(
        self, image: np.ndarray
    ) -> None:
        """A high corruption probability changes a meaningful number of pixels."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.3, max_probability=0.3, seed=0
        )
        result, _ = op._apply(image)

        assert not np.array_equal(result, image)

    def test_only_salt_when_ratio_is_one(self, image: np.ndarray) -> None:
        """A salt_vs_pepper ratio of 1.0 produces only salt (255) corruption."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.3,
            max_probability=0.3,
            salt_vs_pepper=1.0,
            seed=0,
        )
        result, _ = op._apply(image)

        changed_mask = np.any(result != image, axis=-1)
        changed_pixels = result[changed_mask]

        assert changed_pixels.size > 0
        assert np.all(changed_pixels == 255)

    def test_only_pepper_when_ratio_is_zero(self, image: np.ndarray) -> None:
        """A salt_vs_pepper ratio of 0.0 produces only pepper (0) corruption."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.3,
            max_probability=0.3,
            salt_vs_pepper=0.0,
            seed=0,
        )
        result, _ = op._apply(image)

        changed_mask = np.any(result != image, axis=-1)
        changed_pixels = result[changed_mask]

        assert changed_pixels.size > 0
        assert np.all(changed_pixels == 0)

    def test_corrupted_pixels_are_extreme_values(self, image: np.ndarray) -> None:
        """Every corrupted pixel takes either the minimum or maximum intensity."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.3, max_probability=0.3, seed=0
        )
        result, _ = op._apply(image)

        changed_mask = np.any(result != image, axis=-1)
        changed_pixels = result[changed_mask]

        assert np.all((changed_pixels == 0) | (changed_pixels == 255))

    def test_corrupted_pixel_channels_set_together(self, image: np.ndarray) -> None:
        """Corrupted pixels have all three channels set to the same extreme value."""
        op = SaltPepperNoiseAugmentation(
            min_probability=0.3, max_probability=0.3, seed=0
        )
        result, _ = op._apply(image)

        changed_mask = np.any(result != image, axis=-1)
        changed_pixels = result[changed_mask]

        assert np.all(changed_pixels[:, 0] == changed_pixels[:, 1])
        assert np.all(changed_pixels[:, 1] == changed_pixels[:, 2])

    def test_deterministic_with_same_seed(self, image: np.ndarray) -> None:
        """Two operators built with the same seed produce identical output."""
        op1 = SaltPepperNoiseAugmentation(seed=77)
        op2 = SaltPepperNoiseAugmentation(seed=77)

        result1, params1 = op1._apply(image)
        result2, params2 = op2._apply(image)

        assert params1 == params2
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_outputs(self, image: np.ndarray) -> None:
        """Different seeds are expected to yield different corrupted images."""
        op1 = SaltPepperNoiseAugmentation(
            min_probability=0.2, max_probability=0.2, seed=1
        )
        op2 = SaltPepperNoiseAugmentation(
            min_probability=0.2, max_probability=0.2, seed=2
        )

        result1, _ = op1._apply(image)
        result2, _ = op2._apply(image)

        assert not np.array_equal(result1, result2)

    def test_parameters_are_recorded_via_apply(self, image: np.ndarray) -> None:
        """The public apply() lifecycle records the sampled parameters in the result."""
        op = SaltPepperNoiseAugmentation(probability=1.0, seed=5)
        result = op.apply(image)

        assert result.applied is True
        assert "noise_probability" in result.parameters
        assert "salt_vs_pepper" in result.parameters


# ---------------------------------------------------------------------------
# General / inherited BaseAugmentation behaviour
# ---------------------------------------------------------------------------


class TestInheritedBaseAugmentationBehaviour:
    """Tests confirming noise operators correctly inherit base behaviour."""

    @pytest.mark.parametrize(
        "operator_cls",
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
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
        [GaussianNoiseAugmentation, SaltPepperNoiseAugmentation],
    )
    def test_output_dimensions_preserved(
        self, operator_cls: type, image: np.ndarray
    ) -> None:
        """Output image dimensions match the input image dimensions."""
        op = operator_cls(probability=1.0, seed=9)
        result = op.apply(image)

        assert result.image.shape == image.shape