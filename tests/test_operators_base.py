"""Unit tests for :mod:`src.augmentation.operators.base`.

These tests exercise the shared behaviour provided by
:class:`BaseAugmentation` (validation, enabled/probability gating,
deterministic sampling), the immutability and contents of
:class:`AugmentationResult`, and the full lifecycle of
:class:`OperatorRegistry`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.augmentation.operators.base import (
    AugmentationResult,
    BaseAugmentation,
    InvalidImageError,
    InvalidOperatorConfigError,
    OperatorNotFoundError,
    OperatorRegistrationError,
    OperatorRegistry,
    validate_image,
    validate_operator_name,
    validate_probability,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(height: int = 8, width: int = 8, fill: int = 100) -> np.ndarray:
    """Construct a valid uint8 RGB image for use in tests."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


class _AlwaysAppliesOperator(BaseAugmentation):
    """A concrete operator that always increments every pixel by one.

    Used to exercise the successful, deterministic execution path of
    :class:`BaseAugmentation` without needing a real image-processing
    algorithm.
    """

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        transformed = np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        return transformed, {"delta": 1}


class _FailingOperator(BaseAugmentation):
    """A concrete operator whose ``_apply`` always raises."""

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        raise RuntimeError("simulated operator failure")


class _RandomParamOperator(BaseAugmentation):
    """A concrete operator that samples a parameter from its own random_state.

    Used to verify that two operators built with the same seed produce
    identical sampled parameters (determinism), in addition to
    identical probability-gate decisions.
    """

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        magnitude = float(self.random_state.uniform(-5.0, 5.0))
        return image, {"magnitude": magnitude}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidateImage:
    """Tests for the standalone :func:`validate_image` helper."""

    def test_valid_rgb_image_passes(self) -> None:
        """A well-formed uint8 (H, W, 3) ndarray passes validation silently."""
        validate_image(_make_rgb_image())

    def test_non_ndarray_raises(self) -> None:
        """A non-ndarray input (e.g. a list) raises InvalidImageError."""
        with pytest.raises(InvalidImageError, match="numpy.ndarray"):
            validate_image([[1, 2, 3]])

    def test_wrong_dtype_raises(self) -> None:
        """A float32 image raises InvalidImageError mentioning dtype."""
        image = np.zeros((8, 8, 3), dtype=np.float32)
        with pytest.raises(InvalidImageError, match="dtype"):
            validate_image(image)

    def test_grayscale_image_raises(self) -> None:
        """A 2D grayscale image (H, W) raises InvalidImageError."""
        image = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="shape"):
            validate_image(image)

    def test_rgba_image_raises(self) -> None:
        """A 4-channel RGBA image raises InvalidImageError."""
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="shape"):
            validate_image(image)


class TestValidateProbability:
    """Tests for the standalone :func:`validate_probability` helper."""

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_valid_boundary_values_pass(self, value: float) -> None:
        """Boundary probabilities 0.0 and 1.0, plus a mid-range value, pass."""
        validate_probability(value)

    @pytest.mark.parametrize("value", [-0.01, 1.01, 5.0, -5.0])
    def test_invalid_values_raise(self, value: float) -> None:
        """Probabilities outside [0.0, 1.0] raise InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="probability"):
            validate_probability(value)


class TestValidateOperatorName:
    """Tests for the standalone :func:`validate_operator_name` helper."""

    def test_valid_name_passes(self) -> None:
        """A non-empty string name passes validation silently."""
        validate_operator_name("horizontal_flip")

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_or_blank_name_raises(self, name: str) -> None:
        """An empty or whitespace-only name raises InvalidOperatorConfigError."""
        with pytest.raises(InvalidOperatorConfigError, match="operator_name"):
            validate_operator_name(name)


# ---------------------------------------------------------------------------
# BaseAugmentation
# ---------------------------------------------------------------------------


class TestBaseAugmentationAbstract:
    """Tests confirming BaseAugmentation cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Instantiating BaseAugmentation directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseAugmentation(operator_name="base", probability=0.5)  # type: ignore[abstract]


class TestBaseAugmentationConstruction:
    """Tests for BaseAugmentation construction-time validation and exposed properties."""

    def test_exposes_constructor_arguments(self) -> None:
        """operator_name, probability, enabled, and random_state are all exposed."""
        operator = _AlwaysAppliesOperator(
            operator_name="always_applies", probability=0.75, enabled=True, seed=1
        )

        assert operator.operator_name == "always_applies"
        assert operator.probability == 0.75
        assert operator.enabled is True
        assert isinstance(operator.random_state, np.random.Generator)

    def test_invalid_operator_name_raises(self) -> None:
        """An empty operator_name raises InvalidOperatorConfigError at construction."""
        with pytest.raises(InvalidOperatorConfigError, match="operator_name"):
            _AlwaysAppliesOperator(operator_name="", probability=0.5)

    @pytest.mark.parametrize("probability", [-0.5, 1.5])
    def test_invalid_probability_raises(self, probability: float) -> None:
        """An out-of-range probability raises InvalidOperatorConfigError at construction."""
        with pytest.raises(InvalidOperatorConfigError, match="probability"):
            _AlwaysAppliesOperator(operator_name="op", probability=probability)


class TestBaseAugmentationApply:
    """Tests for the apply() lifecycle: validation, gating, and result contents."""

    def test_successful_subclass_execution(self) -> None:
        """A subclass with probability=1.0 always applies and returns the transformed image."""
        image = _make_rgb_image(fill=100)
        operator = _AlwaysAppliesOperator(
            operator_name="always_applies", probability=1.0, seed=7
        )

        result = operator.apply(image)

        assert isinstance(result, AugmentationResult)
        assert result.applied is True
        assert result.success is True
        assert result.operator_name == "always_applies"
        assert result.parameters == {"delta": 1}
        assert np.all(result.image == 101)
        assert result.error_message is None

    def test_disabled_operator_skips_augmentation(self) -> None:
        """A disabled operator never applies, regardless of probability."""
        image = _make_rgb_image(fill=100)
        operator = _AlwaysAppliesOperator(
            operator_name="always_applies", probability=1.0, enabled=False, seed=7
        )

        result = operator.apply(image)

        assert result.applied is False
        assert result.success is True
        assert result.parameters == {}
        assert np.array_equal(result.image, image)

    def test_probability_gate_zero_never_applies(self) -> None:
        """A probability of 0.0 means the operator is never applied."""
        image = _make_rgb_image()
        operator = _AlwaysAppliesOperator(
            operator_name="never_applies", probability=0.0, seed=123
        )

        for _ in range(20):
            result = operator.apply(image)
            assert result.applied is False
            assert result.success is True

    def test_probability_gate_one_always_applies(self) -> None:
        """A probability of 1.0 means the operator always applies."""
        image = _make_rgb_image()
        operator = _AlwaysAppliesOperator(
            operator_name="always_applies", probability=1.0, seed=123
        )

        for _ in range(20):
            result = operator.apply(image)
            assert result.applied is True
            assert result.success is True

    def test_deterministic_execution_using_random_state(self) -> None:
        """Two operators built with the same seed produce identical gate decisions and params."""
        image = _make_rgb_image()
        operator_a = _RandomParamOperator(
            operator_name="random_param", probability=0.5, seed=99
        )
        operator_b = _RandomParamOperator(
            operator_name="random_param", probability=0.5, seed=99
        )

        results_a = [operator_a.apply(image) for _ in range(10)]
        results_b = [operator_b.apply(image) for _ in range(10)]

        for result_a, result_b in zip(results_a, results_b):
            assert result_a.applied == result_b.applied
            assert result_a.parameters == result_b.parameters

    def test_different_seeds_can_diverge(self) -> None:
        """Operators built with different seeds are not guaranteed to match (sanity check)."""
        image = _make_rgb_image()
        operator_a = _RandomParamOperator(
            operator_name="random_param", probability=1.0, seed=1
        )
        operator_b = _RandomParamOperator(
            operator_name="random_param", probability=1.0, seed=2
        )

        result_a = operator_a.apply(image)
        result_b = operator_b.apply(image)

        assert result_a.parameters != result_b.parameters

    def test_apply_failure_returns_unsuccessful_result(self) -> None:
        """An operator whose _apply raises returns a failed, unapplied result."""
        image = _make_rgb_image()
        operator = _FailingOperator(
            operator_name="failing_operator", probability=1.0, seed=1
        )

        result = operator.apply(image)

        assert result.applied is False
        assert result.success is False
        assert result.error_message == "simulated operator failure"
        assert np.array_equal(result.image, image)

    def test_invalid_image_raises_before_gating(self) -> None:
        """An invalid image raises InvalidImageError, even for a disabled operator."""
        operator = _AlwaysAppliesOperator(
            operator_name="always_applies", probability=1.0, enabled=False
        )
        invalid_image = np.zeros((8, 8), dtype=np.uint8)

        with pytest.raises(InvalidImageError, match="shape"):
            operator.apply(invalid_image)

    def test_wrong_dtype_image_raises(self) -> None:
        """An image with a non-uint8 dtype raises InvalidImageError."""
        operator = _AlwaysAppliesOperator(operator_name="op", probability=1.0)
        image = np.zeros((8, 8, 3), dtype=np.float64)

        with pytest.raises(InvalidImageError, match="dtype"):
            operator.apply(image)

    def test_grayscale_image_raises(self) -> None:
        """A grayscale image raises InvalidImageError via apply()."""
        operator = _AlwaysAppliesOperator(operator_name="op", probability=1.0)
        image = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(InvalidImageError):
            operator.apply(image)

    def test_rgba_image_raises(self) -> None:
        """An RGBA image raises InvalidImageError via apply()."""
        operator = _AlwaysAppliesOperator(operator_name="op", probability=1.0)
        image = np.zeros((10, 10, 4), dtype=np.uint8)

        with pytest.raises(InvalidImageError):
            operator.apply(image)

    def test_apply_uses_mocked_random_state_for_gating(self) -> None:
        """apply() consults random_state.random() to decide whether to run _apply."""
        image = _make_rgb_image()
        operator = _AlwaysAppliesOperator(operator_name="op", probability=0.5, seed=1)
        mock_generator = MagicMock(spec=np.random.Generator)
        mock_generator.random.return_value = 0.1
        operator._random_state = mock_generator  # type: ignore[attr-defined]

        result = operator.apply(image)

        mock_generator.random.assert_called_once()
        assert result.applied is True


# ---------------------------------------------------------------------------
# AugmentationResult
# ---------------------------------------------------------------------------


class TestAugmentationResult:
    """Tests for the AugmentationResult dataclass."""

    def test_holds_all_expected_fields(self) -> None:
        """AugmentationResult stores and exposes all documented fields."""
        image = _make_rgb_image()
        result = AugmentationResult(
            image=image,
            operator_name="rotation",
            applied=True,
            success=True,
            parameters={"degrees": 3.2},
            error_message=None,
        )

        assert np.array_equal(result.image, image)
        assert result.operator_name == "rotation"
        assert result.applied is True
        assert result.success is True
        assert result.parameters == {"degrees": 3.2}
        assert result.error_message is None

    def test_default_parameters_is_empty_dict(self) -> None:
        """parameters defaults to an empty dict when not supplied."""
        result = AugmentationResult(
            image=_make_rgb_image(),
            operator_name="noop",
            applied=False,
            success=True,
        )

        assert result.parameters == {}

    def test_is_immutable(self) -> None:
        """AugmentationResult instances cannot have their fields reassigned."""
        result = AugmentationResult(
            image=_make_rgb_image(),
            operator_name="rotation",
            applied=True,
            success=True,
        )

        with pytest.raises(FrozenInstanceError):
            result.applied = False  # type: ignore[misc]

    def test_two_instances_with_equal_fields_are_equal(self) -> None:
        """Dataclass-generated equality compares by field values."""
        image = _make_rgb_image()
        result_a = AugmentationResult(
            image=image, operator_name="op", applied=True, success=True
        )
        result_b = AugmentationResult(
            image=image, operator_name="op", applied=True, success=True
        )

        assert result_a.operator_name == result_b.operator_name
        assert result_a.applied == result_b.applied
        assert result_a.success == result_b.success


# ---------------------------------------------------------------------------
# OperatorRegistry
# ---------------------------------------------------------------------------


class TestOperatorRegistry:
    """Tests for the OperatorRegistry lifecycle."""

    def test_register_and_get(self) -> None:
        """A registered operator class can be retrieved by name."""
        registry = OperatorRegistry()

        registry.register("always_applies", _AlwaysAppliesOperator)

        assert registry.get("always_applies") is _AlwaysAppliesOperator

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises OperatorRegistrationError."""
        registry = OperatorRegistry()
        registry.register("always_applies", _AlwaysAppliesOperator)

        with pytest.raises(OperatorRegistrationError, match="already registered"):
            registry.register("always_applies", _RandomParamOperator)

    def test_unregister_removes_operator(self) -> None:
        """unregister() removes an operator so it can no longer be retrieved."""
        registry = OperatorRegistry()
        registry.register("failing_operator", _FailingOperator)

        registry.unregister("failing_operator")

        assert "failing_operator" not in registry
        with pytest.raises(OperatorNotFoundError):
            registry.get("failing_operator")

    def test_unregister_missing_operator_raises(self) -> None:
        """unregister() on a name that was never registered raises OperatorNotFoundError."""
        registry = OperatorRegistry()

        with pytest.raises(OperatorNotFoundError):
            registry.unregister("does_not_exist")

    def test_get_missing_operator_raises(self) -> None:
        """get() on a name that was never registered raises OperatorNotFoundError."""
        registry = OperatorRegistry()

        with pytest.raises(OperatorNotFoundError, match="does_not_exist"):
            registry.get("does_not_exist")

    def test_list_registered_returns_sorted_names(self) -> None:
        """list_registered() returns all registered names, sorted alphabetically."""
        registry = OperatorRegistry()
        registry.register("rotation", _RandomParamOperator)
        registry.register("always_applies", _AlwaysAppliesOperator)
        registry.register("failing_operator", _FailingOperator)

        assert registry.list_registered() == [
            "always_applies",
            "failing_operator",
            "rotation",
        ]

    def test_clear_removes_all_operators(self) -> None:
        """clear() empties the registry entirely."""
        registry = OperatorRegistry()
        registry.register("always_applies", _AlwaysAppliesOperator)
        registry.register("failing_operator", _FailingOperator)

        registry.clear()

        assert registry.list_registered() == []
        assert len(registry) == 0

    def test_register_invalid_name_raises(self) -> None:
        """register() with an empty name raises InvalidOperatorConfigError."""
        registry = OperatorRegistry()

        with pytest.raises(InvalidOperatorConfigError):
            registry.register("", _AlwaysAppliesOperator)

    def test_register_non_operator_class_raises_type_error(self) -> None:
        """register() with a class that is not a BaseAugmentation subclass raises TypeError."""
        registry = OperatorRegistry()

        class NotAnOperator:
            pass

        with pytest.raises(TypeError):
            registry.register("not_an_operator", NotAnOperator)  # type: ignore[arg-type]

    def test_contains_and_len(self) -> None:
        """__contains__ and __len__ reflect the current registry state."""
        registry = OperatorRegistry()
        assert len(registry) == 0
        assert "always_applies" not in registry

        registry.register("always_applies", _AlwaysAppliesOperator)

        assert len(registry) == 1
        assert "always_applies" in registry

    def test_independent_registry_instances_do_not_share_state(self) -> None:
        """Two OperatorRegistry instances maintain fully independent mappings."""
        registry_a = OperatorRegistry()
        registry_b = OperatorRegistry()

        registry_a.register("always_applies", _AlwaysAppliesOperator)

        assert "always_applies" in registry_a
        assert "always_applies" not in registry_b