"""Reusable base framework for augmentation operators.

This module defines the common abstraction layer that every concrete
augmentation operator (rotation, translation, scaling, horizontal flip,
brightness, contrast, gamma, gaussian blur, motion blur, gaussian noise,
JPEG compression, ...) in the FMAD Data Augmentation stage will inherit
from.

It provides:

* :class:`AugmentationResult` — an immutable record describing the
  outcome of applying a single operator to a single image.
* :class:`BaseAugmentation` — an abstract base class that owns all
  shared behaviour (validation, probability gating, enabled-state
  gating, and deterministic random sampling) so that concrete operators
  only need to implement their own transform logic in ``_apply``.
* :class:`OperatorRegistry` — a small in-memory registry that concrete
  operators can register themselves with, keyed by a unique operator
  name, so that the executor can discover and instantiate operators by
  name.
* Validation helpers used internally by :class:`BaseAugmentation` and
  available for reuse by concrete operator implementations.

No concrete augmentation algorithm is implemented in this module. It is
strictly the reusable foundation that the remainder of the augmentation
stage builds upon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

__all__ = [
    "AugmentationError",
    "InvalidImageError",
    "InvalidOperatorConfigError",
    "OperatorRegistrationError",
    "OperatorNotFoundError",
    "AugmentationResult",
    "BaseAugmentation",
    "OperatorRegistry",
    "validate_image",
    "validate_probability",
    "validate_operator_name",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AugmentationError(Exception):
    """Base exception for all errors raised by the augmentation framework."""


class InvalidImageError(AugmentationError, ValueError):
    """Raised when an image fails validation (type, dtype, or shape)."""


class InvalidOperatorConfigError(AugmentationError, ValueError):
    """Raised when operator construction parameters fail validation."""


class OperatorRegistrationError(AugmentationError, KeyError):
    """Raised when registering an operator name that is already registered."""


class OperatorNotFoundError(AugmentationError, KeyError):
    """Raised when looking up an operator name that is not registered."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_EXPECTED_NDIM: Final[int] = 3
_EXPECTED_CHANNELS: Final[int] = 3
_EXPECTED_DTYPE: Final[np.dtype] = np.dtype("uint8")


def validate_image(image: Any) -> None:
    """Validate that ``image`` is a well-formed augmentation input.

    An acceptable image is a :class:`numpy.ndarray` with dtype
    ``uint8`` and shape ``(H, W, 3)`` (height, width, three colour
    channels). Grayscale images (2 dimensions) and images with an
    alpha channel (4 channels) are rejected, as are non-array inputs
    and arrays with any other dtype.

    Args:
        image: The candidate image to validate.

    Raises:
        InvalidImageError: If ``image`` is not a :class:`numpy.ndarray`,
            if its dtype is not ``uint8``, or if its shape is not
            ``(H, W, 3)``.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            f"'image' must be a numpy.ndarray, got {type(image).__name__!r}."
        )

    if image.dtype != _EXPECTED_DTYPE:
        raise InvalidImageError(
            f"'image' must have dtype {_EXPECTED_DTYPE!s}, got {image.dtype!s}."
        )

    if image.ndim != _EXPECTED_NDIM or image.shape[-1] != _EXPECTED_CHANNELS:
        raise InvalidImageError(
            "'image' must have shape (H, W, 3), got shape "
            f"{image.shape!r} with {image.ndim} dimension(s)."
        )


def validate_probability(value: float, *, field_name: str = "probability") -> None:
    """Validate that ``value`` is a legal probability in ``[0.0, 1.0]``.

    Args:
        value: The probability value to validate.
        field_name: The name of the field being validated, used to
            build a descriptive error message.

    Raises:
        InvalidOperatorConfigError: If ``value`` is not within
            ``[0.0, 1.0]``.
    """
    if not (0.0 <= value <= 1.0):
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be between 0.0 and 1.0 (inclusive), got {value!r}."
        )


def validate_operator_name(name: str, *, field_name: str = "operator_name") -> None:
    """Validate that ``name`` is a non-empty operator name.

    Args:
        name: The operator name to validate.
        field_name: The name of the field being validated, used to
            build a descriptive error message.

    Raises:
        InvalidOperatorConfigError: If ``name`` is not a non-empty
            string (after stripping surrounding whitespace).
    """
    if not isinstance(name, str) or not name.strip():
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be a non-empty string, got {name!r}."
        )


# ---------------------------------------------------------------------------
# AugmentationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationResult:
    """The outcome of applying a single augmentation operator to an image.

    Instances are immutable so that a result, once produced, can be
    safely shared, cached, or logged without risk of accidental
    mutation by downstream consumers.

    Attributes:
        image: The resulting image. If the operator was not applied
            (because it was disabled, or the probability check did not
            pass), this is the original, unmodified input image. If
            application failed, this is also the original input image.
        operator_name: The unique name of the operator that produced
            this result.
        applied: Whether the operator's transform logic (``_apply``)
            was actually invoked and its output used. ``False`` when
            the operator was disabled or the probability gate did not
            pass.
        success: Whether the operator completed without raising an
            exception. ``True`` whenever ``applied`` is ``False``,
            since skipping is not a failure. ``False`` only when
            ``_apply`` raised an exception.
        parameters: A mapping of the concrete, sampled parameter values
            used for this application (e.g. ``{"degrees": 3.2}``).
            Empty when the operator was not applied.
        error_message: A human-readable description of the failure,
            populated only when ``success`` is ``False``.
    """

    image: np.ndarray
    operator_name: str
    applied: bool
    success: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


# ---------------------------------------------------------------------------
# BaseAugmentation
# ---------------------------------------------------------------------------


class BaseAugmentation(ABC):
    """Abstract base class for all augmentation operators.

    ``BaseAugmentation`` owns every piece of behaviour that is common
    to *all* augmentation operators: validating its own configuration
    at construction time, validating incoming images, deciding whether
    it should run at all (enabled-state and probability gating),
    sampling deterministically from its own random state, and wrapping
    the outcome of its transform in an :class:`AugmentationResult`.

    Concrete subclasses implement only :meth:`_apply`, which receives
    an already-validated image and must return the transformed image
    together with the concrete parameter values it sampled and used.

    Attributes:
        operator_name: The unique, human-readable name of this
            operator (e.g. ``"horizontal_flip"``).
        probability: The probability, in ``[0.0, 1.0]``, that this
            operator applies its transform to a given image.
        enabled: Whether this operator is eligible to run at all. When
            ``False``, :meth:`apply` always returns an unmodified,
            unapplied result.
        random_state: The :class:`numpy.random.Generator` used for all
            of this operator's random sampling, seeded deterministically
            from the ``seed`` constructor argument.
    """

    def __init__(
        self,
        operator_name: str,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialise the shared state of an augmentation operator.

        Args:
            operator_name: The unique name of this operator. Must be a
                non-empty string.
            probability: The probability, in ``[0.0, 1.0]``, that this
                operator's transform is applied to a given image.
            enabled: Whether this operator is eligible to run.
            seed: An optional integer seed used to construct this
                operator's private :class:`numpy.random.Generator`.
                Two operators constructed with the same seed produce
                identical sequences of probability-gate decisions and,
                for well-behaved subclasses, identical sampled
                parameters. If ``None``, the generator is seeded from
                entropy and behaviour is non-deterministic.

        Raises:
            InvalidOperatorConfigError: If ``operator_name`` is empty
                or if ``probability`` is outside ``[0.0, 1.0]``.
        """
        validate_operator_name(operator_name)
        validate_probability(probability)

        self._operator_name = operator_name
        self._probability = probability
        self._enabled = enabled
        self._seed = seed
        self._random_state: np.random.Generator = np.random.default_rng(seed)

    @property
    def operator_name(self) -> str:
        """str: The unique name identifying this operator."""
        return self._operator_name

    @property
    def probability(self) -> float:
        """float: The probability, in ``[0.0, 1.0]``, of applying this operator."""
        return self._probability

    @property
    def enabled(self) -> bool:
        """bool: Whether this operator is eligible to run at all."""
        return self._enabled

    @property
    def random_state(self) -> np.random.Generator:
        """numpy.random.Generator: This operator's private random generator."""
        return self._random_state

    def apply(self, image: np.ndarray) -> AugmentationResult:
        """Apply this operator to ``image``, honouring gating and validation.

        The flow is always, in order:

        1. Validate ``image`` (type, dtype, shape).
        2. If :attr:`enabled` is ``False``, return an unapplied,
           successful result carrying the original image.
        3. Draw a probability-gate sample from :attr:`random_state`. If
           the sample does not fall below :attr:`probability`, return
           an unapplied, successful result carrying the original
           image.
        4. Otherwise, invoke :meth:`_apply` on the validated image. If
           it raises, catch the exception and return a failed,
           unapplied result carrying the original image and the error
           message. If it succeeds, return an applied, successful
           result carrying the transformed image and the sampled
           parameters.

        Args:
            image: The input image, expected to be a
                :class:`numpy.ndarray` of dtype ``uint8`` and shape
                ``(H, W, 3)``.

        Returns:
            AugmentationResult: The outcome of attempting to apply this
            operator to ``image``.

        Raises:
            InvalidImageError: If ``image`` fails validation. This is
                raised rather than captured in the result because an
                invalid image indicates a caller bug, not a transient
                or expected operator failure.
        """
        validate_image(image)

        if not self._enabled:
            return AugmentationResult(
                image=image,
                operator_name=self._operator_name,
                applied=False,
                success=True,
                parameters={},
                error_message=None,
            )

        if not self._should_apply():
            return AugmentationResult(
                image=image,
                operator_name=self._operator_name,
                applied=False,
                success=True,
                parameters={},
                error_message=None,
            )

        try:
            transformed_image, parameters = self._apply(image)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, converted to a result
            return AugmentationResult(
                image=image,
                operator_name=self._operator_name,
                applied=False,
                success=False,
                parameters={},
                error_message=str(exc),
            )

        return AugmentationResult(
            image=transformed_image,
            operator_name=self._operator_name,
            applied=True,
            success=True,
            parameters=parameters,
            error_message=None,
        )

    def _should_apply(self) -> bool:
        """Draw a deterministic probability-gate decision.

        Returns:
            bool: ``True`` if a uniform sample drawn from
            :attr:`random_state` falls below :attr:`probability`,
            ``False`` otherwise.
        """
        return bool(self._random_state.random() < self._probability)

    @abstractmethod
    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply this operator's concrete transform to a validated image.

        Subclasses implement only this method. The image passed in has
        already been validated by :meth:`apply`, so implementations do
        not need to re-validate type, dtype, or shape.

        Args:
            image: The validated input image.

        Returns:
            tuple[numpy.ndarray, dict[str, Any]]: A pair of the
            transformed image and a mapping of the concrete parameter
            values sampled and used to produce it (e.g.
            ``{"degrees": 3.2}``).
        """


# ---------------------------------------------------------------------------
# OperatorRegistry
# ---------------------------------------------------------------------------


class OperatorRegistry:
    """An in-memory registry mapping operator names to operator classes.

    The registry allows concrete operator implementations to be
    discovered and instantiated by name, which decouples the
    augmentation executor from the concrete set of operator classes
    available at any given time.

    Each :class:`OperatorRegistry` instance owns its own independent
    mapping; the class is not a singleton, which keeps it trivially
    testable (tests can construct a fresh registry rather than
    mutating shared global state).
    """

    def __init__(self) -> None:
        """Initialise an empty operator registry."""
        self._operators: dict[str, type[BaseAugmentation]] = {}

    def register(
        self, name: str, operator_cls: type[BaseAugmentation]
    ) -> None:
        """Register ``operator_cls`` under the unique key ``name``.

        Args:
            name: The unique name to register the operator class
                under. Must be a non-empty string.
            operator_cls: The operator class to register. Must be a
                subclass of :class:`BaseAugmentation`.

        Raises:
            InvalidOperatorConfigError: If ``name`` is empty.
            TypeError: If ``operator_cls`` is not a subclass of
                :class:`BaseAugmentation`.
            OperatorRegistrationError: If ``name`` is already
                registered.
        """
        validate_operator_name(name)

        if not (isinstance(operator_cls, type) and issubclass(operator_cls, BaseAugmentation)):
            raise TypeError(
                "'operator_cls' must be a subclass of BaseAugmentation, "
                f"got {operator_cls!r}."
            )

        if name in self._operators:
            raise OperatorRegistrationError(
                f"An operator named {name!r} is already registered."
            )

        self._operators[name] = operator_cls

    def unregister(self, name: str) -> None:
        """Remove the operator registered under ``name``.

        Args:
            name: The unique name of the operator to remove.

        Raises:
            OperatorNotFoundError: If no operator is registered under
                ``name``.
        """
        if name not in self._operators:
            raise OperatorNotFoundError(f"No operator named {name!r} is registered.")

        del self._operators[name]

    def get(self, name: str) -> type[BaseAugmentation]:
        """Look up the operator class registered under ``name``.

        Args:
            name: The unique name of the operator to look up.

        Returns:
            type[BaseAugmentation]: The operator class registered
            under ``name``.

        Raises:
            OperatorNotFoundError: If no operator is registered under
                ``name``.
        """
        try:
            return self._operators[name]
        except KeyError as exc:
            raise OperatorNotFoundError(
                f"No operator named {name!r} is registered."
            ) from exc

    def list_registered(self) -> list[str]:
        """List the names of all currently registered operators.

        Returns:
            list[str]: The registered operator names, sorted
            alphabetically for deterministic output.
        """
        return sorted(self._operators.keys())

    def clear(self) -> None:
        """Remove all registered operators from this registry."""
        self._operators.clear()

    def __contains__(self, name: object) -> bool:
        """Support ``name in registry`` membership checks."""
        return name in self._operators

    def __len__(self) -> int:
        """Return the number of currently registered operators."""
        return len(self._operators)