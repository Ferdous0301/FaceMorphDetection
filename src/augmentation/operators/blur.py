"""Blur augmentation operators for the FMAD Data Augmentation stage.

This module implements the concrete blur family of augmentation
operators described by the augmentation architecture: isotropic
Gaussian blur and directional motion blur. Every operator here
inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only NumPy and OpenCV are used for image processing. Unless explicitly
noted otherwise, every operator preserves the input image's dtype,
spatial dimensions (height and width), and channel count.

Classes:
    GaussianBlurAugmentation: Applies isotropic Gaussian blur using a
        randomly sampled odd kernel size (and, optionally, a randomly
        sampled sigma).
    MotionBlurAugmentation: Applies directional motion blur using a
        randomly sampled odd kernel size and blur angle.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

from src.augmentation.operators.base import (
    BaseAugmentation,
    InvalidOperatorConfigError,
)

__all__ = [
    "GaussianBlurAugmentation",
    "MotionBlurAugmentation",
]


_AUTO_SIGMA: Final[float] = 0.0


def _validate_range(minimum: float, maximum: float, *, field_name: str) -> None:
    """Validate that ``minimum`` does not exceed ``maximum``.

    Parameters
    ----------
    minimum : float
        The lower bound of the range.
    maximum : float
        The upper bound of the range.
    field_name : str
        The name of the range being validated, used to build a
        descriptive error message.

    Raises
    ------
    InvalidOperatorConfigError
        If ``minimum`` is greater than ``maximum``.
    """
    if minimum > maximum:
        raise InvalidOperatorConfigError(
            f"'{field_name}' range is invalid: minimum ({minimum!r}) "
            f"exceeds maximum ({maximum!r})."
        )


def _validate_positive_odd_kernel_size(value: int, *, field_name: str) -> None:
    """Validate that ``value`` is a strictly positive, odd integer.

    Parameters
    ----------
    value : int
        The kernel size to validate.
    field_name : str
        The name of the field being validated, used to build a
        descriptive error message.

    Raises
    ------
    InvalidOperatorConfigError
        If ``value`` is not strictly positive, or if it is not odd.
    """
    if value <= 0:
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be a strictly positive integer, got {value!r}."
        )
    if value % 2 == 0:
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be an odd integer, got {value!r}."
        )


def _validate_non_negative(value: float, *, field_name: str) -> None:
    """Validate that ``value`` is not negative.

    Parameters
    ----------
    value : float
        The value to validate.
    field_name : str
        The name of the field being validated, used to build a
        descriptive error message.

    Raises
    ------
    InvalidOperatorConfigError
        If ``value`` is negative.
    """
    if value < 0:
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be non-negative, got {value!r}."
        )


def _sample_odd_kernel_size(
    random_state: np.random.Generator, minimum: int, maximum: int
) -> int:
    """Sample a random odd kernel size, inclusive, from ``[minimum, maximum]``.

    Both ``minimum`` and ``maximum`` are assumed to already be
    validated as strictly positive, odd integers with
    ``minimum <= maximum``. Sampling is performed over the set of odd
    integers in the range (rather than over all integers followed by
    rounding), so that every valid odd kernel size is equally likely.

    Parameters
    ----------
    random_state : numpy.random.Generator
        The deterministic random generator to sample from.
    minimum : int
        The minimum odd kernel size, inclusive.
    maximum : int
        The maximum odd kernel size, inclusive.

    Returns
    -------
    int
        A randomly sampled odd kernel size in ``[minimum, maximum]``.
    """
    odd_step_count = (maximum - minimum) // 2 + 1
    chosen_step = int(random_state.integers(0, odd_step_count))
    return minimum + 2 * chosen_step


class GaussianBlurAugmentation(BaseAugmentation):
    """Apply isotropic Gaussian blur using a randomly sampled odd kernel size.

    The kernel size is sampled uniformly (over the set of valid odd
    integers) from ``[min_kernel_size, max_kernel_size]`` using the
    operator's inherited deterministic random state. If a sigma range
    is configured (via ``min_sigma``/``max_sigma`` strictly greater
    than ``0.0``), a sigma value is additionally sampled uniformly
    from that range; otherwise sigma is left at ``0.0``, letting
    OpenCV derive it automatically from the kernel size.

    Parameters
    ----------
    min_kernel_size : int, default=3
        The minimum kernel size. Must be a strictly positive, odd
        integer.
    max_kernel_size : int, default=9
        The maximum kernel size. Must be a strictly positive, odd
        integer.
    min_sigma : float, default=0.0
        The minimum Gaussian sigma. Must be non-negative. A value of
        ``0.0`` for both ``min_sigma`` and ``max_sigma`` lets OpenCV
        compute sigma automatically from the sampled kernel size.
    max_sigma : float, default=0.0
        The maximum Gaussian sigma. Must be non-negative.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="gaussian_blur"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If either kernel size is not a strictly positive, odd integer,
        if ``min_kernel_size`` exceeds ``max_kernel_size``, if either
        sigma is negative, or if ``min_sigma`` exceeds ``max_sigma``.
    """

    def __init__(
        self,
        min_kernel_size: int = 3,
        max_kernel_size: int = 9,
        min_sigma: float = _AUTO_SIGMA,
        max_sigma: float = _AUTO_SIGMA,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "gaussian_blur",
    ) -> None:
        _validate_positive_odd_kernel_size(
            min_kernel_size, field_name="min_kernel_size"
        )
        _validate_positive_odd_kernel_size(
            max_kernel_size, field_name="max_kernel_size"
        )
        _validate_range(
            min_kernel_size, max_kernel_size, field_name="kernel_size"
        )
        _validate_non_negative(min_sigma, field_name="min_sigma")
        _validate_non_negative(max_sigma, field_name="max_sigma")
        _validate_range(min_sigma, max_sigma, field_name="sigma")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_kernel_size = min_kernel_size
        self._max_kernel_size = max_kernel_size
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma

    @property
    def min_kernel_size(self) -> int:
        """int: The minimum Gaussian blur kernel size."""
        return self._min_kernel_size

    @property
    def max_kernel_size(self) -> int:
        """int: The maximum Gaussian blur kernel size."""
        return self._max_kernel_size

    @property
    def min_sigma(self) -> float:
        """float: The minimum Gaussian blur sigma."""
        return self._min_sigma

    @property
    def max_sigma(self) -> float:
        """float: The maximum Gaussian blur sigma."""
        return self._max_sigma

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply Gaussian blur to ``image`` using a sampled kernel size and sigma.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The blurred image, with identical shape and dtype to the
            input, and a dict containing the sampled
            ``"kernel_size"`` and ``"sigma"``.
        """
        kernel_size = _sample_odd_kernel_size(
            self.random_state, self._min_kernel_size, self._max_kernel_size
        )
        sigma = float(self.random_state.uniform(self._min_sigma, self._max_sigma))

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

        parameters = {"kernel_size": kernel_size, "sigma": sigma}
        return blurred.astype(image.dtype, copy=False), parameters


class MotionBlurAugmentation(BaseAugmentation):
    """Apply directional motion blur using a sampled kernel size and angle.

    A kernel size is sampled uniformly (over the set of valid odd
    integers) from ``[min_kernel_size, max_kernel_size]``, and a blur
    angle, in degrees, is sampled uniformly from
    ``[min_angle, max_angle]``, both using the operator's inherited
    deterministic random state. A horizontal line kernel of the
    sampled size is constructed, rotated by the sampled angle, and
    normalized to sum to ``1.0`` before being convolved with the image
    via 2D filtering.

    Parameters
    ----------
    min_kernel_size : int, default=3
        The minimum kernel size. Must be a strictly positive, odd
        integer.
    max_kernel_size : int, default=15
        The maximum kernel size. Must be a strictly positive, odd
        integer.
    min_angle : float, default=0.0
        The minimum blur angle, in degrees.
    max_angle : float, default=180.0
        The maximum blur angle, in degrees.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="motion_blur"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If either kernel size is not a strictly positive, odd integer,
        if ``min_kernel_size`` exceeds ``max_kernel_size``, or if
        ``min_angle`` exceeds ``max_angle``.
    """

    def __init__(
        self,
        min_kernel_size: int = 3,
        max_kernel_size: int = 15,
        min_angle: float = 0.0,
        max_angle: float = 180.0,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "motion_blur",
    ) -> None:
        _validate_positive_odd_kernel_size(
            min_kernel_size, field_name="min_kernel_size"
        )
        _validate_positive_odd_kernel_size(
            max_kernel_size, field_name="max_kernel_size"
        )
        _validate_range(
            min_kernel_size, max_kernel_size, field_name="kernel_size"
        )
        _validate_range(min_angle, max_angle, field_name="angle")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_kernel_size = min_kernel_size
        self._max_kernel_size = max_kernel_size
        self._min_angle = min_angle
        self._max_angle = max_angle

    @property
    def min_kernel_size(self) -> int:
        """int: The minimum motion blur kernel size."""
        return self._min_kernel_size

    @property
    def max_kernel_size(self) -> int:
        """int: The maximum motion blur kernel size."""
        return self._max_kernel_size

    @property
    def min_angle(self) -> float:
        """float: The minimum motion blur angle, in degrees."""
        return self._min_angle

    @property
    def max_angle(self) -> float:
        """float: The maximum motion blur angle, in degrees."""
        return self._max_angle

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply motion blur to ``image`` using a sampled kernel size and angle.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The motion-blurred image, with identical shape and dtype
            to the input, and a dict containing the sampled
            ``"kernel_size"`` and ``"angle"``.
        """
        kernel_size = _sample_odd_kernel_size(
            self.random_state, self._min_kernel_size, self._max_kernel_size
        )
        angle = float(self.random_state.uniform(self._min_angle, self._max_angle))

        kernel = self._build_motion_kernel(kernel_size, angle)
        blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel)

        parameters = {"kernel_size": kernel_size, "angle": angle}
        return blurred.astype(image.dtype, copy=False), parameters

    @staticmethod
    def _build_motion_kernel(kernel_size: int, angle: float) -> np.ndarray:
        """Build a normalized, rotated motion blur kernel.

        A horizontal line of ones is drawn through the center row of a
        ``(kernel_size, kernel_size)`` matrix, rotated about the
        kernel's center by ``angle`` degrees, and normalized so its
        entries sum to ``1.0``.

        Parameters
        ----------
        kernel_size : int
            The odd, strictly positive size of the (square) kernel.
        angle : float
            The rotation angle, in degrees, applied to the horizontal
            line kernel.

        Returns
        -------
        numpy.ndarray
            A ``(kernel_size, kernel_size)`` ``float64`` kernel whose
            entries sum to ``1.0``.
        """
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        center = kernel_size // 2
        kernel[center, :] = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(
            (center, center), angle, 1.0
        )
        rotated_kernel = cv2.warpAffine(
            kernel, rotation_matrix, (kernel_size, kernel_size)
        )

        kernel_sum = rotated_kernel.sum()
        if kernel_sum == 0.0:
            rotated_kernel[center, center] = 1.0
            kernel_sum = 1.0

        return rotated_kernel / kernel_sum