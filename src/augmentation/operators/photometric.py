"""Photometric augmentation operators for the FMAD Data Augmentation stage.

This module implements the concrete photometric family of augmentation
operators described by the augmentation architecture: brightness
adjustment, contrast adjustment, and gamma correction. Every operator
here inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing. Unless
explicitly noted otherwise, every operator preserves the input image's
dtype, spatial dimensions (height and width), and channel count, and
clips pixel values to the valid ``[0, 255]`` range where required.

Classes:
    BrightnessAugmentation: Scales pixel intensities by a randomly
        sampled brightness factor.
    ContrastAugmentation: Scales pixel intensities about the image's
        mean gray level by a randomly sampled contrast factor.
    GammaAugmentation: Applies gamma correction via a precomputed
        lookup table for efficiency.
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
    "BrightnessAugmentation",
    "ContrastAugmentation",
    "GammaAugmentation",
]


_PIXEL_MIN: Final[int] = 0
_PIXEL_MAX: Final[int] = 255
_LUT_SIZE: Final[int] = 256


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


def _validate_strictly_positive(value: float, *, field_name: str) -> None:
    """Validate that ``value`` is strictly greater than zero.

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
        If ``value`` is less than or equal to zero.
    """
    if value <= 0:
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be strictly positive, got {value!r}."
        )


class BrightnessAugmentation(BaseAugmentation):
    """Scale pixel intensities by a randomly sampled brightness factor.

    The brightness factor is sampled uniformly from
    ``[min_factor, max_factor]`` using the operator's inherited
    deterministic random state. Every pixel value is multiplied by the
    sampled factor and the result is clipped to the valid ``[0, 255]``
    pixel range before being cast back to ``uint8``.

    Parameters
    ----------
    min_factor : float, default=0.7
        The minimum brightness factor. Must be strictly positive.
    max_factor : float, default=1.3
        The maximum brightness factor. Must be strictly positive.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="brightness"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_factor`` or ``max_factor`` is not strictly positive,
        or if ``min_factor`` exceeds ``max_factor``.
    """

    def __init__(
        self,
        min_factor: float = 0.7,
        max_factor: float = 1.3,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "brightness",
    ) -> None:
        _validate_strictly_positive(min_factor, field_name="min_factor")
        _validate_strictly_positive(max_factor, field_name="max_factor")
        _validate_range(min_factor, max_factor, field_name="factor")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_factor = min_factor
        self._max_factor = max_factor

    @property
    def min_factor(self) -> float:
        """float: The minimum brightness factor."""
        return self._min_factor

    @property
    def max_factor(self) -> float:
        """float: The maximum brightness factor."""
        return self._max_factor

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Scale ``image`` pixel intensities by a sampled brightness factor.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The brightness-adjusted image, with identical shape and
            dtype to the input, and a dict containing the sampled
            ``"factor"``.
        """
        factor = float(self.random_state.uniform(self._min_factor, self._max_factor))

        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0.0)

        return adjusted.astype(image.dtype, copy=False), {"factor": factor}


class ContrastAugmentation(BaseAugmentation):
    """Scale pixel intensities about the image mean by a sampled factor.

    The contrast factor is sampled uniformly from
    ``[min_factor, max_factor]`` using the operator's inherited
    deterministic random state. Each pixel value is moved toward or
    away from the image's overall mean gray level by the sampled
    factor, and the result is clipped to the valid ``[0, 255]`` pixel
    range before being cast back to ``uint8``.

    Parameters
    ----------
    min_factor : float, default=0.7
        The minimum contrast factor. Must be strictly positive.
    max_factor : float, default=1.3
        The maximum contrast factor. Must be strictly positive.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="contrast"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_factor`` or ``max_factor`` is not strictly positive,
        or if ``min_factor`` exceeds ``max_factor``.
    """

    def __init__(
        self,
        min_factor: float = 0.7,
        max_factor: float = 1.3,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "contrast",
    ) -> None:
        _validate_strictly_positive(min_factor, field_name="min_factor")
        _validate_strictly_positive(max_factor, field_name="max_factor")
        _validate_range(min_factor, max_factor, field_name="factor")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_factor = min_factor
        self._max_factor = max_factor

    @property
    def min_factor(self) -> float:
        """float: The minimum contrast factor."""
        return self._min_factor

    @property
    def max_factor(self) -> float:
        """float: The maximum contrast factor."""
        return self._max_factor

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Scale ``image`` pixel intensities about the mean by a sampled factor.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The contrast-adjusted image, with identical shape and
            dtype to the input, and a dict containing the sampled
            ``"factor"``.
        """
        factor = float(self.random_state.uniform(self._min_factor, self._max_factor))

        mean_gray = float(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))[0])
        beta = mean_gray * (1.0 - factor)

        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=beta)

        return adjusted.astype(image.dtype, copy=False), {"factor": factor}


class GammaAugmentation(BaseAugmentation):
    """Apply gamma correction to an image via a precomputed lookup table.

    The gamma value is sampled uniformly from ``[min_gamma, max_gamma]``
    using the operator's inherited deterministic random state. Gamma
    correction is applied as ``output = 255 * (input / 255) ** gamma``,
    implemented via a 256-entry lookup table (LUT) for efficiency
    rather than per-pixel exponentiation.

    Parameters
    ----------
    min_gamma : float, default=0.7
        The minimum gamma value. Must be strictly positive.
    max_gamma : float, default=1.3
        The maximum gamma value. Must be strictly positive.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="gamma"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_gamma`` or ``max_gamma`` is not strictly positive, or
        if ``min_gamma`` exceeds ``max_gamma``.
    """

    def __init__(
        self,
        min_gamma: float = 0.7,
        max_gamma: float = 1.3,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "gamma",
    ) -> None:
        _validate_strictly_positive(min_gamma, field_name="min_gamma")
        _validate_strictly_positive(max_gamma, field_name="max_gamma")
        _validate_range(min_gamma, max_gamma, field_name="gamma")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma

    @property
    def min_gamma(self) -> float:
        """float: The minimum gamma value."""
        return self._min_gamma

    @property
    def max_gamma(self) -> float:
        """float: The maximum gamma value."""
        return self._max_gamma

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply gamma correction to ``image`` via a sampled gamma value.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The gamma-corrected image, with identical shape and dtype
            to the input, and a dict containing the sampled
            ``"gamma"`` value.
        """
        gamma = float(self.random_state.uniform(self._min_gamma, self._max_gamma))

        lookup_table = self._build_lookup_table(gamma)
        corrected = cv2.LUT(image, lookup_table)

        return corrected.astype(image.dtype, copy=False), {"gamma": gamma}

    @staticmethod
    def _build_lookup_table(gamma: float) -> np.ndarray:
        """Build a 256-entry ``uint8`` gamma-correction lookup table.

        Parameters
        ----------
        gamma : float
            The gamma value to encode in the lookup table.

        Returns
        -------
        numpy.ndarray
            A ``(256,)`` ``uint8`` array where entry ``i`` holds the
            gamma-corrected value for input intensity ``i``.
        """
        intensities = np.arange(_LUT_SIZE, dtype=np.float64) / float(_PIXEL_MAX)
        corrected_intensities = np.power(intensities, gamma) * _PIXEL_MAX
        return np.clip(corrected_intensities, _PIXEL_MIN, _PIXEL_MAX).astype(np.uint8)