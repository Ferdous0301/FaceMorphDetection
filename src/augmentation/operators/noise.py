"""Noise augmentation operators for the FMAD Data Augmentation stage.

This module implements the concrete noise family of augmentation
operators described by the augmentation architecture: additive
Gaussian noise and salt-and-pepper impulse noise. Every operator here
inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only NumPy and OpenCV are used for image processing. Unless explicitly
noted otherwise, every operator preserves the input image's dtype,
spatial dimensions (height and width), and channel count, and clips
pixel values to the valid ``[0, 255]`` range where required.

Classes:
    GaussianNoiseAugmentation: Adds zero-mean (by default) Gaussian
        noise with a randomly sampled standard deviation.
    SaltPepperNoiseAugmentation: Randomly replaces pixels with either
        the minimum or maximum intensity value ("pepper" and "salt"
        respectively), at a randomly sampled corruption probability.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from src.augmentation.operators.base import (
    BaseAugmentation,
    InvalidOperatorConfigError,
)

__all__ = [
    "GaussianNoiseAugmentation",
    "SaltPepperNoiseAugmentation",
]


_PIXEL_MIN: Final[int] = 0
_PIXEL_MAX: Final[int] = 255


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


def _validate_unit_interval(value: float, *, field_name: str) -> None:
    """Validate that ``value`` lies within the inclusive ``[0.0, 1.0]`` range.

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
        If ``value`` lies outside ``[0.0, 1.0]``.
    """
    if not (0.0 <= value <= 1.0):
        raise InvalidOperatorConfigError(
            f"'{field_name}' must be between 0.0 and 1.0 (inclusive), got {value!r}."
        )


class GaussianNoiseAugmentation(BaseAugmentation):
    """Add zero-mean (by default) Gaussian noise with a sampled standard deviation.

    The noise standard deviation (``sigma``) is sampled uniformly from
    ``[min_sigma, max_sigma]`` using the operator's inherited
    deterministic random state. Independent Gaussian noise, drawn with
    the configured ``mean`` and the sampled ``sigma``, is added to
    every pixel channel, and the result is clipped to the valid
    ``[0, 255]`` pixel range before being cast back to ``uint8``.

    Parameters
    ----------
    min_sigma : float, default=2.0
        The minimum noise standard deviation. Must be strictly
        positive.
    max_sigma : float, default=15.0
        The maximum noise standard deviation. Must be strictly
        positive.
    mean : float, default=0.0
        The mean of the additive Gaussian noise distribution.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="gaussian_noise"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_sigma`` or ``max_sigma`` is not strictly positive, or
        if ``min_sigma`` exceeds ``max_sigma``.
    """

    def __init__(
        self,
        min_sigma: float = 2.0,
        max_sigma: float = 15.0,
        mean: float = 0.0,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "gaussian_noise",
    ) -> None:
        _validate_strictly_positive(min_sigma, field_name="min_sigma")
        _validate_strictly_positive(max_sigma, field_name="max_sigma")
        _validate_range(min_sigma, max_sigma, field_name="sigma")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._mean = mean

    @property
    def min_sigma(self) -> float:
        """float: The minimum noise standard deviation."""
        return self._min_sigma

    @property
    def max_sigma(self) -> float:
        """float: The maximum noise standard deviation."""
        return self._max_sigma

    @property
    def mean(self) -> float:
        """float: The mean of the additive Gaussian noise distribution."""
        return self._mean

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Add sampled Gaussian noise to ``image``.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The noisy image, with identical shape and dtype to the
            input, and a dict containing the sampled ``"sigma"`` and
            the configured ``"mean"``.
        """
        sigma = float(self.random_state.uniform(self._min_sigma, self._max_sigma))

        noise = self.random_state.normal(loc=self._mean, scale=sigma, size=image.shape)
        noisy = image.astype(np.float64) + noise
        clipped = np.clip(noisy, _PIXEL_MIN, _PIXEL_MAX)

        parameters = {"sigma": sigma, "mean": self._mean}
        return clipped.astype(image.dtype, copy=False), parameters


class SaltPepperNoiseAugmentation(BaseAugmentation):
    """Randomly replace pixels with extreme "salt" or "pepper" intensities.

    A per-image corruption probability is sampled uniformly from
    ``[min_probability, max_probability]`` using the operator's
    inherited deterministic random state. Each pixel is independently
    selected for corruption with that probability; corrupted pixels
    are then split between "salt" (set to ``255``) and "pepper" (set
    to ``0``) according to ``salt_vs_pepper``. All three color channels
    of a corrupted pixel are set together, so noise appears as
    grayscale specks rather than colored speckles.

    Parameters
    ----------
    min_probability : float, default=0.01
        The minimum fraction of pixels to corrupt. Must lie within
        ``[0.0, 1.0]``.
    max_probability : float, default=0.05
        The maximum fraction of pixels to corrupt. Must lie within
        ``[0.0, 1.0]``.
    salt_vs_pepper : float, default=0.5
        The fraction of corrupted pixels that become "salt" (value
        ``255``) rather than "pepper" (value ``0``). Must lie within
        ``[0.0, 1.0]``.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="salt_pepper_noise"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_probability`` or ``max_probability`` lies outside
        ``[0.0, 1.0]``, if ``min_probability`` exceeds
        ``max_probability``, or if ``salt_vs_pepper`` lies outside
        ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        min_probability: float = 0.01,
        max_probability: float = 0.05,
        salt_vs_pepper: float = 0.5,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "salt_pepper_noise",
    ) -> None:
        _validate_unit_interval(min_probability, field_name="min_probability")
        _validate_unit_interval(max_probability, field_name="max_probability")
        _validate_range(
            min_probability, max_probability, field_name="noise_probability"
        )
        _validate_unit_interval(salt_vs_pepper, field_name="salt_vs_pepper")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_probability = min_probability
        self._max_probability = max_probability
        self._salt_vs_pepper = salt_vs_pepper

    @property
    def min_probability(self) -> float:
        """float: The minimum fraction of pixels to corrupt."""
        return self._min_probability

    @property
    def max_probability(self) -> float:
        """float: The maximum fraction of pixels to corrupt."""
        return self._max_probability

    @property
    def salt_vs_pepper(self) -> float:
        """float: The fraction of corrupted pixels that become "salt"."""
        return self._salt_vs_pepper

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Corrupt a sampled fraction of ``image`` pixels with salt/pepper noise.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The corrupted image, with identical shape and dtype to the
            input, and a dict containing the sampled
            ``"noise_probability"`` and the configured
            ``"salt_vs_pepper"`` ratio.
        """
        noise_probability = float(
            self.random_state.uniform(self._min_probability, self._max_probability)
        )

        height, width = image.shape[:2]
        corruption_roll = self.random_state.random(size=(height, width))
        corruption_mask = corruption_roll < noise_probability

        salt_mask = corruption_mask & (
            self.random_state.random(size=(height, width)) < self._salt_vs_pepper
        )
        pepper_mask = corruption_mask & ~salt_mask

        noisy = image.copy()
        noisy[salt_mask] = _PIXEL_MAX
        noisy[pepper_mask] = _PIXEL_MIN

        parameters = {
            "noise_probability": noise_probability,
            "salt_vs_pepper": self._salt_vs_pepper,
        }
        return noisy.astype(image.dtype, copy=False), parameters