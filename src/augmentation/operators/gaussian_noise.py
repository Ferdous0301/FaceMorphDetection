"""Gaussian noise augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone noise augmentation
operator, ``GaussianNoiseOperator``, which adds zero-mean additive
Gaussian noise with a randomly sampled standard deviation to an image.
It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only NumPy is used for image processing; no OpenCV filtering
functions are involved.

Classes:
    GaussianNoiseOperator: Adds zero-mean Gaussian noise with a
        randomly sampled standard deviation, clipping the result into
        the valid ``[0, 255]`` pixel range.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["GaussianNoiseOperator"]


_NOISE_MEAN: Final[float] = 0.0
_PIXEL_MIN: Final[int] = 0
_PIXEL_MAX: Final[int] = 255


class GaussianNoiseOperator(BaseAugmentation):
    """Add zero-mean additive Gaussian noise with a sampled standard deviation.

    The noise standard deviation (``sigma``) is sampled uniformly from
    ``[min_sigma, max_sigma]`` using the operator's inherited
    deterministic random state. Independent, zero-mean Gaussian noise
    with the sampled ``sigma`` is added to every pixel channel, and
    the result is clipped to the valid ``[0, 255]`` pixel range before
    being cast back to ``uint8``. A ``sigma`` of ``0.0`` adds no
    noise, leaving the image unchanged.

    Parameters
    ----------
    min_sigma : float, default=2.0
        The minimum noise standard deviation. Must be non-negative.
    max_sigma : float, default=15.0
        The maximum noise standard deviation. Must be non-negative.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical sigma values
        and identical noise, as well as identical probability-gate
        decisions. If ``None``, the generator is seeded from entropy
        and behaviour is non-deterministic.
    operator_name : str, default="gaussian_noise"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_sigma`` is negative, if ``max_sigma`` is negative, or
        if ``min_sigma`` exceeds ``max_sigma``.
    """

    def __init__(
        self,
        min_sigma: float = 2.0,
        max_sigma: float = 15.0,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "gaussian_noise",
    ) -> None:
        self._validate_configuration(min_sigma=min_sigma, max_sigma=max_sigma)

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma

    @staticmethod
    def _validate_configuration(*, min_sigma: float, max_sigma: float) -> None:
        """Validate the sigma range at construction time.

        Parameters
        ----------
        min_sigma : float
            The minimum noise standard deviation to validate.
        max_sigma : float
            The maximum noise standard deviation to validate.

        Raises
        ------
        ValueError
            If ``min_sigma`` is negative, if ``max_sigma`` is
            negative, or if ``min_sigma`` exceeds ``max_sigma``.
        """
        if min_sigma < 0:
            raise ValueError(
                f"'min_sigma' must be non-negative, got {min_sigma!r}."
            )
        if max_sigma < 0:
            raise ValueError(
                f"'max_sigma' must be non-negative, got {max_sigma!r}."
            )
        if min_sigma > max_sigma:
            raise ValueError(
                f"'min_sigma' ({min_sigma!r}) must not exceed 'max_sigma' "
                f"({max_sigma!r})."
            )

    @property
    def min_sigma(self) -> float:
        """float: The minimum noise standard deviation."""
        return self._min_sigma

    @property
    def max_sigma(self) -> float:
        """float: The maximum noise standard deviation."""
        return self._max_sigma

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Add sampled zero-mean Gaussian noise to ``image``.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The noisy image, with identical shape and dtype to the
            input, and a dict containing the sampled ``"sigma"``.
        """
        sigma = float(self.random_state.uniform(self._min_sigma, self._max_sigma))

        noise = self.random_state.normal(loc=_NOISE_MEAN, scale=sigma, size=image.shape)
        noisy = image.astype(np.float64) + noise
        clipped = np.clip(noisy, _PIXEL_MIN, _PIXEL_MAX)

        return clipped.astype(image.dtype, copy=False), {"sigma": sigma}