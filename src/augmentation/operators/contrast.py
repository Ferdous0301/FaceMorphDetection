"""Contrast augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone photometric augmentation
operator, ``ContrastOperator``, which adjusts overall image contrast
by scaling pixel intensities about the image's mean gray level by a
randomly sampled factor. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing.

Classes:
    ContrastOperator: Scales pixel intensities about the image mean by
        a randomly sampled contrast factor, clipping the result into
        the valid ``[0, 255]`` pixel range.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["ContrastOperator"]


class ContrastOperator(BaseAugmentation):
    """Scale pixel intensities about the image mean by a sampled contrast factor.

    The contrast factor is sampled uniformly from
    ``[min_factor, max_factor]`` using the operator's inherited
    deterministic random state. Contrast adjustment is performed using
    the standard formula::

        output = mean + factor * (image - mean)

    where ``mean`` is the mean gray-level intensity of the input
    image. The result is clipped to the valid ``[0, 255]`` pixel range
    before being cast back to ``uint8``. A factor below ``1.0``
    decreases contrast, a factor of exactly ``1.0`` leaves the image
    unchanged, and a factor above ``1.0`` increases contrast.

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
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical contrast
        factors and identical probability-gate decisions. If
        ``None``, the generator is seeded from entropy and behaviour
        is non-deterministic.
    operator_name : str, default="contrast"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_factor`` is not strictly positive, if ``max_factor``
        is not strictly positive, or if ``min_factor`` exceeds
        ``max_factor``.
    """

    def __init__(
        self,
        min_factor: float = 0.7,
        max_factor: float = 1.3,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "contrast",
    ) -> None:
        self._validate_configuration(min_factor=min_factor, max_factor=max_factor)

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_factor = min_factor
        self._max_factor = max_factor

    @staticmethod
    def _validate_configuration(*, min_factor: float, max_factor: float) -> None:
        """Validate the contrast factor range at construction time.

        Parameters
        ----------
        min_factor : float
            The minimum contrast factor to validate.
        max_factor : float
            The maximum contrast factor to validate.

        Raises
        ------
        ValueError
            If ``min_factor`` is not strictly positive, if
            ``max_factor`` is not strictly positive, or if
            ``min_factor`` exceeds ``max_factor``.
        """
        if min_factor <= 0:
            raise ValueError(
                f"'min_factor' must be strictly positive, got {min_factor!r}."
            )
        if max_factor <= 0:
            raise ValueError(
                f"'max_factor' must be strictly positive, got {max_factor!r}."
            )
        if min_factor > max_factor:
            raise ValueError(
                f"'min_factor' ({min_factor!r}) must not exceed 'max_factor' "
                f"({max_factor!r})."
            )

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

        adjusted = mean_gray + factor * (image.astype(np.float64) - mean_gray)
        clipped = np.clip(adjusted, 0, 255)

        return clipped.astype(image.dtype, copy=False), {"factor": factor}