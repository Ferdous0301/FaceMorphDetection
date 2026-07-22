"""Brightness augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone photometric augmentation
operator, ``BrightnessOperator``, which adjusts overall image
brightness by multiplying pixel intensities by a randomly sampled
factor. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing.

Classes:
    BrightnessOperator: Scales pixel intensities by a randomly sampled
        brightness factor, clipping the result into the valid
        ``[0, 255]`` pixel range.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["BrightnessOperator"]


class BrightnessOperator(BaseAugmentation):
    """Scale pixel intensities by a randomly sampled brightness factor.

    The brightness factor is sampled uniformly from
    ``[min_factor, max_factor]`` using the operator's inherited
    deterministic random state. Every pixel value is multiplied by the
    sampled factor and the result is clipped to the valid ``[0, 255]``
    pixel range before being cast back to ``uint8``. A factor below
    ``1.0`` darkens the image, a factor of exactly ``1.0`` leaves it
    unchanged, and a factor above ``1.0`` brightens it.

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
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical brightness
        factors and identical probability-gate decisions. If
        ``None``, the generator is seeded from entropy and behaviour
        is non-deterministic.
    operator_name : str, default="brightness"
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
        operator_name: str = "brightness",
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
        """Validate the brightness factor range at construction time.

        Parameters
        ----------
        min_factor : float
            The minimum brightness factor to validate.
        max_factor : float
            The maximum brightness factor to validate.

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