"""Gamma augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone photometric augmentation
operator, ``GammaOperator``, which applies gamma correction using a
randomly sampled gamma value and an efficient lookup-table (LUT)
implementation. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing.

Classes:
    GammaOperator: Applies gamma correction via a precomputed 256-entry
        lookup table for efficiency.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["GammaOperator"]

_PIXEL_MIN: Final[int] = 0
_PIXEL_MAX: Final[int] = 255
_LUT_SIZE: Final[int] = 256


class GammaOperator(BaseAugmentation):
    """Apply gamma correction to an image via a precomputed lookup table.

    The gamma value is sampled uniformly from
    ``[min_gamma, max_gamma]`` using the operator's inherited
    deterministic random state. Gamma correction is applied as
    ``output = 255 * (input / 255) ** gamma``, implemented via a
    256-entry lookup table (LUT) for efficiency rather than per-pixel
    exponentiation. A gamma value below ``1.0`` brightens the image, a
    value of exactly ``1.0`` leaves it unchanged, and a value above
    ``1.0`` darkens it.

    Parameters
    ----------
    min_gamma : float, default=0.7
        The minimum gamma value. Must be strictly positive.
    max_gamma : float, default=1.5
        The maximum gamma value. Must be strictly positive.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical gamma values
        and identical probability-gate decisions. If ``None``, the
        generator is seeded from entropy and behaviour is
        non-deterministic.
    operator_name : str, default="gamma"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_gamma`` or ``max_gamma`` is not strictly positive, or
        if ``min_gamma`` exceeds ``max_gamma``.
    """

    def __init__(
        self,
        min_gamma: float = 0.7,
        max_gamma: float = 1.5,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "gamma",
    ) -> None:
        self._validate_configuration(min_gamma=min_gamma, max_gamma=max_gamma)

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma

    @staticmethod
    def _validate_configuration(*, min_gamma: float, max_gamma: float) -> None:
        """Validate the gamma range at construction time.

        Parameters
        ----------
        min_gamma : float
            The minimum gamma value to validate.
        max_gamma : float
            The maximum gamma value to validate.

        Raises
        ------
        ValueError
            If ``min_gamma`` or ``max_gamma`` is not strictly
            positive, or if ``min_gamma`` exceeds ``max_gamma``.
        """
        if min_gamma <= 0:
            raise ValueError(
                f"'min_gamma' must be strictly positive, got {min_gamma!r}."
            )
        if max_gamma <= 0:
            raise ValueError(
                f"'max_gamma' must be strictly positive, got {max_gamma!r}."
            )
        if min_gamma > max_gamma:
            raise ValueError(
                f"'min_gamma' ({min_gamma!r}) must not exceed 'max_gamma' "
                f"({max_gamma!r})."
            )

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