"""Sharpen augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone augmentation operator,
``SharpenOperator``, which sharpens an image using an unsharp-mask
technique at a randomly sampled sharpening strength. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing.

Classes:
    SharpenOperator: Sharpens an image via an unsharp mask at a
        randomly sampled strength.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["SharpenOperator"]

_BLUR_KERNEL_SIZE: Final[tuple[int, int]] = (0, 0)
_BLUR_SIGMA: Final[float] = 3.0


class SharpenOperator(BaseAugmentation):
    """Sharpen an image via an unsharp mask at a randomly sampled strength.

    The sharpening strength is sampled uniformly from
    ``[min_strength, max_strength]`` using the operator's inherited
    deterministic random state. Sharpening is implemented as a
    standard unsharp mask: a Gaussian-blurred copy of the image is
    subtracted from a scaled copy of the original, amplifying edges
    in proportion to the sampled strength::

        sharpened = image * (1 + strength) - blurred * strength

    A strength of ``0.0`` leaves the image unchanged; larger strengths
    produce a more pronounced sharpening effect.

    Parameters
    ----------
    min_strength : float, default=0.5
        The minimum sharpening strength. Must be non-negative.
    max_strength : float, default=2.0
        The maximum sharpening strength. Must be non-negative.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical strength
        values and identical probability-gate decisions. If
        ``None``, the generator is seeded from entropy and behaviour
        is non-deterministic.
    operator_name : str, default="sharpen"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_strength`` or ``max_strength`` is negative, or if
        ``min_strength`` exceeds ``max_strength``.
    """

    def __init__(
        self,
        min_strength: float = 0.5,
        max_strength: float = 2.0,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "sharpen",
    ) -> None:
        self._validate_configuration(
            min_strength=min_strength, max_strength=max_strength
        )

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_strength = min_strength
        self._max_strength = max_strength

    @staticmethod
    def _validate_configuration(*, min_strength: float, max_strength: float) -> None:
        """Validate the sharpening strength range at construction time.

        Parameters
        ----------
        min_strength : float
            The minimum sharpening strength to validate.
        max_strength : float
            The maximum sharpening strength to validate.

        Raises
        ------
        ValueError
            If ``min_strength`` or ``max_strength`` is negative, or if
            ``min_strength`` exceeds ``max_strength``.
        """
        if min_strength < 0:
            raise ValueError(
                f"'min_strength' must be non-negative, got {min_strength!r}."
            )
        if max_strength < 0:
            raise ValueError(
                f"'max_strength' must be non-negative, got {max_strength!r}."
            )
        if min_strength > max_strength:
            raise ValueError(
                f"'min_strength' ({min_strength!r}) must not exceed "
                f"'max_strength' ({max_strength!r})."
            )

    @property
    def min_strength(self) -> float:
        """float: The minimum sharpening strength."""
        return self._min_strength

    @property
    def max_strength(self) -> float:
        """float: The maximum sharpening strength."""
        return self._max_strength

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Sharpen ``image`` via an unsharp mask at a sampled strength.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The sharpened image, with identical shape and dtype to
            the input, and a dict containing the sampled
            ``"strength"``.
        """
        strength = float(
            self.random_state.uniform(self._min_strength, self._max_strength)
        )

        blurred = cv2.GaussianBlur(image, _BLUR_KERNEL_SIZE, _BLUR_SIGMA)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, blurred, -strength, 0.0
        )

        return sharpened.astype(image.dtype, copy=False), {"strength": strength}