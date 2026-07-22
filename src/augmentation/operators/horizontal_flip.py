"""Horizontal flip augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone geometric augmentation
operator, ``HorizontalFlipOperator``, which mirrors an image
left-right. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own image transform in ``_apply``;
validation, enabled/probability gating, and result construction are
all handled by the base class.

Only OpenCV (``cv2``) is used for the underlying flip implementation.

Classes:
    HorizontalFlipOperator: Flips an image left-right using
        :func:`cv2.flip`.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["HorizontalFlipOperator"]

_HORIZONTAL_FLIP_CODE: Final[int] = 1


class HorizontalFlipOperator(BaseAugmentation):
    """Flip an image left-right (horizontal mirror only).

    This operator has no additional tunable numeric parameters beyond
    those inherited from
    :class:`~src.augmentation.operators.base.BaseAugmentation`. Whether
    a given image is flipped at all is governed entirely by the
    inherited ``probability`` gate and deterministic random state; the
    flip itself, once triggered, is a fixed, parameter-free
    transform.

    Parameters
    ----------
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` make identical
        probability-gate decisions. If ``None``, the generator is
        seeded from entropy and behaviour is non-deterministic.
    operator_name : str, default="horizontal_flip"
        The unique name under which this operator is identified.
    """

    def __init__(
        self,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "horizontal_flip",
    ) -> None:
        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Flip ``image`` horizontally (left-right).

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The horizontally flipped image, with identical shape and
            dtype to the input, and a dict recording
            ``{"flipped": True}``.
        """
        flipped = cv2.flip(image, _HORIZONTAL_FLIP_CODE)
        return flipped.astype(image.dtype, copy=False), {"flipped": True}