"""JPEG compression augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone augmentation operator,
``JPEGCompressionOperator``, which simulates JPEG compression
artifacts by encoding an image to JPEG at a randomly sampled quality
level and immediately decoding it back. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

OpenCV's :func:`cv2.imencode` and :func:`cv2.imdecode` are used for
the underlying JPEG round trip.

Classes:
    JPEGCompressionOperator: Simulates JPEG compression artifacts
        using a randomly sampled quality factor.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.augmentation.operators.base import (
    BaseAugmentation,
    InvalidOperatorConfigError,
)

__all__ = ["JPEGCompressionOperator"]

_JPEG_EXTENSION = ".jpg"
_JPEG_QUALITY_FLAG = cv2.IMWRITE_JPEG_QUALITY
_MIN_QUALITY = 1
_MAX_QUALITY = 100


class JPEGCompressionOperator(BaseAugmentation):
    """Simulate JPEG compression artifacts at a randomly sampled quality level.

    The JPEG quality factor is sampled uniformly from
    ``[min_quality, max_quality]`` using the operator's inherited
    deterministic random state. The image is encoded to an in-memory
    JPEG byte buffer at the sampled quality and immediately decoded
    back into a pixel array, reproducing the lossy compression
    artifacts (blocking, ringing, chroma subsampling) that a real JPEG
    round trip would introduce.

    Parameters
    ----------
    min_quality : int, default=30
        The minimum JPEG quality factor. Must be an integer in
        ``[1, 100]``.
    max_quality : int, default=90
        The maximum JPEG quality factor. Must be an integer in
        ``[1, 100]``.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical quality
        factors and identical probability-gate decisions. If
        ``None``, the generator is seeded from entropy and behaviour
        is non-deterministic.
    operator_name : str, default="jpeg_compression"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_quality`` or ``max_quality`` lies outside
        ``[1, 100]``, or if ``min_quality`` exceeds ``max_quality``.
    """

    def __init__(
        self,
        min_quality: int = 30,
        max_quality: int = 90,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "jpeg_compression",
    ) -> None:
        self._validate_configuration(
            min_quality=min_quality, max_quality=max_quality
        )

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_quality = min_quality
        self._max_quality = max_quality

    @staticmethod
    def _validate_configuration(*, min_quality: int, max_quality: int) -> None:
        """Validate the JPEG quality range at construction time.

        Parameters
        ----------
        min_quality : int
            The minimum JPEG quality factor to validate.
        max_quality : int
            The maximum JPEG quality factor to validate.

        Raises
        ------
        ValueError
            If ``min_quality`` or ``max_quality`` lies outside
            ``[1, 100]``, or if ``min_quality`` exceeds
            ``max_quality``.
        """
        if not (_MIN_QUALITY <= min_quality <= _MAX_QUALITY):
            raise ValueError(
                f"'min_quality' must be between {_MIN_QUALITY} and "
                f"{_MAX_QUALITY} (inclusive), got {min_quality!r}."
            )
        if not (_MIN_QUALITY <= max_quality <= _MAX_QUALITY):
            raise ValueError(
                f"'max_quality' must be between {_MIN_QUALITY} and "
                f"{_MAX_QUALITY} (inclusive), got {max_quality!r}."
            )
        if min_quality > max_quality:
            raise ValueError(
                f"'min_quality' ({min_quality!r}) must not exceed "
                f"'max_quality' ({max_quality!r})."
            )

    @property
    def min_quality(self) -> int:
        """int: The minimum JPEG quality factor."""
        return self._min_quality

    @property
    def max_quality(self) -> int:
        """int: The maximum JPEG quality factor."""
        return self._max_quality

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode ``image`` to JPEG at a sampled quality and decode it back.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The compressed-and-decompressed image, with identical
            shape and dtype to the input, and a dict containing the
            sampled ``"quality"``.

        Raises
        ------
        InvalidOperatorConfigError
            If OpenCV fails to encode the image to JPEG.
        """
        quality = int(
            self.random_state.integers(self._min_quality, self._max_quality + 1)
        )

        encode_params = [int(_JPEG_QUALITY_FLAG), quality]
        success, encoded_buffer = cv2.imencode(
            _JPEG_EXTENSION, image, encode_params
        )
        if not success:
            raise InvalidOperatorConfigError(
                "Failed to encode image to JPEG during compression augmentation."
            )

        decoded = cv2.imdecode(encoded_buffer, cv2.IMREAD_UNCHANGED)

        return decoded.astype(image.dtype, copy=False), {"quality": quality}