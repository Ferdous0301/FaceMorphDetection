"""Geometric augmentation operators for the FMAD Data Augmentation stage.

This module implements the concrete geometric family of augmentation
operators described by the augmentation architecture: rotation,
translation, scaling, and horizontal flip. Every operator here inherits
from :class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

Only OpenCV (``cv2``) and NumPy are used for image processing. Unless
explicitly noted otherwise, every operator preserves the input image's
dtype, spatial dimensions (height and width), and channel count.

Classes:
    RotationAugmentation: Rotates the image about its center by a
        randomly sampled angle.
    TranslationAugmentation: Shifts the image horizontally and/or
        vertically by randomly sampled percentages of its dimensions.
    ScalingAugmentation: Zooms the image in or out by a randomly
        sampled scale factor, then crops or pads back to the original
        size.
    HorizontalFlipAugmentation: Flips the image left-right.
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
    "RotationAugmentation",
    "TranslationAugmentation",
    "ScalingAugmentation",
    "HorizontalFlipAugmentation",
]


_BORDER_MODE: Final[int] = cv2.BORDER_REFLECT_101


def _validate_range(
    minimum: float, maximum: float, *, field_name: str
) -> None:
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


class RotationAugmentation(BaseAugmentation):
    """Rotate an image about its center by a randomly sampled angle.

    The rotation angle, in degrees, is sampled uniformly from
    ``[min_angle, max_angle]`` using the operator's inherited
    deterministic random state. The image is rotated about its own
    center, the output size is identical to the input size, and pixels
    exposed at the borders are filled using reflected-border padding
    (``cv2.BORDER_REFLECT_101``) so no artificial constant-color
    border is introduced.

    Parameters
    ----------
    min_angle : float, default=-15.0
        The minimum rotation angle, in degrees. Negative values rotate
        clockwise, positive values rotate counter-clockwise (following
        OpenCV's convention for ``cv2.getRotationMatrix2D``).
    max_angle : float, default=15.0
        The maximum rotation angle, in degrees.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="rotation"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_angle`` is greater than ``max_angle``.
    """

    def __init__(
        self,
        min_angle: float = -15.0,
        max_angle: float = 15.0,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "rotation",
    ) -> None:
        _validate_range(min_angle, max_angle, field_name="angle")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_angle = min_angle
        self._max_angle = max_angle

    @property
    def min_angle(self) -> float:
        """float: The minimum rotation angle, in degrees."""
        return self._min_angle

    @property
    def max_angle(self) -> float:
        """float: The maximum rotation angle, in degrees."""
        return self._max_angle

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Rotate ``image`` about its center by a sampled angle.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The rotated image, with identical shape and dtype to the
            input, and a dict containing the sampled ``"angle"`` in
            degrees.
        """
        angle = float(self.random_state.uniform(self._min_angle, self._max_angle))

        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            borderMode=_BORDER_MODE,
        )

        return rotated.astype(image.dtype, copy=False), {"angle": angle}


class TranslationAugmentation(BaseAugmentation):
    """Shift an image horizontally and/or vertically by sampled percentages.

    The horizontal and vertical shift amounts are each sampled
    independently as a percentage of the image's width and height,
    respectively, from their configured ranges. Percentages are
    expressed as fractions (e.g. ``0.1`` means ``10%`` of the
    dimension). The output has identical dimensions to the input, and
    pixels exposed at the borders are filled using reflected-border
    padding.

    Parameters
    ----------
    min_tx_percent : float, default=-0.1
        The minimum horizontal translation, as a fraction of image
        width. Must lie within ``[-1.0, 1.0]``.
    max_tx_percent : float, default=0.1
        The maximum horizontal translation, as a fraction of image
        width. Must lie within ``[-1.0, 1.0]``.
    min_ty_percent : float, default=-0.1
        The minimum vertical translation, as a fraction of image
        height. Must lie within ``[-1.0, 1.0]``.
    max_ty_percent : float, default=0.1
        The maximum vertical translation, as a fraction of image
        height. Must lie within ``[-1.0, 1.0]``.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="translation"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If either range's minimum exceeds its maximum, or if any
        percentage lies outside ``[-1.0, 1.0]``.
    """

    _PERCENT_BOUND: Final[float] = 1.0

    def __init__(
        self,
        min_tx_percent: float = -0.1,
        max_tx_percent: float = 0.1,
        min_ty_percent: float = -0.1,
        max_ty_percent: float = 0.1,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "translation",
    ) -> None:
        _validate_range(min_tx_percent, max_tx_percent, field_name="tx_percent")
        _validate_range(min_ty_percent, max_ty_percent, field_name="ty_percent")
        self._validate_percentage(min_tx_percent, field_name="min_tx_percent")
        self._validate_percentage(max_tx_percent, field_name="max_tx_percent")
        self._validate_percentage(min_ty_percent, field_name="min_ty_percent")
        self._validate_percentage(max_ty_percent, field_name="max_ty_percent")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_tx_percent = min_tx_percent
        self._max_tx_percent = max_tx_percent
        self._min_ty_percent = min_ty_percent
        self._max_ty_percent = max_ty_percent

    @staticmethod
    def _validate_percentage(value: float, *, field_name: str) -> None:
        """Validate that a translation percentage lies within ``[-1.0, 1.0]``.

        Parameters
        ----------
        value : float
            The percentage value to validate.
        field_name : str
            The name of the field being validated, used to build a
            descriptive error message.

        Raises
        ------
        InvalidOperatorConfigError
            If ``value`` lies outside ``[-1.0, 1.0]``.
        """
        bound = TranslationAugmentation._PERCENT_BOUND
        if not (-bound <= value <= bound):
            raise InvalidOperatorConfigError(
                f"'{field_name}' must be between -1.0 and 1.0 (inclusive), "
                f"got {value!r}."
            )

    @property
    def min_tx_percent(self) -> float:
        """float: The minimum horizontal translation, as a fraction of width."""
        return self._min_tx_percent

    @property
    def max_tx_percent(self) -> float:
        """float: The maximum horizontal translation, as a fraction of width."""
        return self._max_tx_percent

    @property
    def min_ty_percent(self) -> float:
        """float: The minimum vertical translation, as a fraction of height."""
        return self._min_ty_percent

    @property
    def max_ty_percent(self) -> float:
        """float: The maximum vertical translation, as a fraction of height."""
        return self._max_ty_percent

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Translate ``image`` by sampled horizontal and vertical percentages.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The translated image, with identical shape and dtype to
            the input, and a dict containing the sampled
            ``"tx_percent"``, ``"ty_percent"``, ``"tx_pixels"``, and
            ``"ty_pixels"`` values.
        """
        tx_percent = float(
            self.random_state.uniform(self._min_tx_percent, self._max_tx_percent)
        )
        ty_percent = float(
            self.random_state.uniform(self._min_ty_percent, self._max_ty_percent)
        )

        height, width = image.shape[:2]
        tx_pixels = tx_percent * width
        ty_pixels = ty_percent * height

        translation_matrix = np.array(
            [[1.0, 0.0, tx_pixels], [0.0, 1.0, ty_pixels]], dtype=np.float32
        )

        translated = cv2.warpAffine(
            image,
            translation_matrix,
            (width, height),
            borderMode=_BORDER_MODE,
        )

        parameters = {
            "tx_percent": tx_percent,
            "ty_percent": ty_percent,
            "tx_pixels": tx_pixels,
            "ty_pixels": ty_pixels,
        }
        return translated.astype(image.dtype, copy=False), parameters


class ScalingAugmentation(BaseAugmentation):
    """Zoom an image in or out, then crop or pad back to its original size.

    A scale factor is sampled uniformly from ``[min_scale, max_scale]``.
    A scale factor greater than ``1.0`` zooms in (the resized image is
    larger than the original and is then center-cropped back down); a
    scale factor less than ``1.0`` zooms out (the resized image is
    smaller than the original and is then reflect-padded back up). In
    both cases the returned image has exactly the same height and
    width as the input.

    Parameters
    ----------
    min_scale : float, default=0.9
        The minimum scale factor. Must be strictly positive.
    max_scale : float, default=1.1
        The maximum scale factor. Must be strictly positive.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="scaling"
        The unique name under which this operator is identified.

    Raises
    ------
    InvalidOperatorConfigError
        If ``min_scale`` or ``max_scale`` is not strictly positive, or
        if ``min_scale`` exceeds ``max_scale``.
    """

    def __init__(
        self,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "scaling",
    ) -> None:
        if min_scale <= 0:
            raise InvalidOperatorConfigError(
                f"'min_scale' must be strictly positive, got {min_scale!r}."
            )
        if max_scale <= 0:
            raise InvalidOperatorConfigError(
                f"'max_scale' must be strictly positive, got {max_scale!r}."
            )
        _validate_range(min_scale, max_scale, field_name="scale")

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
        )
        self._min_scale = min_scale
        self._max_scale = max_scale

    @property
    def min_scale(self) -> float:
        """float: The minimum scale factor."""
        return self._min_scale

    @property
    def max_scale(self) -> float:
        """float: The maximum scale factor."""
        return self._max_scale

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Zoom ``image`` by a sampled scale factor, preserving its final size.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The scaled image, cropped or padded back to the input's
            original shape and dtype, and a dict containing the
            sampled ``"scale"`` factor.
        """
        scale = float(self.random_state.uniform(self._min_scale, self._max_scale))

        height, width = image.shape[:2]
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )

        if scale >= 1.0:
            output = self._center_crop(resized, target_height=height, target_width=width)
        else:
            output = self._reflect_pad(resized, target_height=height, target_width=width)

        return output.astype(image.dtype, copy=False), {"scale": scale}

    @staticmethod
    def _center_crop(
        image: np.ndarray, *, target_height: int, target_width: int
    ) -> np.ndarray:
        """Crop ``image`` down to ``(target_height, target_width)`` about its center.

        Parameters
        ----------
        image : numpy.ndarray
            The (larger or equal) image to crop.
        target_height : int
            The desired output height.
        target_width : int
            The desired output width.

        Returns
        -------
        numpy.ndarray
            The center-cropped image of shape
            ``(target_height, target_width, C)``.
        """
        source_height, source_width = image.shape[:2]
        top = max(0, (source_height - target_height) // 2)
        left = max(0, (source_width - target_width) // 2)
        return np.ascontiguousarray(
            image[top : top + target_height, left : left + target_width]
        )

    @staticmethod
    def _reflect_pad(
        image: np.ndarray, *, target_height: int, target_width: int
    ) -> np.ndarray:
        """Reflect-pad ``image`` up to ``(target_height, target_width)``, centered.

        Parameters
        ----------
        image : numpy.ndarray
            The (smaller or equal) image to pad.
        target_height : int
            The desired output height.
        target_width : int
            The desired output width.

        Returns
        -------
        numpy.ndarray
            The padded image of shape ``(target_height, target_width, C)``.
        """
        source_height, source_width = image.shape[:2]
        total_pad_height = max(0, target_height - source_height)
        total_pad_width = max(0, target_width - source_width)

        top = total_pad_height // 2
        bottom = total_pad_height - top
        left = total_pad_width // 2
        right = total_pad_width - left

        return cv2.copyMakeBorder(
            image, top, bottom, left, right, borderType=_BORDER_MODE
        )


class HorizontalFlipAugmentation(BaseAugmentation):
    """Flip an image left-right (horizontal mirror only).

    This operator performs only a horizontal (left-right) flip; it
    never flips vertically. It has no additional tunable numeric
    parameters beyond those inherited from
    :class:`~src.augmentation.operators.base.BaseAugmentation`.

    Parameters
    ----------
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    seed : int, optional
        Seed for this operator's private random generator.
    operator_name : str, default="horizontal_flip"
        The unique name under which this operator is identified.
    """

    def __init__(
        self,
        probability: float = 0.5,
        enabled: bool = True,
        seed: int | None = None,
        operator_name: str = "horizontal_flip",
    ) -> None:
        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=seed,
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
        flipped = cv2.flip(image, 1)
        return flipped.astype(image.dtype, copy=False), {"flipped": True}