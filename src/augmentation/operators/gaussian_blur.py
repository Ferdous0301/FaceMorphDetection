"""Gaussian blur augmentation operator for the FMAD Data Augmentation stage.

This module implements a single, standalone blur augmentation
operator, ``GaussianBlurOperator``, which applies isotropic Gaussian
blur using a randomly sampled odd kernel size and a configurable
sigma. It inherits from
:class:`~src.augmentation.operators.base.BaseAugmentation` and
therefore only implements its own parameter sampling and image
transform in ``_apply``; validation, enabled/probability gating, and
result construction are all handled by the base class.

OpenCV's :func:`cv2.GaussianBlur` is used for the underlying blur
implementation.

Classes:
    GaussianBlurOperator: Applies Gaussian blur using a randomly
        sampled odd kernel size and a configurable sigma.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = ["GaussianBlurOperator"]


class GaussianBlurOperator(BaseAugmentation):
    """Apply Gaussian blur using a randomly sampled odd kernel size.

    The kernel size is sampled uniformly (over the set of valid odd
    integers) from ``[min_kernel_size, max_kernel_size]`` using the
    operator's inherited deterministic random state. For example, with
    ``min_kernel_size=3`` and ``max_kernel_size=11``, the sampled
    kernel size is always one of ``3, 5, 7, 9, 11``. The configured
    ``sigma`` is used directly as OpenCV's ``sigmaX``; a ``sigma`` of
    ``0.0`` lets OpenCV compute an appropriate sigma automatically
    from the sampled kernel size.

    Parameters
    ----------
    min_kernel_size : int, default=3
        The minimum kernel size. Must be a strictly positive, odd
        integer.
    max_kernel_size : int, default=11
        The maximum kernel size. Must be a strictly positive, odd
        integer.
    sigma : float, default=0.0
        The Gaussian sigma passed to OpenCV as ``sigmaX``. Must be
        non-negative. A value of ``0.0`` lets OpenCV derive sigma
        automatically from the sampled kernel size.
    probability : float, default=0.5
        The probability that this operator is applied to a given
        image.
    enabled : bool, default=True
        Whether this operator is eligible to run.
    random_state : int, optional
        Seed used to construct this operator's private, deterministic
        :class:`numpy.random.Generator`. Two operators constructed
        with the same ``random_state`` sample identical kernel sizes
        and identical probability-gate decisions. If ``None``, the
        generator is seeded from entropy and behaviour is
        non-deterministic.
    operator_name : str, default="gaussian_blur"
        The unique name under which this operator is identified.

    Raises
    ------
    ValueError
        If ``min_kernel_size`` or ``max_kernel_size`` is not a
        strictly positive, odd integer, if ``min_kernel_size`` exceeds
        ``max_kernel_size``, or if ``sigma`` is negative.
    """

    def __init__(
        self,
        min_kernel_size: int = 3,
        max_kernel_size: int = 11,
        sigma: float = 0.0,
        probability: float = 0.5,
        enabled: bool = True,
        random_state: int | None = None,
        operator_name: str = "gaussian_blur",
    ) -> None:
        self._validate_configuration(
            min_kernel_size=min_kernel_size,
            max_kernel_size=max_kernel_size,
            sigma=sigma,
        )

        super().__init__(
            operator_name=operator_name,
            probability=probability,
            enabled=enabled,
            seed=random_state,
        )
        self._min_kernel_size = min_kernel_size
        self._max_kernel_size = max_kernel_size
        self._sigma = sigma

    @staticmethod
    def _validate_configuration(
        *, min_kernel_size: int, max_kernel_size: int, sigma: float
    ) -> None:
        """Validate the kernel size range and sigma at construction time.

        Parameters
        ----------
        min_kernel_size : int
            The minimum kernel size to validate.
        max_kernel_size : int
            The maximum kernel size to validate.
        sigma : float
            The Gaussian sigma to validate.

        Raises
        ------
        ValueError
            If ``min_kernel_size`` or ``max_kernel_size`` is not a
            strictly positive, odd integer, if ``min_kernel_size``
            exceeds ``max_kernel_size``, or if ``sigma`` is negative.
        """
        GaussianBlurOperator._validate_positive_odd_kernel_size(
            min_kernel_size, field_name="min_kernel_size"
        )
        GaussianBlurOperator._validate_positive_odd_kernel_size(
            max_kernel_size, field_name="max_kernel_size"
        )
        if min_kernel_size > max_kernel_size:
            raise ValueError(
                f"'min_kernel_size' ({min_kernel_size!r}) must not exceed "
                f"'max_kernel_size' ({max_kernel_size!r})."
            )
        if sigma < 0:
            raise ValueError(f"'sigma' must be non-negative, got {sigma!r}.")

    @staticmethod
    def _validate_positive_odd_kernel_size(value: int, *, field_name: str) -> None:
        """Validate that ``value`` is a strictly positive, odd integer.

        Parameters
        ----------
        value : int
            The kernel size to validate.
        field_name : str
            The name of the field being validated, used to build a
            descriptive error message.

        Raises
        ------
        ValueError
            If ``value`` is not strictly positive, or if it is not
            odd.
        """
        if value <= 0:
            raise ValueError(
                f"'{field_name}' must be a strictly positive integer, "
                f"got {value!r}."
            )
        if value % 2 == 0:
            raise ValueError(
                f"'{field_name}' must be an odd integer, got {value!r}."
            )

    @property
    def min_kernel_size(self) -> int:
        """int: The minimum Gaussian blur kernel size."""
        return self._min_kernel_size

    @property
    def max_kernel_size(self) -> int:
        """int: The maximum Gaussian blur kernel size."""
        return self._max_kernel_size

    @property
    def sigma(self) -> float:
        """float: The configured Gaussian sigma (``0.0`` means auto-derived)."""
        return self._sigma

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply Gaussian blur to ``image`` using a sampled odd kernel size.

        Parameters
        ----------
        image : numpy.ndarray
            The validated input image, of shape ``(H, W, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, Any]]
            The blurred image, with identical shape and dtype to the
            input, and a dict containing the sampled
            ``"kernel_size"`` and the configured ``"sigma"``.
        """
        kernel_size = self._sample_odd_kernel_size()

        blurred = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), sigmaX=self._sigma
        )

        parameters = {"kernel_size": kernel_size, "sigma": self._sigma}
        return blurred.astype(image.dtype, copy=False), parameters

    def _sample_odd_kernel_size(self) -> int:
        """Sample a random odd kernel size from the configured range.

        Sampling is performed over the set of valid odd integers in
        ``[min_kernel_size, max_kernel_size]`` (rather than over all
        integers followed by rounding), so that every valid odd
        kernel size is equally likely.

        Returns
        -------
        int
            A randomly sampled odd kernel size in
            ``[min_kernel_size, max_kernel_size]``.
        """
        odd_step_count = (self._max_kernel_size - self._min_kernel_size) // 2 + 1
        chosen_step = int(self.random_state.integers(0, odd_step_count))
        return self._min_kernel_size + 2 * chosen_step