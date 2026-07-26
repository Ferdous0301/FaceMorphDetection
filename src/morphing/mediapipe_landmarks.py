"""
src/morphing/mediapipe_landmarks.py
=====================================

MediaPipe Face Landmarker (Tasks API) landmark detection module.

This module wraps Google's MediaPipe Face Landmarker task so that the rest of
the morphing pipeline never interacts with the MediaPipe API directly.
A single ``LandmarkDetector`` instance is created once and reused across all
images, which avoids the overhead of re-initialising the underlying model on
every call.

NOTE: This module was migrated from the legacy ``mp.solutions.face_mesh`` API
to the modern ``mp.tasks.vision.FaceLandmarker`` API, because mediapipe
removed ``mp.solutions`` in newer releases. The Tasks API requires a
downloaded ``.task`` model file (see MODEL_PATH / detector construction).

Design decisions
----------------
* MediaPipe is imported inside ``_build_face_mesh`` rather than at module
  level, so importing this module is cheap and tests can patch the method
  without installing MediaPipe.
* All coordinates are returned in **pixel space** (float32 ``(x, y)`` pairs)
  rather than MediaPipe's normalised ``[0, 1]`` range, matching the contract
  expected by ``delaunay.py``.
* Failures are logged and surfaced as ``None`` returns rather than exceptions
  so that ``morph_generator.py`` can continue processing other pairs.
* The detector is intentionally NOT a global singleton; callers own the
  lifecycle and can control resource cleanup via the context-manager interface.

Public API
----------
LandmarkDetector
    Class-based API with explicit lifecycle management.
detect_landmarks(image, detector=None)
    Convenience function for callers that do not want to manage a detector.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Number of landmarks produced by MediaPipe Face Landmarker (same topology
#: as the legacy Face Mesh model).
NUM_LANDMARKS: Final[int] = 468

#: Expected number of colour channels in a valid input image.
_EXPECTED_CHANNELS: Final[int] = 3

#: Valid range for MediaPipe confidence thresholds.
_CONFIDENCE_MIN: Final[float] = 0.0
_CONFIDENCE_MAX: Final[float] = 1.0

#: Default path to the downloaded FaceLandmarker .task model file.
#: Override via the LandmarkDetector(model_path=...) argument or the
#: MEDIAPIPE_FACE_LANDMARKER_MODEL environment variable.
DEFAULT_MODEL_PATH: Final[str] = os.environ.get(
    "MEDIAPIPE_FACE_LANDMARKER_MODEL",
    "/kaggle/working/face_landmarker.task",
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LandmarkResult:
    """Detected face landmarks for a single image.

    Attributes
    ----------
    points : NDArray[np.float32]
        Array of shape ``(NUM_LANDMARKS, 2)`` containing ``(x, y)`` pixel
        coordinates.  All values are in image-pixel space (not normalised).
    image_shape : tuple[int, int]
        ``(height, width)`` of the source image, stored for downstream
        validation and debugging.

    Raises
    ------
    ValueError
        If ``points`` does not have shape ``(N, 2)`` for any positive ``N``.
    """

    points: NDArray[np.float32]
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2), got {self.points.shape}."
            )


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class LandmarkDetector:
    """MediaPipe Face Landmarker wrapper with single-initialisation lifecycle.

    One instance should be created per processing run and shared across all
    ``detect`` calls.  Use it as a context manager to ensure resources are
    released cleanly::

        with LandmarkDetector() as detector:
            for image in images:
                result = detector.detect(image)

    Parameters
    ----------
    max_num_faces : int
        Maximum number of faces MediaPipe will attempt to detect per image.
        For morphing, this must be 1 to avoid ambiguity.  Must be ≥ 1.
    refine_landmarks : bool
        When ``True``, requests blend-shape / attention-mesh refinement
        (iris etc.) from the Tasks API. Defaults to ``False``; the additional
        model increases CPU time without benefiting the Delaunay morphing
        pipeline.
    min_detection_confidence : float
        Minimum confidence score in ``[0, 1]`` for the face-detection stage.
    min_tracking_confidence : float
        Minimum confidence score in ``[0, 1]`` for the landmark-tracking
        stage. (Kept for backward-compatible signature; passed through to
        the Tasks API's min_tracking_confidence equivalent.)
    model_path : str, optional
        Path to the downloaded FaceLandmarker ``.task`` model file. Defaults
        to ``DEFAULT_MODEL_PATH``.

    Raises
    ------
    ValueError
        If ``max_num_faces < 1`` or either confidence value is outside
        ``[0, 1]``.
    ImportError
        If the ``mediapipe`` package is not installed.
    FileNotFoundError
        If the ``.task`` model file cannot be found at ``model_path``.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        self._validate_init_args(
            max_num_faces, min_detection_confidence, min_tracking_confidence
        )
        self._max_num_faces = max_num_faces
        self._refine_landmarks = refine_landmarks
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_path = model_path
        self._face_mesh = self._build_face_mesh()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_init_args(
        max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """Validate constructor arguments and raise ``ValueError`` if invalid."""
        if max_num_faces < 1:
            raise ValueError(
                f"max_num_faces must be ≥ 1, got {max_num_faces}."
            )
        for name, val in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not (_CONFIDENCE_MIN <= val <= _CONFIDENCE_MAX):
                raise ValueError(
                    f"{name} must be in [{_CONFIDENCE_MIN}, {_CONFIDENCE_MAX}], "
                    f"got {val}."
                )

    def _build_face_mesh(self):
        """Initialise and return a ``mediapipe.tasks.python.vision.FaceLandmarker``.

        Separated from ``__init__`` so that tests can patch this method
        without mocking the entire constructor.

        Returns
        -------
        mediapipe.tasks.python.vision.FaceLandmarker

        Raises
        ------
        ImportError
            If ``mediapipe`` is not installed.
        FileNotFoundError
            If the ``.task`` model file is missing.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for landmark detection. "
                "Install it with:  pip install mediapipe"
            ) from exc

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at '{self._model_path}'. "
                "Download it with:\n"
                "  wget -O face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=self._model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=self._max_num_faces,
            min_face_detection_confidence=self._min_detection_confidence,
            min_face_presence_confidence=self._min_tracking_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp  # keep a reference for Image construction in detect()
        return mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image: NDArray[np.uint8]) -> Optional[LandmarkResult]:
        """Detect face-mesh landmarks in ``image``.

        Parameters
        ----------
        image : NDArray[np.uint8]
            BGR image as returned by ``cv2.imread`` or the face-alignment
            module.  Must have dtype ``uint8`` and shape ``(H, W, 3)``.

        Returns
        -------
        LandmarkResult or None
            ``None`` is returned (and a warning is logged) when:

            * ``image`` is ``None`` or has an unexpected shape.
            * MediaPipe does not detect any face in the image.
            * MediaPipe returns an unexpected number of landmarks.
            * Any runtime error occurs during processing.
        """
        if image is None:
            logger.warning("detect() received None; skipping.")
            return None

        if image.ndim != 3 or image.shape[2] != _EXPECTED_CHANNELS:
            logger.warning(
                "detect() expects a 3-channel BGR image, got shape %s; skipping.",
                image.shape,
            )
            return None

        h, w = image.shape[:2]

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb
            )
            results = self._face_mesh.detect(mp_image)
        except Exception:
            logger.exception(
                "MediaPipe processing failed for image of shape (%d, %d, %d).",
                h, w, image.shape[2],
            )
            return None

        if not results.face_landmarks:
            logger.warning(
                "No face detected in %d×%d image. "
                "Ensure the image is aligned and the face is clearly visible.",
                w, h,
            )
            return None

        raw = results.face_landmarks[0]
        n_detected = len(raw)

        if n_detected < NUM_LANDMARKS:
            logger.warning(
                "Expected at least %d landmarks, got %d in %d×%d image; skipping.",
                NUM_LANDMARKS, n_detected, w, h,
            )
            return None
        raw = raw[:NUM_LANDMARKS]
        # Convert normalised [0, 1] coordinates to pixel space in one
        # vectorised operation to avoid a Python-level loop.
        scale = np.array([w, h], dtype=np.float32)
        points: NDArray[np.float32] = (
            np.array([[lm.x, lm.y] for lm in raw], dtype=np.float32) * scale
        )

        logger.debug(
            "Detected %d landmarks in %d×%d image.", NUM_LANDMARKS, w, h
        )
        return LandmarkResult(points=points, image_shape=(h, w))

    def close(self) -> None:
        """Release MediaPipe resources.

        Safe to call multiple times; subsequent calls after the first are
        no-ops.
        """
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.debug("LandmarkDetector closed.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> LandmarkDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_landmarks(
    image: NDArray[np.uint8],
    detector: Optional[LandmarkDetector] = None,
) -> Optional[LandmarkResult]:
    """Detect face landmarks, optionally reusing an existing detector.

    This is a convenience wrapper for one-shot callers that do not want to
    manage a ``LandmarkDetector`` lifecycle.  When ``detector`` is ``None``,
    a fresh detector is created, used once, and immediately closed.

    For batch processing, always prefer creating a single
    ``LandmarkDetector`` and calling ``detector.detect(image)`` in a loop;
    that avoids the MediaPipe model initialisation overhead on every call.

    Parameters
    ----------
    image : NDArray[np.uint8]
        BGR image, dtype ``uint8``, shape ``(H, W, 3)``.
    detector : LandmarkDetector, optional
        Existing detector to reuse.  When provided, its lifecycle is **not**
        managed here; the caller remains responsible for closing it.

    Returns
    -------
    LandmarkResult or None
        Detected landmarks, or ``None`` on failure.

    Examples
    --------
    >>> result = detect_landmarks(image)
    >>> if result is not None:
    ...     print(result.points.shape)  # (468, 2)
    """
    if detector is not None:
        return detector.detect(image)

    with LandmarkDetector() as det:
        return det.detect(image)
