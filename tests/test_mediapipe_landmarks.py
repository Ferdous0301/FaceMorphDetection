"""
tests/test_mediapipe_landmarks.py
===================================

Tests for ``src/morphing/mediapipe_landmarks.py``.

MediaPipe is mocked throughout so that tests run without the package being
installed and without any GPU or real images.

Coverage
--------
* NUM_LANDMARKS constant
* LandmarkResult – valid construction, invalid shapes
* LandmarkDetector.__init__ – calls _build_face_mesh once; validates args
* LandmarkDetector.detect   – happy path, None image, wrong channel count,
                              no face, wrong landmark count, runtime exception
* LandmarkDetector lifecycle – close, idempotent close, context manager
* detect_landmarks           – with and without an explicit detector
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.morphing.mediapipe_landmarks import (
    NUM_LANDMARKS,
    LandmarkDetector,
    LandmarkResult,
    detect_landmarks,
)


# ---------------------------------------------------------------------------
# Test-level helpers
# ---------------------------------------------------------------------------

def _bgr_image(h: int = 120, w: int = 120) -> np.ndarray:
    """Return a blank uint8 BGR image of the requested size."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fake_face_mesh(num_landmarks: int = NUM_LANDMARKS) -> MagicMock:
    """Return a mock FaceMesh that returns ``num_landmarks`` landmarks.

    Each landmark has x, y, z attributes.  x and y are spread across a small
    range so that pixel-space conversion produces values > 1.
    """
    landmarks = [MagicMock(x=0.5 + i * 1e-4, y=0.5 + i * 1e-4, z=0.0)
                 for i in range(num_landmarks)]

    face = MagicMock()
    face.landmark = landmarks

    result = MagicMock()
    result.multi_face_landmarks = [face]

    mesh = MagicMock()
    mesh.process.return_value = result
    return mesh


def _no_face_mesh() -> MagicMock:
    """Return a mock FaceMesh that detects no faces."""
    result = MagicMock()
    result.multi_face_landmarks = None

    mesh = MagicMock()
    mesh.process.return_value = result
    return mesh


def _detector(mesh: MagicMock) -> LandmarkDetector:
    """Return a ``LandmarkDetector`` with ``_face_mesh`` replaced by ``mesh``."""
    det = LandmarkDetector.__new__(LandmarkDetector)
    det._max_num_faces = 1
    det._refine_landmarks = False
    det._min_detection_confidence = 0.5
    det._min_tracking_confidence = 0.5
    det._face_mesh = mesh
    return det


def _patched_build(num_landmarks: int = NUM_LANDMARKS):
    """Context manager that patches ``_build_face_mesh`` for detector construction."""
    return patch.object(
        LandmarkDetector,
        "_build_face_mesh",
        return_value=_fake_face_mesh(num_landmarks),
    )


# ---------------------------------------------------------------------------
# NUM_LANDMARKS constant
# ---------------------------------------------------------------------------

class TestNumLandmarksConstant:
    def test_value_is_468(self) -> None:
        assert NUM_LANDMARKS == 468


# ---------------------------------------------------------------------------
# LandmarkResult
# ---------------------------------------------------------------------------

class TestLandmarkResult:
    def test_valid_construction(self) -> None:
        pts = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        result = LandmarkResult(points=pts, image_shape=(120, 120))
        assert result.points.shape == (NUM_LANDMARKS, 2)
        assert result.image_shape == (120, 120)

    @pytest.mark.parametrize("bad_shape", [
        (NUM_LANDMARKS, 3),   # extra column
        (NUM_LANDMARKS, 1),   # too few columns
        (NUM_LANDMARKS,),     # 1-D
    ])
    def test_invalid_points_shape_raises(self, bad_shape: tuple[int, ...]) -> None:
        pts = np.zeros(bad_shape, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            LandmarkResult(points=pts, image_shape=(120, 120))


# ---------------------------------------------------------------------------
# LandmarkDetector – constructor argument validation
# ---------------------------------------------------------------------------

class TestLandmarkDetectorValidation:
    @pytest.mark.parametrize("max_num_faces", [0, -1])
    def test_max_num_faces_less_than_1_raises(self, max_num_faces: int) -> None:
        with _patched_build():
            with pytest.raises(ValueError, match="max_num_faces"):
                LandmarkDetector(max_num_faces=max_num_faces)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_invalid_detection_confidence_raises(self, confidence: float) -> None:
        with _patched_build():
            with pytest.raises(ValueError, match="min_detection_confidence"):
                LandmarkDetector(min_detection_confidence=confidence)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_invalid_tracking_confidence_raises(self, confidence: float) -> None:
        with _patched_build():
            with pytest.raises(ValueError, match="min_tracking_confidence"):
                LandmarkDetector(min_tracking_confidence=confidence)


# ---------------------------------------------------------------------------
# LandmarkDetector – initialisation
# ---------------------------------------------------------------------------

class TestLandmarkDetectorInit:
    def test_build_face_mesh_called_exactly_once(self) -> None:
        with _patched_build() as mock_build:
            det = LandmarkDetector()
            mock_build.assert_called_once()
            det.close()

    def test_missing_mediapipe_raises_import_error(self) -> None:
        import builtins
        real_import = builtins.__import__

        def _block_mediapipe(name: str, *args: object, **kwargs: object):
            if name == "mediapipe":
                raise ImportError("no mediapipe")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_mediapipe):
            with pytest.raises(ImportError, match="mediapipe"):
                LandmarkDetector()


# ---------------------------------------------------------------------------
# LandmarkDetector – detect()
# ---------------------------------------------------------------------------

class TestLandmarkDetectorDetect:

    # Happy path -----------------------------------------------------------

    def test_returns_landmark_result_on_success(self) -> None:
        det = _detector(_fake_face_mesh())
        result = det.detect(_bgr_image(120, 120))
        assert result is not None
        assert isinstance(result, LandmarkResult)
        assert result.points.shape == (NUM_LANDMARKS, 2)

    def test_coordinates_scaled_to_pixel_space(self) -> None:
        """x-coordinates must be multiplied by image width (not stay in [0,1])."""
        det = _detector(_fake_face_mesh())
        result = det.detect(_bgr_image(120, 200))  # w=200
        assert result is not None
        assert result.points[:, 0].max() > 1.0
        assert result.image_shape == (120, 200)

    # Invalid inputs -------------------------------------------------------

    def test_none_image_returns_none(self) -> None:
        det = _detector(_fake_face_mesh())
        assert det.detect(None) is None

    @pytest.mark.parametrize("shape", [
        (120, 120),         # grayscale (2-D)
        (120, 120, 1),      # single-channel
        (120, 120, 4),      # BGRA
    ])
    def test_wrong_channel_count_returns_none(self, shape: tuple) -> None:
        det = _detector(_fake_face_mesh())
        assert det.detect(np.zeros(shape, dtype=np.uint8)) is None

    # Face detection failures ----------------------------------------------

    def test_no_face_detected_returns_none(self) -> None:
        det = _detector(_no_face_mesh())
        assert det.detect(_bgr_image()) is None

    @pytest.mark.parametrize("n", [1, 100, 467, 469])
    def test_wrong_landmark_count_returns_none(self, n: int) -> None:
        det = _detector(_fake_face_mesh(num_landmarks=n))
        assert det.detect(_bgr_image()) is None

    def test_mediapipe_runtime_exception_returns_none(self) -> None:
        mesh = MagicMock()
        mesh.process.side_effect = RuntimeError("model crashed")
        det = _detector(mesh)
        assert det.detect(_bgr_image()) is None


# ---------------------------------------------------------------------------
# LandmarkDetector – lifecycle
# ---------------------------------------------------------------------------

class TestLandmarkDetectorLifecycle:
    def _make_detector(self) -> LandmarkDetector:
        with _patched_build():
            return LandmarkDetector()

    def test_close_releases_face_mesh(self) -> None:
        det = self._make_detector()
        det.close()
        assert det._face_mesh is None

    def test_close_is_idempotent(self) -> None:
        det = self._make_detector()
        det.close()
        det.close()  # must not raise

    def test_context_manager_closes_on_exit(self) -> None:
        with _patched_build():
            with LandmarkDetector() as det:
                assert det._face_mesh is not None
        assert det._face_mesh is None

    def test_context_manager_returns_detector_instance(self) -> None:
        with _patched_build():
            with LandmarkDetector() as det:
                assert isinstance(det, LandmarkDetector)


# ---------------------------------------------------------------------------
# detect_landmarks convenience function
# ---------------------------------------------------------------------------

class TestDetectLandmarks:

    def test_creates_and_closes_detector_when_none_given(self) -> None:
        with _patched_build():
            result = detect_landmarks(_bgr_image())
        assert result is not None
        assert result.points.shape == (NUM_LANDMARKS, 2)

    def test_delegates_to_provided_detector(self) -> None:
        """When an existing detector is supplied, its detect() is called."""
        det = _detector(_fake_face_mesh())
        image = _bgr_image()
        with patch.object(det, "detect", wraps=det.detect) as mock_detect:
            detect_landmarks(image, detector=det)
        mock_detect.assert_called_once_with(image)

    def test_none_image_returns_none_without_raising(self) -> None:
        with _patched_build():
            assert detect_landmarks(None) is None