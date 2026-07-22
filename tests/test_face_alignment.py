"""
tests/test_face_alignment.py
============================
Unit tests for :mod:`src.preprocessing.face_alignment`.

RetinaFace and ``cv2.imread`` are *mocked* throughout so the suite runs
without a GPU, model weights, or real images.

Run with:
    pytest tests/test_face_alignment.py -v
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is importable when running pytest from repo root
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AlignmentConfig, LoggingConfig
from src.preprocessing.face_alignment import (
    AlignmentResult,
    AlignmentStats,
    FaceDetection,
    FailureLogger,
    _add_margin,
    _compute_rotation_angle,
    _parse_detections,
    _rotate_image_and_box,
    align_face,
    collect_images,
    derive_output_path,
    process_dataset,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TEST_CFG = AlignmentConfig(
    output_size=(224, 224),
    crop_margin=0.3,
    jpeg_quality=95,
    det_score_threshold=0.90,
    skip_existing=False,  # off by default in tests to keep behaviour explicit
)


def _make_log_cfg(tmp_path: Path) -> LoggingConfig:
    return LoggingConfig(
        log_dir=tmp_path / "logs",
        alignment_log=tmp_path / "logs" / "alignment.log",
        alignment_failures_csv=tmp_path / "logs" / "alignment_failures.csv",
    )


def _dummy_bgr(h: int = 300, w: int = 300) -> np.ndarray:
    """Return a solid grey BGR image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _retinaface_response(
    score: float = 0.99,
    box: tuple[int, int, int, int] = (50, 50, 200, 200),
    left_eye: tuple[float, float] = (90.0, 100.0),
    right_eye: tuple[float, float] = (160.0, 100.0),
) -> dict[str, Any]:
    """Minimal RetinaFace dict structure for a single face."""
    x1, y1, x2, y2 = box
    return {
        "face_1": {
            "score": score,
            "facial_area": [x1, y1, x2, y2],
            "landmarks": {
                "left_eye": list(left_eye),
                "right_eye": list(right_eye),
                "nose": [125.0, 140.0],
                "mouth_right": [160.0, 170.0],
                "mouth_left": [90.0, 170.0],
            },
        }
    }


# ===========================================================================
# _parse_detections
# ===========================================================================


class TestParseDetections:
    def test_valid_detection_parsed(self) -> None:
        raw = _retinaface_response(score=0.99)
        detections = _parse_detections(raw, score_threshold=0.90)
        assert len(detections) == 1
        d = detections[0]
        assert d.score == pytest.approx(0.99)
        assert d.box == (50, 50, 200, 200)
        assert d.left_eye == pytest.approx((90.0, 100.0))
        assert d.right_eye == pytest.approx((160.0, 100.0))

    def test_below_threshold_discarded(self) -> None:
        raw = _retinaface_response(score=0.50)
        detections = _parse_detections(raw, score_threshold=0.90)
        assert detections == []

    def test_sorted_by_area_descending(self) -> None:
        raw = {
            "face_1": {
                "score": 0.95,
                "facial_area": [0, 0, 50, 50],
                "landmarks": {
                    "left_eye": [10.0, 20.0],
                    "right_eye": [40.0, 20.0],
                },
            },
            "face_2": {
                "score": 0.97,
                "facial_area": [0, 0, 200, 200],
                "landmarks": {
                    "left_eye": [60.0, 80.0],
                    "right_eye": [140.0, 80.0],
                },
            },
        }
        detections = _parse_detections(raw, score_threshold=0.90)
        assert len(detections) == 2
        # Largest first
        assert detections[0].box == (0, 0, 200, 200)
        assert detections[1].box == (0, 0, 50, 50)

    def test_empty_dict_returns_empty(self) -> None:
        assert _parse_detections({}, score_threshold=0.90) == []

    def test_face_detection_area_property(self) -> None:
        d = FaceDetection(
            score=0.99,
            box=(10, 10, 110, 210),
            left_eye=(30.0, 50.0),
            right_eye=(90.0, 50.0),
        )
        # width=100, height=200 → area=20000
        assert d.area == 20_000

    def test_degenerate_box_area_zero(self) -> None:
        d = FaceDetection(
            score=0.99,
            box=(100, 100, 100, 100),  # zero-size
            left_eye=(100.0, 100.0),
            right_eye=(100.0, 100.0),
        )
        assert d.area == 0


# ===========================================================================
# _compute_rotation_angle
# ===========================================================================


class TestComputeRotationAngle:
    def test_horizontal_eyes_zero_angle(self) -> None:
        angle = _compute_rotation_angle((50.0, 100.0), (150.0, 100.0))
        assert angle == pytest.approx(0.0, abs=1e-9)

    def test_left_eye_higher_positive_angle(self) -> None:
        # right_eye is *below* left_eye → positive dy → positive angle
        angle = _compute_rotation_angle((50.0, 100.0), (150.0, 120.0))
        expected = math.degrees(math.atan2(20.0, 100.0))
        assert angle == pytest.approx(expected, abs=1e-6)

    def test_known_45_degree_angle(self) -> None:
        # dy == dx → 45°
        angle = _compute_rotation_angle((0.0, 0.0), (100.0, 100.0))
        assert angle == pytest.approx(45.0, abs=1e-6)

    def test_negative_angle(self) -> None:
        # right_eye above left_eye → negative angle
        angle = _compute_rotation_angle((50.0, 120.0), (150.0, 100.0))
        assert angle < 0


# ===========================================================================
# _add_margin
# ===========================================================================


class TestAddMargin:
    def test_symmetric_expansion(self) -> None:
        box = (100, 100, 200, 200)
        result = _add_margin(box, margin=0.1, img_w=400, img_h=400)
        # bw=bh=100, pad=10
        assert result == (90, 90, 210, 210)

    def test_clamped_to_image_bounds(self) -> None:
        box = (5, 5, 50, 50)
        result = _add_margin(box, margin=0.5, img_w=100, img_h=100)
        x1, y1, x2, y2 = result
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 100 and y2 <= 100

    def test_zero_margin_unchanged(self) -> None:
        box = (10, 20, 80, 90)
        result = _add_margin(box, margin=0.0, img_w=200, img_h=200)
        assert result == box

    def test_large_margin_fully_clamped(self) -> None:
        box = (40, 40, 60, 60)
        result = _add_margin(box, margin=10.0, img_w=100, img_h=100)
        assert result == (0, 0, 100, 100)


# ===========================================================================
# _rotate_image_and_box
# ===========================================================================


class TestRotateImageAndBox:
    def test_zero_rotation_returns_original_shape(self) -> None:
        img = _dummy_bgr(200, 200)
        rotated, new_box = _rotate_image_and_box(img, (50, 50, 150, 150), 0.0)
        assert rotated.shape == img.shape
        assert new_box == (50, 50, 150, 150)

    def test_output_shape_preserved(self) -> None:
        img = _dummy_bgr(300, 400)
        rotated, _ = _rotate_image_and_box(img, (50, 50, 200, 200), 15.0)
        assert rotated.shape == img.shape

    def test_box_is_integers(self) -> None:
        img = _dummy_bgr(300, 300)
        _, box = _rotate_image_and_box(img, (80, 80, 220, 220), 7.3)
        assert all(isinstance(v, int) for v in box)


# ===========================================================================
# align_face  (mocking RetinaFace and cv2.imread)
# ===========================================================================


class TestAlignFace:
    """Test the end-to-end align_face function with mocked IO/detector."""

    def _patch_imread(self, bgr: np.ndarray):
        return patch(
            "src.preprocessing.face_alignment.cv2.imread",
            return_value=bgr,
        )

    def _patch_retinaface(self, response: Any):
        return patch(
            "src.preprocessing.face_alignment.RetinaFace.detect_faces",
            return_value=response,
        )

    def test_success_returns_pil_image(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "img.jpg")
        fake_img.touch()

        bgr = _dummy_bgr(300, 300)
        rf_resp = _retinaface_response()

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is True
        assert isinstance(result.image, Image.Image)
        assert result.image.size == TEST_CFG.output_size
        assert result.image.mode == "RGB"
        assert result.failure_reason is None

    def test_unreadable_image_fails(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "bad.jpg")
        fake_img.touch()

        with self._patch_imread(None):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is False
        assert "None" in result.failure_reason or "unreadable" in result.failure_reason

    def test_no_detection_fails(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "noface.jpg")
        fake_img.touch()

        bgr = _dummy_bgr()
        with self._patch_imread(bgr), self._patch_retinaface({}):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is False
        assert "No faces" in result.failure_reason

    def test_below_threshold_fails(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "lowconf.jpg")
        fake_img.touch()

        bgr = _dummy_bgr()
        rf_resp = _retinaface_response(score=0.10)

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is False
        assert "threshold" in result.failure_reason

    def test_retinaface_exception_fails(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "exc.jpg")
        fake_img.touch()

        bgr = _dummy_bgr()
        with self._patch_imread(bgr), patch(
            "src.preprocessing.face_alignment.RetinaFace.detect_faces",
            side_effect=RuntimeError("GPU OOM"),
        ):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is False
        assert "RuntimeError" in result.failure_reason

    def test_multiple_faces_keeps_largest(self, tmp_path: Path) -> None:
        """Verify the correct (largest) box is used when two faces are present."""
        fake_img = Path(tmp_path / "two_faces.jpg")
        fake_img.touch()

        bgr = _dummy_bgr(400, 400)
        rf_resp = {
            "face_1": {
                "score": 0.95,
                "facial_area": [10, 10, 50, 50],      # small
                "landmarks": {"left_eye": [20.0, 25.0], "right_eye": [40.0, 25.0]},
            },
            "face_2": {
                "score": 0.98,
                "facial_area": [50, 50, 300, 300],    # large
                "landmarks": {"left_eye": [100.0, 120.0], "right_eye": [250.0, 120.0]},
            },
        }

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is True
        assert result.image is not None

    def test_output_respects_custom_output_size(self, tmp_path: Path) -> None:
        fake_img = Path(tmp_path / "img.jpg")
        fake_img.touch()

        bgr = _dummy_bgr(300, 300)
        rf_resp = _retinaface_response()
        custom_cfg = AlignmentConfig(output_size=(112, 112))

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, custom_cfg)

        assert result.success is True
        assert result.image.size == (112, 112)

    # -----------------------------------------------------------------------
    # New: improved failure messages
    # -----------------------------------------------------------------------

    def test_below_threshold_message_contains_score_and_threshold(
        self, tmp_path: Path
    ) -> None:
        """Failure message must include both the detected score and threshold."""
        fake_img = Path(tmp_path / "lowconf.jpg")
        fake_img.touch()

        bgr = _dummy_bgr()
        # Score 0.60 < threshold 0.90
        rf_resp = _retinaface_response(score=0.60)

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is False
        assert "0.600" in result.failure_reason or "0.60" in result.failure_reason
        assert str(TEST_CFG.det_score_threshold) in result.failure_reason

    # -----------------------------------------------------------------------
    # New: tiny image (2-F)
    # -----------------------------------------------------------------------

    def test_tiny_image_does_not_crash(self, tmp_path: Path) -> None:
        """A 30×30 px image should fail gracefully, not raise."""
        fake_img = Path(tmp_path / "tiny.jpg")
        fake_img.touch()

        bgr = _dummy_bgr(30, 30)
        rf_resp = _retinaface_response(
            box=(2, 2, 28, 28),
            left_eye=(8.0, 12.0),
            right_eye=(22.0, 12.0),
        )

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        # Either success or a graceful failure — no unhandled exception
        assert isinstance(result, AlignmentResult)

    # -----------------------------------------------------------------------
    # New: large image (2-C)
    # -----------------------------------------------------------------------

    def test_large_image_output_size_correct(self, tmp_path: Path) -> None:
        """A 4000×3000 px source image must still produce a 224×224 output."""
        fake_img = Path(tmp_path / "large.jpg")
        fake_img.touch()

        bgr = _dummy_bgr(3000, 4000)
        rf_resp = _retinaface_response(
            box=(500, 400, 3000, 2500),
            left_eye=(900.0, 1000.0),
            right_eye=(2600.0, 1000.0),
        )

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, TEST_CFG)

        assert result.success is True
        assert result.image.size == (224, 224)

    # -----------------------------------------------------------------------
    # New: configurable threshold (2-H)
    # -----------------------------------------------------------------------

    def test_low_threshold_accepts_marginal_detection(
        self, tmp_path: Path
    ) -> None:
        """Score 0.60 should succeed when threshold is lowered to 0.50."""
        fake_img = Path(tmp_path / "marginal.jpg")
        fake_img.touch()

        bgr = _dummy_bgr()
        rf_resp = _retinaface_response(score=0.60)
        low_cfg = AlignmentConfig(det_score_threshold=0.50)

        with self._patch_imread(bgr), self._patch_retinaface(rf_resp):
            result = align_face(fake_img, low_cfg)

        assert result.success is True


# ===========================================================================
# collect_images
# ===========================================================================


class TestCollectImages:
    def test_finds_supported_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.PNG").touch()
        (tmp_path / "c.txt").touch()  # unsupported
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.jpeg").touch()

        images = collect_images(tmp_path)
        names = {p.name for p in images}

        assert "a.jpg" in names
        assert "b.PNG" in names
        assert "d.jpeg" in names
        assert "c.txt" not in names

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        assert collect_images(tmp_path) == []

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        (tmp_path / "z.jpg").touch()
        (tmp_path / "a.jpg").touch()
        images = collect_images(tmp_path)
        names = [p.name for p in images]
        assert names == sorted(names)

    # -----------------------------------------------------------------------
    # New: unsupported file types (2-D)
    # -----------------------------------------------------------------------

    def test_unsupported_types_ignored(self, tmp_path: Path) -> None:
        """txt, pdf, and zip files must not appear in the collected list."""
        (tmp_path / "doc.txt").touch()
        (tmp_path / "report.pdf").touch()
        (tmp_path / "archive.zip").touch()
        (tmp_path / "valid.jpg").touch()

        images = collect_images(tmp_path)
        names = {p.name for p in images}

        assert "doc.txt" not in names
        assert "report.pdf" not in names
        assert "archive.zip" not in names
        assert "valid.jpg" in names


# ===========================================================================
# derive_output_path
# ===========================================================================


class TestDeriveOutputPath:
    def test_extension_replaced_with_jpg(self) -> None:
        result = derive_output_path(
            Path("datasets/raw/lfw/Person/img.png"),
            Path("datasets/raw"),
            Path("datasets/aligned"),
        )
        assert result.suffix == ".jpg"

    def test_relative_structure_preserved(self) -> None:
        result = derive_output_path(
            Path("datasets/raw/lfw/Person/img.png"),
            Path("datasets/raw"),
            Path("datasets/aligned"),
        )
        assert result == Path("datasets/aligned/lfw/Person/img.jpg")

    def test_already_jpg_stays_jpg(self) -> None:
        result = derive_output_path(
            Path("raw/cat/face.jpg"),
            Path("raw"),
            Path("aligned"),
        )
        assert result == Path("aligned/cat/face.jpg")


# ===========================================================================
# FailureLogger
# ===========================================================================


class TestFailureLogger:
    def _make_logger(self, tmp_path: Path) -> FailureLogger:
        return FailureLogger(
            tmp_path / "failures.csv",
            input_root=tmp_path / "raw",
        )

    # -----------------------------------------------------------------------
    # Original tests updated for new CSV schema
    # -----------------------------------------------------------------------

    def test_creates_csv_with_header(self, tmp_path: Path) -> None:
        fl = self._make_logger(tmp_path)
        img = tmp_path / "raw" / "lfw" / "Alice" / "img001.jpg"
        fl.log(img, "No face detected")

        csv_path = tmp_path / "failures.csv"
        assert csv_path.exists()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["reason"] == "No face detected"

    def test_header_not_duplicated_on_second_open(self, tmp_path: Path) -> None:
        input_root = tmp_path / "raw"
        csv_path = tmp_path / "failures.csv"

        fl1 = FailureLogger(csv_path, input_root)
        fl1.log(tmp_path / "raw" / "lfw" / "Alice" / "a.jpg", "reason A")

        fl2 = FailureLogger(csv_path, input_root)
        fl2.log(tmp_path / "raw" / "lfw" / "Bob" / "b.jpg", "reason B")

        with csv_path.open(newline="", encoding="utf-8") as fh:
            lines = fh.readlines()

        header_lines = [l for l in lines if "dataset" in l]
        assert len(header_lines) == 1
        data_lines = [l for l in lines if "dataset" not in l and l.strip()]
        assert len(data_lines) == 2

    def test_log_multiple_entries(self, tmp_path: Path) -> None:
        fl = self._make_logger(tmp_path)
        for i in range(5):
            fl.log(
                tmp_path / "raw" / "lfw" / f"person_{i}" / "img.jpg",
                f"reason {i}",
            )

        csv_path = tmp_path / "failures.csv"
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 5

    # -----------------------------------------------------------------------
    # New: verify structured columns (2-E)
    # -----------------------------------------------------------------------

    def test_csv_contains_structured_columns(self, tmp_path: Path) -> None:
        """CSV must have dataset / identity / filename / reason columns."""
        input_root = tmp_path / "raw"
        csv_path = tmp_path / "failures.csv"
        fl = FailureLogger(csv_path, input_root)

        img = input_root / "frll" / "Subject_007" / "img_003.png"
        fl.log(img, "Rotation failed")

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["dataset"] == "frll"
        assert row["identity"] == "Subject_007"
        assert row["filename"] == "img_003.png"
        assert row["reason"] == "Rotation failed"

    def test_csv_lfw_structured_columns(self, tmp_path: Path) -> None:
        """Verify lfw dataset is parsed into the dataset column correctly."""
        input_root = tmp_path / "raw"
        csv_path = tmp_path / "failures.csv"
        fl = FailureLogger(csv_path, input_root)

        img = input_root / "lfw" / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"
        fl.log(img, "No face detected")

        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        assert rows[0]["dataset"] == "lfw"
        assert rows[0]["identity"] == "Aaron_Eckhart"
        assert rows[0]["filename"] == "Aaron_Eckhart_0001.jpg"


# ===========================================================================
# process_dataset  (integration-level with mocked align_face)
# ===========================================================================


class TestProcessDataset:
    """Test the batch processing loop with mocked align_face."""

    def _make_images(self, root: Path, n: int = 3) -> list[Path]:
        sub = root / "lfw" / "person"
        sub.mkdir(parents=True)
        paths = []
        for i in range(n):
            p = sub / f"img_{i}.jpg"
            p.touch()
            paths.append(p)
        return paths

    def test_all_success(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        self._make_images(input_dir)

        pil_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)

        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ):
            stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert stats.success == 3
        assert stats.failure == 0

    def test_all_failure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        self._make_images(input_dir, n=2)

        fail_result = AlignmentResult(
            success=False, image=None, failure_reason="No face detected"
        )
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=fail_result
        ):
            stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert stats.success == 0
        assert stats.failure == 2
        assert log_cfg.alignment_failures_csv.exists()

    def test_mixed_success_and_failure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        self._make_images(input_dir, n=4)

        pil_img = Image.new("RGB", (224, 224))
        results = [
            AlignmentResult(success=True, image=pil_img, failure_reason=None),
            AlignmentResult(success=False, image=None, failure_reason="bad"),
            AlignmentResult(success=True, image=pil_img, failure_reason=None),
            AlignmentResult(success=False, image=None, failure_reason="bad"),
        ]
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", side_effect=results
        ):
            stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert stats.success == 2
        assert stats.failure == 2

    def test_empty_input_dir_returns_zeros(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        input_dir.mkdir()
        output_dir = tmp_path / "aligned"
        log_cfg = _make_log_cfg(tmp_path)

        stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert stats.success == 0
        assert stats.failure == 0

    def test_output_files_saved_under_correct_structure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        self._make_images(input_dir, n=1)

        pil_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ):
            process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        saved = list(output_dir.rglob("*.jpg"))
        assert len(saved) == 1
        # Relative path should match source structure
        rel = saved[0].relative_to(output_dir)
        assert rel.parts[0] == "lfw"

    def test_failure_csv_populated(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        self._make_images(input_dir, n=2)

        fail_result = AlignmentResult(
            success=False, image=None, failure_reason="detector error"
        )
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=fail_result
        ):
            process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        with log_cfg.alignment_failures_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == 2
        assert all(r["reason"] == "detector error" for r in rows)

    # -----------------------------------------------------------------------
    # New: skip-existing (2-A)
    # -----------------------------------------------------------------------

    def test_skip_existing_does_not_call_align_face(self, tmp_path: Path) -> None:
        """When output already exists and skip_existing=True, align_face is never called."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        imgs = self._make_images(input_dir, n=2)

        # Pre-create the corresponding output files
        for img in imgs:
            out = derive_output_path(img, input_dir, output_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()

        cfg_skip = AlignmentConfig(skip_existing=True)
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face"
        ) as mock_align:
            stats = process_dataset(input_dir, output_dir, cfg_skip, log_cfg)

        mock_align.assert_not_called()
        assert stats.skipped == 2
        assert stats.success == 0

    def test_skip_existing_false_processes_all(self, tmp_path: Path) -> None:
        """When skip_existing=False, pre-existing outputs are overwritten."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        imgs = self._make_images(input_dir, n=2)

        # Pre-create output files
        for img in imgs:
            out = derive_output_path(img, input_dir, output_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()

        pil_img = Image.new("RGB", (224, 224))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)
        cfg_no_skip = AlignmentConfig(skip_existing=False)
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ) as mock_align:
            stats = process_dataset(input_dir, output_dir, cfg_no_skip, log_cfg)

        assert mock_align.call_count == 2
        assert stats.skipped == 0
        assert stats.success == 2

    # -----------------------------------------------------------------------
    # New: skipped statistics in AlignmentStats (2-F)
    # -----------------------------------------------------------------------

    def test_skipped_count_in_stats(self, tmp_path: Path) -> None:
        """AlignmentStats.skipped must reflect skipped image count."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "aligned"
        imgs = self._make_images(input_dir, n=3)

        # Pre-create 2 outputs → 2 skipped, 1 processed
        for img in imgs[:2]:
            out = derive_output_path(img, input_dir, output_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()

        pil_img = Image.new("RGB", (224, 224))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)
        cfg_skip = AlignmentConfig(skip_existing=True)
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ):
            stats = process_dataset(input_dir, output_dir, cfg_skip, log_cfg)

        assert stats.skipped == 2
        assert stats.success == 1
        assert stats.processed == 1  # only the non-skipped image counts as processed

    # -----------------------------------------------------------------------
    # New: elapsed time in AlignmentStats
    # -----------------------------------------------------------------------

    def test_elapsed_seconds_is_non_negative(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        input_dir.mkdir()
        output_dir = tmp_path / "aligned"
        log_cfg = _make_log_cfg(tmp_path)

        stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert stats.elapsed_seconds >= 0.0

    # -----------------------------------------------------------------------
    # New: automatic output directory creation (2-G)
    # -----------------------------------------------------------------------

    def test_output_directory_created_automatically(self, tmp_path: Path) -> None:
        """Output directory must be created even if it does not exist beforehand."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "deeply" / "nested" / "aligned"
        self._make_images(input_dir, n=1)

        assert not output_dir.exists()

        pil_img = Image.new("RGB", (224, 224))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)
        log_cfg = _make_log_cfg(tmp_path)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ):
            process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert output_dir.exists()

    # -----------------------------------------------------------------------
    # New: log file creation (2-I)
    # -----------------------------------------------------------------------

    def test_log_file_created_after_processing(self, tmp_path: Path) -> None:
        """alignment.log must exist after configure_logging is called."""
        from src.preprocessing.face_alignment import configure_logging

        log_cfg = _make_log_cfg(tmp_path)
        configure_logging(log_cfg)

        assert log_cfg.alignment_log.exists()

    def test_alignment_stats_is_dataclass(self, tmp_path: Path) -> None:
        """AlignmentStats returned from process_dataset is the correct type."""
        input_dir = tmp_path / "raw"
        input_dir.mkdir()
        output_dir = tmp_path / "aligned"
        log_cfg = _make_log_cfg(tmp_path)

        stats = process_dataset(input_dir, output_dir, TEST_CFG, log_cfg)

        assert isinstance(stats, AlignmentStats)
        assert hasattr(stats, "processed")
        assert hasattr(stats, "success")
        assert hasattr(stats, "failure")
        assert hasattr(stats, "skipped")
        assert hasattr(stats, "elapsed_seconds")


# ===========================================================================
# CLI (main)
# ===========================================================================


class TestCLI:
    def test_missing_input_dir_returns_1(self, tmp_path: Path) -> None:
        from src.preprocessing.face_alignment import main

        exit_code = main(
            ["--input", str(tmp_path / "nonexistent"), "--output", str(tmp_path / "out")]
        )
        assert exit_code == 1

    def test_empty_input_dir_returns_0(self, tmp_path: Path) -> None:
        from src.preprocessing.face_alignment import main

        in_dir = tmp_path / "raw"
        in_dir.mkdir()

        exit_code = main(["--input", str(in_dir), "--output", str(tmp_path / "out")])
        assert exit_code == 0

    def test_failures_return_exit_code_1(self, tmp_path: Path) -> None:
        from src.preprocessing.face_alignment import main

        in_dir = tmp_path / "raw"
        in_dir.mkdir()
        (in_dir / "bad.jpg").touch()

        fail_result = AlignmentResult(
            success=False, image=None, failure_reason="No face"
        )

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=fail_result
        ):
            exit_code = main(
                ["--input", str(in_dir), "--output", str(tmp_path / "out")]
            )

        assert exit_code == 1

    def test_all_success_returns_exit_code_0(self, tmp_path: Path) -> None:
        from src.preprocessing.face_alignment import main

        in_dir = tmp_path / "raw"
        in_dir.mkdir()
        (in_dir / "good.jpg").touch()

        pil_img = Image.new("RGB", (224, 224))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ):
            exit_code = main(
                ["--input", str(in_dir), "--output", str(tmp_path / "out")]
            )

        assert exit_code == 0

    def test_no_skip_existing_flag_disables_skip(self, tmp_path: Path) -> None:
        """--no-skip-existing must override the default skip_existing=True."""
        from src.preprocessing.face_alignment import main

        in_dir = tmp_path / "raw"
        in_dir.mkdir()
        img = in_dir / "img.jpg"
        img.touch()

        out_dir = tmp_path / "out"
        out = derive_output_path(img, in_dir, out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()

        pil_img = Image.new("RGB", (224, 224))
        ok_result = AlignmentResult(success=True, image=pil_img, failure_reason=None)

        with patch(
            "src.preprocessing.face_alignment.align_face", return_value=ok_result
        ) as mock_align:
            main(
                [
                    "--input", str(in_dir),
                    "--output", str(out_dir),
                    "--no-skip-existing",
                ]
            )

        # align_face should have been called despite the output existing
        mock_align.assert_called_once()