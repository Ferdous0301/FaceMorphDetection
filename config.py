"""
config.py
=========
Central configuration for the FaceMorphDetection project.

All tuneable parameters live here so that no magic numbers appear
anywhere else in the codebase.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Project root (one level above *this* file's parent directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DirectoryConfig:
    """Canonical paths used across the project."""

    project_root: Path = PROJECT_ROOT
    datasets_root: Path = PROJECT_ROOT / "datasets"
    raw_root: Path = PROJECT_ROOT / "datasets" / "raw"
    aligned_root: Path = PROJECT_ROOT / "datasets" / "aligned"
    outputs_root: Path = PROJECT_ROOT / "outputs"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"


DIRS = DirectoryConfig()


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlignmentConfig:
    """Parameters that control the face-alignment pipeline."""

    # Output spatial size (width, height) in pixels
    output_size: tuple[int, int] = (224, 224)

    # Fractional margin added around the bounding box on each side.
    # 0.3  →  30 % of the box's dimension is added as padding.
    crop_margin: float = 0.3

    # JPEG save quality (1–95)
    jpeg_quality: int = 95

    # Minimum RetinaFace detection confidence to accept a face.
    # Lowered to 0.80 to improve recall on difficult LFW / FRLL images.
    det_score_threshold: float = 0.80

    # OpenCV interpolation flag used when resizing the aligned crop.
    resize_interpolation: int = cv2.INTER_AREA

    # When True, images whose output path already exists are skipped so
    # interrupted preprocessing runs can resume without reprocessing.
    skip_existing: bool = True

    # Supported input image extensions shared by all preprocessing modules.
    supported_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

    # RetinaFace internal resize scale passed to `detect_faces`
    # (None lets the library choose automatically)
    retinaface_resize: float | None = None

    # Global random seed shared by preprocessing and morph-generation modules.
    random_seed: int = 42


ALIGNMENT = AlignmentConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related paths and format strings."""

    log_dir: Path = DIRS.logs_dir
    alignment_log: Path = DIRS.logs_dir / "alignment.log"
    alignment_failures_csv: Path = DIRS.logs_dir / "alignment_failures.csv"

    log_format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format: str = "%Y-%m-%d %H:%M:%S"


LOGGING = LoggingConfig()