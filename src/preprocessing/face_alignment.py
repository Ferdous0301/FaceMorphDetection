"""
face_alignment.py
=================
Align every face image under ``datasets/raw/`` and write the results to
``datasets/aligned/``, preserving the original directory structure.

Pipeline (per image)
--------------------
1.  Detect faces with RetinaFace.
2.  If multiple detections, keep the one with the largest bounding box.
3.  Extract left-eye / right-eye landmark centroids.
4.  Rotate the image so the inter-ocular axis is horizontal.
5.  Crop with a configurable margin around the bounding box.
6.  Resize to ``AlignmentConfig.output_size`` (default 224 × 224).
7.  Convert to RGB.
8.  Save as JPEG at ``AlignmentConfig.jpeg_quality`` (default 95).

Failures (unreadable image, no detection, alignment error) are logged and
appended to ``outputs/logs/alignment_failures.csv``; processing continues.

Usage
-----
    python -m src.preprocessing.face_alignment \\
        --input  datasets/raw \\
        --output datasets/aligned
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import result

import src.augmentation.config as config
import cv2
import numpy as np
from PIL import Image
from retinaface import RetinaFace
 # type: ignore[import]
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Support both `python face_alignment.py` and `python -m src.preprocessing…`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ALIGNMENT, LOGGING, AlignmentConfig, LoggingConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def configure_logging(log_cfg: LoggingConfig = LOGGING) -> None:
    """Configure root logger to write to both *stderr* and the alignment log.

    Parameters
    ----------
    log_cfg:
        Logging paths and format strings from :class:`~src.config.LoggingConfig`.
    """
    log_cfg.log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=log_cfg.log_format, datefmt=log_cfg.date_format
    )

    # File handler
    fh = logging.FileHandler(log_cfg.alignment_log, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers when the module is re-imported in tests
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(ch)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FaceDetection:
    """Structured representation of a single RetinaFace detection.

    Attributes
    ----------
    score:
        Confidence score in ``[0, 1]``.
    box:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    left_eye:
        ``(x, y)`` centroid of the left-eye landmark.
    right_eye:
        ``(x, y)`` centroid of the right-eye landmark.
    area:
        Pixel area of the bounding box (derived).
    """

    score: float
    box: tuple[int, int, int, int]
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]

    @property
    def area(self) -> int:
        """Return bounding-box area in pixels."""
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class AlignmentResult:
    """Outcome returned by :func:`align_face`.

    Attributes
    ----------
    success:
        ``True`` when alignment produced a valid image.
    image:
        The aligned ``PIL.Image`` (``None`` on failure).
    failure_reason:
        Human-readable explanation of failure (``None`` on success).
    """

    success: bool
    image: Optional[Image.Image]
    failure_reason: Optional[str]


@dataclass
class AlignmentStats:
    """Aggregated statistics returned by :func:`process_dataset`.

    Attributes
    ----------
    processed:
        Total number of images encountered (excluding skipped).
    success:
        Images successfully aligned and saved.
    failure:
        Images that failed alignment or saving.
    skipped:
        Images skipped because their output already existed and
        ``AlignmentConfig.skip_existing`` was ``True``.
    elapsed_seconds:
        Wall-clock time in seconds for the entire batch.
    """

    processed: int
    success: int
    failure: int
    skipped: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Core alignment helpers
# ---------------------------------------------------------------------------


def _parse_detections(
    raw: dict,
    score_threshold: float,
) -> list[FaceDetection]:
    """Convert the raw RetinaFace dict into a list of :class:`FaceDetection`.

    Parameters
    ----------
    raw:
        Dict returned by ``RetinaFace.detect_faces``.  Keys are face labels
        (e.g. ``"face_1"``); values contain ``"score"``, ``"facial_area"``,
        and ``"landmarks"``.
    score_threshold:
        Detections with ``score < score_threshold`` are discarded.

    Returns
    -------
    list[FaceDetection]
        Parsed detections that exceed *score_threshold*, sorted by area
        descending so index-0 is always the largest face.
    """
    detections: list[FaceDetection] = []

    for face_key, face_data in raw.items():
        score: float = float(face_data.get("score", 0.0))
        if score < score_threshold:
            logger.debug(
                "Skipping detection '%s' (score=%.3f < threshold=%.3f)",
                face_key,
                score,
                score_threshold,
            )
            continue

        facial_area: list[int] = face_data["facial_area"]
        x1, y1, x2, y2 = (
            int(facial_area[0]),
            int(facial_area[1]),
            int(facial_area[2]),
            int(facial_area[3]),
        )

        landmarks: dict = face_data.get("landmarks", {})
        left_eye: tuple[float, float] = tuple(landmarks.get("left_eye", (0.0, 0.0)))  # type: ignore[assignment]
        right_eye: tuple[float, float] = tuple(landmarks.get("right_eye", (0.0, 0.0)))  # type: ignore[assignment]

        detections.append(
            FaceDetection(
                score=score,
                box=(x1, y1, x2, y2),
                left_eye=left_eye,
                right_eye=right_eye,
            )
        )

    detections.sort(key=lambda d: d.area, reverse=True)
    return detections


def _compute_rotation_angle(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> float:
    """Return the angle (degrees) needed to rotate *right_eye* level with *left_eye*.

    Parameters
    ----------
    left_eye:
        ``(x, y)`` of the left-eye centroid.
    right_eye:
        ``(x, y)`` of the right-eye centroid.

    Returns
    -------
    float
        Rotation angle in degrees; positive = counter-clockwise.
    """
    dx: float = right_eye[0] - left_eye[0]
    dy: float = right_eye[1] - left_eye[1]
    angle_rad: float = math.atan2(dy, dx)
    angle_deg: float = math.degrees(angle_rad)
    return angle_deg


def _rotate_image_and_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    angle: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rotate *image* by *angle* degrees about its centre and transform *box*.

    Parameters
    ----------
    image:
        BGR image as a NumPy array.
    box:
        Bounding box ``(x1, y1, x2, y2)`` before rotation.
    angle:
        Counter-clockwise rotation angle in degrees.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int]]
        Rotated image and the transformed bounding box.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Transform the four corners of the original bounding box
    x1, y1, x2, y2 = box
    corners = np.array(
        [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64
    )
    transformed = (M @ corners.T).T  # shape (4, 2)

    new_x1 = int(np.floor(transformed[:, 0].min()))
    new_y1 = int(np.floor(transformed[:, 1].min()))
    new_x2 = int(np.ceil(transformed[:, 0].max()))
    new_y2 = int(np.ceil(transformed[:, 1].max()))

    return rotated, (new_x1, new_y1, new_x2, new_y2)


def _add_margin(
    box: tuple[int, int, int, int],
    margin: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand *box* by *margin* (fraction of box size) and clamp to image bounds.

    Parameters
    ----------
    box:
        Original bounding box ``(x1, y1, x2, y2)``.
    margin:
        Fractional padding, e.g. ``0.3`` adds 30 % of the box width/height.
    img_w:
        Image width in pixels (used for clamping).
    img_h:
        Image height in pixels (used for clamping).

    Returns
    -------
    tuple[int, int, int, int]
        Expanded and clamped bounding box.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Public alignment function
# ---------------------------------------------------------------------------


def align_face(
    image_path: Path,
    cfg: AlignmentConfig = ALIGNMENT,
) -> AlignmentResult:
    """Detect, align, crop, and resize the dominant face in *image_path*.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the source image.
    cfg:
        Alignment configuration (output size, margin, quality, …).

    Returns
    -------
    AlignmentResult
        ``success=True`` and a ``PIL.Image`` on success; ``success=False``
        and a ``failure_reason`` string on any error.
    """
    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason="cv2.imread returned None – unreadable or missing file",
        )

    logger.debug("Loaded image '%s' shape=%s", image_path.name, bgr.shape)

    # ------------------------------------------------------------------
    # 2. Detect faces with RetinaFace
    # ------------------------------------------------------------------
    try:
        raw_detections = RetinaFace.detect_faces(str(image_path))
    except Exception as exc:  # noqa: BLE001
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=f"RetinaFace.detect_faces raised {type(exc).__name__}: {exc}",
        )

    if not raw_detections or not isinstance(raw_detections, dict):
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason="No faces detected by RetinaFace",
        )

    detections = _parse_detections(raw_detections, cfg.det_score_threshold)
    if not detections:
        # Include the highest observed score so callers can tune the threshold
        best_score = max(
            float(v.get("score", 0.0)) for v in raw_detections.values()
        )
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=(
                f"All detections below score threshold "
                f"(best={best_score:.3f}, threshold={cfg.det_score_threshold:.3f})"
            ),
        )

    # ------------------------------------------------------------------
    # 3. Keep the largest face
    # ------------------------------------------------------------------
    face = detections[0]
    if len(detections) > 1:
        logger.debug(
            "Multiple faces (%d) detected in '%s'; keeping largest (area=%d px²)",
            len(detections),
            image_path.name,
            face.area,
        )

    # ------------------------------------------------------------------
    # 4. Compute rotation angle from eye landmarks
    # ------------------------------------------------------------------
    try:
        angle = _compute_rotation_angle(face.left_eye, face.right_eye)
    except Exception as exc:  # noqa: BLE001
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=f"Angle computation failed: {exc}",
        )

    logger.debug("Rotation angle for '%s': %.2f°", image_path.name, angle)

    # ------------------------------------------------------------------
    # 5. Rotate image + transform bounding box
    # ------------------------------------------------------------------
    try:
        rotated_bgr, rotated_box = _rotate_image_and_box(bgr, face.box, angle)
    except Exception as exc:  # noqa: BLE001
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=f"Rotation failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 6. Add margin and crop
    # ------------------------------------------------------------------
    h_rot, w_rot = rotated_bgr.shape[:2]
    x1, y1, x2, y2 = _add_margin(rotated_box, cfg.crop_margin, w_rot, h_rot)

    if x2 <= x1 or y2 <= y1:
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=(
                f"Degenerate crop region after margin: "
                f"({x1},{y1}) → ({x2},{y2})"
            ),
        )

    cropped_bgr = rotated_bgr[y1:y2, x1:x2]

    if cropped_bgr.size == 0:
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason="Crop produced an empty array",
        )

    # ------------------------------------------------------------------
    # 7. Resize to target size
    # ------------------------------------------------------------------
    out_w, out_h = cfg.output_size
    try:
        resized_bgr = cv2.resize(
            cropped_bgr,
            (out_w, out_h),
            interpolation=cfg.resize_interpolation,
        )
    except Exception as exc:  # noqa: BLE001
        return AlignmentResult(
            success=False,
            image=None,
            failure_reason=f"cv2.resize failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 8. Convert BGR → RGB → PIL
    # ------------------------------------------------------------------
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb, mode="RGB")

    return AlignmentResult(success=True, image=pil_image, failure_reason=None)


# ---------------------------------------------------------------------------
# Failure tracking
# ---------------------------------------------------------------------------


class FailureLogger:
    """Append failure records to a CSV file, writing the header once.

    The CSV schema stores structured path components rather than a single
    ``image_path`` column so downstream analysis can filter by dataset or
    identity without string parsing.

    Columns
    -------
    dataset:
        First directory component below the raw root (e.g. ``lfw``, ``frll``).
    identity:
        Person / subject sub-directory name.
    filename:
        Image filename including extension.
    reason:
        Human-readable failure explanation.

    Parameters
    ----------
    csv_path:
        Path to the failures CSV.  Created (with header) on first write.
    input_root:
        Raw dataset root used to derive the structured path components.
    """

    _FIELDS: tuple[str, ...] = ("dataset", "identity", "filename", "reason")

    def __init__(self, csv_path: Path, input_root: Path) -> None:
        self._csv_path = csv_path
        self._input_root = input_root
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = csv_path.exists() and csv_path.stat().st_size > 0

    def log(self, image_path: Path, reason: str) -> None:
        """Append one failure row to the CSV.

        Parameters
        ----------
        image_path:
            Absolute path of the failed image.
        reason:
            Short human-readable explanation.
        """
        try:
            parts = image_path.relative_to(self._input_root).parts
            dataset = parts[0] if len(parts) > 0 else ""
            identity = parts[1] if len(parts) > 1 else ""
            filename = parts[-1] if len(parts) > 0 else image_path.name
        except ValueError:
            dataset = ""
            identity = ""
            filename = image_path.name

        with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._FIELDS)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "dataset": dataset,
                    "identity": identity,
                    "filename": filename,
                    "reason": reason,
                }
            )


# ---------------------------------------------------------------------------
# Directory traversal and batch processing
# ---------------------------------------------------------------------------


def collect_images(
    input_dir: Path,
    cfg: AlignmentConfig = ALIGNMENT,
) -> list[Path]:
    """Recursively collect all supported image files under *input_dir*.

    Supported extensions are read from ``cfg.supported_extensions`` so the
    list is consistent across all preprocessing modules.

    Parameters
    ----------
    input_dir:
        Root directory to search.
    cfg:
        Alignment configuration (provides ``supported_extensions``).

    Returns
    -------
    list[Path]
        Sorted list of absolute paths to image files.
    """
    extensions: frozenset[str] = frozenset(cfg.supported_extensions)
    images = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    logger.info("Found %d image(s) under '%s'", len(images), input_dir)
    return images


def derive_output_path(
    image_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Mirror *image_path*'s position under *input_root* into *output_root*.

    The file extension is always replaced with ``.jpg``.

    Parameters
    ----------
    image_path:
        Absolute path of the source image.
    input_root:
        Source directory root.
    output_root:
        Destination directory root.

    Returns
    -------
    Path
        Computed output path under *output_root*.

    Examples
    --------
    >>> derive_output_path(
    ...     Path("datasets/raw/lfw/Aaron_Eckhart/img_001.png"),
    ...     Path("datasets/raw"),
    ...     Path("datasets/aligned"),
    ... )
    PosixPath('datasets/aligned/lfw/Aaron_Eckhart/img_001.jpg')
    """
    relative = image_path.relative_to(input_root)
    return (output_root / relative).with_suffix(".jpg")


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    cfg: AlignmentConfig = ALIGNMENT,
    log_cfg: LoggingConfig = LOGGING,
) -> AlignmentStats:
    """Align all images found under *input_dir* and save them to *output_dir*.

    When ``cfg.skip_existing`` is ``True``, any image whose output path
    already exists is skipped and counted separately so interrupted runs can
    resume without reprocessing.

    Parameters
    ----------
    input_dir:
        Root of the raw dataset tree.
    output_dir:
        Destination root; sub-directories are created automatically.
    cfg:
        Alignment configuration.
    log_cfg:
        Logging / failure-CSV configuration.

    Returns
    -------
    AlignmentStats
        Counts of processed, succeeded, failed, and skipped images plus
        total elapsed wall-clock time.
    """
    t_start = time.perf_counter()

    images = collect_images(input_dir, cfg)
    if not images:
        logger.warning("No images found under '%s'. Nothing to do.", input_dir)
        return AlignmentStats(
            processed=0,
            success=0,
            failure=0,
            skipped=0,
            elapsed_seconds=time.perf_counter() - t_start,
        )

    failure_logger = FailureLogger(log_cfg.alignment_failures_csv, input_dir)
    n_success = 0
    n_failure = 0
    n_skipped = 0

    for image_path in tqdm(images, desc="Aligning faces", unit="img"):
        # ------------------------------------------------------------------
        # Skip-existing check
        # ------------------------------------------------------------------
        out_path = derive_output_path(image_path, input_dir, output_dir)
        if cfg.skip_existing and out_path.exists():
            logger.debug("SKIPPED (exists) '%s'", out_path)
            n_skipped += 1
            continue

        logger.debug("Processing '%s'", image_path)

        result = align_face(image_path, cfg)

        if not result.success:
            logger.warning(
                "FAILED '%s': %s", image_path, result.failure_reason
            )
            failure_logger.log(image_path, result.failure_reason or "unknown")
            n_failure += 1
            continue

        # Save aligned image
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if result.image is None:
                reason = "Alignment succeeded but returned no image."
                logger.error(reason)
                failure_logger.log(image_path, reason)
                n_failure += 1
                continue

            result.image.save(out_path, quality=config.jpeg_quality)
            logger.debug("Saved → '%s'", out_path)
            n_success += 1
        except Exception as exc:  # noqa: BLE001
            reason = f"PIL save failed: {exc}"
            logger.warning("FAILED '%s': %s", image_path, reason)
            failure_logger.log(image_path, reason)
            n_failure += 1

    elapsed = time.perf_counter() - t_start
    n_processed = n_success + n_failure

    summary = (
        f"\n"
        f"  Processed : {n_processed}\n"
        f"  Succeeded : {n_success}\n"
        f"  Failed    : {n_failure}\n"
        f"  Skipped   : {n_skipped}\n"
        f"  Elapsed   : {elapsed:.2f} seconds"
    )
    logger.info("Alignment complete%s", summary)

    return AlignmentStats(
        processed=n_processed,
        success=n_success,
        failure=n_failure,
        skipped=n_skipped,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_alignment",
        description=(
            "Align face images under INPUT_DIR and write them to OUTPUT_DIR, "
            "preserving the original directory structure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/raw"),
        metavar="INPUT_DIR",
        help="Root directory of raw images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/aligned"),
        metavar="OUTPUT_DIR",
        help="Root directory for aligned output images.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=ALIGNMENT.crop_margin,
        dest="crop_margin",
        metavar="MARGIN",
        help="Fractional bounding-box expansion (e.g. 0.3 = 30 %%).",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=list(ALIGNMENT.output_size),
        dest="output_size",
        metavar=("WIDTH", "HEIGHT"),
        help="Target width and height in pixels.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=ALIGNMENT.jpeg_quality,
        dest="jpeg_quality",
        metavar="QUALITY",
        help="JPEG save quality (1–95).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=ALIGNMENT.det_score_threshold,
        dest="det_score_threshold",
        metavar="THRESHOLD",
        help="Minimum RetinaFace confidence score to accept a detection.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        default=ALIGNMENT.skip_existing,
        help="Reprocess images even when the output file already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, 1 if any failures occurred.
    """
    configure_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build a config from CLI overrides
    cfg = AlignmentConfig(
        output_size=tuple(args.output_size),  # type: ignore[arg-type]
        crop_margin=args.crop_margin,
        jpeg_quality=args.jpeg_quality,
        det_score_threshold=args.det_score_threshold,
        skip_existing=args.skip_existing,
    )

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_dir.exists():
        logger.error("Input directory does not exist: '%s'", input_dir)
        return 1

    logger.info("Input  : %s", input_dir)
    logger.info("Output : %s", output_dir)
    logger.info("Config : %s", cfg)

    stats = process_dataset(input_dir, output_dir, cfg)
    return 1 if stats.failure > 0 else 0


if __name__ == "__main__":
    sys.exit(main())