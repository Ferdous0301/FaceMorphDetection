"""
dataset_inspector.py
====================
Production-ready module for inspecting image datasets organized in
identity-based directory hierarchies.

Usage
-----
    python dataset_inspector.py /path/to/dataset1 /path/to/dataset2 \\
        [--output-dir ./reports] \\
        [--image-exts .jpg .jpeg .png .bmp .webp .tiff] \\
        [--log-level INFO]

Outputs
-------
    dataset_report.csv   – one row per identity with per-identity statistics
    dataset_summary.json – aggregate statistics across all datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional dependency: Pillow.  We degrade gracefully if it is absent, but
# resolution/format detection and corruption checks require it.
# ---------------------------------------------------------------------------
try:
    from PIL import Image, UnidentifiedImageError  # type: ignore

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".gif",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    """Metadata for a single image file."""

    path: Path
    identity: str
    dataset_root: Path
    format: Optional[str] = None          # e.g. "JPEG", "PNG"
    width: Optional[int] = None
    height: Optional[int] = None
    corrupted: bool = False
    error_message: Optional[str] = None


@dataclass
class IdentityStats:
    """Aggregated statistics for one identity folder."""

    identity: str
    dataset_root: str
    image_count: int
    formats: List[str] = field(default_factory=list)
    resolutions: List[str] = field(default_factory=list)
    corrupted_count: int = 0


@dataclass
class DatasetSummary:
    """Top-level summary across one or more dataset roots."""

    dataset_roots: List[str]
    total_identities: int
    total_images: int
    total_corrupted: int
    images_per_identity_min: float
    images_per_identity_max: float
    images_per_identity_mean: float
    images_per_identity_median: float
    unique_formats: List[str]
    unique_resolutions: List[str]


# ---------------------------------------------------------------------------
# Core inspection helpers
# ---------------------------------------------------------------------------


def _is_image_file(path: Path, extensions: Tuple[str, ...]) -> bool:
    """Return True when *path* has a recognised image extension."""
    return path.suffix.lower() in extensions


def collect_image_paths(
    identity_dir: Path,
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> Iterator[Path]:
    """
    Yield all image paths found recursively inside *identity_dir*.

    Parameters
    ----------
    identity_dir:
        Root directory for a single identity.
    extensions:
        Tuple of lower-cased file extensions to treat as images.

    Yields
    ------
    Path
        Absolute path to each image file.
    """
    for path in sorted(identity_dir.rglob("*")):
        if path.is_file() and _is_image_file(path, extensions):
            yield path


def inspect_image(
    path: Path,
    identity: str,
    dataset_root: Path,
) -> ImageRecord:
    """
    Open *path* with Pillow and extract metadata.

    Parameters
    ----------
    path:
        Absolute path to the candidate image file.
    identity:
        Identity (person) that owns the image.
    dataset_root:
        Root directory of the dataset containing the image.

    Returns
    -------
    ImageRecord
        Populated record; ``corrupted=True`` when the file cannot be read.
    """
    record = ImageRecord(
        path=path,
        identity=identity,
        dataset_root=dataset_root,
    )

    if not _PIL_AVAILABLE:
        logger.warning(
            "Pillow is not installed – skipping format/resolution/corruption "
            "checks for %s",
            path,
        )
        return record

    try:
        with Image.open(path) as img:
            img.verify()

        # Re-open after verify() (verify() closes the file internally)
        with Image.open(path) as img:
            record.format = img.format
            record.width, record.height = img.size

    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        record.corrupted = True
        record.error_message = str(exc)
        logger.warning("Corrupted image detected: %s — %s", path, exc)

    except Exception as exc:  # noqa: BLE001
        record.corrupted = True
        record.error_message = f"Unexpected error: {exc}"
        logger.error("Unexpected error reading %s: %s", path, exc)

    return record

def inspect_identity(
    identity_dir: Path,
    dataset_root: Path,
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> Tuple[IdentityStats, List[ImageRecord]]:
    """
    Inspect all images under a single identity directory.

    Parameters
    ----------
    identity_dir:
        Directory representing one identity (person / class).
    dataset_root:
        The top-level dataset root that contains this identity.
    extensions:
        Recognised image extensions.

    Returns
    -------
    (IdentityStats, list[ImageRecord])
        Aggregated stats and raw per-image records.
    """
    identity_name = identity_dir.name
    records: List[ImageRecord] = []

    for img_path in collect_image_paths(identity_dir, extensions):
        record = inspect_image(
            img_path,
            identity=identity_name,
            dataset_root=dataset_root,
        )
    records.append(record)

    formats = sorted(
        {r.format for r in records if r.format is not None}
    )
    resolutions = sorted(
        {f"{r.width}x{r.height}" for r in records if r.width and r.height}
    )
    corrupted_count = sum(1 for r in records if r.corrupted)

    stats = IdentityStats(
        identity=identity_name,
        dataset_root=str(dataset_root),
        image_count=len(records),
        formats=formats,
        resolutions=resolutions,
        corrupted_count=corrupted_count,
    )

    logger.debug(
        "Identity '%s': %d images (%d corrupted)",
        identity_name,
        len(records),
        corrupted_count,
    )
    return stats, records


def discover_identity_dirs(dataset_root: Path) -> List[Path]:
    """
    Return all immediate child directories of *dataset_root* that contain at
    least one file (treating each as one identity).

    Parameters
    ----------
    dataset_root:
        Top-level directory of the dataset.

    Returns
    -------
    list[Path]
        Sorted list of identity directories.

    Raises
    ------
    NotADirectoryError
        When *dataset_root* does not exist or is not a directory.
    """
    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )
    identities = sorted(p for p in dataset_root.iterdir() if p.is_dir())

    if not identities:
        logger.warning(
            "No identity directories found under '%s'.",
            dataset_root,
        )
    else:
        logger.info(
            "Discovered %d identity folders under '%s'",
            len(identities),
            dataset_root,
        )

    return identities


def inspect_dataset(
    dataset_root: Path,
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> Tuple[List[IdentityStats], List[ImageRecord]]:
    """
    Traverse *dataset_root* and collect statistics for every identity.

    Parameters
    ----------
    dataset_root:
        Top-level dataset directory.
    extensions:
        Recognised image extensions.

    Returns
    -------
    (list[IdentityStats], list[ImageRecord])
        Per-identity statistics and all raw image records.
    """
    identity_dirs = discover_identity_dirs(dataset_root)
    all_stats: List[IdentityStats] = []
    all_records: List[ImageRecord] = []

    for idx, identity_dir in enumerate(identity_dirs, start=1):
        logger.info(
            "[%d/%d] Inspecting identity: %s",
            idx,
            len(identity_dirs),
            identity_dir.name,
        )
        stats, records = inspect_identity(identity_dir, dataset_root, extensions)
        all_stats.append(stats)
        all_records.extend(records)

    return all_stats, all_records


def build_summary(
    all_stats: List[IdentityStats],
    dataset_roots: List[Path],
) -> DatasetSummary:
    """
    Compute aggregate statistics from per-identity results.

    Parameters
    ----------
    all_stats:
        Flat list of :class:`IdentityStats` from one or more datasets.
    dataset_roots:
        The top-level paths that were inspected.

    Returns
    -------
    DatasetSummary
    """
    counts = [s.image_count for s in all_stats]

    all_formats: List[str] = sorted(
        {fmt for s in all_stats for fmt in s.formats}
    )
    all_resolutions: List[str] = sorted(
        {res for s in all_stats for res in s.resolutions}
    )

    if counts:
        img_min = float(min(counts))
        img_max = float(max(counts))
        img_mean = round(mean(counts), 2)
        img_median = round(median(counts), 2)
    else:
        img_min = img_max = img_mean = img_median = 0.0

    return DatasetSummary(
        dataset_roots=[str(r) for r in dataset_roots],
        total_identities=len(all_stats),
        total_images=sum(counts),
        total_corrupted=sum(s.corrupted_count for s in all_stats),
        images_per_identity_min=img_min,
        images_per_identity_max=img_max,
        images_per_identity_mean=img_mean,
        images_per_identity_median=img_median,
        unique_formats=all_formats,
        unique_resolutions=all_resolutions,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_csv(
    all_stats: List[IdentityStats],
    output_path: Path,
) -> None:
    """
    Write per-identity statistics to a CSV file.

    Parameters
    ----------
    all_stats:
        Flat list of :class:`IdentityStats`.
    output_path:
        Destination file path (created or overwritten).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "identity",
        "dataset_root",
        "image_count",
        "corrupted_count",
        "formats",
        "resolutions",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow(
                {
                    "identity": s.identity,
                    "dataset_root": s.dataset_root,
                    "image_count": s.image_count,
                    "corrupted_count": s.corrupted_count,
                    "formats": "|".join(s.formats),
                    "resolutions": "|".join(s.resolutions),
                }
            )

    logger.info("CSV report written to '%s'", output_path)


def export_json(summary: DatasetSummary, output_path: Path) -> None:
    """
    Serialise :class:`DatasetSummary` to a JSON file.

    Parameters
    ----------
    summary:
        Aggregate dataset statistics.
    output_path:
        Destination file path (created or overwritten).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2, ensure_ascii=False)

    logger.info("JSON summary written to '%s'", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="dataset_inspector",
        description=(
            "Inspect one or more image datasets organised as "
            "<dataset_root>/<identity>/<images> and export statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_roots",
        metavar="DATASET_ROOT",
        nargs="+",
        type=Path,
        help="One or more root directories of identity-based datasets.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("reports"),
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--image-exts",
        metavar="EXT",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="File extensions to treat as images (include the leading dot).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser


def configure_logging(level: str) -> None:
    """Configure root logger with a timestamp-prefixed format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point: parse arguments, run inspection, export reports.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when *None*).

    Returns
    -------
    int
        Exit code (0 = success, 1 = at least one error encountered).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    extensions: Tuple[str, ...] = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.image_exts
    )

    logger.info("Dataset Inspector starting")
    logger.info("Dataset roots : %s", [str(r) for r in args.dataset_roots])
    logger.info("Output dir    : %s", args.output_dir)
    logger.info("Image exts    : %s", extensions)

    if not _PIL_AVAILABLE:
        logger.warning(
            "Pillow is not installed. Install it with: pip install Pillow\n"
            "Format, resolution, and corruption checks will be skipped."
        )

    all_stats: List[IdentityStats] = []
    exit_code = 0

    for root in args.dataset_roots:
        try:
            stats, _ = inspect_dataset(root, extensions)
            all_stats.extend(stats)
        except NotADirectoryError as exc:
            logger.error("%s", exc)
            exit_code = 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error processing '%s': %s", root, exc)
            exit_code = 1

    summary = build_summary(all_stats, args.dataset_roots)

    # Log headline numbers
    logger.info(
        "Results — identities: %d | images: %d | corrupted: %d",
        summary.total_identities,
        summary.total_images,
        summary.total_corrupted,
    )
    logger.info(
        "Images per identity — min: %.0f | max: %.0f | mean: %.2f | median: %.2f",
        summary.images_per_identity_min,
        summary.images_per_identity_max,
        summary.images_per_identity_mean,
        summary.images_per_identity_median,
    )

    csv_path = args.output_dir / "dataset_report.csv"
    json_path = args.output_dir / "dataset_summary.json"

    try:
        export_csv(all_stats, csv_path)
        export_json(summary, json_path)
    except OSError as exc:
        logger.error("Failed to write output files: %s", exc)
        exit_code = 1

    logger.info("Dataset Inspector finished (exit code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
