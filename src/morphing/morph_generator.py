"""
src/morphing/morph_generator.py
================================

End-to-end morph generation pipeline with resume support and a CLI.

This module ties together all other morphing sub-modules:

* Traverses the aligned-images directory tree produced by the Face Alignment
  module.
* Selects one representative image per identity using a seeded RNG.
* Pairs up identities and calls ``LandmarkDetector`` + ``warp_and_blend``.
* Saves each morph image and appends a ``MorphRecord`` to the metadata CSV.
* Supports **resume mode**: pairs whose output file already exists are skipped
  when ``skip_existing=True``.
* Logs every failure with its reason and prints a full summary at the end.
* Exposes ``main()`` as a CLI entry-point.

Configuration
-------------
Morph-specific settings are collected in ``MorphConfig``.  Paths and
hyper-parameters that already exist in ``src/config.py`` should be read from
there by the calling script; ``MorphConfig`` defines only the fields that
belong to the morphing module.

CLI usage::

    python -m src.morphing.morph_generator \\
        --dataset lfw \\
        --alpha 0.5 \\
        --skip-existing \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2

from src.morphing.delaunay import warp_and_blend
from src.morphing.mediapipe_landmarks import LandmarkDetector
from src.morphing.metadata import MetadataWriter, MorphRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: One identity entry: (identity_name, representative_image_path).
_Identity = tuple[str, Path]

#: One pair entry: (id_a, path_a, id_b, path_b).
_Pair = tuple[str, Path, str, Path]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MorphConfig:
    """Morph-generation configuration.

    All fields have sensible defaults and may be overridden by callers or the
    CLI.  ``__post_init__`` validates the values that have well-defined
    constraints.

    Attributes
    ----------
    aligned_root : Path
        Root directory of the aligned images produced by the Face Alignment
        module.  Expected layout::

            aligned_root/
            └── <dataset>/
                └── <identity>/
                    └── *.jpg  (or *.png)

    output_root : Path
        Root directory where morph images are saved.  Dataset sub-directories
        are created automatically.
    metadata_path : Path
        Full path to the morph metadata CSV file.
    alpha : float
        Blend weight for all morphs in this run, in ``[0, 1]``.
    skip_existing : bool
        When ``True``, pairs whose output file already exists are silently
        skipped, enabling resume of interrupted runs.
    seed : int
        Random seed for reproducible representative-image selection.
    image_extensions : tuple[str, ...]
        Lower-case file extensions (with leading dot) treated as images when
        scanning the aligned directory.
    max_pairs_per_dataset : int or None
        Maximum pairs to process per dataset.  ``None`` means no cap.

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1]``, or ``max_pairs_per_dataset`` is
        not ``None`` and is less than 1.
    """

    aligned_root: Path = Path("datasets/aligned")
    output_root: Path = Path("datasets/morph/images")
    metadata_path: Path = Path("datasets/morph/metadata/morph_metadata.csv")
    alpha: float = 0.5
    skip_existing: bool = True
    seed: int = 42
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_pairs_per_dataset: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"MorphConfig.alpha must be in [0, 1], got {self.alpha}."
            )
        if self.max_pairs_per_dataset is not None and self.max_pairs_per_dataset < 1:
            raise ValueError(
                f"MorphConfig.max_pairs_per_dataset must be ≥ 1 or None, "
                f"got {self.max_pairs_per_dataset}."
            )


# ---------------------------------------------------------------------------
# Processing summary
# ---------------------------------------------------------------------------

@dataclass
class ProcessingSummary:
    """Accumulated counters and failure details for one morph-generation run.

    Attributes
    ----------
    total_pairs : int
        Number of identity pairs considered (including skipped and failed).
    success : int
        Pairs that produced a saved morph image.
    skipped : int
        Pairs skipped because the output already existed.
    failed : int
        Pairs that failed for any reason.
    failures : list[tuple[str, str, str]]
        One entry per failure: ``(identity_a, identity_b, reason)``.
    """

    total_pairs: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def log(self) -> None:
        """Print the summary to stdout and emit it at INFO level."""
        lines = [
            "=" * 60,
            "Morph Generation Summary",
            "=" * 60,
            f"  Total pairs considered : {self.total_pairs}",
            f"  Successfully generated : {self.success}",
            f"  Skipped (existing)     : {self.skipped}",
            f"  Failed                 : {self.failed}",
        ]
        if self.failures:
            lines.append("\nFailure details:")
            for id_a, id_b, reason in self.failures:
                lines.append(f"    [{id_a} × {id_b}]  {reason}")
        lines.append("=" * 60)
        summary_str = "\n".join(lines)
        print(summary_str)
        logger.info(summary_str)


# ---------------------------------------------------------------------------
# Core generator class
# ---------------------------------------------------------------------------

class MorphGenerator:
    """Orchestrates the full morph-generation pipeline for one or more datasets.

    Parameters
    ----------
    config : MorphConfig
        All configuration for this run.

    Examples
    --------
    >>> cfg = MorphConfig(alpha=0.5, skip_existing=True, seed=42)
    >>> gen = MorphGenerator(cfg)
    >>> summary = gen.run(datasets=["lfw"])
    >>> summary.log()
    """

    def __init__(self, config: MorphConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, datasets: Optional[Sequence[str]] = None) -> ProcessingSummary:
        """Run the morph generator across one or more datasets.

        Opens a single ``LandmarkDetector`` for the entire run and closes it
        when all datasets have been processed.

        Parameters
        ----------
        datasets : Sequence[str] or None
            Names of sub-directories under ``config.aligned_root`` to
            process.  ``None`` discovers all sub-directories automatically.

        Returns
        -------
        ProcessingSummary
            Counts and failure details for the completed run.
        """
        summary = ProcessingSummary()
        writer = MetadataWriter(self._cfg.metadata_path)

        target_datasets = self._resolve_datasets(datasets)
        if not target_datasets:
            logger.warning("No datasets found under %s.", self._cfg.aligned_root)
            return summary

        t0 = time.monotonic()
        with LandmarkDetector() as detector:
            for dataset_name in target_datasets:
                logger.info("Processing dataset: %s", dataset_name)
                self._process_dataset(dataset_name, detector, writer, summary)

        elapsed = time.monotonic() - t0
        logger.info(
            "Run complete in %.1f s: %d success, %d skipped, %d failed.",
            elapsed,
            summary.success,
            summary.skipped,
            summary.failed,
        )
        summary.log()
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_datasets(
        self, datasets: Optional[Sequence[str]]
    ) -> list[str]:
        """Return the list of dataset names to process.

        Parameters
        ----------
        datasets : Sequence[str] or None
            Explicit list of names, or ``None`` to auto-discover from
            ``config.aligned_root``.

        Returns
        -------
        list[str]
            Sorted dataset names, or an empty list if the root is missing.
        """
        root = self._cfg.aligned_root
        if not root.exists():
            logger.error("Aligned root does not exist: %s", root)
            return []

        if datasets is not None:
            return sorted(datasets)

        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def _process_dataset(
        self,
        dataset_name: str,
        detector: LandmarkDetector,
        writer: MetadataWriter,
        summary: ProcessingSummary,
    ) -> None:
        """Process all selected identity pairs in one dataset directory.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset sub-directory under ``config.aligned_root``.
        detector : LandmarkDetector
            Shared landmark detector for the run.
        writer : MetadataWriter
            Shared metadata writer for the run.
        summary : ProcessingSummary
            Accumulator updated in-place.
        """
        dataset_dir = self._cfg.aligned_root / dataset_name
        if not dataset_dir.is_dir():
            logger.warning(
            "Dataset directory does not exist: %s",
            dataset_dir,
        )
        return
        output_dir = self._cfg.output_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        identities = self._collect_identities(dataset_dir)
        if len(identities) < 2:
            logger.warning(
                "Dataset '%s' has %d identity/identities; need ≥ 2 to form pairs.",
                dataset_name,
                len(identities),
            )
            return

        pairs = self._build_pairs(identities)
        if self._cfg.max_pairs_per_dataset is not None:
            pairs = pairs[: self._cfg.max_pairs_per_dataset]

        logger.info(
            "Dataset '%s': %d identities → %d pair(s) to process.",
            dataset_name,
            len(identities),
            len(pairs),
        )

        for id_a, path_a, id_b, path_b in pairs:
            summary.total_pairs += 1
            self._process_pair(
                id_a, path_a, id_b, path_b,
                dataset_name, output_dir, detector, writer, summary,
            )

    def _collect_identities(self, dataset_dir: Path) -> list[_Identity]:
        """Return one representative image per identity sub-directory.

        Identities are sorted alphabetically for determinism, and one image is
        selected per identity using the seeded RNG.

        Parameters
        ----------
        dataset_dir : Path
            Dataset directory containing one sub-directory per identity.

        Returns
        -------
        list[_Identity]
            Sorted list of ``(identity_name, representative_image_path)``
            pairs.  Sub-directories with no matching images are skipped.
        """
        identities: list[_Identity] = []

        for id_dir in sorted(dataset_dir.iterdir()):
            if id_dir.name.startswith("."):
                continue
            if not id_dir.is_dir():
                continue

            images = sorted(
                p
                for p in id_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in self._cfg.image_extensions
            )

            if not images:
                logger.debug(
                    "Identity directory '%s' contains no images; skipping.",
                    id_dir.name,
                )
                continue

            chosen = self._rng.choice(images)
            identities.append((id_dir.name, chosen))

        return identities

    def _build_pairs(
        self, identities: list[_Identity]
    ) -> list[_Pair]:
        """Return all unique ordered pairs (A < B) from ``identities``.

        Parameters
        ----------
        identities : list[_Identity]
            Sorted identity list.

        Returns
        -------
        list[_Pair]
            List of ``(id_a, path_a, id_b, path_b)`` tuples; length is
            ``len(identities) * (len(identities) - 1) / 2``.
        """
        pairs: list[_Pair] = []
        n = len(identities)
        for i in range(n):
            for j in range(i + 1, n):
                id_a, path_a = identities[i]
                id_b, path_b = identities[j]
                pairs.append((id_a, path_a, id_b, path_b))
        return pairs

    def _process_pair(
        self,
        id_a: str,
        path_a: Path,
        id_b: str,
        path_b: Path,
        dataset_name: str,
        output_dir: Path,
        detector: LandmarkDetector,
        writer: MetadataWriter,
        summary: ProcessingSummary,
    ) -> None:
        """Generate, save, and record one morph for an identity pair.

        All failures are caught, logged with a reason string, and recorded in
        ``summary`` without raising.

        Parameters
        ----------
        id_a, path_a : str, Path
            Identity name and image path for subject A.
        id_b, path_b : str, Path
            Identity name and image path for subject B.
        dataset_name : str
            Dataset label written to the metadata record.
        output_dir : Path
            Directory where the morph image is saved.
        detector : LandmarkDetector
            Shared landmark detector.
        writer : MetadataWriter
            Shared metadata writer.
        summary : ProcessingSummary
            Accumulator updated in-place.
        """
        alpha_tag = f"{int(self._cfg.alpha * 100):03d}"
        morph_filename = f"morph_{id_a}_{id_b}_a{alpha_tag}.jpg"
        out_path = output_dir / morph_filename

        # -- Resume mode ---------------------------------------------------
        if self._cfg.skip_existing and out_path.exists():
            logger.debug("Skipping existing morph: %s", morph_filename)
            summary.skipped += 1
            return

        # -- Load images ---------------------------------------------------
        img_a = cv2.imread(str(path_a))
        if img_a is None:
            self._record_failure(
                summary, id_a, id_b,
                f"Could not load image A: {path_a.name}"
            )
            return

        img_b = cv2.imread(str(path_b))
        if img_b is None:
            self._record_failure(
                summary, id_a, id_b,
                f"Could not load image B: {path_b.name}"
            )
            return

        # -- Guard against shape mismatch (should not happen post-alignment) -
        if img_a.shape != img_b.shape:
            logger.warning(
                "[%s × %s] Shape mismatch (%s vs %s); resizing B to match A.",
                id_a, id_b, img_a.shape, img_b.shape,
            )
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

        h, w = img_a.shape[:2]

        # -- Landmark detection --------------------------------------------
        result_a = detector.detect(img_a)
        if result_a is None:
            self._record_failure(
                summary, id_a, id_b,
                f"Landmark detection failed for A: {path_a.name} ({w}×{h})"
            )
            return

        result_b = detector.detect(img_b)
        if result_b is None:
            self._record_failure(
                summary, id_a, id_b,
                f"Landmark detection failed for B: {path_b.name} ({w}×{h})"
            )
            return

        # -- Morph generation ----------------------------------------------
        try:
            morphed = warp_and_blend(
                img_a, img_b,
                result_a.points, result_b.points,
                self._cfg.alpha,
            )
        except Exception:
            logger.exception(
                "warp_and_blend raised an exception [%s × %s].", id_a, id_b
            )
            self._record_failure(
                summary, id_a, id_b, "warp_and_blend raised an unexpected exception"
            )
            return

        # -- Save morph image ----------------------------------------------
        try:
            saved = cv2.imwrite(str(out_path), morphed)
            if not saved:
                raise IOError(f"cv2.imwrite returned False for {out_path}")
        except Exception:
            logger.exception("Failed to save morph to %s.", out_path)
            self._record_failure(
                summary, id_a, id_b,
                f"Failed to save morph image to {out_path.name}"
            )
            return

        # -- Write metadata ------------------------------------------------
        record = MorphRecord(
            morph_filename=morph_filename,
            source_image_a=str(path_a),
            source_image_b=str(path_b),
            identity_a=id_a,
            identity_b=id_b,
            dataset=dataset_name,
            alpha=self._cfg.alpha,
        )
        try:
            writer.append(record)
        except Exception:
            # Do not count as a failure – the image was saved successfully.
            logger.exception(
                "Failed to write metadata for %s; image was saved.", morph_filename
            )

        summary.success += 1
        logger.info(
            "Saved morph: %s  [%s × %s, %d×%d, α=%.2f]",
            morph_filename, id_a, id_b, w, h, self._cfg.alpha,
        )

    @staticmethod
    def _record_failure(
        summary: ProcessingSummary,
        id_a: str,
        id_b: str,
        reason: str,
    ) -> None:
        """Log a failure at WARNING level and append it to ``summary``.

        Parameters
        ----------
        summary : ProcessingSummary
            Accumulator updated in-place.
        id_a : str
            Identity A label.
        id_b : str
            Identity B label.
        reason : str
            Human-readable failure description.
        """
        logger.warning("FAILED [%s × %s]: %s", id_a, id_b, reason)
        summary.failed += 1
        summary.failures.append((id_a, id_b, reason))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    _defaults = MorphConfig()

    parser = argparse.ArgumentParser(
        prog="morph_generator",
        description=(
            "Generate face morph attack images using MediaPipe Face Mesh "
            "and Delaunay triangulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=_defaults.aligned_root,
        help="Root directory of aligned face images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_defaults.output_root,
        help="Root directory where morph images are saved.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=_defaults.metadata_path,
        help="Path to the morph metadata CSV file.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "One or more dataset sub-directory names to process.  "
            "Omit to process all datasets found under --aligned-root."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_defaults.alpha,
        metavar="FLOAT",
        help="Blend weight for morph generation, in [0, 1].",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip pairs whose output file already exists (resume mode).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-generate morphs even if the output file already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_defaults.seed,
        help="Random seed for reproducible pair selection.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        metavar="N",
        help="Cap on the number of pairs processed per dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry-point for the morph generator.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        ``0`` if all pairs succeeded (or none were attempted);
        ``1`` if at least one pair failed.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = MorphConfig(
        aligned_root=args.aligned_root,
        output_root=args.output_root,
        metadata_path=args.metadata_path,
        alpha=args.alpha,
        skip_existing=args.skip_existing,
        seed=args.seed,
        max_pairs_per_dataset=args.max_pairs,
    )

    summary = MorphGenerator(cfg).run(datasets=args.dataset)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())