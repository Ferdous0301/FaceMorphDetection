"""Command-line orchestration for the Dataset Split stage.

This module wires together the previously implemented Dataset Split
components — configuration, the identity graph, the splitter, the
manifest writer, the verifier, and the statistics reporter — into a
single command-line entry point.

Usage (illustrative):

    python -m src.dataset_split.cli \\
        --input /path/to/images \\
        --metadata /path/to/metadata.json \\
        --output /path/to/output \\
        --seed 42 \\
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \\
        --verify --statistics

The metadata file is expected to be JSON of the form::

    {
      "bona_fide": [
        {"image_id": "...", "identity": "...", "label": "bona_fide"}
      ],
      "morphs": [
        {
          "image_id": "...",
          "identity_a": "...",
          "identity_b": "...",
          "label": "morph"
        }
      ]
    }

This module performs orchestration only: argument parsing, wiring
components together, and reporting results. All actual splitting,
writing, verification, and statistics logic lives in the modules it
calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.dataset_split.config import DatasetSplitConfig
from src.dataset_split.exceptions import DatasetSplitError
from src.dataset_split.identity_graph import IdentityGraph
from src.dataset_split.manifest_writer import ManifestWriter
from src.dataset_split.splitter import (
    BonaFideSample,
    DatasetSplitter,
    MorphSample,
)
from src.dataset_split.statistics_reporter import StatisticsReporter
from src.dataset_split.verifier import ManifestVerifier

#: Process exit code for successful completion.
EXIT_SUCCESS = 0
#: Process exit code for a general/unexpected runtime error.
EXIT_ERROR = 1
#: Process exit code reserved for argparse's own invalid-argument errors.
EXIT_INVALID_ARGS = 2
#: Process exit code for missing or unreadable input paths.
EXIT_MISSING_PATH = 3
#: Process exit code for post-split verification failures.
EXIT_VERIFICATION_FAILED = 4
#: Process exit code for invalid split configuration (e.g. bad ratios).
EXIT_CONFIG_ERROR = 5


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        A fully configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="dataset-split",
        description=(
            "Split a face morphing attack detection dataset into "
            "identity-disjoint train/validation/test sets."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Directory containing the dataset's image files.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=str,
        help=(
            "Path to a JSON metadata file describing bona fide and "
            "morph samples."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Directory into which manifests and reports are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling deterministic split ordering.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of the dataset allocated to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset allocated to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset allocated to the test split.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run manifest verification after splitting.",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Compute and print dataset statistics after splitting.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: The argument list to parse, or ``None`` to use
            ``sys.argv[1:]``.

    Returns:
        The parsed argument namespace.

    Raises:
        SystemExit: Raised by ``argparse`` itself (with code
            :data:`EXIT_INVALID_ARGS`) when arguments are missing,
            malformed, or otherwise invalid.
    """
    parser = build_parser()
    return parser.parse_args(argv)


def _load_metadata(metadata_path: Path) -> tuple[
    list[BonaFideSample], list[MorphSample]
]:
    """Load bona fide and morph samples from a JSON metadata file.

    Args:
        metadata_path: Path to the JSON metadata file.

    Returns:
        A tuple of ``(bona_fide_samples, morph_samples)``.

    Raises:
        ValueError: If the metadata file contains malformed JSON or is
            missing required fields on an entry.
    """
    with metadata_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    bona_fide_samples = [
        BonaFideSample(
            image_id=entry["image_id"],
            identity=entry["identity"],
            label=entry.get("label", "bona_fide"),
        )
        for entry in payload.get("bona_fide", [])
    ]
    morph_samples = [
        MorphSample(
            image_id=entry["image_id"],
            identity_a=entry["identity_a"],
            identity_b=entry["identity_b"],
            label=entry.get("label", "morph"),
        )
        for entry in payload.get("morphs", [])
    ]
    return bona_fide_samples, morph_samples


def _build_identity_graph(
    bona_fide_samples: list[BonaFideSample],
    morph_samples: list[MorphSample],
) -> IdentityGraph:
    """Build the identity graph used for splitting, verification, and stats.

    Args:
        bona_fide_samples: All bona fide samples.
        morph_samples: All morph samples.

    Returns:
        The populated :class:`IdentityGraph`.
    """
    graph = IdentityGraph()
    for sample in bona_fide_samples:
        graph.add_identity(sample.identity)
    for morph in morph_samples:
        graph.add_morph(morph.identity_a, morph.identity_b)
    return graph


def run(args: argparse.Namespace) -> int:
    """Execute the dataset split pipeline stage from parsed arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        A process exit code: :data:`EXIT_SUCCESS` on success, or one
        of the other ``EXIT_*`` constants describing the failure mode.
    """
    input_dir = Path(args.input)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return EXIT_MISSING_PATH

    if not metadata_path.is_file():
        print(f"error: metadata file not found: {metadata_path}", file=sys.stderr)
        return EXIT_MISSING_PATH

    try:
        bona_fide_samples, morph_samples = _load_metadata(metadata_path)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: failed to load metadata: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        config = DatasetSplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            output_directory=output_dir,
            manifest_directory=output_dir,
            statistics_directory=output_dir,
        )
    except DatasetSplitError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        splitter = DatasetSplitter(config)
        result = splitter.split(bona_fide_samples, morph_samples)
    except DatasetSplitError as exc:
        print(f"error: splitting failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    writer = ManifestWriter(config.output_directory)
    writer.write_result(result)
    print(f"Manifests written to {config.output_directory}")

    identity_graph = _build_identity_graph(bona_fide_samples, morph_samples)

    if args.verify:
        verifier = ManifestVerifier()
        expected_ids = {s.image_id for s in bona_fide_samples} | {
            m.image_id for m in morph_samples
        }
        report = verifier.verify(
            result,
            identity_graph=identity_graph,
            image_root=input_dir,
            expected_image_ids=expected_ids,
        )
        if report.passed:
            print("Verification: PASSED")
        else:
            print(
                f"Verification: FAILED ({report.error_count} error(s))",
                file=sys.stderr,
            )
            for issue in report.issues:
                print(f"  [{issue.check}] {issue.message}", file=sys.stderr)
            return EXIT_VERIFICATION_FAILED

    if args.statistics:
        reporter = StatisticsReporter()
        stats_report = reporter.generate(result, identity_graph=identity_graph)
        print(reporter.format_console(stats_report))

    return EXIT_SUCCESS


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: The argument list to parse, or ``None`` to use
            ``sys.argv[1:]``.

    Returns:
        The process exit code produced by :func:`run`.
    """
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())