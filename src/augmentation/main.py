"""Command-line interface for the FMAD Data Augmentation stage.

This module wires the dataset augmentation engine
(:func:`src.augmentation.augment_dataset.augment_dataset`) up to a
production-quality command-line interface. It owns *no* image
processing or augmentation logic of its own: its sole responsibilities
are argument parsing, operator selection, logging configuration, and
translating the outcome of a run into a process exit code.

Usage
-----
As an installed console script or via ``python -m``::

    python -m src.augmentation.main \\
        --input-dir /path/to/train \\
        --output-dir /path/to/train_augmented \\
        --operators brightness contrast horizontal_flip \\
        --seed 42 \\
        --overwrite \\
        --verbose

Exit codes
----------
``0``
    The run completed successfully (even if individual images were
    skipped or failed; per-image failures are expected and reported in
    the run summary, not treated as CLI failures).
``2``
    The command-line arguments themselves were invalid (unknown
    operator name, out-of-range JPEG quality, malformed extension,
    and so on). This mirrors :mod:`argparse`'s own convention of
    exiting with status ``2`` on a parsing failure.
``1``
    The run could not be started or failed at the engine level (for
    example, because the input directory does not exist, or because
    an unexpected error occurred while orchestrating the run).
``130``
    The run was interrupted by the user (``Ctrl+C`` / ``SIGINT``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

from src.augmentation.augment_dataset import (
    DEFAULT_FILENAME_TEMPLATE,
    DEFAULT_SUPPORTED_EXTENSIONS,
    DatasetAugmentationError,
    DatasetProcessingSummary,
    augment_dataset,
)
from src.augmentation.operators.base import BaseAugmentation
from src.augmentation.operators.brightness import BrightnessOperator
from src.augmentation.operators.contrast import ContrastOperator
from src.augmentation.operators.gamma import GammaOperator
from src.augmentation.operators.gaussian_blur import GaussianBlurOperator
from src.augmentation.operators.gaussian_noise import GaussianNoiseOperator
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator
from src.augmentation.operators.jpeg_compression import JPEGCompressionOperator
from src.augmentation.operators.sharpen import SharpenOperator

__all__ = [
    "EXIT_SUCCESS",
    "EXIT_CONFIGURATION_ERROR",
    "EXIT_RUNTIME_ERROR",
    "EXIT_INTERRUPTED",
    "OPERATOR_FACTORIES",
    "NAMING_STRATEGIES",
    "build_arg_parser",
    "parse_args",
    "configure_logging",
    "resolve_operators",
    "run",
    "main",
]


_LOGGER_NAME: Final[str] = "src.augmentation.main"

#: Process exit code used when the run completed (per-image failures
#: are reported in the summary and do not affect this code).
EXIT_SUCCESS: Final[int] = 0

#: Process exit code used when the command-line arguments themselves
#: are invalid.
EXIT_CONFIGURATION_ERROR: Final[int] = 2

#: Process exit code used when the run could not be started, or failed
#: at the engine level, due to something other than bad arguments.
EXIT_RUNTIME_ERROR: Final[int] = 1

#: Process exit code used when the run was interrupted by the user.
EXIT_INTERRUPTED: Final[int] = 130

#: Mapping from a stable, user-facing operator name to the concrete
#: operator class that implements it. Used to resolve the
#: ``--operators`` command-line argument to actual operator instances.
OPERATOR_FACTORIES: Final[Mapping[str, type[BaseAugmentation]]] = {
    "brightness": BrightnessOperator,
    "contrast": ContrastOperator,
    "gamma": GammaOperator,
    "gaussian_blur": GaussianBlurOperator,
    "gaussian_noise": GaussianNoiseOperator,
    "horizontal_flip": HorizontalFlipOperator,
    "jpeg_compression": JPEGCompressionOperator,
    "sharpen": SharpenOperator,
}

#: Mapping from a stable, user-facing output-naming strategy name to
#: the ``str.format``-style filename template it corresponds to. Every
#: template must support the ``{stem}``, ``{operators}``, and
#: ``{ext}`` fields, exactly like
#: :data:`~src.augmentation.augment_dataset.DEFAULT_FILENAME_TEMPLATE`.
NAMING_STRATEGIES: Final[Mapping[str, str]] = {
    "operator_suffix": DEFAULT_FILENAME_TEMPLATE,
    "operator_prefix": "aug_{operators}_{stem}{ext}",
    "flat": "{stem}{ext}",
}

_DEFAULT_NAMING_STRATEGY: Final[str] = "operator_suffix"
_DEFAULT_PROBABILITY: Final[float] = 0.5
_DEFAULT_JPEG_QUALITY: Final[int] = 95
_DEFAULT_SEED: Final[int] = 42


def _positive_int(raw_value: str) -> int:
    """Parse and validate a strictly positive integer CLI argument.

    Args:
        raw_value: The raw string value provided on the command line.

    Returns:
        int: The parsed, strictly positive integer.

    Raises:
        argparse.ArgumentTypeError: If ``raw_value`` is not a valid
            integer, or is not strictly positive.
    """
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{raw_value!r} is not a valid integer."
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a strictly positive integer, got {value!r}."
        )
    return value


def _jpeg_quality(raw_value: str) -> int:
    """Parse and validate a JPEG quality factor CLI argument.

    Args:
        raw_value: The raw string value provided on the command line.

    Returns:
        int: The parsed JPEG quality factor, guaranteed to lie within
        ``[1, 100]``.

    Raises:
        argparse.ArgumentTypeError: If ``raw_value`` is not a valid
            integer, or does not lie within ``[1, 100]``.
    """
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{raw_value!r} is not a valid integer."
        ) from exc

    if not (1 <= value <= 100):
        raise argparse.ArgumentTypeError(
            f"JPEG quality must be between 1 and 100 (inclusive), got {value!r}."
        )
    return value


def _probability(raw_value: str) -> float:
    """Parse and validate a probability CLI argument.

    Args:
        raw_value: The raw string value provided on the command line.

    Returns:
        float: The parsed probability, guaranteed to lie within
        ``[0.0, 1.0]``.

    Raises:
        argparse.ArgumentTypeError: If ``raw_value`` is not a valid
            float, or does not lie within ``[0.0, 1.0]``.
    """
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{raw_value!r} is not a valid float."
        ) from exc

    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(
            f"probability must be between 0.0 and 1.0 (inclusive), got {value!r}."
        )
    return value


def _extension(raw_value: str) -> str:
    """Normalise a single file-extension CLI argument.

    Args:
        raw_value: The raw string value provided on the command line,
            with or without a leading dot, in any case.

    Returns:
        str: The normalised, lower-case, dotted extension.

    Raises:
        argparse.ArgumentTypeError: If ``raw_value`` is empty.
    """
    candidate = raw_value.strip().lower()
    if not candidate:
        raise argparse.ArgumentTypeError("extension values must not be empty.")
    if not candidate.startswith("."):
        candidate = f".{candidate}"
    return candidate


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the augmentation CLI.

    Returns:
        argparse.ArgumentParser: A fully configured parser. Calling
        ``parser.parse_args(...)`` with invalid input raises
        :class:`SystemExit` with code ``2``, matching
        :data:`EXIT_CONFIGURATION_ERROR`.
    """
    parser = argparse.ArgumentParser(
        prog="fmad-augment",
        description=(
            "Run the FMAD Data Augmentation stage over a dataset, producing "
            "augmented images under an output directory while preserving "
            "the input directory's structure."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory of the input dataset to augment. Read-only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory that augmented images are written under.",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=sorted(OPERATOR_FACTORIES),
        default=None,
        metavar="OPERATOR",
        help=(
            "Names of the operators to apply, in application order. "
            f"Choices: {', '.join(sorted(OPERATOR_FACTORIES))}. "
            "Defaults to every available operator, in alphabetical order."
        ),
    )
    parser.add_argument(
        "--probability",
        type=_probability,
        default=_DEFAULT_PROBABILITY,
        help=(
            "Probability, in [0.0, 1.0], that each selected operator is "
            "applied to a given image (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed used to construct every selected operator (default: %(default)s).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        type=_extension,
        default=None,
        metavar="EXT",
        help=(
            "File extensions treated as supported images (default: "
            f"{', '.join(sorted(DEFAULT_SUPPORTED_EXTENSIONS))})."
        ),
    )
    parser.add_argument(
        "--naming-strategy",
        choices=sorted(NAMING_STRATEGIES),
        default=_DEFAULT_NAMING_STRATEGY,
        help="Output filename naming strategy (default: %(default)s).",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        metavar="FORMAT",
        help=(
            "Optional output image format override (e.g. 'png', 'jpg'). "
            "If omitted, each image keeps its original extension."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=_jpeg_quality,
        default=_DEFAULT_JPEG_QUALITY,
        help="JPEG quality factor, in [1, 100], used when saving JPEG images (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a full dry run: discover and simulate processing without writing any files.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories of --input-dir (default).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan --input-dir itself, ignoring subdirectories.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity. Repeatable (-v for INFO, -vv for DEBUG).",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the augmentation CLI.

    Args:
        argv: The argument vector to parse, excluding the program
            name. If ``None``, :data:`sys.argv[1:]` is used.

    Returns:
        argparse.Namespace: The parsed arguments.

    Raises:
        SystemExit: If ``argv`` is invalid. The exit code is always
            :data:`EXIT_CONFIGURATION_ERROR` (``2``), matching
            :mod:`argparse`'s standard behaviour.
    """
    parser = build_arg_parser()
    return parser.parse_args(argv)


def configure_logging(verbosity: int, *, logger: logging.Logger | None = None) -> logging.Logger:
    """Configure and return the logger used by the augmentation CLI.

    Args:
        verbosity: The number of times ``-v``/``--verbose`` was
            supplied. ``0`` maps to ``WARNING``, ``1`` maps to
            ``INFO``, and ``2`` or more maps to ``DEBUG``.
        logger: An optional logger to configure. If ``None``, the
            module-level CLI logger is used.

    Returns:
        logging.Logger: The configured logger.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    root_logger.setLevel(level)

    active_logger = logger if logger is not None else logging.getLogger(_LOGGER_NAME)
    active_logger.setLevel(level)
    return active_logger


def resolve_operators(
    operator_names: Sequence[str] | None,
    *,
    probability: float,
    seed: int,
    factories: Mapping[str, type[BaseAugmentation]] = OPERATOR_FACTORIES,
) -> list[BaseAugmentation]:
    """Resolve CLI operator names into constructed operator instances.

    Args:
        operator_names: The operator names selected on the command
            line, in the order they should be applied. If ``None`` or
            empty, every operator in ``factories`` is used, in
            alphabetical order.
        probability: The probability, in ``[0.0, 1.0]``, applied to
            every constructed operator.
        seed: The random seed used to construct every operator's
            private, deterministic random generator.
        factories: The mapping from operator name to operator class
            used to resolve names. Defaults to
            :data:`OPERATOR_FACTORIES`.

    Returns:
        list[BaseAugmentation]: The constructed operator instances, in
        application order.

    Raises:
        DatasetAugmentationError: If any name in ``operator_names`` is
            not a recognised operator.
    """
    selected_names = list(operator_names) if operator_names else sorted(factories)

    operators: list[BaseAugmentation] = []
    for name in selected_names:
        try:
            operator_cls = factories[name]
        except KeyError as exc:
            raise DatasetAugmentationError(
                f"Unknown operator {name!r}. Available operators: "
                f"{sorted(factories)}."
            ) from exc
        operators.append(
            operator_cls(probability=probability, random_state=seed)
        )

    return operators


def run(args: argparse.Namespace, *, logger: logging.Logger | None = None) -> DatasetProcessingSummary:
    """Execute an augmentation run described by parsed CLI arguments.

    This function performs no image processing itself: it resolves
    the selected operators and naming strategy, then delegates
    directly to
    :func:`~src.augmentation.augment_dataset.augment_dataset`.

    Args:
        args: The parsed command-line arguments, as returned by
            :func:`parse_args`.
        logger: An optional logger to use for diagnostic output. If
            ``None``, the module-level CLI logger is used.

    Returns:
        DatasetProcessingSummary: The summary describing the outcome
        of the run.

    Raises:
        DatasetAugmentationError: If the run cannot be started or
            fails at the engine level (for example, because the input
            directory does not exist, an unknown operator was
            requested, or the configuration is otherwise invalid).
    """
    active_logger = logger if logger is not None else logging.getLogger(_LOGGER_NAME)

    operators = resolve_operators(
        args.operators, probability=args.probability, seed=args.seed
    )
    active_logger.info(
        "Resolved %d operator(s): %s",
        len(operators),
        ", ".join(operator.operator_name for operator in operators) or "<none>",
    )

    supported_extensions = args.extensions if args.extensions else DEFAULT_SUPPORTED_EXTENSIONS
    filename_template = NAMING_STRATEGIES[args.naming_strategy]

    input_dir = args.input_dir
    if not args.recursive:
        active_logger.info(
            "Non-recursive mode requested; only top-level files under %s will be scanned.",
            input_dir,
        )

    with _scoped_input_directory(
        input_dir, recursive=args.recursive, extensions=frozenset(supported_extensions)
    ) as scoped_input_dir:
        active_logger.info(
            "Starting augmentation run: input=%s output=%s dry_run=%s overwrite=%s",
            input_dir,
            args.output_dir,
            args.dry_run,
            args.overwrite,
        )

        summary = augment_dataset(
            scoped_input_dir,
            args.output_dir,
            operators=operators,
            supported_extensions=supported_extensions,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            output_image_format=args.output_format,
            jpeg_quality=args.jpeg_quality,
            filename_template=filename_template,
            logger=active_logger,
        )

    active_logger.info(
        "Run finished: total=%d processed=%d augmented=%d skipped=%d failed=%d "
        "elapsed=%.3fs",
        summary.total_images,
        summary.processed_images,
        summary.augmented_images,
        summary.skipped_images,
        summary.failed_images,
        summary.elapsed_time,
    )

    return summary


def _scoped_input_directory(
    input_dir: Path, *, recursive: bool, extensions: frozenset[str]
) -> "_ScopedInputDirectory":
    """Build a context manager exposing the directory to scan for images.

    When ``recursive`` is ``True``, ``input_dir`` itself is exposed
    unchanged, and the underlying engine's own recursive discovery is
    used. When ``recursive`` is ``False``, a temporary directory
    containing symlinks to only the supported image files directly
    inside ``input_dir`` (not its subdirectories) is created and
    exposed instead, so that the engine's recursive discovery
    effectively only "sees" the top-level files. This keeps all actual
    augmentation and file-writing logic inside the engine, with no
    duplication here.

    Args:
        input_dir: The real input dataset root requested by the user.
        recursive: Whether subdirectories of ``input_dir`` should be
            scanned.
        extensions: The set of lower-case, dotted extensions treated
            as supported images, used only to select which top-level
            files to expose in non-recursive mode.

    Returns:
        _ScopedInputDirectory: A context manager yielding the
        directory that should actually be passed to
        :func:`~src.augmentation.augment_dataset.augment_dataset`.
    """
    return _ScopedInputDirectory(input_dir, recursive=recursive, extensions=extensions)


class _ScopedInputDirectory:
    """Context manager yielding the effective input directory to scan.

    See :func:`_scoped_input_directory` for the rationale behind this
    class. In recursive mode this is a no-op wrapper around
    ``input_dir``; in non-recursive mode it manages the lifetime of a
    temporary staging directory of symlinks.
    """

    def __init__(self, input_dir: Path, *, recursive: bool, extensions: frozenset[str]) -> None:
        self._input_dir = input_dir
        self._recursive = recursive
        self._extensions = extensions
        self._temporary_directory: "Callable[[], None] | None" = None
        self._temp_dir_context = None

    def __enter__(self) -> Path:
        if self._recursive:
            return self._input_dir

        import tempfile

        self._temp_dir_context = tempfile.TemporaryDirectory(
            prefix="fmad_augment_scope_"
        )
        scope_root = Path(self._temp_dir_context.name)

        if self._input_dir.is_dir():
            for entry in sorted(self._input_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in self._extensions:
                    (scope_root / entry.name).symlink_to(entry.resolve())

        return scope_root

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._temp_dir_context is not None:
            self._temp_dir_context.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the augmentation CLI end to end and return a process exit code.

    Args:
        argv: The argument vector to parse, excluding the program
            name. If ``None``, :data:`sys.argv[1:]` is used.

    Returns:
        int: The process exit code: :data:`EXIT_SUCCESS`,
        :data:`EXIT_CONFIGURATION_ERROR`, :data:`EXIT_RUNTIME_ERROR`,
        or :data:`EXIT_INTERRUPTED`.
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_CONFIGURATION_ERROR
        return code

    logger = configure_logging(args.verbose)

    try:
        run(args, logger=logger)
    except KeyboardInterrupt:
        logger.warning("Augmentation run interrupted by user.")
        return EXIT_INTERRUPTED
    except DatasetAugmentationError as exc:
        logger.error("Augmentation run failed: %s", exc)
        return EXIT_RUNTIME_ERROR
    except Exception:  # noqa: BLE001 - top-level CLI safety net
        logger.exception("Augmentation run failed with an unexpected error.")
        return EXIT_RUNTIME_ERROR

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())