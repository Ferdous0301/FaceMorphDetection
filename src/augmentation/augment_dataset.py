"""Dataset augmentation engine for the FMAD Data Augmentation stage.

This module is the central orchestration point for the augmentation
stage of the Face Morphing Attack Detection (FMAD) pipeline. It ties
together the augmentation framework defined in
:mod:`src.augmentation.operators.base` (and the concrete operators
built on top of it) with the filesystem: it recursively discovers
supported images under an input dataset root, loads each image
safely, runs it through a configured chain of
:class:`~src.augmentation.operators.base.BaseAugmentation` operators,
and writes the result to an output dataset root while preserving the
original directory hierarchy.

The module never mutates the input dataset: every read from
``input_dataset_root`` is read-only, and all writes go to
``output_dataset_root``.

Public API
----------
:func:`augment_dataset`
    A high-level, production-quality function that performs a full
    augmentation run over a dataset and returns a
    :class:`DatasetProcessingSummary` describing the outcome. This is
    the function intended for reuse by the CLI and the training
    pipeline.
:class:`DatasetProcessingSummary`
    An immutable dataclass summarising the outcome of a run.
:class:`DatasetAugmentationEngine`
    The underlying engine class that implements the orchestration
    logic. ``augment_dataset`` is a thin convenience wrapper around
    this class; callers that want to run multiple augmentation passes
    while reusing configuration may instantiate the engine directly.

Design notes
------------
* Images are loaded and decoded with OpenCV, converted to RGB channel
  order (to match the channel-order assumptions baked into the
  existing photometric operators, e.g. their use of
  ``cv2.COLOR_RGB2GRAY``), and converted back to BGR immediately
  before being written back out with OpenCV.
* A single bad (corrupted, unreadable, or unwritable) file never
  aborts a run: the failure is logged, the file is skipped, and
  processing continues with the next file.
* Filenames for augmented images are generated deterministically from
  the names of the operators that were actually applied to that
  image. Because every operator's probability gate is itself
  deterministic (seeded), two runs configured with identically seeded
  operators over the same input dataset produce byte-for-byte
  identical output filenames.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

import cv2
import numpy as np

from src.augmentation.operators.base import BaseAugmentation

__all__ = [
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "DEFAULT_FILENAME_TEMPLATE",
    "DatasetAugmentationError",
    "DatasetProcessingSummary",
    "DatasetAugmentationEngine",
    "augment_dataset",
]


_LOGGER_NAME: Final[str] = "src.augmentation.augment_dataset"

#: File extensions (lower-case, including the leading dot) that are
#: recognised as loadable images. Any file whose suffix is not in this
#: set is ignored during dataset discovery.
DEFAULT_SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

#: Default deterministic filename template. ``{stem}`` is the original
#: file's stem (filename without extension), ``{operators}`` is an
#: underscore-joined list of the operator names that were actually
#: applied to the image (or ``"noop"`` if none were applied), and
#: ``{ext}`` is the output file extension (including the leading dot).
DEFAULT_FILENAME_TEMPLATE: Final[str] = "{stem}_aug_{operators}{ext}"

_NOOP_TOKEN: Final[str] = "noop"

#: Mapping from a normalised output format name to the file extension
#: (including the leading dot) that should be used when saving.
_FORMAT_TO_EXTENSION: Final[dict[str, str]] = {
    "png": ".png",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "bmp": ".bmp",
    "tif": ".tif",
    "tiff": ".tiff",
}


class DatasetAugmentationError(Exception):
    """Base exception for errors raised by the dataset augmentation engine.

    Raised only for configuration-time problems that make it
    impossible to perform a run at all (for example, a missing input
    directory). Per-image failures (corrupted files, encode/decode
    errors, and so on) are never raised as exceptions; they are
    logged, counted, and skipped so that a single bad image never
    aborts an entire run.
    """


@dataclass(frozen=True)
class DatasetProcessingSummary:
    """An immutable summary of a single dataset augmentation run.

    Attributes:
        total_images: The total number of supported image files
            discovered under the input dataset root.
        processed_images: The number of discovered images that were
            successfully loaded and run through the operator chain
            (regardless of whether the result was ultimately written
            to disk).
        augmented_images: The number of images for which an augmented
            output file was successfully written (or, in
            :attr:`dry_run` mode, would have been written).
        skipped_images: The number of images that were not processed
            because a corresponding output file already existed and
            ``overwrite`` was ``False``.
        failed_images: The number of images that could not be loaded
            (corrupted or unreadable) or could not be saved, and were
            therefore skipped after logging the failure.
        elapsed_time: The wall-clock time, in seconds, taken by the
            run.
        output_directory: The root directory that augmented images
            were (or, in dry-run mode, would have been) written under.
        dry_run: Whether this run was executed in dry-run mode, in
            which no files were actually written to disk.
        failures: An immutable tuple of ``(source_path, reason)``
            pairs describing every failure encountered during the run,
            in the order they occurred.
    """

    total_images: int
    processed_images: int
    augmented_images: int
    skipped_images: int
    failed_images: int
    elapsed_time: float
    output_directory: Path
    dry_run: bool = False
    failures: tuple[tuple[str, str], ...] = ()


def _normalise_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """Normalise an iterable of file extensions to a lower-case, dotted set.

    Args:
        extensions: An iterable of file extensions, with or without a
            leading dot, in any case.

    Returns:
        frozenset[str]: A frozenset of lower-case extensions, each
        with a leading dot (e.g. ``{".jpg", ".png"}``).
    """
    normalised: set[str] = set()
    for extension in extensions:
        candidate = extension.lower().strip()
        if not candidate.startswith("."):
            candidate = f".{candidate}"
        normalised.add(candidate)
    return frozenset(normalised)


def _discover_images(
    input_root: Path, extensions: frozenset[str]
) -> list[Path]:
    """Recursively discover supported image files under ``input_root``.

    Args:
        input_root: The root directory to scan recursively.
        extensions: The set of lower-case, dotted extensions to treat
            as supported image files.

    Returns:
        list[pathlib.Path]: The discovered image file paths, sorted
        for deterministic ordering across runs and platforms.
    """
    discovered = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(discovered)


def _load_image_as_rgb(path: Path) -> np.ndarray | None:
    """Safely load an image from disk as an RGB ``uint8`` array.

    Args:
        path: The path of the image file to load.

    Returns:
        numpy.ndarray | None: The loaded image as an ``(H, W, 3)``
        ``uint8`` array in RGB channel order, or ``None`` if the file
        could not be decoded (for example, because it is corrupted,
        empty, or not actually an image).
    """
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None

    if raw_bytes.size == 0:
        return None

    decoded_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        return None

    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _save_image_from_rgb(
    path: Path, image_rgb: np.ndarray, *, jpeg_quality: int
) -> bool:
    """Safely save an RGB ``uint8`` image to disk, encoding by file extension.

    Args:
        path: The destination path, including the desired file
            extension.
        image_rgb: The image to save, as an ``(H, W, 3)`` ``uint8``
            array in RGB channel order.
        jpeg_quality: The JPEG quality factor, in ``[1, 100]``, used
            when ``path`` has a JPEG extension. Ignored for other
            formats.

    Returns:
        bool: ``True`` if the image was encoded and written
        successfully, ``False`` otherwise.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    extension = path.suffix.lower()
    encode_params: list[int] = []
    if extension in (".jpg", ".jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    success, encoded_buffer = cv2.imencode(extension, image_bgr, encode_params)
    if not success:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded_buffer.tofile(str(path))
    except OSError:
        return False

    return True


def _resolve_output_extension(
    original_extension: str, output_image_format: str | None
) -> str:
    """Resolve the file extension to use for a saved augmented image.

    Args:
        original_extension: The lower-case, dotted extension of the
            source image (e.g. ``".jpg"``).
        output_image_format: An optional format override (e.g.
            ``"png"``). If ``None``, ``original_extension`` is used
            unchanged.

    Returns:
        str: The lower-case, dotted extension to use for the output
        file.

    Raises:
        DatasetAugmentationError: If ``output_image_format`` is
            provided but is not a recognised image format.
    """
    if output_image_format is None:
        return original_extension

    normalised_format = output_image_format.lower().strip().lstrip(".")
    try:
        return _FORMAT_TO_EXTENSION[normalised_format]
    except KeyError as exc:
        raise DatasetAugmentationError(
            f"Unsupported 'output_image_format' {output_image_format!r}. "
            f"Supported formats are: {sorted(_FORMAT_TO_EXTENSION)}."
        ) from exc


def _build_relative_output_path(
    input_root: Path, output_root: Path, source_path: Path
) -> Path:
    """Compute the output path that mirrors ``source_path``'s location.

    Args:
        input_root: The root directory that ``source_path`` was
            discovered under.
        output_root: The root directory that augmented images are
            written under.
        source_path: The path of the source image, somewhere under
            ``input_root``.

    Returns:
        pathlib.Path: The directory, under ``output_root``, that
        mirrors the directory ``source_path`` lives in under
        ``input_root``.
    """
    relative_directory = source_path.parent.relative_to(input_root)
    return output_root / relative_directory


def _generate_output_filename(
    *,
    stem: str,
    applied_operator_names: Sequence[str],
    extension: str,
    filename_template: str,
) -> str:
    """Generate a deterministic output filename for an augmented image.

    Args:
        stem: The original image's filename stem (without extension).
        applied_operator_names: The names of the operators that were
            actually applied to this image, in application order.
        extension: The output file extension, including the leading
            dot.
        filename_template: A ``str.format``-style template supporting
            the ``{stem}``, ``{operators}``, and ``{ext}`` fields.

    Returns:
        str: The generated, deterministic output filename.
    """
    operators_token = "_".join(applied_operator_names) if applied_operator_names else _NOOP_TOKEN
    return filename_template.format(stem=stem, operators=operators_token, ext=extension)


class DatasetAugmentationEngine:
    """Orchestrates a full augmentation pass over a dataset on disk.

    The engine recursively discovers supported images under an input
    dataset root, loads each one, applies a configured chain of
    augmentation operators, and writes the augmented result to an
    output dataset root, mirroring the input directory hierarchy. The
    input dataset is never modified.

    Instances are stateless between calls to :meth:`run` other than
    their fixed configuration, so a single engine instance may safely
    be reused for multiple runs (for example, over different input
    directories) as long as the same operator instances are
    acceptable to reuse; note, however, that reusing the same operator
    instances across runs continues to draw from their shared random
    state, so filenames will differ between repeated runs unless the
    operators are reconstructed with the same seed beforehand.

    Attributes:
        operators: The ordered sequence of augmentation operators
            applied to every discovered image.
        supported_extensions: The set of lower-case, dotted file
            extensions treated as supported images.
        overwrite: Whether existing output files may be overwritten.
        dry_run: Whether this engine performs a dry run (no files
            written).
        output_image_format: An optional format override applied to
            every saved image.
        jpeg_quality: The JPEG quality factor used when saving in JPEG
            format.
        filename_template: The template used to generate deterministic
            output filenames.
        logger: The logger used for all diagnostic output.
    """

    def __init__(
        self,
        operators: Sequence[BaseAugmentation],
        *,
        supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
        overwrite: bool = False,
        dry_run: bool = False,
        output_image_format: str | None = None,
        jpeg_quality: int = 95,
        filename_template: str = DEFAULT_FILENAME_TEMPLATE,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the augmentation engine with a fixed configuration.

        Args:
            operators: The ordered sequence of augmentation operators
                to apply to every discovered image. May be empty, in
                which case every image is copied through unmodified
                (and its output filename uses the ``"noop"`` token).
            supported_extensions: File extensions treated as supported
                images. Defaults to :data:`DEFAULT_SUPPORTED_EXTENSIONS`.
            overwrite: Whether existing output files may be
                overwritten. When ``False``, an image whose output
                file already exists is skipped.
            dry_run: When ``True``, the engine performs every step
                except actually writing files to disk.
            output_image_format: An optional format override (one of
                ``"png"``, ``"jpg"``/``"jpeg"``, ``"bmp"``,
                ``"tif"``/``"tiff"``) applied to every saved image. If
                ``None``, each image keeps its original extension.
            jpeg_quality: The JPEG quality factor, in ``[1, 100]``,
                used when saving images in JPEG format.
            filename_template: A ``str.format``-style template used to
                generate deterministic output filenames, supporting
                the ``{stem}``, ``{operators}``, and ``{ext}`` fields.
            logger: An optional logger to use. If ``None``, a module
                logger is created.

        Raises:
            DatasetAugmentationError: If ``jpeg_quality`` is outside
                ``[1, 100]``.
        """
        if not (1 <= jpeg_quality <= 100):
            raise DatasetAugmentationError(
                f"'jpeg_quality' must be between 1 and 100 (inclusive), "
                f"got {jpeg_quality!r}."
            )

        self.operators: Sequence[BaseAugmentation] = tuple(operators)
        self.supported_extensions: frozenset[str] = _normalise_extensions(
            supported_extensions
        )
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.output_image_format = output_image_format
        self.jpeg_quality = jpeg_quality
        self.filename_template = filename_template
        self.logger = logger if logger is not None else logging.getLogger(_LOGGER_NAME)

    def run(self, input_dataset_root: Path | str, output_dataset_root: Path | str) -> DatasetProcessingSummary:
        """Execute a full augmentation pass over the input dataset.

        Args:
            input_dataset_root: The root directory of the input
                dataset. Read-only: nothing under this directory is
                ever modified.
            output_dataset_root: The root directory that augmented
                images are written under, mirroring the input
                dataset's directory hierarchy.

        Returns:
            DatasetProcessingSummary: An immutable summary describing
            the outcome of the run.

        Raises:
            DatasetAugmentationError: If ``input_dataset_root`` does
                not exist or is not a directory.
        """
        start_time = time.perf_counter()

        input_root = Path(input_dataset_root)
        output_root = Path(output_dataset_root)

        if not input_root.is_dir():
            raise DatasetAugmentationError(
                f"Input dataset root {input_root!s} does not exist or is not "
                "a directory."
            )

        if not self.dry_run:
            output_root.mkdir(parents=True, exist_ok=True)

        discovered_images = _discover_images(input_root, self.supported_extensions)
        self.logger.info(
            "Discovered %d supported image(s) under %s.",
            len(discovered_images),
            input_root,
        )

        processed_count = 0
        augmented_count = 0
        skipped_count = 0
        failed_count = 0
        failures: list[tuple[str, str]] = []

        for source_path in discovered_images:
            outcome = self._process_single_image(
                input_root=input_root,
                output_root=output_root,
                source_path=source_path,
            )

            if outcome == "processed":
                processed_count += 1
                augmented_count += 1
            elif outcome == "skipped":
                skipped_count += 1
            else:
                failed_count += 1
                failures.append((str(source_path), outcome))

        elapsed_time = time.perf_counter() - start_time

        self.logger.info(
            "Augmentation run complete: %d total, %d processed, %d augmented, "
            "%d skipped, %d failed, %.3fs elapsed.",
            len(discovered_images),
            processed_count,
            augmented_count,
            skipped_count,
            failed_count,
            elapsed_time,
        )

        return DatasetProcessingSummary(
            total_images=len(discovered_images),
            processed_images=processed_count,
            augmented_images=augmented_count,
            skipped_images=skipped_count,
            failed_images=failed_count,
            elapsed_time=elapsed_time,
            output_directory=output_root,
            dry_run=self.dry_run,
            failures=tuple(failures),
        )

    def _process_single_image(
        self, *, input_root: Path, output_root: Path, source_path: Path
    ) -> str:
        """Process a single discovered image end to end.

        Args:
            input_root: The input dataset root.
            output_root: The output dataset root.
            source_path: The path of the image being processed.

        Returns:
            str: ``"processed"`` if the image was successfully
            augmented and (in non-dry-run mode) saved; ``"skipped"``
            if an existing output file was left untouched because
            ``overwrite`` is ``False``; otherwise a short, human
            readable failure reason string describing why the image
            was not processed.
        """
        image = _load_image_as_rgb(source_path)
        if image is None:
            self.logger.warning(
                "Skipping unreadable/corrupted image: %s", source_path
            )
            return "failed to load image"

        output_extension = _resolve_output_extension(
            source_path.suffix.lower(), self.output_image_format
        )
        output_directory = _build_relative_output_path(
            input_root, output_root, source_path
        )

        augmented_image, applied_operator_names = self._apply_operator_chain(
            image, source_path=source_path
        )

        output_filename = _generate_output_filename(
            stem=source_path.stem,
            applied_operator_names=applied_operator_names,
            extension=output_extension,
            filename_template=self.filename_template,
        )
        output_path = output_directory / output_filename

        if not self.overwrite and output_path.exists():
            self.logger.info(
                "Skipping existing output file (overwrite=False): %s",
                output_path,
            )
            return "skipped"

        if self.dry_run:
            self.logger.info("[DRY RUN] Would write augmented image to: %s", output_path)
            return "processed"

        saved = _save_image_from_rgb(
            output_path, augmented_image, jpeg_quality=self.jpeg_quality
        )
        if not saved:
            self.logger.error("Failed to save augmented image to: %s", output_path)
            return "failed to save image"

        self.logger.debug("Saved augmented image: %s", output_path)
        return "processed"

    def _apply_operator_chain(
        self, image: np.ndarray, *, source_path: Path
    ) -> tuple[np.ndarray, list[str]]:
        """Run ``image`` through the configured chain of operators.

        Args:
            image: The loaded, validated source image, in RGB channel
                order.
            source_path: The path the image was loaded from, used
                only for logging context.

        Returns:
            tuple[numpy.ndarray, list[str]]: The image after every
            configured operator has had a chance to apply itself, and
            the ordered list of operator names that were actually
            applied (i.e. whose probability gate passed and which
            completed without error).
        """
        current_image = image
        applied_operator_names: list[str] = []

        for operator in self.operators:
            result = operator.apply(current_image)

            if not result.success:
                self.logger.warning(
                    "Operator %r failed on %s: %s",
                    operator.operator_name,
                    source_path,
                    result.error_message,
                )
                continue

            current_image = result.image
            if result.applied:
                applied_operator_names.append(operator.operator_name)

        return current_image, applied_operator_names


def augment_dataset(
    input_dataset_root: Path | str,
    output_dataset_root: Path | str,
    operators: Sequence[BaseAugmentation],
    *,
    supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    overwrite: bool = False,
    dry_run: bool = False,
    output_image_format: str | None = None,
    jpeg_quality: int = 95,
    filename_template: str = DEFAULT_FILENAME_TEMPLATE,
    logger: logging.Logger | None = None,
) -> DatasetProcessingSummary:
    """Augment every supported image in a dataset, preserving its structure.

    This is the high-level, public entry point into the dataset
    augmentation engine, intended for reuse by the CLI and by the
    training pipeline. It recursively scans ``input_dataset_root`` for
    supported images, applies ``operators`` (in order) to each one,
    and writes the results under ``output_dataset_root`` at the same
    relative path as the source image. The input dataset is never
    modified.

    Args:
        input_dataset_root: The root directory of the input dataset.
            Read-only.
        output_dataset_root: The root directory that augmented images
            are written under. Created automatically if it does not
            already exist (unless ``dry_run`` is ``True``).
        operators: The ordered sequence of augmentation operators to
            apply to every discovered image. May be empty.
        supported_extensions: File extensions treated as supported
            images. Defaults to :data:`DEFAULT_SUPPORTED_EXTENSIONS`
            (``.jpg``, ``.jpeg``, ``.png``, ``.bmp``, ``.tif``,
            ``.tiff``).
        overwrite: Whether existing output files may be overwritten.
            When ``False`` (the default), an image whose output file
            already exists is left untouched and counted as skipped.
        dry_run: When ``True``, every step is performed except
            actually writing files to disk; useful for previewing the
            outcome (including the exact output filenames) of a run.
        output_image_format: An optional format override (one of
            ``"png"``, ``"jpg"``/``"jpeg"``, ``"bmp"``,
            ``"tif"``/``"tiff"``) applied to every saved image. If
            ``None`` (the default), each image keeps its original
            file extension.
        jpeg_quality: The JPEG quality factor, in ``[1, 100]``, used
            when saving images in JPEG format.
        filename_template: A ``str.format``-style template used to
            generate deterministic output filenames, supporting the
            ``{stem}``, ``{operators}``, and ``{ext}`` fields. Defaults
            to :data:`DEFAULT_FILENAME_TEMPLATE`.
        logger: An optional logger to use for diagnostic output. If
            ``None``, a module-level logger is used.

    Returns:
        DatasetProcessingSummary: An immutable summary of the run,
        including per-category image counts and the elapsed time.

    Raises:
        DatasetAugmentationError: If ``input_dataset_root`` does not
            exist or is not a directory, if ``jpeg_quality`` is
            outside ``[1, 100]``, or if ``output_image_format`` is
            provided but not a recognised image format.
    """
    engine = DatasetAugmentationEngine(
        operators,
        supported_extensions=supported_extensions,
        overwrite=overwrite,
        dry_run=dry_run,
        output_image_format=output_image_format,
        jpeg_quality=jpeg_quality,
        filename_template=filename_template,
        logger=logger,
    )
    return engine.run(input_dataset_root, output_dataset_root)