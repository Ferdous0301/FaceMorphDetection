"""Configuration module for the FMAD Data Augmentation stage.

This module defines the complete, immutable-by-convention configuration
surface for the augmentation stage of the Face Morphing Attack Detection
(FMAD) pipeline. All configuration is expressed as ``dataclasses`` with
full type hints, sensible defaults, and validation performed eagerly in
``__post_init__`` so that invalid configurations fail fast, before any
image processing begins.

The module exposes four public classes:

* :class:`OperatorConfig` — generic, reusable configuration for a single
  augmentation operator (e.g. horizontal flip, gaussian blur), including
  its enabled state, application probability, and numeric parameter
  ranges.
* :class:`LoggingConfig` — configuration for the augmentation stage's
  logging behaviour (console/file handlers, log level, rotation policy).
* :class:`StatisticsConfig` — configuration for the statistics report
  produced at the end of an augmentation run.
* :class:`AugmentationConfig` — the top-level configuration object that
  composes all of the above along with dataset paths, execution
  parameters, and per-family enable flags.

No I/O, image processing, or execution logic lives in this module by
design: it is pure configuration and validation, which keeps it fast to
import, trivial to unit test, and safe to reuse across the CLI, tests,
and notebooks-free scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ConfigurationError",
    "OperatorConfig",
    "LoggingConfig",
    "StatisticsConfig",
    "AugmentationConfig",
]


class ConfigurationError(ValueError):
    """Raised when an augmentation configuration value fails validation.

    This exception is a specialised :class:`ValueError` so that callers
    relying on standard ``ValueError`` handling continue to work, while
    still allowing callers who want augmentation-specific error handling
    to catch this type explicitly.
    """


def _ensure_probability(value: float, *, field_name: str) -> None:
    """Validate that ``value`` is a legal probability in the range ``[0.0, 1.0]``.

    Args:
        value: The probability value to validate.
        field_name: The name of the field being validated, used to build
            a descriptive error message.

    Raises:
        ConfigurationError: If ``value`` is not within ``[0.0, 1.0]``.
    """
    if not (0.0 <= value <= 1.0):
        raise ConfigurationError(
            f"'{field_name}' must be between 0.0 and 1.0 (inclusive), got {value!r}."
        )


def _ensure_non_negative_int(value: int, *, field_name: str) -> None:
    """Validate that ``value`` is a non-negative integer.

    Args:
        value: The integer value to validate.
        field_name: The name of the field being validated, used to build
            a descriptive error message.

    Raises:
        ConfigurationError: If ``value`` is negative.
    """
    if value < 0:
        raise ConfigurationError(
            f"'{field_name}' must be a non-negative integer, got {value!r}."
        )


def _ensure_positive_int(value: int, *, field_name: str) -> None:
    """Validate that ``value`` is a strictly positive integer.

    Args:
        value: The integer value to validate.
        field_name: The name of the field being validated, used to build
            a descriptive error message.

    Raises:
        ConfigurationError: If ``value`` is less than or equal to zero.
    """
    if value <= 0:
        raise ConfigurationError(
            f"'{field_name}' must be a positive integer greater than 0, got {value!r}."
        )


@dataclass
class OperatorConfig:
    """Generic, reusable configuration for a single augmentation operator.

    An "operator" is one concrete augmentation transform, such as
    horizontal flip, small rotation, brightness adjustment, gaussian
    noise, gaussian blur, or JPEG compression simulation. Every operator
    in the augmentation stage shares the same three configuration
    concerns — whether it is enabled, how likely it is to be applied,
    and the numeric ranges its randomised parameters may be sampled
    from — which this class captures generically so that new operators
    can be configured without introducing a new dataclass.

    Attributes:
        enabled: Whether this specific operator is eligible for
            selection by the sampler. Disabled operators are never
            applied, regardless of their probability or parent family's
            enabled state.
        probability: The probability, in ``[0.0, 1.0]``, that this
            operator is applied to a given augmented copy of an image,
            conditional on the operator being eligible for selection.
        parameters: A mapping from parameter name to an inclusive
            ``(minimum, maximum)`` range that the operator's randomised
            parameter(s) are sampled from. For example, a rotation
            operator might use ``{"degrees": (-5.0, 5.0)}`` and a JPEG
            compression operator might use ``{"quality": (60.0, 90.0)}``.
            Every range must satisfy ``minimum <= maximum``.
    """

    enabled: bool = True
    probability: float = 0.5
    parameters: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate probability and parameter ranges after initialisation.

        Raises:
            ConfigurationError: If ``probability`` is outside
                ``[0.0, 1.0]``, or if any entry in ``parameters`` is not
                a valid ``(minimum, maximum)`` pair with
                ``minimum <= maximum``.
        """
        _ensure_probability(self.probability, field_name="probability")

        for parameter_name, value_range in self.parameters.items():
            if len(value_range) != 2:
                raise ConfigurationError(
                    f"Parameter range for '{parameter_name}' must be a "
                    f"(minimum, maximum) pair, got {value_range!r}."
                )
            minimum, maximum = value_range
            if minimum > maximum:
                raise ConfigurationError(
                    f"Parameter range for '{parameter_name}' is invalid: "
                    f"minimum ({minimum!r}) exceeds maximum ({maximum!r})."
                )


@dataclass
class LoggingConfig:
    """Configuration for the augmentation stage's logging behaviour.

    Attributes:
        log_dir: Directory where log files are written. Converted to a
            :class:`pathlib.Path` if a string is supplied.
        log_level: The minimum severity of messages to emit, expressed
            as a standard ``logging`` level name (e.g. ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
        console_logging: Whether log messages are emitted to the
            console (stdout/stderr) in addition to any file handlers.
        file_logging: Whether log messages are written to a rotating
            log file inside ``log_dir``.
        log_filename: The filename used for the rotating log file,
            relative to ``log_dir``.
        max_bytes: The maximum size, in bytes, of a single log file
            before it is rotated. Must be a positive integer.
        backup_count: The number of rotated log file backups to retain.
            Must be a non-negative integer.
    """

    log_dir: Path
    log_level: str = "INFO"
    console_logging: bool = True
    file_logging: bool = True
    log_filename: str = "augmentation.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5

    _VALID_LOG_LEVELS: tuple[str, ...] = field(
        default=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalise and validate logging configuration values.

        Raises:
            ConfigurationError: If ``log_level`` is not a recognised
                logging level name, if ``max_bytes`` is not positive,
                or if ``backup_count`` is negative.
        """
        self.log_dir = Path(self.log_dir)

        normalised_level = self.log_level.upper()
        if normalised_level not in self._VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"'log_level' must be one of {self._VALID_LOG_LEVELS}, "
                f"got {self.log_level!r}."
            )
        self.log_level = normalised_level

        _ensure_positive_int(self.max_bytes, field_name="max_bytes")
        _ensure_non_negative_int(self.backup_count, field_name="backup_count")


@dataclass
class StatisticsConfig:
    """Configuration for the statistics report produced by an augmentation run.

    Attributes:
        enabled: Whether statistics collection and reporting is
            performed at all. When ``False``, the executor should skip
            statistics accumulation entirely.
        output_dir: Directory where statistics report files are
            written. Converted to a :class:`pathlib.Path` if a string
            is supplied.
        output_json_filename: The filename used for the JSON statistics
            report, relative to ``output_dir``.
        output_csv_filename: The filename used for the flattened CSV
            statistics report, relative to ``output_dir``.
        track_per_operator_counts: Whether the statistics report
            includes a breakdown of how many times each operator was
            applied.
        track_per_class_counts: Whether the statistics report includes
            a breakdown of augmented image counts per class label
            (e.g. ``bonafide`` vs ``morph``).
        track_per_dataset_counts: Whether the statistics report
            includes a breakdown of augmented image counts per source
            dataset.
    """

    output_dir: Path
    enabled: bool = True
    output_json_filename: str = "augmentation_stats.json"
    output_csv_filename: str = "augmentation_stats.csv"
    track_per_operator_counts: bool = True
    track_per_class_counts: bool = True
    track_per_dataset_counts: bool = True

    def __post_init__(self) -> None:
        """Normalise the output directory to a :class:`pathlib.Path`."""
        self.output_dir = Path(self.output_dir)


@dataclass
class AugmentationConfig:
    """Top-level configuration for the FMAD Data Augmentation stage.

    This class composes dataset paths, execution parameters, per-family
    enable flags, per-operator configurations, and the logging and
    statistics sub-configurations into a single, validated object that
    fully specifies one augmentation run.

    Attributes:
        input_dataset_root: Root directory containing the input images
            referenced by the manifests. Converted to a
            :class:`pathlib.Path` if a string is supplied.
        output_dataset_root: Root directory under which augmented
            images (and, optionally, copied originals) are written.
            Converted to a :class:`pathlib.Path` if a string is
            supplied.
        train_manifest_path: Path to the training split manifest CSV
            produced by the Dataset Split stage. This is the only
            manifest whose records are eligible for augmentation.
        validation_manifest_path: Path to the validation split manifest
            CSV. Records in this manifest are never augmented.
        test_manifest_path: Path to the test split manifest CSV.
            Records in this manifest are never augmented.
        output_manifest_path: Path where the final merged manifest
            (original validation/test rows plus original and augmented
            training rows) is written.
        seed: The master random seed used to derive deterministic
            per-record seeds. Must be non-negative.
        deterministic: Whether the augmentation stage must produce
            byte-identical output when re-run with the same manifests,
            configuration, and seed. When ``True``, execution must not
            depend on multiprocessing worker count or processing order.
        overwrite_existing: Whether existing files in
            ``output_dataset_root`` may be overwritten. When ``False``,
            the executor should refuse to run if the output directory
            already contains files.
        copy_original_images: Whether original (non-augmented) training
            images are copied into the output dataset alongside the
            augmented copies, so that the output directory is
            self-contained.
        output_image_format: The file format used when saving augmented
            images, one of ``"png"``, ``"jpg"``, or ``"jpeg"``.
        jpeg_quality: The JPEG quality factor, in ``[1, 100]``, used
            when saving images in ``"jpg"``/``"jpeg"`` format. This is
            distinct from the JPEG compression *simulation* operator's
            quality parameter range, which models compression as an
            augmentation artifact rather than a save-time encoding
            setting.
        augmentations_per_image: The number of augmented copies
            generated per eligible training image. Must be a positive
            integer.
        geometric_enabled: Family-level toggle for geometric operators
            (horizontal flip, small rotation). When ``False``, all
            geometric operators are skipped regardless of their
            individual ``enabled`` flags.
        photometric_enabled: Family-level toggle for photometric
            operators (brightness, contrast, gamma, color jitter).
        blur_enabled: Family-level toggle for the gaussian blur
            operator.
        noise_enabled: Family-level toggle for the gaussian noise
            operator.
        compression_enabled: Family-level toggle for the JPEG
            compression simulation operator.
        horizontal_flip: Configuration for the horizontal flip
            operator (geometric family).
        rotation: Configuration for the small rotation operator
            (geometric family).
        brightness: Configuration for the brightness adjustment
            operator (photometric family).
        contrast: Configuration for the contrast adjustment operator
            (photometric family).
        gamma: Configuration for the gamma adjustment operator
            (photometric family).
        color_jitter: Configuration for the color jitter operator
            (photometric family).
        gaussian_noise: Configuration for the gaussian noise operator
            (noise family).
        gaussian_blur: Configuration for the gaussian blur operator
            (blur family).
        jpeg_compression: Configuration for the JPEG compression
            simulation operator (compression family).
        logging: Logging configuration for the augmentation stage.
        statistics: Statistics reporting configuration for the
            augmentation stage.
    """

    input_dataset_root: Path
    output_dataset_root: Path
    train_manifest_path: Path
    validation_manifest_path: Path
    test_manifest_path: Path
    output_manifest_path: Path
    logging: LoggingConfig
    statistics: StatisticsConfig

    seed: int = 42
    deterministic: bool = True
    overwrite_existing: bool = False
    copy_original_images: bool = True
    output_image_format: str = "png"
    jpeg_quality: int = 95
    augmentations_per_image: int = 2

    geometric_enabled: bool = True
    photometric_enabled: bool = True
    blur_enabled: bool = True
    noise_enabled: bool = True
    compression_enabled: bool = True

    horizontal_flip: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.5,
            parameters={},
        )
    )
    rotation: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.5,
            parameters={"degrees": (-5.0, 5.0)},
        )
    )
    brightness: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.5,
            parameters={"factor": (0.85, 1.15)},
        )
    )
    contrast: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.5,
            parameters={"factor": (0.85, 1.15)},
        )
    )
    gamma: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={"gamma": (0.85, 1.15)},
        )
    )
    color_jitter: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={
                "hue_shift": (-0.02, 0.02),
                "saturation_factor": (0.9, 1.1),
            },
        )
    )
    gaussian_noise: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={"sigma": (2.0, 8.0)},
        )
    )
    gaussian_blur: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={"sigma": (0.3, 1.2)},
        )
    )
    jpeg_compression: OperatorConfig = field(
        default_factory=lambda: OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={"quality": (60.0, 90.0)},
        )
    )

    _VALID_IMAGE_FORMATS: tuple[str, ...] = field(
        default=("png", "jpg", "jpeg"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalise paths and validate the full configuration.

        Performs the following, in order:

        1. Converts every path-like field to :class:`pathlib.Path`.
        2. Validates ``seed`` is non-negative.
        3. Validates ``jpeg_quality`` is within ``[1, 100]``.
        4. Validates ``augmentations_per_image`` is a positive integer.
        5. Validates ``output_image_format`` is a supported format.

        Raises:
            ConfigurationError: If any validated field holds an
                out-of-range or unsupported value.
        """
        self.input_dataset_root = Path(self.input_dataset_root)
        self.output_dataset_root = Path(self.output_dataset_root)
        self.train_manifest_path = Path(self.train_manifest_path)
        self.validation_manifest_path = Path(self.validation_manifest_path)
        self.test_manifest_path = Path(self.test_manifest_path)
        self.output_manifest_path = Path(self.output_manifest_path)

        _ensure_non_negative_int(self.seed, field_name="seed")

        if not (1 <= self.jpeg_quality <= 100):
            raise ConfigurationError(
                f"'jpeg_quality' must be between 1 and 100 (inclusive), "
                f"got {self.jpeg_quality!r}."
            )

        _ensure_positive_int(
            self.augmentations_per_image, field_name="augmentations_per_image"
        )

        normalised_format = self.output_image_format.lower()
        if normalised_format not in self._VALID_IMAGE_FORMATS:
            raise ConfigurationError(
                f"'output_image_format' must be one of "
                f"{self._VALID_IMAGE_FORMATS}, got {self.output_image_format!r}."
            )
        self.output_image_format = normalised_format