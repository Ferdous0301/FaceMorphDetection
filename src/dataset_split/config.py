"""Immutable configuration objects for the Dataset Split stage.

This module defines frozen dataclasses that hold all configuration
required to run the dataset split stage of the face morphing attack
detection pipeline. Configuration objects validate their own invariants
upon construction but contain no split, verification, or statistics
logic themselves.

All directory-like fields are normalized to :class:`pathlib.Path`
instances, and all configuration objects are immutable (frozen) to
prevent accidental mutation after construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset_split.exceptions import (
    InvalidConfigurationError,
    InvalidSplitRatioError,
)

#: Absolute tolerance used when validating that split ratios sum to 1.0.
_RATIO_SUM_TOLERANCE: float = 1e-6


def _to_path(value: Path | str) -> Path:
    """Convert a string or Path value into a normalized ``pathlib.Path``.

    Args:
        value: A filesystem path expressed as a string or Path instance.

    Returns:
        The corresponding :class:`pathlib.Path` instance.

    Raises:
        InvalidConfigurationError: If ``value`` is not a str or Path.
    """
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise InvalidConfigurationError(
        f"Expected a str or pathlib.Path, got {type(value).__name__!r}."
    )


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Configuration governing how the dataset is partitioned into splits.

    Attributes:
        train_ratio: Fraction of the dataset allocated to the training
            split. Must be non-negative.
        val_ratio: Fraction of the dataset allocated to the validation
            split. Must be non-negative.
        test_ratio: Fraction of the dataset allocated to the test split.
            Must be non-negative.
        random_seed: Seed used for any deterministic random operations
            performed during splitting (e.g. shuffling).
        shuffle: Whether the dataset should be shuffled prior to
            splitting.
        output_directory: Directory into which split artifacts are
            written.
        manifest_directory: Directory containing or receiving dataset
            manifests.
        statistics_directory: Directory into which split statistics are
            written.

    Raises:
        InvalidSplitRatioError: If any ratio is negative or if the
            ratios do not sum to 1.0 within tolerance.
        InvalidConfigurationError: If a directory field cannot be
            converted to a ``pathlib.Path``.
    """

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    shuffle: bool = True
    output_directory: Path = field(default_factory=lambda: Path("output"))
    manifest_directory: Path = field(default_factory=lambda: Path("manifests"))
    statistics_directory: Path = field(
        default_factory=lambda: Path("statistics")
    )

    def __post_init__(self) -> None:
        """Validate and normalize field values after initialization.

        Raises:
            InvalidSplitRatioError: If any ratio is negative or the
                ratios do not sum to 1.0 within tolerance.
            InvalidConfigurationError: If a directory field cannot be
                converted to a ``pathlib.Path``.
        """
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            ratio_value = getattr(self, name)
            if ratio_value < 0:
                raise InvalidSplitRatioError(
                    f"{name} must be non-negative, got {ratio_value!r}."
                )

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(
            ratio_sum, 1.0, abs_tol=_RATIO_SUM_TOLERANCE
        ):
            raise InvalidSplitRatioError(
                "train_ratio, val_ratio, and test_ratio must sum to 1.0 "
                f"(within tolerance {_RATIO_SUM_TOLERANCE}); got sum "
                f"{ratio_sum!r}."
            )

        for name in (
            "output_directory",
            "manifest_directory",
            "statistics_directory",
        ):
            raw_value = getattr(self, name)
            object.__setattr__(self, name, _to_path(raw_value))


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration governing post-split verification checks.

    Attributes:
        check_identity_leakage: Whether to verify that no identity
            appears across multiple splits.
        check_duplicate_images: Whether to verify that no duplicate
            images exist within or across splits.
        check_missing_metadata: Whether to verify that all required
            metadata fields are present for every sample.
        fail_fast: Whether verification should stop at the first
            detected violation rather than collecting all violations.

    Raises:
        InvalidConfigurationError: If none of the verification checks
            are enabled.
    """

    check_identity_leakage: bool = True
    check_duplicate_images: bool = True
    check_missing_metadata: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        """Validate that at least one verification check is enabled.

        Raises:
            InvalidConfigurationError: If all verification checks are
                disabled, making the verification stage a no-op.
        """
        if not (
            self.check_identity_leakage
            or self.check_duplicate_images
            or self.check_missing_metadata
        ):
            raise InvalidConfigurationError(
                "At least one verification check must be enabled."
            )


@dataclass(frozen=True)
class StatisticsConfig:
    """Configuration governing computation and export of split statistics.

    Attributes:
        compute_class_balance: Whether to compute per-split class
            balance statistics.
        compute_identity_counts: Whether to compute per-split identity
            counts.
        export_format: File format used when exporting statistics
            (e.g. "json" or "csv").
        output_directory: Directory into which statistics files are
            written.

    Raises:
        InvalidConfigurationError: If ``export_format`` is not one of
            the supported formats, or if the output directory cannot be
            converted to a ``pathlib.Path``.
    """

    compute_class_balance: bool = True
    compute_identity_counts: bool = True
    export_format: str = "json"
    output_directory: Path = field(
        default_factory=lambda: Path("statistics")
    )

    _SUPPORTED_FORMATS: tuple[str, ...] = field(
        default=("json", "csv"), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate and normalize field values after initialization.

        Raises:
            InvalidConfigurationError: If ``export_format`` is
                unsupported or ``output_directory`` cannot be converted
                to a ``pathlib.Path``.
        """
        if self.export_format not in self._SUPPORTED_FORMATS:
            raise InvalidConfigurationError(
                f"export_format must be one of {self._SUPPORTED_FORMATS}, "
                f"got {self.export_format!r}."
            )
        object.__setattr__(
            self, "output_directory", _to_path(self.output_directory)
        )