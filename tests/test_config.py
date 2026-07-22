"""Unit tests for the Dataset Split stage configuration dataclasses.

These tests verify:
    * Valid configurations construct successfully.
    * Invalid ratios (negative, non-summing) raise the correct exceptions.
    * Directory fields are normalized to ``pathlib.Path``.
    * All configuration dataclasses are immutable.
    * Default values are deterministic and correct.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.dataset_split.config import (
    DatasetSplitConfig,
    StatisticsConfig,
    VerificationConfig,
)
from src.dataset_split.exceptions import (
    InvalidConfigurationError,
    InvalidSplitRatioError,
)


class TestDatasetSplitConfigValid:
    """Tests for valid DatasetSplitConfig construction."""

    def test_default_construction_succeeds(self) -> None:
        """Default construction succeeds with deterministic defaults."""
        config = DatasetSplitConfig()
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.random_seed == 42
        assert config.shuffle is True

    def test_custom_valid_ratios(self) -> None:
        """Custom ratios that sum to 1.0 construct successfully."""
        config = DatasetSplitConfig(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert config.train_ratio == 0.6
        assert config.val_ratio == 0.2
        assert config.test_ratio == 0.2

    def test_ratio_sum_within_tolerance_succeeds(self) -> None:
        """Ratios summing to 1.0 within floating point tolerance succeed."""
        config = DatasetSplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15000001
        )
        assert config.train_ratio == 0.7

    def test_zero_ratio_is_valid(self) -> None:
        """A zero ratio is valid as long as ratios still sum to 1.0."""
        config = DatasetSplitConfig(
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        assert config.val_ratio == 0.0
        assert config.test_ratio == 0.0


class TestDatasetSplitConfigInvalidRatios:
    """Tests for invalid ratio handling in DatasetSplitConfig."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_ratio": -0.1, "val_ratio": 0.5, "test_ratio": 0.6},
            {"train_ratio": 0.5, "val_ratio": -0.5, "test_ratio": 1.0},
            {"train_ratio": 0.5, "val_ratio": 0.5, "test_ratio": -0.1},
        ],
    )
    def test_negative_ratio_raises(self, kwargs: dict[str, float]) -> None:
        """Any negative ratio raises InvalidSplitRatioError."""
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitConfig(**kwargs)

    def test_ratio_sum_too_low_raises(self) -> None:
        """Ratios summing to less than 1.0 raise InvalidSplitRatioError."""
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitConfig(
                train_ratio=0.5, val_ratio=0.2, test_ratio=0.2
            )

    def test_ratio_sum_too_high_raises(self) -> None:
        """Ratios summing to more than 1.0 raise InvalidSplitRatioError."""
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitConfig(
                train_ratio=0.6, val_ratio=0.3, test_ratio=0.3
            )

    def test_error_message_mentions_offending_field(self) -> None:
        """The error message identifies which ratio was negative."""
        with pytest.raises(InvalidSplitRatioError, match="train_ratio"):
            DatasetSplitConfig(
                train_ratio=-0.5, val_ratio=0.75, test_ratio=0.75
            )


class TestDatasetSplitConfigPathConversion:
    """Tests for directory field normalization to pathlib.Path."""

    def test_default_directories_are_paths(self) -> None:
        """Default directory fields are pathlib.Path instances."""
        config = DatasetSplitConfig()
        assert isinstance(config.output_directory, Path)
        assert isinstance(config.manifest_directory, Path)
        assert isinstance(config.statistics_directory, Path)

    def test_string_directories_are_converted(self) -> None:
        """String directory arguments are converted to pathlib.Path."""
        config = DatasetSplitConfig(
            output_directory="out",
            manifest_directory="manifests",
            statistics_directory="stats",
        )
        assert config.output_directory == Path("out")
        assert config.manifest_directory == Path("manifests")
        assert config.statistics_directory == Path("stats")

    def test_path_directories_remain_paths(self) -> None:
        """Path directory arguments remain pathlib.Path instances."""
        config = DatasetSplitConfig(output_directory=Path("custom_out"))
        assert config.output_directory == Path("custom_out")


class TestDatasetSplitConfigImmutability:
    """Tests verifying DatasetSplitConfig is immutable."""

    def test_is_frozen_dataclass(self) -> None:
        """DatasetSplitConfig is declared as a frozen dataclass."""
        assert dataclasses.fields(DatasetSplitConfig)
        with pytest.raises(dataclasses.FrozenInstanceError):
            DatasetSplitConfig().train_ratio = 0.5  # type: ignore[misc]

    def test_cannot_mutate_directory_field(self) -> None:
        """Attempting to mutate a directory field raises an error."""
        config = DatasetSplitConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.output_directory = Path("new_dir")  # type: ignore[misc]


class TestVerificationConfig:
    """Tests for VerificationConfig."""

    def test_default_construction_succeeds(self) -> None:
        """Default construction enables all checks."""
        config = VerificationConfig()
        assert config.check_identity_leakage is True
        assert config.check_duplicate_images is True
        assert config.check_missing_metadata is True
        assert config.fail_fast is False

    def test_single_enabled_check_is_valid(self) -> None:
        """A config with only one check enabled is valid."""
        config = VerificationConfig(
            check_identity_leakage=True,
            check_duplicate_images=False,
            check_missing_metadata=False,
        )
        assert config.check_identity_leakage is True

    def test_all_checks_disabled_raises(self) -> None:
        """Disabling all verification checks raises InvalidConfigurationError."""
        with pytest.raises(InvalidConfigurationError):
            VerificationConfig(
                check_identity_leakage=False,
                check_duplicate_images=False,
                check_missing_metadata=False,
            )

    def test_is_immutable(self) -> None:
        """VerificationConfig fields cannot be reassigned after creation."""
        config = VerificationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.fail_fast = True  # type: ignore[misc]


class TestStatisticsConfig:
    """Tests for StatisticsConfig."""

    def test_default_construction_succeeds(self) -> None:
        """Default construction succeeds with expected defaults."""
        config = StatisticsConfig()
        assert config.compute_class_balance is True
        assert config.compute_identity_counts is True
        assert config.export_format == "json"
        assert isinstance(config.output_directory, Path)

    def test_csv_export_format_is_valid(self) -> None:
        """The 'csv' export format is accepted."""
        config = StatisticsConfig(export_format="csv")
        assert config.export_format == "csv"

    def test_unsupported_export_format_raises(self) -> None:
        """An unsupported export format raises InvalidConfigurationError."""
        with pytest.raises(InvalidConfigurationError):
            StatisticsConfig(export_format="xml")

    def test_string_directory_is_converted_to_path(self) -> None:
        """String output_directory is converted to pathlib.Path."""
        config = StatisticsConfig(output_directory="my_stats")
        assert config.output_directory == Path("my_stats")

    def test_is_immutable(self) -> None:
        """StatisticsConfig fields cannot be reassigned after creation."""
        config = StatisticsConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.export_format = "csv"  # type: ignore[misc]


class TestDefaultsAreDeterministic:
    """Tests ensuring default values are stable and deterministic."""

    def test_dataset_split_config_defaults_are_stable(self) -> None:
        """Two default DatasetSplitConfig instances are equal."""
        assert DatasetSplitConfig() == DatasetSplitConfig()

    def test_verification_config_defaults_are_stable(self) -> None:
        """Two default VerificationConfig instances are equal."""
        assert VerificationConfig() == VerificationConfig()

    def test_statistics_config_defaults_are_stable(self) -> None:
        """Two default StatisticsConfig instances are equal."""
        assert StatisticsConfig() == StatisticsConfig()