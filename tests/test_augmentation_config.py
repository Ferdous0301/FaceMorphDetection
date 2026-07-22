"""Unit tests for :mod:`src.augmentation.config`.

These tests validate the dataclass defaults, ``Path`` normalisation, and
the eager validation performed in ``__post_init__`` for every
configuration class in the augmentation stage's configuration module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.augmentation.config import (
    AugmentationConfig,
    ConfigurationError,
    LoggingConfig,
    OperatorConfig,
    StatisticsConfig,
)


def _build_logging_config(tmp_path: Path) -> LoggingConfig:
    """Construct a minimal, valid :class:`LoggingConfig` for tests."""
    return LoggingConfig(log_dir=tmp_path / "logs")


def _build_statistics_config(tmp_path: Path) -> StatisticsConfig:
    """Construct a minimal, valid :class:`StatisticsConfig` for tests."""
    return StatisticsConfig(output_dir=tmp_path / "stats")


def _build_valid_augmentation_config(tmp_path: Path) -> AugmentationConfig:
    """Construct a fully valid, default-heavy :class:`AugmentationConfig`."""
    return AugmentationConfig(
        input_dataset_root=tmp_path / "input",
        output_dataset_root=tmp_path / "output",
        train_manifest_path=tmp_path / "splits" / "train.csv",
        validation_manifest_path=tmp_path / "splits" / "val.csv",
        test_manifest_path=tmp_path / "splits" / "test.csv",
        output_manifest_path=tmp_path / "output" / "final_manifest.csv",
        logging=_build_logging_config(tmp_path),
        statistics=_build_statistics_config(tmp_path),
    )


# ---------------------------------------------------------------------------
# OperatorConfig
# ---------------------------------------------------------------------------


class TestOperatorConfig:
    """Tests for :class:`OperatorConfig`."""

    def test_default_values(self) -> None:
        """Default OperatorConfig is enabled, has probability 0.5, and no parameters."""
        config = OperatorConfig()

        assert config.enabled is True
        assert config.probability == 0.5
        assert config.parameters == {}

    def test_valid_configuration_with_parameters(self) -> None:
        """A valid probability and well-formed parameter ranges construct cleanly."""
        config = OperatorConfig(
            enabled=True,
            probability=0.3,
            parameters={"degrees": (-5.0, 5.0), "sigma": (0.5, 1.5)},
        )

        assert config.probability == 0.3
        assert config.parameters["degrees"] == (-5.0, 5.0)
        assert config.parameters["sigma"] == (0.5, 1.5)

    @pytest.mark.parametrize("probability", [-0.01, -1.0, 1.01, 2.0, 100.0])
    def test_invalid_probability_raises(self, probability: float) -> None:
        """Probabilities outside [0.0, 1.0] raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="probability"):
            OperatorConfig(probability=probability)

    @pytest.mark.parametrize("probability", [0.0, 1.0, 0.5])
    def test_boundary_probabilities_are_valid(self, probability: float) -> None:
        """Boundary probability values 0.0 and 1.0 are accepted (inclusive range)."""
        config = OperatorConfig(probability=probability)

        assert config.probability == probability

    def test_invalid_parameter_range_min_greater_than_max_raises(self) -> None:
        """A parameter range with minimum > maximum raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="degrees"):
            OperatorConfig(parameters={"degrees": (5.0, -5.0)})

    def test_invalid_parameter_range_wrong_length_raises(self) -> None:
        """A parameter range that is not a 2-tuple raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="sigma"):
            OperatorConfig(parameters={"sigma": (0.1, 0.2, 0.3)})  # type: ignore[arg-type]

    def test_equal_min_max_parameter_range_is_valid(self) -> None:
        """A degenerate parameter range where minimum == maximum is accepted."""
        config = OperatorConfig(parameters={"quality": (75.0, 75.0)})

        assert config.parameters["quality"] == (75.0, 75.0)


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Tests for :class:`LoggingConfig`."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Default LoggingConfig has expected level, filename, and rotation policy."""
        config = LoggingConfig(log_dir=tmp_path / "logs")

        assert config.log_level == "INFO"
        assert config.console_logging is True
        assert config.file_logging is True
        assert config.log_filename == "augmentation.log"
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5

    def test_log_dir_converted_to_path(self, tmp_path: Path) -> None:
        """A string log_dir is converted to a pathlib.Path instance."""
        config = LoggingConfig(log_dir=str(tmp_path / "logs"))

        assert isinstance(config.log_dir, Path)
        assert config.log_dir == tmp_path / "logs"

    def test_log_level_is_normalised_to_uppercase(self, tmp_path: Path) -> None:
        """A lowercase log_level string is normalised to uppercase."""
        config = LoggingConfig(log_dir=tmp_path / "logs", log_level="debug")

        assert config.log_level == "DEBUG"

    def test_invalid_log_level_raises(self, tmp_path: Path) -> None:
        """An unrecognised log_level raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="log_level"):
            LoggingConfig(log_dir=tmp_path / "logs", log_level="VERBOSE")

    def test_invalid_max_bytes_raises(self, tmp_path: Path) -> None:
        """A non-positive max_bytes raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_bytes"):
            LoggingConfig(log_dir=tmp_path / "logs", max_bytes=0)

    def test_invalid_backup_count_raises(self, tmp_path: Path) -> None:
        """A negative backup_count raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="backup_count"):
            LoggingConfig(log_dir=tmp_path / "logs", backup_count=-1)


# ---------------------------------------------------------------------------
# StatisticsConfig
# ---------------------------------------------------------------------------


class TestStatisticsConfig:
    """Tests for :class:`StatisticsConfig`."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Default StatisticsConfig is enabled with expected filenames and trackers."""
        config = StatisticsConfig(output_dir=tmp_path / "stats")

        assert config.enabled is True
        assert config.output_json_filename == "augmentation_stats.json"
        assert config.output_csv_filename == "augmentation_stats.csv"
        assert config.track_per_operator_counts is True
        assert config.track_per_class_counts is True
        assert config.track_per_dataset_counts is True

    def test_output_dir_converted_to_path(self, tmp_path: Path) -> None:
        """A string output_dir is converted to a pathlib.Path instance."""
        config = StatisticsConfig(output_dir=str(tmp_path / "stats"))

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == tmp_path / "stats"


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    """Tests for :class:`AugmentationConfig`."""

    def test_valid_configuration_constructs_successfully(self, tmp_path: Path) -> None:
        """A fully valid configuration constructs without raising."""
        config = _build_valid_augmentation_config(tmp_path)

        assert config.seed == 42
        assert config.deterministic is True
        assert config.augmentations_per_image == 2

    def test_default_family_toggles_are_enabled(self, tmp_path: Path) -> None:
        """All augmentation family toggles default to enabled."""
        config = _build_valid_augmentation_config(tmp_path)

        assert config.geometric_enabled is True
        assert config.photometric_enabled is True
        assert config.blur_enabled is True
        assert config.noise_enabled is True
        assert config.compression_enabled is True

    def test_default_operator_configs_are_present(self, tmp_path: Path) -> None:
        """All nine default operator configurations are populated with expected defaults."""
        config = _build_valid_augmentation_config(tmp_path)

        assert config.horizontal_flip.enabled is True
        assert config.rotation.parameters["degrees"] == (-5.0, 5.0)
        assert config.brightness.parameters["factor"] == (0.85, 1.15)
        assert config.contrast.parameters["factor"] == (0.85, 1.15)
        assert config.gamma.parameters["gamma"] == (0.85, 1.15)
        assert config.color_jitter.parameters["hue_shift"] == (-0.02, 0.02)
        assert config.gaussian_noise.parameters["sigma"] == (2.0, 8.0)
        assert config.gaussian_blur.parameters["sigma"] == (0.3, 1.2)
        assert config.jpeg_compression.parameters["quality"] == (60.0, 90.0)

    def test_default_output_format_and_quality(self, tmp_path: Path) -> None:
        """Default output image format is 'png' and jpeg_quality defaults to 95."""
        config = _build_valid_augmentation_config(tmp_path)

        assert config.output_image_format == "png"
        assert config.jpeg_quality == 95

    @pytest.mark.parametrize(
        "path_field",
        [
            "input_dataset_root",
            "output_dataset_root",
            "train_manifest_path",
            "validation_manifest_path",
            "test_manifest_path",
            "output_manifest_path",
        ],
    )
    def test_paths_converted_to_path_objects(
        self, tmp_path: Path, path_field: str
    ) -> None:
        """All dataset/manifest path fields are converted to pathlib.Path, even given as str."""
        kwargs: dict[str, object] = {
            "input_dataset_root": tmp_path / "input",
            "output_dataset_root": tmp_path / "output",
            "train_manifest_path": tmp_path / "splits" / "train.csv",
            "validation_manifest_path": tmp_path / "splits" / "val.csv",
            "test_manifest_path": tmp_path / "splits" / "test.csv",
            "output_manifest_path": tmp_path / "output" / "final_manifest.csv",
            "logging": _build_logging_config(tmp_path),
            "statistics": _build_statistics_config(tmp_path),
        }
        kwargs[path_field] = str(kwargs[path_field])

        config = AugmentationConfig(**kwargs)  # type: ignore[arg-type]

        assert isinstance(getattr(config, path_field), Path)

    def test_invalid_seed_raises(self, tmp_path: Path) -> None:
        """A negative seed raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="seed"):
            AugmentationConfig(
                input_dataset_root=tmp_path / "input",
                output_dataset_root=tmp_path / "output",
                train_manifest_path=tmp_path / "splits" / "train.csv",
                validation_manifest_path=tmp_path / "splits" / "val.csv",
                test_manifest_path=tmp_path / "splits" / "test.csv",
                output_manifest_path=tmp_path / "output" / "final_manifest.csv",
                logging=_build_logging_config(tmp_path),
                statistics=_build_statistics_config(tmp_path),
                seed=-1,
            )

    @pytest.mark.parametrize("jpeg_quality", [0, -1, 101, 1000])
    def test_invalid_jpeg_quality_raises(
        self, tmp_path: Path, jpeg_quality: int
    ) -> None:
        """A jpeg_quality outside [1, 100] raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="jpeg_quality"):
            AugmentationConfig(
                input_dataset_root=tmp_path / "input",
                output_dataset_root=tmp_path / "output",
                train_manifest_path=tmp_path / "splits" / "train.csv",
                validation_manifest_path=tmp_path / "splits" / "val.csv",
                test_manifest_path=tmp_path / "splits" / "test.csv",
                output_manifest_path=tmp_path / "output" / "final_manifest.csv",
                logging=_build_logging_config(tmp_path),
                statistics=_build_statistics_config(tmp_path),
                jpeg_quality=jpeg_quality,
            )

    @pytest.mark.parametrize("jpeg_quality", [1, 100, 50])
    def test_boundary_jpeg_quality_is_valid(
        self, tmp_path: Path, jpeg_quality: int
    ) -> None:
        """Boundary jpeg_quality values 1 and 100 are accepted (inclusive range)."""
        config = AugmentationConfig(
            input_dataset_root=tmp_path / "input",
            output_dataset_root=tmp_path / "output",
            train_manifest_path=tmp_path / "splits" / "train.csv",
            validation_manifest_path=tmp_path / "splits" / "val.csv",
            test_manifest_path=tmp_path / "splits" / "test.csv",
            output_manifest_path=tmp_path / "output" / "final_manifest.csv",
            logging=_build_logging_config(tmp_path),
            statistics=_build_statistics_config(tmp_path),
            jpeg_quality=jpeg_quality,
        )

        assert config.jpeg_quality == jpeg_quality

    @pytest.mark.parametrize("augmentations_per_image", [0, -1, -100])
    def test_invalid_augmentations_per_image_raises(
        self, tmp_path: Path, augmentations_per_image: int
    ) -> None:
        """A non-positive augmentations_per_image raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="augmentations_per_image"):
            AugmentationConfig(
                input_dataset_root=tmp_path / "input",
                output_dataset_root=tmp_path / "output",
                train_manifest_path=tmp_path / "splits" / "train.csv",
                validation_manifest_path=tmp_path / "splits" / "val.csv",
                test_manifest_path=tmp_path / "splits" / "test.csv",
                output_manifest_path=tmp_path / "output" / "final_manifest.csv",
                logging=_build_logging_config(tmp_path),
                statistics=_build_statistics_config(tmp_path),
                augmentations_per_image=augmentations_per_image,
            )

    def test_invalid_output_image_format_raises(self, tmp_path: Path) -> None:
        """An unsupported output_image_format raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="output_image_format"):
            AugmentationConfig(
                input_dataset_root=tmp_path / "input",
                output_dataset_root=tmp_path / "output",
                train_manifest_path=tmp_path / "splits" / "train.csv",
                validation_manifest_path=tmp_path / "splits" / "val.csv",
                test_manifest_path=tmp_path / "splits" / "test.csv",
                output_manifest_path=tmp_path / "output" / "final_manifest.csv",
                logging=_build_logging_config(tmp_path),
                statistics=_build_statistics_config(tmp_path),
                output_image_format="bmp",
            )

    def test_output_image_format_is_normalised_to_lowercase(
        self, tmp_path: Path
    ) -> None:
        """An uppercase output_image_format is normalised to lowercase."""
        config = AugmentationConfig(
            input_dataset_root=tmp_path / "input",
            output_dataset_root=tmp_path / "output",
            train_manifest_path=tmp_path / "splits" / "train.csv",
            validation_manifest_path=tmp_path / "splits" / "val.csv",
            test_manifest_path=tmp_path / "splits" / "test.csv",
            output_manifest_path=tmp_path / "output" / "final_manifest.csv",
            logging=_build_logging_config(tmp_path),
            statistics=_build_statistics_config(tmp_path),
            output_image_format="JPEG",
        )

        assert config.output_image_format == "jpeg"

    def test_family_toggle_can_be_disabled(self, tmp_path: Path) -> None:
        """Family-level toggles can be independently disabled."""
        config = AugmentationConfig(
            input_dataset_root=tmp_path / "input",
            output_dataset_root=tmp_path / "output",
            train_manifest_path=tmp_path / "splits" / "train.csv",
            validation_manifest_path=tmp_path / "splits" / "val.csv",
            test_manifest_path=tmp_path / "splits" / "test.csv",
            output_manifest_path=tmp_path / "output" / "final_manifest.csv",
            logging=_build_logging_config(tmp_path),
            statistics=_build_statistics_config(tmp_path),
            geometric_enabled=False,
            noise_enabled=False,
        )

        assert config.geometric_enabled is False
        assert config.noise_enabled is False
        assert config.photometric_enabled is True
        assert config.blur_enabled is True
        assert config.compression_enabled is True

    def test_individual_operator_can_be_disabled(self, tmp_path: Path) -> None:
        """Individual operator configs can be overridden to be disabled."""
        config = AugmentationConfig(
            input_dataset_root=tmp_path / "input",
            output_dataset_root=tmp_path / "output",
            train_manifest_path=tmp_path / "splits" / "train.csv",
            validation_manifest_path=tmp_path / "splits" / "val.csv",
            test_manifest_path=tmp_path / "splits" / "test.csv",
            output_manifest_path=tmp_path / "output" / "final_manifest.csv",
            logging=_build_logging_config(tmp_path),
            statistics=_build_statistics_config(tmp_path),
            horizontal_flip=OperatorConfig(enabled=False),
        )

        assert config.horizontal_flip.enabled is False
        assert config.rotation.enabled is True

    def test_invalid_nested_operator_probability_propagates(
        self, tmp_path: Path
    ) -> None:
        """An invalid probability on a nested OperatorConfig raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="probability"):
            AugmentationConfig(
                input_dataset_root=tmp_path / "input",
                output_dataset_root=tmp_path / "output",
                train_manifest_path=tmp_path / "splits" / "train.csv",
                validation_manifest_path=tmp_path / "splits" / "val.csv",
                test_manifest_path=tmp_path / "splits" / "test.csv",
                output_manifest_path=tmp_path / "output" / "final_manifest.csv",
                logging=_build_logging_config(tmp_path),
                statistics=_build_statistics_config(tmp_path),
                gaussian_blur=OperatorConfig(probability=1.5),
            )