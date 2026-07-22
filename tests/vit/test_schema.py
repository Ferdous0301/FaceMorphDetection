"""Unit tests for vit.config.schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from vit.config.schema import (
    ConfigValidationError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_experiment_config,
    save_experiment_config,
)


def _minimal_data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        train_csv=tmp_path / "train.csv",
        val_csv=tmp_path / "val.csv",
        test_csv=tmp_path / "test.csv",
        image_root=tmp_path,
    )


def _minimal_experiment_config(tmp_path: Path, epochs: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        data=_minimal_data_config(tmp_path),
        model=ModelConfig(),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(total_epochs=epochs, warmup_epochs=1),
        training=TrainingConfig(epochs=epochs),
    )


class TestDataConfig:
    def test_valid_construction(self, tmp_path: Path) -> None:
        cfg = _minimal_data_config(tmp_path)
        assert cfg.image_size == 224
        assert cfg.batch_size == 32

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        cfg = _minimal_data_config(tmp_path)
        assert isinstance(cfg.train_csv, Path)

    @pytest.mark.parametrize("image_size", [0, -1])
    def test_rejects_non_positive_image_size(self, tmp_path: Path, image_size: int) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig(
                train_csv=tmp_path / "a.csv",
                val_csv=tmp_path / "b.csv",
                test_csv=tmp_path / "c.csv",
                image_root=tmp_path,
                image_size=image_size,
            )

    def test_rejects_non_positive_batch_size(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig(
                train_csv=tmp_path / "a.csv",
                val_csv=tmp_path / "b.csv",
                test_csv=tmp_path / "c.csv",
                image_root=tmp_path,
                batch_size=0,
            )

    def test_is_frozen(self, tmp_path: Path) -> None:
        cfg = _minimal_data_config(tmp_path)
        with pytest.raises(Exception):
            cfg.batch_size = 64  # type: ignore[misc]


class TestModelConfig:
    def test_default_backbone_supported(self) -> None:
        cfg = ModelConfig()
        assert cfg.backbone == "vit_b_16"

    def test_rejects_unsupported_backbone(self) -> None:
        with pytest.raises(ConfigValidationError):
            ModelConfig(backbone="resnet50")

    def test_rejects_num_classes_below_two(self) -> None:
        with pytest.raises(ConfigValidationError):
            ModelConfig(num_classes=1)

    @pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
    def test_rejects_invalid_dropout(self, dropout: float) -> None:
        with pytest.raises(ConfigValidationError):
            ModelConfig(dropout=dropout)


class TestOptimizerConfig:
    def test_rejects_unsupported_optimizer(self) -> None:
        with pytest.raises(ConfigValidationError):
            OptimizerConfig(name="rmsprop")

    def test_rejects_non_positive_lr(self) -> None:
        with pytest.raises(ConfigValidationError):
            OptimizerConfig(lr=0.0)

    def test_rejects_negative_weight_decay(self) -> None:
        with pytest.raises(ConfigValidationError):
            OptimizerConfig(weight_decay=-0.1)


class TestSchedulerConfig:
    def test_rejects_unsupported_scheduler(self) -> None:
        with pytest.raises(ConfigValidationError):
            SchedulerConfig(name="linear")

    def test_rejects_warmup_geq_total(self) -> None:
        with pytest.raises(ConfigValidationError):
            SchedulerConfig(warmup_epochs=10, total_epochs=10)


class TestTrainingConfig:
    def test_rejects_unsupported_loss(self) -> None:
        with pytest.raises(ConfigValidationError):
            TrainingConfig(loss_name="hinge")

    def test_rejects_invalid_early_stopping_mode(self) -> None:
        with pytest.raises(ConfigValidationError):
            TrainingConfig(early_stopping_mode="best")

    def test_rejects_non_positive_grad_clip_norm(self) -> None:
        with pytest.raises(ConfigValidationError):
            TrainingConfig(grad_clip_norm=0.0)

    def test_none_grad_clip_norm_allowed(self) -> None:
        cfg = TrainingConfig(grad_clip_norm=None)
        assert cfg.grad_clip_norm is None


class TestExperimentConfig:
    def test_valid_construction(self, tmp_path: Path) -> None:
        cfg = _minimal_experiment_config(tmp_path)
        assert cfg.experiment_name == "vit_morph_detection"

    def test_rejects_mismatched_epochs(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError):
            ExperimentConfig(
                data=_minimal_data_config(tmp_path),
                model=ModelConfig(),
                optimizer=OptimizerConfig(),
                scheduler=SchedulerConfig(total_epochs=10),
                training=TrainingConfig(epochs=20),
            )

    def test_rejects_empty_experiment_name(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError):
            ExperimentConfig(
                data=_minimal_data_config(tmp_path),
                model=ModelConfig(),
                optimizer=OptimizerConfig(),
                scheduler=SchedulerConfig(total_epochs=5),
                training=TrainingConfig(epochs=5),
                experiment_name="   ",
            )

    def test_with_overrides_returns_new_validated_instance(self, tmp_path: Path) -> None:
        cfg = _minimal_experiment_config(tmp_path)
        new_cfg = cfg.with_overrides(experiment_name="renamed")
        assert new_cfg.experiment_name == "renamed"
        assert cfg.experiment_name == "vit_morph_detection"  # original untouched

    def test_with_overrides_still_validates(self, tmp_path: Path) -> None:
        cfg = _minimal_experiment_config(tmp_path)
        with pytest.raises(ConfigValidationError):
            cfg.with_overrides(experiment_name="")


class TestYamlRoundTrip:
    def test_save_then_load_preserves_values(self, tmp_path: Path) -> None:
        cfg = _minimal_experiment_config(tmp_path, epochs=7)
        yaml_path = tmp_path / "experiment.yaml"

        save_experiment_config(cfg, yaml_path)
        assert yaml_path.is_file()

        loaded = load_experiment_config(yaml_path)

        assert loaded.experiment_name == cfg.experiment_name
        assert loaded.training.epochs == cfg.training.epochs
        assert loaded.model.backbone == cfg.model.backbone
        assert loaded.optimizer.lr == pytest.approx(cfg.optimizer.lr)
        assert loaded.data.train_csv == cfg.data.train_csv
        assert isinstance(loaded.data.train_csv, Path)
        assert isinstance(loaded.optimizer.betas, tuple)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment_config(tmp_path / "does_not_exist.yaml")

    def test_load_missing_section_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("data:\n  train_csv: a\n  val_csv: b\n  test_csv: c\n  image_root: d\n")
        with pytest.raises(ConfigValidationError):
            load_experiment_config(yaml_path)

    def test_load_unknown_key_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            """
data:
  train_csv: a
  val_csv: b
  test_csv: c
  image_root: d
  not_a_real_field: 123
model: {}
optimizer: {}
scheduler:
  total_epochs: 5
training:
  epochs: 5
"""
        )
        with pytest.raises(ConfigValidationError):
            load_experiment_config(yaml_path)

    def test_load_real_default_config_file(self) -> None:
        # Exercises the shipped configs/vit_experiment.yaml, catching drift
        # between the example config and the schema early.
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "configs" / "vit_experiment.yaml"
        cfg = load_experiment_config(config_path)
        assert cfg.model.backbone in ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32")