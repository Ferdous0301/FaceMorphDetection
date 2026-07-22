"""End-to-end integration smoke test for the ViT stage.

Exercises the full pipeline exactly as ``train_cli.py`` / ``evaluate_cli.py``
/ ``predict_cli.py`` would, but wired together directly (rather than via
subprocess) against the tiny synthetic dataset from ``conftest.py``, so it
runs in seconds rather than requiring the real face-morph dataset or a GPU.

Covered end to end:
    config loading & validation -> ViTDataModule -> ViTMorphClassifier
    -> loss/optimizer/scheduler factories -> Trainer.fit (with checkpointing
    and CSV/TensorBoard logging) -> training-curve plot -> Evaluator on the
    test split -> confusion-matrix plot -> CheckpointManager.load_best ->
    ViTPredictor single-image inference.

This is the one test in the suite that uses the real ViT-B/16 backbone
(pretrained=False, since no network access to download real weights is
available in this sandbox), so it is slower than the rest of the suite
(order of a minute on CPU) but is the strongest guarantee that every module
actually composes correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch

from vit.checkpoint.checkpoint_manager import CheckpointManager
from vit.configs.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_experiment_config,
    save_experiment_config,
)
from vit.data.datamodule import ViTDataModule
from vit.engine.evaluator import Evaluator
from vit.engine.loss import build_loss
from vit.engine.optimizer_factory import build_optimizer
from vit.engine.scheduler_factory import build_scheduler
from vit.engine.trainer import Trainer
from vit.inference.predictor import ViTPredictor
from vit.logging_utils.csv_logger import CSVLogger
from vit.logging_utils.experiment_logger import ExperimentLogger
from vit.logging_utils.tb_logger import TensorBoardLogger
from vit.models.vit_model import ViTMorphClassifier
from vit.utils.device import resolve_device
from vit.utils.seed import set_global_seed
from vit.visualization.confusion_matrix import plot_confusion_matrix
from vit.visualization.curves import plot_training_curves


def _build_experiment_config(tmp_path: Path, synthetic_dataset: Dict[str, object]) -> ExperimentConfig:
    epochs = 1
    return ExperimentConfig(
        experiment_name="smoke_test_experiment",
        data=DataConfig(
            train_csv=synthetic_dataset["train_csv"],
            val_csv=synthetic_dataset["val_csv"],
            test_csv=synthetic_dataset["test_csv"],
            image_root=synthetic_dataset["image_root"],
            image_size=224,  # required by the real ViT-B/16 positional embeddings
            batch_size=2,
            num_workers=0,
        ),
        model=ModelConfig(backbone="vit_b_16", pretrained=False, num_classes=2, dropout=0.1),
        optimizer=OptimizerConfig(name="sgd", lr=0.1, weight_decay=0.0),
        scheduler=SchedulerConfig(name="none", total_epochs=epochs, warmup_epochs=0),
        training=TrainingConfig(
            epochs=epochs,
            loss_name="cross_entropy",
            use_class_weights=True,
            mixed_precision=False,
            grad_clip_norm=1.0,
            early_stopping_patience=5,
            early_stopping_metric="val_eer",
            early_stopping_mode="min",
            seed=123,
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            results_dir=tmp_path / "results",
            device="cpu",
        ),
    )


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path: Path, synthetic_dataset: Dict[str, object]) -> None:
    """Run config -> train -> evaluate -> predict, asserting every artifact exists."""
    config = _build_experiment_config(tmp_path, synthetic_dataset)

    # --- Config round-trip (as a real experiment would snapshot it) -------
    config_path = tmp_path / "experiment.yaml"
    save_experiment_config(config, config_path)
    reloaded_config = load_experiment_config(config_path)
    assert reloaded_config.experiment_name == config.experiment_name

    # --- Reproducibility + device resolution -------------------------------
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    assert device.type == "cpu"

    # --- Data ---------------------------------------------------------------
    datamodule = ViTDataModule(reloaded_config.data, seed=reloaded_config.training.seed)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # --- Model / loss / optimizer / scheduler -------------------------------
    model = ViTMorphClassifier(reloaded_config.model).to(device)

    class_weights = (
        datamodule.class_weights().to(device) if reloaded_config.training.use_class_weights else None
    )
    loss_fn = build_loss(name=reloaded_config.training.loss_name, class_weights=class_weights)
    optimizer = build_optimizer(model, reloaded_config.optimizer)
    scheduler = build_scheduler(
        optimizer, reloaded_config.scheduler, steps_per_epoch=len(train_loader)
    )

    # --- Checkpointing + logging --------------------------------------------
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=reloaded_config.training.checkpoint_dir,
        monitor_metric=reloaded_config.training.early_stopping_metric,
        mode=reloaded_config.training.early_stopping_mode,
    )
    logger = ExperimentLogger(
        csv_logger=CSVLogger(reloaded_config.training.log_dir / "train_log.csv"),
        tb_logger=TensorBoardLogger(reloaded_config.training.log_dir / "tensorboard"),
        name=reloaded_config.experiment_name,
    )

    # --- Training ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=reloaded_config.training,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        experiment_config=reloaded_config,
    )
    history = trainer.fit(train_loader, val_loader)
    logger.close()

    assert len(history.train_results) == 1
    assert len(history.val_results) == 1
    assert history.best_epoch == 0

    best_checkpoint_path = reloaded_config.training.checkpoint_dir / "best.pt"
    last_checkpoint_path = reloaded_config.training.checkpoint_dir / "last.pt"
    assert best_checkpoint_path.is_file()
    assert last_checkpoint_path.is_file()

    train_log_path = reloaded_config.training.log_dir / "train_log.csv"
    val_log_path = reloaded_config.training.log_dir / "val_log.csv"
    assert train_log_path.is_file()
    assert val_log_path.is_file()

    # --- Training curves plot -------------------------------------------------
    curves_path = reloaded_config.training.results_dir / "training_curves.png"
    plot_training_curves(history, curves_path)
    assert curves_path.is_file()
    assert curves_path.stat().st_size > 0

    # --- Standalone test-set evaluation via the best checkpoint ---------------
    eval_model = ViTMorphClassifier(reloaded_config.model).to(device)
    best_state = checkpoint_manager.load(best_checkpoint_path, map_location=device)
    eval_model.load_state_dict(best_state.model_state_dict)

    evaluator = Evaluator(model=eval_model, device=device, loss_fn=loss_fn)
    eval_result = evaluator.evaluate(test_loader)

    assert 0.0 <= eval_result.accuracy <= 1.0
    assert 0.0 <= eval_result.eer <= 1.0
    assert eval_result.confusion_matrix.shape == (2, 2)
    assert len(eval_result.image_ids) == len(synthetic_dataset["labels"]["test"])

    cm_path = reloaded_config.training.results_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(
        eval_result.confusion_matrix, class_names=["bonafide", "morph"], output_path=cm_path
    )
    assert cm_path.is_file()

    # --- Standalone single-image inference via ViTPredictor --------------------
    predictor = ViTPredictor.from_checkpoint(
        checkpoint_path=best_checkpoint_path, config=reloaded_config.model, device=device
    )
    sample_image_path = next(
        (reloaded_config.data.image_root).glob("test_*.png")
    )
    prediction = predictor.predict_image(sample_image_path)

    assert prediction.predicted_label in (0, 1)
    assert prediction.probability_morph == pytest.approx(
        1.0 - prediction.probability_bonafide, abs=1e-5
    )

    # --- Checkpoint rotation sanity: at least one epoch checkpoint retained ----
    assert len(checkpoint_manager.list_epoch_checkpoints()) >= 1