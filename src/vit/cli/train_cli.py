"""Training entry point.

Usage:
    python -m vit.cli.train_cli --config configs/vit_experiment.yaml [--resume checkpoints/vit/last.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vit.checkpoint.checkpoint_manager import CheckpointManager
from vit.configs.schema import ConfigValidationError, ExperimentConfig, load_experiment_config
from vit.data.datamodule import ViTDataModule
from vit.engine.loss import build_loss
from vit.engine.optimizer_factory import build_optimizer
from vit.engine.scheduler_factory import build_scheduler
from vit.engine.trainer import Trainer
from vit.logging_utils.csv_logger import CSVLogger
from vit.logging_utils.experiment_logger import ExperimentLogger
from vit.logging_utils.tb_logger import TensorBoardLogger
from vit.models.vit_model import ViTMorphClassifier
from vit.utils.device import get_device_info, resolve_device
from vit.utils.seed import set_global_seed
from vit.visualization.curves import plot_training_curves

__all__ = ["main"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ViT morph-attack-detection classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from (e.g. checkpoints/vit/last.pt).",
    )
    return parser.parse_args()


def _flatten_hparams(config: ExperimentConfig) -> dict:
    flat = {"experiment_name": config.experiment_name}
    for section_name in ("data", "model", "optimizer", "scheduler", "training"):
        section = getattr(config, section_name)
        for key, value in vars(section).items():
            flat[f"{section_name}.{key}"] = value
    return flat


def main() -> None:
    """Run a full training experiment end to end.

    Loads the experiment config, seeds every RNG, builds the data module /
    model / loss / optimizer / scheduler / checkpoint manager / logger,
    optionally resumes from a checkpoint, then delegates the actual loop to
    :class:`~vit.engine.trainer.Trainer`. On completion, saves a training
    curves plot and prints a summary of the best epoch/metric and where the
    best checkpoint was written.
    """
    args = _parse_args()
    try:
        config = load_experiment_config(args.config)
    except ConfigValidationError as exc:
        raise SystemExit(f"Invalid config {args.config}: {exc}") from exc

    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    print(f"Device info: {get_device_info(device)}")

    datamodule = ViTDataModule(config.data, seed=config.training.seed)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = ViTMorphClassifier(config.model).to(device)

    class_weights = datamodule.class_weights().to(device) if config.training.use_class_weights else None
    loss_fn = build_loss(name=config.training.loss_name, class_weights=class_weights)

    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, steps_per_epoch=len(train_loader))

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        monitor_metric=config.training.early_stopping_metric,
        mode=config.training.early_stopping_mode,
    )

    logger = ExperimentLogger(
        csv_logger=CSVLogger(config.training.log_dir / "train_log.csv"),
        tb_logger=TensorBoardLogger(config.training.log_dir / "tensorboard"),
        name=config.experiment_name,
    )
    logger.log_hparams(_flatten_hparams(config))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config.training,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        experiment_config=config,
    )

    if args.resume is not None:
        state = checkpoint_manager.load(args.resume, map_location=device)
        model.load_state_dict(state.model_state_dict)
        optimizer.load_state_dict(state.optimizer_state_dict)
        scheduler.load_state_dict(state.scheduler_state_dict)
        print(f"Resumed from {args.resume} at epoch {state.epoch}")

    history = trainer.fit(train_loader, val_loader)

    curves_path = config.training.results_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    logger.close()

    best_checkpoint = config.training.checkpoint_dir / "best.pt"
    print(f"Training complete. Best epoch: {history.best_epoch}")
    print(f"Best {config.training.early_stopping_metric}: {history.best_metric_value:.6f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training curves saved to: {curves_path}")


if __name__ == "__main__":
    main()