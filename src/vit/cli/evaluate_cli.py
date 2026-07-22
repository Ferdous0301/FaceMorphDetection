"""Standalone evaluation entry point.

Usage:
    python -m vit.cli.evaluate_cli --config configs/vit_experiment.yaml \
        --checkpoint checkpoints/vit/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from vit.checkpoint.checkpoint_manager import CheckpointManager
from vit.configs.schema import ConfigValidationError, load_experiment_config
from vit.data.datamodule import ViTDataModule
from vit.engine.evaluator import EvaluationResult, Evaluator
from vit.engine.loss import build_loss
from vit.models.vit_model import ViTMorphClassifier
from vit.utils.device import resolve_device
from vit.utils.seed import set_global_seed
from vit.visualization.confusion_matrix import plot_confusion_matrix

__all__ = ["main"]

_SPLIT_CHOICES = ("train", "val", "test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT morph-attack-detection checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--split",
        type=str,
        choices=_SPLIT_CHOICES,
        default="test",
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write metrics/predictions/plots to. "
        "Defaults to the config's training.results_dir.",
    )
    return parser.parse_args()


def _write_results(result: EvaluationResult, split: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "split": split,
        "loss": result.loss,
        "accuracy": result.accuracy,
        "eer": result.eer,
        "eer_threshold": result.eer_threshold,
        "auc": result.auc,
        "apcer": result.apcer,
        "bpcer": result.bpcer,
    }
    metrics_path = output_dir / f"{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = output_dir / f"{split}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "prediction", "score_morph"])
        for image_id, label, pred, score in zip(
            result.image_ids,
            result.labels.tolist(),
            result.predictions.tolist(),
            result.scores.tolist(),
        ):
            writer.writerow([image_id, label, pred, score])

    cm_path = output_dir / f"{split}_confusion_matrix.png"
    plot_confusion_matrix(
        result.confusion_matrix, class_names=["bonafide", "morph"], output_path=cm_path
    )

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote per-sample predictions to {predictions_path}")
    print(f"Wrote confusion matrix plot to {cm_path}")


def main() -> None:
    """Load a checkpoint and evaluate it on the requested split.

    Prints a concise metrics summary to the console and writes full results
    (metrics JSON, per-sample predictions CSV, confusion matrix PNG) to
    ``--output-dir``.
    """
    args = _parse_args()
    try:
        config = load_experiment_config(args.config)
    except ConfigValidationError as exc:
        raise SystemExit(f"Invalid config {args.config}: {exc}") from exc

    output_dir = args.output_dir if args.output_dir is not None else config.training.results_dir

    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)

    datamodule = ViTDataModule(config.data, seed=config.training.seed)
    datamodule.setup()

    loader_by_split = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }
    loader = loader_by_split[args.split]()

    model = ViTMorphClassifier(config.model).to(device)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        monitor_metric=config.training.early_stopping_metric,
        mode=config.training.early_stopping_mode,
    )
    state = checkpoint_manager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.model_state_dict)

    class_weights = datamodule.class_weights().to(device) if config.training.use_class_weights else None
    loss_fn = build_loss(name=config.training.loss_name, class_weights=class_weights)

    evaluator = Evaluator(model=model, device=device, loss_fn=loss_fn)
    result = evaluator.evaluate(loader)

    print(f"[{args.split}] loss={result.loss:.4f} accuracy={result.accuracy:.4f} "
          f"eer={result.eer:.4f} auc={result.auc:.4f} "
          f"apcer={result.apcer:.4f} bpcer={result.bpcer:.4f}")

    _write_results(result, split=args.split, output_dir=output_dir)


if __name__ == "__main__":
    main()