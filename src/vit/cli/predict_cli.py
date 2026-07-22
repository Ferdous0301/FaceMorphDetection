"""Single/batch inference entry point.

Usage:
    python -m vit.cli.predict_cli --checkpoint checkpoints/vit/best.pt --config configs/vit_experiment.yaml --image path/to/image.jpg
    python -m vit.cli.predict_cli --checkpoint checkpoints/vit/best.pt --config configs/vit_experiment.yaml --image-dir path/to/images/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from vit.configs.schema import load_experiment_config
from vit.inference.predictor import PredictionResult, ViTPredictor
from vit.utils.device import resolve_device

__all__ = ["main"]

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained ViT morph-attack-detection model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config (for model architecture).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint file.")

    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=Path, help="Path to a single image file.")
    image_group.add_argument("--image-dir", type=Path, help="Directory of images to run batch inference on.")

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write batch predictions as CSV (only used with --image-dir).",
    )
    return parser.parse_args()


def _print_result(result: PredictionResult) -> None:
    label_name = "morph" if result.predicted_label == 1 else "bonafide"
    print(
        f"{result.image_path}: predicted={label_name} "
        f"(p_morph={result.probability_morph:.4f}, p_bonafide={result.probability_bonafide:.4f})"
    )


def _write_csv(results: List[PredictionResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label", "probability_morph", "probability_bonafide"])
        for r in results:
            writer.writerow([r.image_path, r.predicted_label, r.probability_morph, r.probability_bonafide])
    print(f"Wrote predictions to {output_csv}")


def main() -> None:
    """Run inference on a single image or every image in a directory.

    For ``--image``, prints one prediction line. For ``--image-dir``, runs
    a single batched forward pass over every image with a supported
    extension in that directory, prints one line per image, and optionally
    writes a CSV via ``--output-csv``.
    """
    args = _parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(config.training.device)

    predictor = ViTPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint, config=config.model, device=device
    )

    if args.image is not None:
        result = predictor.predict_image(args.image)
        _print_result(result)
        return

    image_paths = sorted(
        p for p in args.image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No images with extensions {_IMAGE_EXTENSIONS} found in {args.image_dir}")

    results = predictor.predict_batch(image_paths)
    for result in results:
        _print_result(result)

    if args.output_csv is not None:
        _write_csv(results, args.output_csv)


if __name__ == "__main__":
    main()