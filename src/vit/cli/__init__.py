"""Command-line entry points for training, evaluation, and inference.

Run as modules:

    python -m vit.cli.train_cli --config configs/vit_experiment.yaml
    python -m vit.cli.evaluate_cli --config configs/vit_experiment.yaml --checkpoint checkpoints/vit/best.pt
    python -m vit.cli.predict_cli --config configs/vit_experiment.yaml --checkpoint checkpoints/vit/best.pt --image path/to/image.jpg
"""

from __future__ import annotations

__all__: list[str] = []