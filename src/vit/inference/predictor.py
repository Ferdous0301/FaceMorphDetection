"""Standalone inference on single images or batches, decoupled from training internals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from src.vit.checkpoint.checkpoint_manager import CheckpointManager
from src.vit.configs.schema import ModelConfig
from src.vit.data.transforms import build_eval_transforms
from src.vit.models.vit_model import ViTMorphClassifier

__all__ = ["ViTPredictor", "PredictionResult"]

# Matches the label convention in vit.metrics.classification_metrics:
# 0 = bona fide, 1 = attack (morph).
_BONAFIDE_CLASS_INDEX = 0
_MORPH_CLASS_INDEX = 1


@dataclass
class PredictionResult:
    """Single-image prediction output."""

    image_path: Path
    predicted_label: int
    probability_morph: float
    probability_bonafide: float


class ViTPredictor:
    """Loads a trained model from a checkpoint and runs single/batch inference.

    Fully decoupled from :class:`~vit.engine.trainer.Trainer` internals —
    only needs a model, device, and eval transform — so it is usable
    standalone in a downstream demo/app.

    Args:
        model: A model already loaded with trained weights and moved to
            ``device``.
        device: Device to run inference on.
        transform: Preprocessing transform applied to each loaded PIL image,
            matching what the model was trained with (see
            :func:`vit.data.transforms.build_eval_transforms`).
    """

    def __init__(self, model: nn.Module, device: torch.device, transform: Callable) -> None:
        self._model = model
        self._device = device
        self._transform = transform
        self._model.eval()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, config: ModelConfig, device: torch.device
    ) -> "ViTPredictor":
        """Build a predictor by loading model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint written by
                :class:`~vit.checkpoint.checkpoint_manager.CheckpointManager`
                (e.g. ``checkpoints/vit/best.pt``).
            config: Model architecture configuration matching the checkpoint
                (backbone variant, num_classes, etc.) — used to construct
                the model shell before loading weights into it.
            device: Device to load the model onto and run inference on.

        Returns:
            A ready-to-use :class:`ViTPredictor`.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # CheckpointManager.load handles both the FileNotFoundError check and
        # the weights_only=False deserialization needed since checkpoints
        # store a full CheckpointState dataclass (not a plain dict).
        state = CheckpointManager(
            checkpoint_dir=checkpoint_path.parent, monitor_metric="", mode="min"
        ).load(checkpoint_path, map_location=device)

        model = ViTMorphClassifier(config)
        model.load_state_dict(state.model_state_dict)
        model.to(device)

        transform = build_eval_transforms(image_size=224)
        return cls(model=model, device=device, transform=transform)

    @torch.no_grad()
    def predict_image(self, image_path: Path) -> PredictionResult:
        """Run inference on a single image file.

        Args:
            image_path: Path to an image file readable by PIL.

        Returns:
            A :class:`PredictionResult` with the predicted label and both
            class probabilities.

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        logits = self._model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        predicted_label = int(torch.argmax(probs).item())
        return PredictionResult(
            image_path=image_path,
            predicted_label=predicted_label,
            probability_morph=float(probs[_MORPH_CLASS_INDEX]),
            probability_bonafide=float(probs[_BONAFIDE_CLASS_INDEX]),
        )

    @torch.no_grad()
    def predict_batch(self, image_paths: List[Path]) -> List[PredictionResult]:
        """Run inference on a list of image files as a single batch.

        Args:
            image_paths: Paths to image files readable by PIL.

        Returns:
            A list of :class:`PredictionResult`, one per input path, in the
            same order as ``image_paths``.

        Raises:
            ValueError: If ``image_paths`` is empty.
            FileNotFoundError: If any path in ``image_paths`` does not exist.
        """
        if not image_paths:
            raise ValueError("predict_batch received an empty list of image paths")

        paths = [Path(p) for p in image_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        tensors: List[Tensor] = []
        for p in paths:
            image = Image.open(p).convert("RGB")
            tensors.append(self._transform(image))

        batch = torch.stack(tensors, dim=0).to(self._device)
        logits = self._model(batch)
        probs = F.softmax(logits, dim=1).cpu()
        predicted_labels = torch.argmax(probs, dim=1)

        return [
            PredictionResult(
                image_path=paths[i],
                predicted_label=int(predicted_labels[i].item()),
                probability_morph=float(probs[i, _MORPH_CLASS_INDEX]),
                probability_bonafide=float(probs[i, _BONAFIDE_CLASS_INDEX]),
            )
            for i in range(len(paths))
        ]