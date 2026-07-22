"""Unit tests for vit.inference.predictor.ViTPredictor.

Uses pretrained=False (no network access to download real weights in this
sandbox) and only a couple of tiny forward passes, since ViT-B/16 forward
passes are the slow part; the point here is exercising the checkpoint ->
predictor wiring, not model quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch
from PIL import Image

from vit.checkpoint.checkpoint_manager import CheckpointManager, CheckpointState
from vit.configs.schema import ModelConfig
from vit.inference.predictor import PredictionResult, ViTPredictor
from vit.models.vit_model import ViTMorphClassifier


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and save a tiny ViT-B/16-based checkpoint once for this module."""
    tmp_path = tmp_path_factory.mktemp("predictor_ckpt")
    config = ModelConfig(backbone="vit_b_16", pretrained=False, num_classes=2)
    model = ViTMorphClassifier(config)

    manager = CheckpointManager(checkpoint_dir=tmp_path, monitor_metric="val_loss", mode="min")
    state = CheckpointState(
        epoch=0,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict={},
        scaler_state_dict=None,
        best_metric_value=0.0,
        config=None,
        rng_state={},
    )
    manager.save(state, is_best=True)
    return tmp_path / "best.pt"


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(120, 40, 40)).save(path)
    return path


class TestFromCheckpoint:
    def test_builds_predictor(self, checkpoint_path: Path) -> None:
        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        assert isinstance(predictor, ViTPredictor)

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ViTPredictor.from_checkpoint(
                checkpoint_path=tmp_path / "nope.pt",
                config=ModelConfig(backbone="vit_b_16", pretrained=False),
                device=torch.device("cpu"),
            )


class TestPredictImage:
    def test_returns_valid_prediction_result(
        self, checkpoint_path: Path, sample_image: Path
    ) -> None:
        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        result = predictor.predict_image(sample_image)

        assert isinstance(result, PredictionResult)
        assert result.predicted_label in (0, 1)
        assert 0.0 <= result.probability_morph <= 1.0
        assert 0.0 <= result.probability_bonafide <= 1.0
        assert result.probability_morph == pytest.approx(
            1.0 - result.probability_bonafide, abs=1e-5
        )

    def test_missing_image_raises(self, checkpoint_path: Path, tmp_path: Path) -> None:
        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        with pytest.raises(FileNotFoundError):
            predictor.predict_image(tmp_path / "missing.png")


class TestPredictBatch:
    def test_batch_matches_single_image_order(
        self, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        paths = []
        for i, color in enumerate([(200, 30, 30), (30, 200, 30)]):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (64, 64), color=color).save(p)
            paths.append(p)

        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        results = predictor.predict_batch(paths)

        assert len(results) == 2
        assert [r.image_path for r in results] == paths

    def test_empty_list_raises(self, checkpoint_path: Path) -> None:
        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError):
            predictor.predict_batch([])

    def test_missing_file_in_batch_raises(self, checkpoint_path: Path, tmp_path: Path) -> None:
        predictor = ViTPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=ModelConfig(backbone="vit_b_16", pretrained=False),
            device=torch.device("cpu"),
        )
        with pytest.raises(FileNotFoundError):
            predictor.predict_batch([tmp_path / "missing.png"])