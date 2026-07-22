"""Unit tests for vit.visualization.*"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vit.engine.trainer import EpochResult, TrainingHistory
from vit.visualization.attention_maps import plot_attention_rollout
from vit.visualization.confusion_matrix import plot_confusion_matrix
from vit.visualization.curves import plot_training_curves


def _make_history(num_epochs: int = 3) -> TrainingHistory:
    train_results = [
        EpochResult(epoch=i, loss=1.0 - i * 0.1, accuracy=0.5 + i * 0.1, eer=0.3 - i * 0.05, auc=0.6 + i * 0.05, learning_rate=1e-4)
        for i in range(num_epochs)
    ]
    val_results = [
        EpochResult(epoch=i, loss=1.1 - i * 0.1, accuracy=0.45 + i * 0.1, eer=0.35 - i * 0.05, auc=0.55 + i * 0.05, learning_rate=1e-4)
        for i in range(num_epochs)
    ]
    return TrainingHistory(
        train_results=train_results,
        val_results=val_results,
        best_epoch=num_epochs - 1,
        best_metric_value=val_results[-1].eer,
    )


class TestPlotTrainingCurves:
    def test_creates_png_file(self, tmp_path: Path) -> None:
        history = _make_history()
        output_path = tmp_path / "curves.png"
        plot_training_curves(history, output_path)
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        history = _make_history()
        output_path = tmp_path / "nested" / "dir" / "curves.png"
        plot_training_curves(history, output_path)
        assert output_path.is_file()

    def test_empty_history_raises(self, tmp_path: Path) -> None:
        empty_history = TrainingHistory()
        with pytest.raises(ValueError):
            plot_training_curves(empty_history, tmp_path / "curves.png")


class TestPlotConfusionMatrix:
    def test_creates_png_file(self, tmp_path: Path) -> None:
        cm = torch.tensor([[8, 2], [1, 9]])
        output_path = tmp_path / "cm.png"
        plot_confusion_matrix(cm, class_names=["bonafide", "morph"], output_path=output_path)
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_mismatched_class_names_raises(self, tmp_path: Path) -> None:
        cm = torch.tensor([[8, 2], [1, 9]])
        with pytest.raises(ValueError):
            plot_confusion_matrix(cm, class_names=["only_one"], output_path=tmp_path / "cm.png")

    def test_non_square_matrix_raises(self, tmp_path: Path) -> None:
        cm = torch.tensor([[8, 2, 0], [1, 9, 0]])
        with pytest.raises(ValueError):
            plot_confusion_matrix(cm, class_names=["a", "b"], output_path=tmp_path / "cm.png")

    def test_accepts_numpy_array(self, tmp_path: Path) -> None:
        import numpy as np

        cm = np.array([[5, 1], [0, 6]])
        output_path = tmp_path / "cm.png"
        plot_confusion_matrix(cm, class_names=["bonafide", "morph"], output_path=output_path)
        assert output_path.is_file()


class TestPlotAttentionRollout:
    def test_creates_png_file_for_square_patch_grid(self, tmp_path: Path) -> None:
        # 4 patch tokens (2x2 grid) + 1 class token = 5 tokens.
        attention = torch.rand(5, 5)
        attention = attention / attention.sum(dim=-1, keepdim=True)  # normalize rows
        output_path = tmp_path / "attn.png"
        plot_attention_rollout(attention, output_path)
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_realistic_vit_b_16_token_count(self, tmp_path: Path) -> None:
        # 196 patch tokens (14x14) + 1 class token = 197, matching ViT-B/16 @ 224.
        attention = torch.rand(197, 197)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        output_path = tmp_path / "attn.png"
        plot_attention_rollout(attention, output_path)
        assert output_path.is_file()

    def test_non_square_attention_raises(self, tmp_path: Path) -> None:
        attention = torch.rand(5, 6)
        with pytest.raises(ValueError):
            plot_attention_rollout(attention, tmp_path / "attn.png")

    def test_too_few_tokens_raises(self, tmp_path: Path) -> None:
        attention = torch.rand(1, 1)
        with pytest.raises(ValueError):
            plot_attention_rollout(attention, tmp_path / "attn.png")

    def test_non_perfect_square_patch_count_raises(self, tmp_path: Path) -> None:
        # 5 tokens total -> 4 patch tokens would be fine (2x2), but 6 tokens
        # -> 5 patch tokens is not a perfect square.
        attention = torch.rand(6, 6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        with pytest.raises(ValueError):
            plot_attention_rollout(attention, tmp_path / "attn.png")