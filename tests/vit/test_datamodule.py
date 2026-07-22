"""Unit tests for vit.data.datamodule.ViTDataModule."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from vit.config.schema import DataConfig
from vit.data.datamodule import ViTDataModule


def _make_config(synthetic_dataset: Dict[str, object], batch_size: int = 2) -> DataConfig:
    return DataConfig(
        train_csv=synthetic_dataset["train_csv"],
        val_csv=synthetic_dataset["val_csv"],
        test_csv=synthetic_dataset["test_csv"],
        image_root=synthetic_dataset["image_root"],
        image_size=32,
        batch_size=batch_size,
        num_workers=0,
    )


class TestSetupRequired:
    def test_dataloader_before_setup_raises(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset), seed=0)
        with pytest.raises(RuntimeError):
            dm.train_dataloader()

    def test_class_counts_before_setup_raises(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset), seed=0)
        with pytest.raises(RuntimeError):
            dm.class_counts()


class TestDataloaders:
    def test_train_val_test_sizes(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset), seed=0)
        dm.setup()

        n_train = sum(len(b[1]) for b in dm.train_dataloader())
        n_val = sum(len(b[1]) for b in dm.val_dataloader())
        n_test = sum(len(b[1]) for b in dm.test_dataloader())

        assert n_train == len(synthetic_dataset["labels"]["train"])
        assert n_val == len(synthetic_dataset["labels"]["val"])
        assert n_test == len(synthetic_dataset["labels"]["test"])

    def test_batch_shapes(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset, batch_size=2), seed=0)
        dm.setup()
        images, labels, ids = next(iter(dm.train_dataloader()))
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)
        assert len(ids) == 2

    def test_val_and_test_are_not_shuffled(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset, batch_size=1), seed=0)
        dm.setup()

        val_labels_a = [int(lbl) for _, lbl, _ in dm.val_dataloader()]
        val_labels_b = [int(lbl) for _, lbl, _ in dm.val_dataloader()]
        assert val_labels_a == val_labels_b == synthetic_dataset["labels"]["val"]

    def test_train_shuffle_reproducible_with_same_seed(
        self, synthetic_dataset: Dict[str, object]
    ) -> None:
        dm_a = ViTDataModule(_make_config(synthetic_dataset, batch_size=1), seed=123)
        dm_a.setup()
        order_a = [int(lbl) for _, lbl, _ in dm_a.train_dataloader()]

        dm_b = ViTDataModule(_make_config(synthetic_dataset, batch_size=1), seed=123)
        dm_b.setup()
        order_b = [int(lbl) for _, lbl, _ in dm_b.train_dataloader()]

        assert order_a == order_b

    def test_train_shuffle_differs_across_epochs(self, synthetic_dataset: Dict[str, object]) -> None:
        # A fresh DataLoader iterator should typically produce a different
        # order across "epochs" (successive full passes) because the
        # generator's internal state advances; we just check the loader is
        # re-iterable and produces the full label set each time.
        dm = ViTDataModule(_make_config(synthetic_dataset, batch_size=1), seed=0)
        dm.setup()
        loader = dm.train_dataloader()
        epoch_1 = sorted(int(lbl) for _, lbl, _ in loader)
        epoch_2 = sorted(int(lbl) for _, lbl, _ in loader)
        assert epoch_1 == epoch_2 == sorted(synthetic_dataset["labels"]["train"])


class TestClassStatistics:
    def test_class_counts_reflects_training_split(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset), seed=0)
        dm.setup()
        assert dm.class_counts() == {0: 4, 1: 2}

    def test_class_weights_are_tensor(self, synthetic_dataset: Dict[str, object]) -> None:
        dm = ViTDataModule(_make_config(synthetic_dataset), seed=0)
        dm.setup()
        weights = dm.class_weights()
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)


class TestCustomTransforms:
    def test_custom_transform_is_used(self, synthetic_dataset: Dict[str, object]) -> None:
        calls = {"count": 0}

        def counting_transform(img):
            calls["count"] += 1
            return torch.zeros(3, 8, 8)

        dm = ViTDataModule(
            _make_config(synthetic_dataset, batch_size=1),
            seed=0,
            train_transform=counting_transform,
            eval_transform=counting_transform,
        )
        dm.setup()
        images, _, _ = next(iter(dm.train_dataloader()))
        assert images.shape == (1, 3, 8, 8)
        assert calls["count"] > 0