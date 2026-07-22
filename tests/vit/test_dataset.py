"""Unit tests for vit.data.dataset.MorphDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch
from PIL import Image

from vit.data.dataset import MorphDataset
from vit.data.transforms import build_eval_transforms


class TestConstruction:
    def test_valid_dataset_length(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        assert len(ds) == len(synthetic_dataset["labels"]["train"])

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MorphDataset(csv_path=tmp_path / "nope.csv", image_root=tmp_path)

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"path": ["a.png"], "label": [0]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            MorphDataset(csv_path=csv_path, image_root=tmp_path)  # expects "image_path"

    def test_missing_image_file_raises_when_validating(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "manifest.csv"
        pd.DataFrame({"image_path": ["does_not_exist.png"], "label": [0]}).to_csv(
            csv_path, index=False
        )
        with pytest.raises(FileNotFoundError):
            MorphDataset(csv_path=csv_path, image_root=tmp_path, validate_files=True)

    def test_missing_image_file_allowed_when_not_validating(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "manifest.csv"
        pd.DataFrame({"image_path": ["does_not_exist.png"], "label": [0]}).to_csv(
            csv_path, index=False
        )
        ds = MorphDataset(csv_path=csv_path, image_root=tmp_path, validate_files=False)
        assert len(ds) == 1

    def test_negative_label_raises(self, tmp_path: Path) -> None:
        img_path = tmp_path / "a.png"
        Image.new("RGB", (10, 10)).save(img_path)
        csv_path = tmp_path / "manifest.csv"
        pd.DataFrame({"image_path": ["a.png"], "label": [-1]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            MorphDataset(csv_path=csv_path, image_root=tmp_path)

    def test_custom_column_names(self, tmp_path: Path) -> None:
        img_path = tmp_path / "a.png"
        Image.new("RGB", (10, 10)).save(img_path)
        csv_path = tmp_path / "manifest.csv"
        pd.DataFrame({"path_col": ["a.png"], "label_col": [1]}).to_csv(csv_path, index=False)
        ds = MorphDataset(
            csv_path=csv_path,
            image_root=tmp_path,
            path_column="path_col",
            label_column="label_col",
        )
        assert len(ds) == 1
        assert ds.labels == [1]

    def test_absolute_paths_supported(self, tmp_path: Path) -> None:
        img_path = tmp_path / "abs.png"
        Image.new("RGB", (10, 10)).save(img_path)
        csv_path = tmp_path / "manifest.csv"
        pd.DataFrame({"image_path": [str(img_path)], "label": [0]}).to_csv(csv_path, index=False)
        # image_root deliberately points somewhere irrelevant to prove
        # absolute paths bypass it.
        ds = MorphDataset(csv_path=csv_path, image_root=tmp_path / "irrelevant", validate_files=True)
        assert len(ds) == 1


class TestGetItem:
    def test_returns_tensor_label_id_with_transform(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
            transform=build_eval_transforms(32),
        )
        image, label, image_id = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert label == synthetic_dataset["labels"]["train"][0]
        assert isinstance(image_id, str)

    def test_returns_pil_image_without_transform(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        image, _, _ = ds[0]
        assert isinstance(image, Image.Image)

    def test_out_of_range_index_raises(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_labels_match_manifest_order(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        assert ds.labels == synthetic_dataset["labels"]["train"]


class TestClassStatistics:
    def test_class_counts(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        # train fixture: 4 bona fide (0), 2 morph (1)
        assert ds.class_counts == {0: 4, 1: 2}

    def test_class_weights_shape_and_balance(self, synthetic_dataset: Dict[str, object]) -> None:
        ds = MorphDataset(
            csv_path=synthetic_dataset["train_csv"],
            image_root=synthetic_dataset["image_root"],
        )
        weights = ds.class_weights()
        assert weights.shape == (2,)
        # Minority class (1, count=2) should get a larger weight than
        # majority class (0, count=4).
        assert weights[1] > weights[0]

    def test_class_weights_raises_on_empty_dataset(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame({"image_path": [], "label": []}).to_csv(csv_path, index=False)
        ds = MorphDataset(csv_path=csv_path, image_root=tmp_path)
        with pytest.raises(RuntimeError):
            ds.class_weights()