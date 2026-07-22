"""Shared pytest fixtures for the ``vit`` test suite.

Provides a tiny, fully synthetic image dataset (a handful of small PNGs plus
matching CSV manifests) so that dataset/datamodule/model tests never depend
on the real face-morph dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest
from PIL import Image


def _make_solid_image(path: Path, color: Tuple[int, int, int], size: Tuple[int, int] = (32, 32)) -> None:
    """Write a tiny solid-color RGB PNG to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Dict[str, object]:
    """Build a tiny synthetic bona-fide/morph dataset with train/val/test manifests.

    Layout:
        tmp_path/images/{train,val,test}_XXX.png
        tmp_path/train.csv, val.csv, test.csv  (columns: image_path, label)

    Returns:
        A dict with keys ``"image_root"``, ``"train_csv"``, ``"val_csv"``,
        ``"test_csv"``, and ``"labels"`` (a dict split -> list[int] of the
        labels written, for use in assertions).
    """
    image_root = tmp_path / "images"

    # 6 train (4 bona fide / 2 morph -> deliberately imbalanced),
    # 4 val (2/2), 4 test (2/2).
    split_specs: Dict[str, List[int]] = {
        "train": [0, 0, 0, 0, 1, 1],
        "val": [0, 0, 1, 1],
        "test": [0, 0, 1, 1],
    }

    csv_paths: Dict[str, Path] = {}
    labels_by_split: Dict[str, List[int]] = {}

    for split, labels in split_specs.items():
        rows = []
        for i, label in enumerate(labels):
            rel_path = f"{split}_{i:03d}.png"
            color = (200, 50, 50) if label == 1 else (50, 200, 50)
            _make_solid_image(image_root / rel_path, color=color)
            rows.append({"image_path": rel_path, "label": label})

        frame = pd.DataFrame(rows)
        csv_path = tmp_path / f"{split}.csv"
        frame.to_csv(csv_path, index=False)
        csv_paths[split] = csv_path
        labels_by_split[split] = labels

    return {
        "image_root": image_root,
        "train_csv": csv_paths["train"],
        "val_csv": csv_paths["val"],
        "test_csv": csv_paths["test"],
        "labels": labels_by_split,
    }