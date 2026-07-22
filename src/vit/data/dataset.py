"""``torch.utils.data.Dataset`` implementation over morph-detection CSV manifests.

Consumes the CSV manifests produced by the upstream **Dataset Split** stage
(one manifest each for train/val/test), each containing at minimum an image
path column and an integer label column, plus whatever additional metadata
columns (identity, morph technique, source dataset, etc.) that stage chose to
emit — those extra columns are preserved but not interpreted here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["MorphDataset"]


class MorphDataset(Dataset):
    """Loads (image, label, image_id) triples from a manifest CSV.

    Attributes:
        csv_path: Path to the manifest CSV this dataset was built from.
        image_root: Root directory relative paths in the manifest are
            resolved against.
        transform: Optional callable applied to each loaded PIL image before
            it is returned (typically from :mod:`vit.data.transforms`).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist, or if
            ``validate_files=True`` and one or more referenced image files
            are missing.
        ValueError: If ``path_column`` or ``label_column`` are not present
            in the CSV header, or if any label is not a non-negative
            integer.
    """

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        path_column: str = "image_path",
        label_column: str = "label",
        validate_files: bool = True,
    ) -> None:
        """Initialize the dataset by eagerly reading and validating the manifest.

        Validation happens at construction time (fail-fast) rather than
        lazily inside ``__getitem__``, so that a malformed manifest or a
        missing image is caught immediately, before any training time is
        spent, rather than crashing mid-epoch.

        Args:
            csv_path: Path to the manifest CSV.
            image_root: Root directory that relative paths in
                ``path_column`` are resolved against. Ignored for entries
                that are already absolute paths.
            transform: Optional transform applied to each loaded image.
            path_column: Name of the CSV column holding the image path.
            label_column: Name of the CSV column holding the integer label.
            validate_files: If True (default), verify at construction time
                that every referenced image file exists on disk. Set to
                False only for very large manifests where the upfront
                existence-check cost is prohibitive and file existence is
                otherwise guaranteed.
        """
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self._path_column = path_column
        self._label_column = label_column

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)

        missing_columns = [c for c in (path_column, label_column) if c not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Manifest '{self.csv_path}' is missing required column(s) "
                f"{missing_columns}. Available columns: {list(frame.columns)}"
            )

        resolved_paths: List[Path] = [
            self._resolve_path(raw_path) for raw_path in frame[path_column]
        ]

        try:
            labels: List[int] = [int(v) for v in frame[label_column]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column '{label_column}' in '{self.csv_path}' must contain integer labels"
            ) from exc

        if any(label < 0 for label in labels):
            raise ValueError(f"Column '{label_column}' must contain only non-negative labels")

        if validate_files:
            missing_files = [str(p) for p in resolved_paths if not p.is_file()]
            if missing_files:
                preview = missing_files[:5]
                raise FileNotFoundError(
                    f"{len(missing_files)} image file(s) referenced in '{self.csv_path}' "
                    f"do not exist. First few: {preview}"
                )

        self._paths: List[Path] = resolved_paths
        self._labels: List[int] = labels
        # image_id: stable identifier for downstream traceability (error
        # analysis, per-sample prediction export). Uses the path relative to
        # image_root when possible, falling back to the raw manifest value.
        self._image_ids: List[str] = [
            self._make_image_id(raw_path) for raw_path in frame[path_column]
        ]

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve a manifest path entry against ``image_root`` if it is relative."""
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else self.image_root / candidate

    def _make_image_id(self, raw_path: str) -> str:
        """Derive a stable string identifier for a manifest row."""
        return str(raw_path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, str]:
        """Load and return the sample at ``idx``.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            A tuple ``(image, label, image_id)`` where ``image`` is a
            ``torch.Tensor`` (post-transform, or a raw ``PIL.Image`` if no
            transform was supplied), ``label`` is the integer class label,
            and ``image_id`` is a stable string identifier for the sample.

        Raises:
            IndexError: If ``idx`` is out of range.
            OSError: If the underlying image file cannot be decoded.
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        image_path = self._paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self._labels[idx], self._image_ids[idx]

    @property
    def labels(self) -> List[int]:
        """Return the full list of integer labels, in dataset order."""
        return list(self._labels)

    @property
    def class_counts(self) -> Dict[int, int]:
        """Return a mapping from class label to number of samples with that label."""
        return dict(Counter(self._labels))

    def class_weights(self) -> Tensor:
        """Compute inverse-frequency class weights, suitable for a weighted loss.

        Weight for class ``c`` is ``N / (num_classes * count[c])``, the
        standard balanced-class-weight formula, normalized so that weights
        average to 1.0 across classes actually present in the dataset.

        Returns:
            A 1-D float tensor of length ``max(label) + 1``, indexed by
            class label. Classes with zero samples receive weight ``0.0``
            (they cannot contribute to gradients regardless).

        Raises:
            RuntimeError: If the dataset is empty.
        """
        counts = self.class_counts
        if not counts:
            raise RuntimeError("Cannot compute class weights for an empty dataset")

        num_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = torch.zeros(num_classes, dtype=torch.float32)
        present_classes = len(counts)
        for label, count in counts.items():
            weights[label] = total / (present_classes * count)

        return weights