"""CSV manifest writing for the Dataset Split stage.

This module provides :class:`ManifestWriter`, which persists the split
assignments produced by :mod:`src.dataset_split.splitter` to
``train.csv``, ``validation.csv``, and ``test.csv`` manifest files.

Writing is deterministic (rows are always emitted in a stable,
sorted order), append-safe (headers are never duplicated when appending
to an existing file), UTF-8 encoded, and uses :mod:`pathlib` throughout.
This module contains manifest I/O only; it performs no splitting logic.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment

#: CSV column headers, written exactly once per manifest file.
MANIFEST_FIELDNAMES: tuple[str, str, str, str] = (
    "image_id",
    "split",
    "label",
    "identities",
)

#: Mapping from logical split name to its manifest filename.
MANIFEST_FILENAMES: dict[str, str] = {
    "train": "train.csv",
    "val": "validation.csv",
    "test": "test.csv",
}

#: Separator used to join multiple identities within a single CSV cell.
_IDENTITY_SEPARATOR = "|"

WriteMode = Literal["overwrite", "append"]


class ManifestWriter:
    """Writes dataset split manifests to CSV files in a stable, safe manner.

    Manifests are written to a configurable output directory as
    ``train.csv``, ``validation.csv``, and ``test.csv``. Rows within each
    manifest are always emitted in a deterministic order (sorted by
    ``image_id``), regardless of the order in which assignments are
    supplied. Writing is append-safe: the CSV header row is written
    exactly once per file, whether the file is being created,
    overwritten, or appended to.
    """

    def __init__(self, output_directory: Path | str) -> None:
        """Initialize the manifest writer.

        Args:
            output_directory: Directory into which manifest CSV files
                are written. Created automatically (including parent
                directories) if it does not already exist.
        """
        self._output_directory = Path(output_directory)

    @property
    def output_directory(self) -> Path:
        """Return the configured output directory as a ``pathlib.Path``."""
        return self._output_directory

    def manifest_path(self, split_name: str) -> Path:
        """Return the manifest file path for a given split name.

        Args:
            split_name: One of ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            The full path to that split's manifest CSV file.

        Raises:
            ValueError: If ``split_name`` is not recognized.
        """
        if split_name not in MANIFEST_FILENAMES:
            raise ValueError(f"Unknown split name: {split_name!r}.")
        return self._output_directory / MANIFEST_FILENAMES[split_name]

    def write_result(
        self, result: DatasetSplitResult, mode: WriteMode = "overwrite"
    ) -> None:
        """Write all three manifests from a complete split result.

        Args:
            result: The dataset split result containing train, val, and
                test assignments.
            mode: ``"overwrite"`` truncates and rewrites each manifest
                from scratch; ``"append"`` adds rows to any existing
                manifest without duplicating the header row.
        """
        self.write_split("train", result.train, mode=mode)
        self.write_split("val", result.val, mode=mode)
        self.write_split("test", result.test, mode=mode)

    def write_split(
        self,
        split_name: str,
        assignments: Sequence[SplitAssignment],
        mode: WriteMode = "overwrite",
    ) -> None:
        """Write a single split's assignments to its manifest CSV file.

        Args:
            split_name: One of ``"train"``, ``"val"``, or ``"test"``.
            assignments: The split assignments to write. May be empty,
                in which case only the header row is written (or, in
                append mode with an existing non-empty file, nothing
                is written at all).
            mode: ``"overwrite"`` truncates and rewrites the manifest
                from scratch; ``"append"`` adds rows to any existing
                manifest without duplicating the header row.

        Raises:
            ValueError: If ``split_name`` is not recognized or ``mode``
                is not one of ``"overwrite"`` or ``"append"``.
        """
        if mode not in ("overwrite", "append"):
            raise ValueError(f"Unknown write mode: {mode!r}.")

        path = self.manifest_path(split_name)
        self._output_directory.mkdir(parents=True, exist_ok=True)

        ordered_rows = self._ordered_rows(assignments)
        write_header = self._should_write_header(path, mode)
        file_mode = "a" if mode == "append" else "w"

        with path.open(mode=file_mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(ordered_rows)

    def _should_write_header(self, path: Path, mode: WriteMode) -> bool:
        """Determine whether a header row needs to be written.

        Args:
            path: The manifest file path being written to.
            mode: The write mode in effect.

        Returns:
            ``True`` if a header row should be written: always true for
            ``"overwrite"`` mode, and true for ``"append"`` mode only
            when the target file does not yet exist or is empty.
        """
        if mode == "overwrite":
            return True
        return not path.exists() or path.stat().st_size == 0

    def _ordered_rows(
        self, assignments: Sequence[SplitAssignment]
    ) -> list[dict[str, str]]:
        """Convert assignments into deterministically ordered CSV rows.

        Args:
            assignments: The split assignments to convert.

        Returns:
            A list of row dictionaries, sorted by ``image_id`` to
            guarantee deterministic output regardless of input order.
        """
        sorted_assignments = sorted(assignments, key=lambda a: a.image_id)
        return [
            {
                "image_id": assignment.image_id,
                "split": assignment.split,
                "label": assignment.label,
                "identities": _IDENTITY_SEPARATOR.join(assignment.identities),
            }
            for assignment in sorted_assignments
        ]