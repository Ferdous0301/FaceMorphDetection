"""Append-only CSV logging for epoch-level metrics.

``CSVLogger`` writes one row per call to :meth:`CSVLogger.log`, lazily
determining the header from the keys of the first row and validating that
every subsequent row uses the exact same set of keys (a mismatch usually
indicates a bug upstream, e.g. a metric silently dropped for one split).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["CSVLogger"]


class CSVLogger:
    """Writes dict rows to a CSV file, one row per call to :meth:`log`.

    Args:
        path: Destination CSV file path. Parent directories are created if
            they do not exist. If the file already exists it is truncated
            (fresh log per run).

    Example:
        >>> logger = CSVLogger(Path("logs/vit/train_log.csv"))
        >>> logger.log({"epoch": 0, "loss": 0.693, "accuracy": 0.5})
        >>> logger.log({"epoch": 1, "loss": 0.421, "accuracy": 0.71})
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: Optional[List[str]] = None

        # Truncate any existing file so each run starts a fresh log.
        self._path.write_text("", encoding="utf-8")

    def log(self, row: Dict[str, Any]) -> None:
        """Append a single row to the CSV file.

        On the first call, the header is derived from ``row``'s keys (in
        insertion order) and written before the row itself.

        Args:
            row: Mapping of column name to value for this row.

        Raises:
            ValueError: If ``row`` is empty, or if its keys do not match the
                header established by the first call to :meth:`log`.
        """
        if not row:
            raise ValueError("Cannot log an empty row")

        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self._path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

        if set(row.keys()) != set(self._fieldnames):
            raise ValueError(
                f"Row keys {sorted(row.keys())} do not match established "
                f"header {sorted(self._fieldnames)}"
            )

        with self._path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    @property
    def path(self) -> Path:
        """The CSV file path this logger writes to."""
        return self._path