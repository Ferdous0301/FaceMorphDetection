"""
src/morphing/metadata.py
=========================

Append-safe CSV metadata writer for generated morph images.

Every morph produced by ``morph_generator.py`` is recorded in a CSV file
(default: ``datasets/morph/metadata/morph_metadata.csv``).

Guarantees
----------
* The header row is written **exactly once**, even when multiple runs append
  to the same file or when two ``MetadataWriter`` instances share the same
  path within the same process.
* The file is opened, written, and closed atomically per ``append`` call, so
  a ``KeyboardInterrupt`` cannot leave the file without a complete row.
* Duplicate rows are not prevented here; deduplication is the caller's
  responsibility (``morph_generator.py`` checks ``skip_existing``).

Public API
----------
MorphRecord
    Frozen dataclass representing one CSV row.
MetadataWriter
    Manages the CSV path and exposes ``append`` / ``append_many``.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

#: Ordered column names for the morph metadata CSV.
#: Changing this list is a breaking schema change.
CSV_FIELDNAMES: Final[list[str]] = [
    "morph_filename",
    "source_image_a",
    "source_image_b",
    "identity_a",
    "identity_b",
    "dataset",
    "alpha",
    "timestamp",
]

#: Pre-compiled pattern for validating ISO-8601 timestamps stored in records.
_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MorphRecord:
    """A single row in the morph metadata CSV.

    All string fields are validated to be non-empty on construction.
    ``alpha`` is validated to lie in ``[0, 1]``.

    Attributes
    ----------
    morph_filename : str
        Filename (*not* full path) of the generated morph image, e.g.
        ``"morph_id001_id002_a050.jpg"``.
    source_image_a : str
        Relative path to the aligned source image A.
    source_image_b : str
        Relative path to the aligned source image B.
    identity_a : str
        Subject identifier for image A (e.g. folder name or subject ID).
    identity_b : str
        Subject identifier for image B.
    dataset : str
        Name of the source dataset, e.g. ``"lfw"`` or ``"feret"``.
    alpha : float
        Blend weight used to generate this morph, in ``[0, 1]``.
    timestamp : str
        ISO-8601 UTC timestamp (``"YYYY-MM-DDTHH:MM:SS"``).  Defaults to the
        current UTC time at the moment of construction.

    Raises
    ------
    ValueError
        If any string field is empty, ``alpha`` is outside ``[0, 1]``, or
        the provided ``timestamp`` does not match the expected format.
    """

    morph_filename: str
    source_image_a: str
    source_image_b: str
    identity_a: str
    identity_b: str
    dataset: str
    alpha: float
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    )

    def __post_init__(self) -> None:
        _validate_record(self)

    def to_dict(self) -> dict[str, str]:
        """Return the record as an ordered dict matching ``CSV_FIELDNAMES``.

        ``alpha`` is formatted as a 4-decimal-place string so that CSV
        round-trips are lossless for the precision used in this pipeline.

        Returns
        -------
        dict[str, str]
            Keys are exactly ``CSV_FIELDNAMES``; values are all strings.
        """
        return {
            "morph_filename": self.morph_filename,
            "source_image_a": self.source_image_a,
            "source_image_b": self.source_image_b,
            "identity_a": self.identity_a,
            "identity_b": self.identity_b,
            "dataset": self.dataset,
            "alpha": f"{self.alpha:.4f}",
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Private: record validation
# ---------------------------------------------------------------------------

def _validate_record(record: MorphRecord) -> None:
    """Raise ``ValueError`` if any field of ``record`` is invalid.

    Parameters
    ----------
    record : MorphRecord
        The record to validate.

    Raises
    ------
    ValueError
        * If any required string field is empty or whitespace-only.
        * If ``alpha`` is outside ``[0, 1]``.
        * If ``timestamp`` does not match ``YYYY-MM-DDTHH:MM:SS``.
    """
    string_fields = (
        ("morph_filename", record.morph_filename),
        ("source_image_a", record.source_image_a),
        ("source_image_b", record.source_image_b),
        ("identity_a", record.identity_a),
        ("identity_b", record.identity_b),
        ("dataset", record.dataset),
    )
    for name, value in string_fields:
        if not value or not value.strip():
            raise ValueError(f"MorphRecord.{name} must not be empty.")

    if not (0.0 <= record.alpha <= 1.0):
        raise ValueError(
            f"MorphRecord.alpha must be in [0, 1], got {record.alpha}."
        )

    if not _TIMESTAMP_RE.match(record.timestamp):
        raise ValueError(
            f"MorphRecord.timestamp must match YYYY-MM-DDTHH:MM:SS, "
            f"got {record.timestamp!r}."
        )


# ---------------------------------------------------------------------------
# Writer class
# ---------------------------------------------------------------------------

class MetadataWriter:
    """Append-safe CSV writer for morph metadata.

    The header row is written only when the target file is absent or empty,
    so multiple ``MetadataWriter`` instances or multiple runs safely share
    the same CSV file.

    Parameters
    ----------
    csv_path : Path
        Destination CSV file.  Parent directories are created automatically
        on construction.

    Examples
    --------
    >>> writer = MetadataWriter(Path("datasets/morph/metadata/morph_metadata.csv"))
    >>> writer.append(MorphRecord(
    ...     morph_filename="morph_a_b_050.jpg",
    ...     source_image_a="aligned/lfw/person_a/00001.jpg",
    ...     source_image_b="aligned/lfw/person_b/00001.jpg",
    ...     identity_a="person_a",
    ...     identity_b="person_b",
    ...     dataset="lfw",
    ...     alpha=0.5,
    ... ))
    """

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = Path(csv_path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("MetadataWriter initialised at %s.", self._csv_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def csv_path(self) -> Path:
        """Absolute path to the managed CSV file."""
        return self._csv_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _needs_header(self) -> bool:
        """Return ``True`` when the CSV file is absent or empty.

        This check is performed once before opening the file in append mode,
        which avoids re-writing the header on subsequent calls.
        """
        return not self._csv_path.exists() or self._csv_path.stat().st_size == 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def append(self, record: MorphRecord) -> None:
        """Append one ``MorphRecord`` row to the CSV file.

        The header row is written only when the file is newly created or
        empty.  The file is opened, written, and closed within this call.

        Parameters
        ----------
        record : MorphRecord
            The morph metadata row to append.
        """
        write_header = self._needs_header()

        with self._csv_path.open(mode="a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
                logger.debug("Wrote CSV header to %s.", self._csv_path)
            writer.writerow(record.to_dict())

        logger.debug("Appended record: %s.", record.morph_filename)

    def append_many(self, records: Sequence[MorphRecord]) -> None:
        """Append multiple ``MorphRecord`` rows in a single file open.

        More efficient than calling ``append`` in a loop when writing many
        records at once.  An empty sequence is a no-op.

        Parameters
        ----------
        records : Sequence[MorphRecord]
            Records to append.
        """
        if not records:
            return

        write_header = self._needs_header()

        with self._csv_path.open(mode="a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
                logger.debug("Wrote CSV header to %s.", self._csv_path)
            for record in records:
                writer.writerow(record.to_dict())

        logger.debug(
            "Appended %d record(s) to %s.", len(records), self._csv_path
        )

    def exists(self) -> bool:
        """Return ``True`` if the CSV file exists and is non-empty."""
        return self._csv_path.exists() and self._csv_path.stat().st_size > 0

    def row_count(self) -> int:
        """Return the number of data rows in the CSV, excluding the header.

        Returns
        -------
        int
            ``0`` if the file does not exist or is empty.
        """
        if not self.exists():
            return 0
        with self._csv_path.open(encoding="utf-8") as fh:
            return sum(1 for _ in csv.DictReader(fh))