"""Unit tests for the ManifestWriter class.

These tests verify:
    * The header row is written exactly once.
    * Row counts match the number of assignments written.
    * Row ordering is deterministic (sorted by image_id).
    * Overwrite mode truncates and replaces existing content.
    * Append mode adds rows without duplicating the header.
    * Empty manifests still write a valid header-only file.
    * The output directory is created automatically if missing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.dataset_split.manifest_writer import (
    MANIFEST_FIELDNAMES,
    ManifestWriter,
)
from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment


def _assignment(image_id: str, split: str = "train") -> SplitAssignment:
    """Build a simple SplitAssignment for test purposes."""
    return SplitAssignment(
        image_id=image_id,
        split=split,
        label="bona_fide",
        identities=(f"identity_{image_id}",),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read all rows of a CSV manifest file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestHeaderWrittenOnce:
    """Tests verifying the CSV header is written exactly once."""

    def test_header_present_on_fresh_write(self, tmp_path: Path) -> None:
        """A fresh write produces exactly one header row."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("img1")])

        path = writer.manifest_path("train")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(MANIFEST_FIELDNAMES)
        assert lines.count(",".join(MANIFEST_FIELDNAMES)) == 1

    def test_header_not_duplicated_on_append(self, tmp_path: Path) -> None:
        """Appending twice does not duplicate the header row."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("img1")], mode="overwrite")
        writer.write_split("train", [_assignment("img2")], mode="append")

        path = writer.manifest_path("train")
        content = path.read_text(encoding="utf-8")
        header_line = ",".join(MANIFEST_FIELDNAMES)
        assert content.count(header_line) == 1

    def test_header_written_for_empty_split_via_overwrite(
        self, tmp_path: Path
    ) -> None:
        """Overwriting with an empty assignment list still writes a header."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [])
        path = writer.manifest_path("train")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == [",".join(MANIFEST_FIELDNAMES)]


class TestRowCount:
    """Tests verifying the correct number of rows are written."""

    def test_row_count_matches_assignments(self, tmp_path: Path) -> None:
        """The number of data rows matches the number of assignments."""
        writer = ManifestWriter(tmp_path)
        assignments = [_assignment(f"img{i}") for i in range(5)]
        writer.write_split("train", assignments)

        rows = _read_csv_rows(writer.manifest_path("train"))
        assert len(rows) == 5

    def test_row_count_after_append(self, tmp_path: Path) -> None:
        """Row count accumulates correctly across an append operation."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("img1"), _assignment("img2")])
        writer.write_split("train", [_assignment("img3")], mode="append")

        rows = _read_csv_rows(writer.manifest_path("train"))
        assert len(rows) == 3


class TestOrdering:
    """Tests verifying deterministic row ordering."""

    def test_rows_sorted_by_image_id(self, tmp_path: Path) -> None:
        """Rows are written in sorted image_id order regardless of input order."""
        writer = ManifestWriter(tmp_path)
        assignments = [
            _assignment("img3"),
            _assignment("img1"),
            _assignment("img2"),
        ]
        writer.write_split("train", assignments)

        rows = _read_csv_rows(writer.manifest_path("train"))
        image_ids = [row["image_id"] for row in rows]
        assert image_ids == ["img1", "img2", "img3"]

    def test_ordering_is_stable_across_multiple_writes(
        self, tmp_path: Path
    ) -> None:
        """Repeated overwrites with the same input yield identical ordering."""
        writer = ManifestWriter(tmp_path)
        assignments = [_assignment("img5"), _assignment("img1"), _assignment("img3")]

        writer.write_split("train", assignments)
        first_rows = _read_csv_rows(writer.manifest_path("train"))

        writer.write_split("train", assignments)
        second_rows = _read_csv_rows(writer.manifest_path("train"))

        assert first_rows == second_rows


class TestOverwriteBehavior:
    """Tests verifying overwrite mode truncates prior content."""

    def test_overwrite_replaces_previous_rows(self, tmp_path: Path) -> None:
        """Overwriting replaces old rows entirely rather than appending."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("old1"), _assignment("old2")])
        writer.write_split("train", [_assignment("new1")], mode="overwrite")

        rows = _read_csv_rows(writer.manifest_path("train"))
        assert [row["image_id"] for row in rows] == ["new1"]

    def test_overwrite_is_default_mode(self, tmp_path: Path) -> None:
        """Calling write_split without a mode argument overwrites."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("old1")])
        writer.write_split("train", [_assignment("new1")])

        rows = _read_csv_rows(writer.manifest_path("train"))
        assert [row["image_id"] for row in rows] == ["new1"]


class TestAppendBehavior:
    """Tests verifying append mode preserves and extends prior content."""

    def test_append_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Appending keeps previously written rows intact."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("img1")])
        writer.write_split("train", [_assignment("img2")], mode="append")

        rows = _read_csv_rows(writer.manifest_path("train"))
        image_ids = {row["image_id"] for row in rows}
        assert image_ids == {"img1", "img2"}

    def test_append_to_nonexistent_file_creates_it_with_header(
        self, tmp_path: Path
    ) -> None:
        """Appending to a file that does not yet exist creates it properly."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("train", [_assignment("img1")], mode="append")

        path = writer.manifest_path("train")
        assert path.exists()
        rows = _read_csv_rows(path)
        assert len(rows) == 1


class TestEmptyManifest:
    """Tests for writing manifests with no assignments."""

    def test_empty_result_writes_header_only_files(
        self, tmp_path: Path
    ) -> None:
        """Writing an empty DatasetSplitResult produces header-only files."""
        writer = ManifestWriter(tmp_path)
        result = DatasetSplitResult()
        writer.write_result(result)

        for split_name in ("train", "val", "test"):
            rows = _read_csv_rows(writer.manifest_path(split_name))
            assert rows == []

    def test_empty_manifest_has_valid_header(self, tmp_path: Path) -> None:
        """An empty manifest's header still matches the expected fieldnames."""
        writer = ManifestWriter(tmp_path)
        writer.write_split("val", [])
        with writer.manifest_path("val").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader)
        assert tuple(header) == MANIFEST_FIELDNAMES


class TestPathCreation:
    """Tests verifying output directories are created automatically."""

    def test_nested_output_directory_is_created(self, tmp_path: Path) -> None:
        """A deeply nested, non-existent output directory is created."""
        nested_dir = tmp_path / "a" / "b" / "c"
        writer = ManifestWriter(nested_dir)
        writer.write_split("train", [_assignment("img1")])

        assert nested_dir.exists()
        assert writer.manifest_path("train").exists()

    def test_output_directory_accepts_string_path(self, tmp_path: Path) -> None:
        """A string output directory is normalized to a pathlib.Path."""
        writer = ManifestWriter(str(tmp_path / "manifests"))
        assert isinstance(writer.output_directory, Path)
        writer.write_split("test", [_assignment("img1")])
        assert writer.manifest_path("test").exists()


class TestWriteResultIntegration:
    """Integration tests exercising write_result with a full split result."""

    def test_write_result_writes_all_three_files(self, tmp_path: Path) -> None:
        """write_result produces train.csv, validation.csv, and test.csv."""
        writer = ManifestWriter(tmp_path)
        result = DatasetSplitResult(
            train=(_assignment("t1", "train"),),
            val=(_assignment("v1", "val"),),
            test=(_assignment("s1", "test"),),
        )
        writer.write_result(result)

        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "validation.csv").exists()
        assert (tmp_path / "test.csv").exists()

    def test_write_result_content_matches_assignments(
        self, tmp_path: Path
    ) -> None:
        """Written manifest content matches the provided split result."""
        writer = ManifestWriter(tmp_path)
        result = DatasetSplitResult(
            train=(_assignment("t1", "train"), _assignment("t2", "train")),
        )
        writer.write_result(result)

        rows = _read_csv_rows(writer.manifest_path("train"))
        assert {row["image_id"] for row in rows} == {"t1", "t2"}


class TestInvalidInputs:
    """Tests for invalid split names and write modes."""

    def test_unknown_split_name_raises(self, tmp_path: Path) -> None:
        """An unrecognized split name raises ValueError."""
        writer = ManifestWriter(tmp_path)
        with pytest.raises(ValueError):
            writer.manifest_path("bogus")

    def test_unknown_write_mode_raises(self, tmp_path: Path) -> None:
        """An unrecognized write mode raises ValueError."""
        writer = ManifestWriter(tmp_path)
        with pytest.raises(ValueError):
            writer.write_split("train", [], mode="bogus")  # type: ignore[arg-type]