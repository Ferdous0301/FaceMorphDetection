"""
Unit tests for src/preprocessing/dataset_inspector.py
Run with:  pytest tests/test_dataset_inspector.py -v
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest  # type: ignore[import]

# Make the src tree importable regardless of working directory
# Insert the project root (two levels up) to ensure the `src` package is found
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.dataset_inspector import (  # noqa: E402  # type: ignore[import]
    DatasetSummary,
    IdentityStats,
    ImageRecord,
    build_summary,
    collect_image_paths,
    discover_identity_dirs,
    export_csv,
    export_json,
    inspect_identity,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    """
    Build a minimal on-disk dataset layout:

    tmp_path/
        alice/
            img1.jpg
            img2.png
        bob/
            img3.jpg
        carol/          ← empty (no images)
    """
    root = tmp_path / "dataset"
    (root / "alice").mkdir(parents=True)
    (root / "bob").mkdir(parents=True)
    (root / "carol").mkdir(parents=True)

    # Create 1-byte placeholder files (not real images, PIL will call them
    # corrupted – that is intentional for corruption-detection tests)
    (root / "alice" / "img1.jpg").write_bytes(b"\xff")
    (root / "alice" / "img2.png").write_bytes(b"\xff")
    (root / "bob" / "img3.jpg").write_bytes(b"\xff")

    return root


@pytest.fixture()
def identity_stats_list() -> list[IdentityStats]:
    return [
        IdentityStats(
            identity="alice",
            dataset_root="/data",
            image_count=10,
            formats=["JPEG"],
            resolutions=["640x480"],
            corrupted_count=1,
        ),
        IdentityStats(
            identity="bob",
            dataset_root="/data",
            image_count=5,
            formats=["PNG"],
            resolutions=["1920x1080"],
            corrupted_count=0,
        ),
        IdentityStats(
            identity="carol",
            dataset_root="/data",
            image_count=20,
            formats=["JPEG", "PNG"],
            resolutions=["640x480", "1280x720"],
            corrupted_count=2,
        ),
    ]


# ---------------------------------------------------------------------------
# collect_image_paths
# ---------------------------------------------------------------------------


class TestCollectImagePaths:
    def test_finds_images_with_default_extensions(self, fake_dataset: Path) -> None:
        paths = list(collect_image_paths(fake_dataset / "alice"))
        names = {p.name for p in paths}
        assert names == {"img1.jpg", "img2.png"}

    def test_respects_custom_extensions(self, fake_dataset: Path) -> None:
        paths = list(collect_image_paths(fake_dataset / "alice", (".png",)))
        names = {p.name for p in paths}
        assert names == {"img2.png"}

    def test_empty_dir_returns_no_paths(self, fake_dataset: Path) -> None:
        paths = list(collect_image_paths(fake_dataset / "carol"))
        assert paths == []

    def test_nested_images_are_found(self, tmp_path: Path) -> None:
        nested = tmp_path / "id" / "subdir"
        nested.mkdir(parents=True)
        (nested / "deep.jpg").write_bytes(b"")
        paths = list(collect_image_paths(tmp_path / "id"))
        assert any(p.name == "deep.jpg" for p in paths)


# ---------------------------------------------------------------------------
# discover_identity_dirs
# ---------------------------------------------------------------------------


class TestDiscoverIdentityDirs:
    def test_returns_subdirectories(self, fake_dataset: Path) -> None:
        dirs = discover_identity_dirs(fake_dataset)
        names = {d.name for d in dirs}
        assert names == {"alice", "bob", "carol"}

    def test_raises_for_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(NotADirectoryError):
            discover_identity_dirs(missing)

    def test_raises_for_file_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("x")
        with pytest.raises(NotADirectoryError):
            discover_identity_dirs(file_path)


# ---------------------------------------------------------------------------
# inspect_identity
# ---------------------------------------------------------------------------


class TestInspectIdentity:
    def test_corrupted_images_are_detected(
    self,
    fake_dataset: Path,
) -> None:
    stats, records = inspect_identity(
        fake_dataset / "alice",
        fake_dataset,
    )

    assert stats.corrupted_count == len(records)
    assert all(record.corrupted for record in records)
    def test_image_count(self, fake_dataset: Path) -> None:
        stats, records = inspect_identity(
            fake_dataset / "alice", fake_dataset
        )
        assert stats.image_count == 2
        assert len(records) == 2

    def test_identity_name(self, fake_dataset: Path) -> None:
        stats, _ = inspect_identity(fake_dataset / "bob", fake_dataset)
        assert stats.identity == "bob"

    def test_empty_identity_has_zero_images(self, fake_dataset: Path) -> None:
        stats, records = inspect_identity(
            fake_dataset / "carol", fake_dataset
        )
        assert stats.image_count == 0
        assert records == []

    def test_records_carry_correct_identity(self, fake_dataset: Path) -> None:
        _, records = inspect_identity(fake_dataset / "alice", fake_dataset)
        assert all(r.identity == "alice" for r in records)


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_total_identities(
        self, identity_stats_list: list[IdentityStats]
    ) -> None:
        summary = build_summary(identity_stats_list, [Path("/data")])
        assert summary.total_identities == 3

    def test_total_images(
        self, identity_stats_list: list[IdentityStats]
    ) -> None:
        summary = build_summary(identity_stats_list, [Path("/data")])
        assert summary.total_images == 35  # 10 + 5 + 20

    def test_total_corrupted(
        self, identity_stats_list: list[IdentityStats]
    ) -> None:
        summary = build_summary(identity_stats_list, [Path("/data")])
        assert summary.total_corrupted == 3  # 1 + 0 + 2

    def test_images_per_identity_stats(
        self, identity_stats_list: list[IdentityStats]
    ) -> None:
        summary = build_summary(identity_stats_list, [Path("/data")])
        assert summary.images_per_identity_min == 5.0
        assert summary.images_per_identity_max == 20.0
        assert summary.images_per_identity_mean == pytest.approx(11.67, abs=0.01)

    def test_unique_formats(
        self, identity_stats_list: list[IdentityStats]
    ) -> None:
        summary = build_summary(identity_stats_list, [Path("/data")])
        assert set(summary.unique_formats) == {"JPEG", "PNG"}

    def test_empty_stats_list(self) -> None:
        summary = build_summary([], [Path("/data")])
        assert summary.total_identities == 0
        assert summary.total_images == 0
        assert summary.images_per_identity_min == 0.0


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_csv_has_header_and_rows(
        self,
        tmp_path: Path,
        identity_stats_list: list[IdentityStats],
    ) -> None:
        out = tmp_path / "out" / "report.csv"
        export_csv(identity_stats_list, out)

        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 3
        assert rows[0]["identity"] == "alice"
        assert rows[0]["image_count"] == "10"
        assert rows[0]["corrupted_count"] == "1"

    def test_creates_parent_directory(
        self,
        tmp_path: Path,
        identity_stats_list: list[IdentityStats],
    ) -> None:
        out = tmp_path / "deeply" / "nested" / "report.csv"
        export_csv(identity_stats_list, out)
        assert out.exists()

    def test_formats_are_pipe_separated(
        self,
        tmp_path: Path,
        identity_stats_list: list[IdentityStats],
    ) -> None:
        out = tmp_path / "report.csv"
        export_csv(identity_stats_list, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        carol_row = next(r for r in rows if r["identity"] == "carol")
        assert "|" in carol_row["formats"]


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_json_is_valid_and_contains_keys(self, tmp_path: Path) -> None:
        summary = DatasetSummary(
            dataset_roots=["/data"],
            total_identities=2,
            total_images=15,
            total_corrupted=1,
            images_per_identity_min=5.0,
            images_per_identity_max=10.0,
            images_per_identity_mean=7.5,
            images_per_identity_median=7.5,
            unique_formats=["JPEG"],
            unique_resolutions=["640x480"],
        )
        out = tmp_path / "summary.json"
        export_json(summary, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_identities"] == 2
        assert data["total_images"] == 15
        assert "unique_formats" in data

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        summary = DatasetSummary(
            dataset_roots=[],
            total_identities=0,
            total_images=0,
            total_corrupted=0,
            images_per_identity_min=0,
            images_per_identity_max=0,
            images_per_identity_mean=0,
            images_per_identity_median=0,
            unique_formats=[],
            unique_resolutions=[],
        )
        out = tmp_path / "sub" / "summary.json"
        export_json(summary, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliRun:
    def test_run_returns_zero_on_valid_dataset(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "reports"
        exit_code = run(
            [str(fake_dataset), "--output-dir", str(out_dir), "--log-level", "WARNING"]
        )
        assert exit_code == 0

    def test_run_creates_output_files(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "reports"
        run([str(fake_dataset), "--output-dir", str(out_dir), "--log-level", "WARNING"])
        assert (out_dir / "dataset_report.csv").exists()
        assert (out_dir / "dataset_summary.json").exists()

    def test_run_returns_nonzero_for_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        out_dir = tmp_path / "reports"
        exit_code = run(
            [str(missing), "--output-dir", str(out_dir), "--log-level", "WARNING"]
        )
        assert exit_code == 1

    def test_run_multiple_roots(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        # Build a second dataset
        ds2 = tmp_path / "dataset2"
        (ds2 / "dave").mkdir(parents=True)
        (ds2 / "dave" / "img.jpg").write_bytes(b"\xff")

        out_dir = tmp_path / "reports"
        exit_code = run(
            [
                str(fake_dataset),
                str(ds2),
                "--output-dir",
                str(out_dir),
                "--log-level",
                "WARNING",
            ]
        )
        assert exit_code == 0
        summary = json.loads((out_dir / "dataset_summary.json").read_text())
        assert summary["total_identities"] == 4  # alice, bob, carol, dave
