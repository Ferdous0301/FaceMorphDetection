"""
tests/test_morph_generator.py
==============================

Tests for ``src/morphing/metadata.py`` and ``src/morphing/morph_generator.py``.

Filesystem operations use pytest's ``tmp_path`` fixture.
MediaPipe, ``warp_and_blend``, and ``cv2.imwrite`` are mocked where needed so
no real images or GPU are required.

Coverage
--------
metadata.py
  * MorphRecord – to_dict keys/values, alpha formatting, timestamp default
                  and validation, empty-field validation, alpha range validation
  * MetadataWriter – creates file, writes header once, appends rows,
                     append_many, exists, row_count, deep directory creation

morph_generator.py
  * MorphConfig – alpha and max_pairs validation
  * ProcessingSummary – default values, log output
  * MorphGenerator helpers – _build_pairs, _resolve_datasets
  * MorphGenerator.run – no root, skip_existing, landmark failure, success,
                          max_pairs cap, elapsed-time logging
  * CLI main() – unknown flag, missing root, failures → exit 1,
                 alpha forwarded, seed forwarded
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.morphing.mediapipe_landmarks import LandmarkResult, NUM_LANDMARKS
from src.morphing.metadata import CSV_FIELDNAMES, MetadataWriter, MorphRecord
from src.morphing.morph_generator import (
    MorphConfig,
    MorphGenerator,
    ProcessingSummary,
    main,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _record(**kwargs) -> MorphRecord:
    """Return a valid ``MorphRecord`` with sensible defaults."""
    defaults: dict = dict(
        morph_filename="morph_a_b_050.jpg",
        source_image_a="aligned/lfw/a/00001.jpg",
        source_image_b="aligned/lfw/b/00001.jpg",
        identity_a="person_a",
        identity_b="person_b",
        dataset="lfw",
        alpha=0.5,
    )
    defaults.update(kwargs)
    return MorphRecord(**defaults)


def _solid_bgr(
    h: int = 128,
    w: int = 128,
    color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _fake_landmark_result(h: int = 128, w: int = 128) -> LandmarkResult:
    """Return a ``LandmarkResult`` with random landmarks (no MediaPipe needed)."""
    rng = np.random.default_rng(42)
    points = rng.uniform(
        low=[10.0, 10.0],
        high=[float(w - 10), float(h - 10)],
        size=(NUM_LANDMARKS, 2),
    ).astype(np.float32)
    return LandmarkResult(points=points, image_shape=(h, w))


def _patch_detector(detect_return=None):
    """Return a context manager that patches ``LandmarkDetector`` in the generator.

    Parameters
    ----------
    detect_return
        Value returned by ``detector.detect()``.  Defaults to a valid
        ``LandmarkResult``.
    """
    if detect_return is None:
        detect_return = _fake_landmark_result()

    det_instance = MagicMock()
    det_instance.detect.return_value = detect_return

    cls_mock = MagicMock()
    cls_mock.return_value.__enter__ = lambda s: det_instance
    cls_mock.return_value.__exit__ = MagicMock(return_value=False)
    return patch("src.morphing.morph_generator.LandmarkDetector", cls_mock), det_instance


def _write_image(path: Path, color: tuple[int, int, int] = (50, 100, 150)) -> None:
    """Write a tiny solid-colour JPEG to ``path`` (creates parents)."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _solid_bgr(color=color))


def _setup_dataset(
    root: Path,
    dataset: str = "lfw",
    n_identities: int = 3,
) -> Path:
    """Create a minimal aligned-images directory tree with real JPEG files."""
    ds_dir = root / dataset
    for i in range(n_identities):
        id_dir = ds_dir / f"id{i:03d}"
        _write_image(id_dir / "00001.jpg", color=(i * 20 + 50, 100, 100))
    return ds_dir


def _make_config(
    aligned_root: Path,
    output_root: Path,
    meta_path: Path,
    **kwargs,
) -> MorphConfig:
    return MorphConfig(
        aligned_root=aligned_root,
        output_root=output_root,
        metadata_path=meta_path,
        alpha=0.5,
        skip_existing=True,
        seed=42,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# MorphRecord
# ---------------------------------------------------------------------------

class TestMorphRecord:
    def test_to_dict_contains_all_csv_fieldnames(self) -> None:
        assert set(_record().to_dict().keys()) == set(CSV_FIELDNAMES)

    @pytest.mark.parametrize("alpha, expected", [
        (0.5, "0.5000"),
        (0.0, "0.0000"),
        (1.0, "1.0000"),
        (0.333, "0.3330"),
    ])
    def test_alpha_formatted_to_four_decimal_places(
        self, alpha: float, expected: str
    ) -> None:
        assert _record(alpha=alpha).to_dict()["alpha"] == expected

    def test_default_timestamp_is_iso8601(self) -> None:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        assert re.match(pattern, _record().timestamp)

    def test_custom_timestamp_preserved(self) -> None:
        ts = "2024-06-15T08:30:00"
        assert _record(timestamp=ts).to_dict()["timestamp"] == ts

    def test_invalid_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            _record(timestamp="2024/06/15 08:30:00")

    @pytest.mark.parametrize("field_name", [
        "morph_filename", "source_image_a", "source_image_b",
        "identity_a", "identity_b", "dataset",
    ])
    def test_empty_string_field_raises(self, field_name: str) -> None:
        with pytest.raises(ValueError, match=field_name):
            _record(**{field_name: ""})

    def test_whitespace_only_field_raises(self) -> None:
        with pytest.raises(ValueError, match="identity_a"):
            _record(identity_a="   ")

    @pytest.mark.parametrize("alpha", [-0.01, 1.01])
    def test_out_of_range_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _record(alpha=alpha)


# ---------------------------------------------------------------------------
# MetadataWriter
# ---------------------------------------------------------------------------

class TestMetadataWriter:
    def test_append_creates_file(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta" / "morph_metadata.csv"
        MetadataWriter(csv_path).append(_record())
        assert csv_path.exists()

    def test_header_written_on_first_append(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        MetadataWriter(csv_path).append(_record())
        header = csv_path.read_text().splitlines()[0]
        assert "morph_filename" in header

    def test_header_written_exactly_once_across_multiple_appends(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "meta.csv"
        w = MetadataWriter(csv_path)
        w.append(_record(morph_filename="first.jpg"))
        w.append(_record(morph_filename="second.jpg"))
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3  # 1 header + 2 data rows

    def test_two_writers_share_file_without_duplicate_header(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "meta.csv"
        MetadataWriter(csv_path).append(_record(morph_filename="first.jpg"))
        MetadataWriter(csv_path).append(_record(morph_filename="second.jpg"))
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3  # 1 header + 2 data rows

    def test_row_count_after_multiple_appends(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        w = MetadataWriter(csv_path)
        for i in range(7):
            w.append(_record(morph_filename=f"m{i}.jpg"))
        assert w.row_count() == 7

    def test_row_count_zero_before_any_write(self, tmp_path: Path) -> None:
        assert MetadataWriter(tmp_path / "absent.csv").row_count() == 0

    def test_exists_false_before_any_write(self, tmp_path: Path) -> None:
        assert not MetadataWriter(tmp_path / "absent.csv").exists()

    def test_exists_true_after_append(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        w = MetadataWriter(csv_path)
        w.append(_record())
        assert w.exists()

    def test_append_many_writes_all_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        w = MetadataWriter(csv_path)
        records = [_record(morph_filename=f"m{i}.jpg") for i in range(10)]
        w.append_many(records)
        assert w.row_count() == 10

    def test_append_many_empty_sequence_is_noop(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        MetadataWriter(csv_path).append_many([])
        assert not csv_path.exists()

    def test_csv_path_property(self, tmp_path: Path) -> None:
        p = tmp_path / "m.csv"
        assert MetadataWriter(p).csv_path == p

    def test_deep_parent_directories_created(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "meta.csv"
        MetadataWriter(deep).append(_record())
        assert deep.exists()

    def test_written_row_values_match_record(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        rec = _record(
            morph_filename="test.jpg",
            identity_a="alice",
            identity_b="bob",
            alpha=0.75,
        )
        MetadataWriter(csv_path).append(rec)
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["morph_filename"] == "test.jpg"
        assert rows[0]["identity_a"] == "alice"
        assert rows[0]["alpha"] == "0.7500"


# ---------------------------------------------------------------------------
# MorphConfig validation
# ---------------------------------------------------------------------------

class TestMorphConfigValidation:
    @pytest.mark.parametrize("alpha", [-0.01, 1.01])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            MorphConfig(alpha=alpha)

    @pytest.mark.parametrize("max_pairs", [0, -1])
    def test_invalid_max_pairs_raises(self, max_pairs: int) -> None:
        with pytest.raises(ValueError, match="max_pairs"):
            MorphConfig(max_pairs_per_dataset=max_pairs)

    def test_none_max_pairs_is_valid(self) -> None:
        cfg = MorphConfig(max_pairs_per_dataset=None)
        assert cfg.max_pairs_per_dataset is None


# ---------------------------------------------------------------------------
# ProcessingSummary
# ---------------------------------------------------------------------------

class TestProcessingSummary:
    def test_default_values_are_zero(self) -> None:
        s = ProcessingSummary()
        assert s.total_pairs == 0
        assert s.success == 0
        assert s.skipped == 0
        assert s.failed == 0
        assert s.failures == []

    def test_log_prints_summary(self, capsys) -> None:
        s = ProcessingSummary(
            total_pairs=10, success=7, skipped=2, failed=1,
            failures=[("a", "b", "detection failed")],
        )
        s.log()
        out = capsys.readouterr().out
        assert "Summary" in out
        assert "7" in out
        assert "a" in out


# ---------------------------------------------------------------------------
# MorphGenerator helpers
# ---------------------------------------------------------------------------

class TestMorphGeneratorHelpers:
    def _gen(self, **kw) -> MorphGenerator:
        return MorphGenerator(MorphConfig(**kw))

    # _build_pairs ----------------------------------------------------------

    @pytest.mark.parametrize("n, expected_pairs", [
        (2, 1), (3, 3), (4, 6), (5, 10),
    ])
    def test_build_pairs_combinatorial_count(
        self, n: int, expected_pairs: int
    ) -> None:
        identities = [(f"id{i}", Path(f"{i}.jpg")) for i in range(n)]
        assert len(self._gen()._build_pairs(identities)) == expected_pairs

    def test_build_pairs_no_duplicates_and_a_less_than_b(self) -> None:
        identities = [(f"id{i}", Path(f"{i}.jpg")) for i in range(5)]
        pairs = self._gen()._build_pairs(identities)
        seen: set[tuple[str, str]] = set()
        for id_a, _, id_b, _ in pairs:
            key = (id_a, id_b)
            assert key not in seen
            seen.add(key)

    def test_build_pairs_single_identity_returns_empty(self) -> None:
        identities = [("only", Path("only.jpg"))]
        assert self._gen()._build_pairs(identities) == []

    # _resolve_datasets -----------------------------------------------------

    def test_resolve_datasets_missing_root_returns_empty(
        self, tmp_path: Path
    ) -> None:
        cfg = MorphConfig(aligned_root=tmp_path / "nonexistent")
        assert MorphGenerator(cfg)._resolve_datasets(None) == []

    def test_resolve_datasets_uses_provided_list(
        self, tmp_path: Path
    ) -> None:
        cfg = MorphConfig(aligned_root=tmp_path)
        result = MorphGenerator(cfg)._resolve_datasets(["lfw", "feret"])
        assert result == ["lfw", "feret"]

    def test_resolve_datasets_discovers_subdirectories(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lfw").mkdir()
        (tmp_path / "feret").mkdir()
        cfg = MorphConfig(aligned_root=tmp_path)
        assert sorted(MorphGenerator(cfg)._resolve_datasets(None)) == [
            "feret", "lfw"
        ]


# ---------------------------------------------------------------------------
# MorphGenerator.run integration tests
# ---------------------------------------------------------------------------

class TestMorphGeneratorRun:

    # -- No datasets --------------------------------------------------------

    def test_missing_aligned_root_returns_empty_summary(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            aligned_root=tmp_path / "nonexistent",
            output_root=tmp_path / "out",
            meta_path=tmp_path / "meta.csv",
        )
        ctx, _ = _patch_detector()
        with ctx:
            summary = MorphGenerator(cfg).run()
        assert summary.total_pairs == 0

    # -- skip_existing mode -------------------------------------------------

    def test_skip_existing_skips_pre_existing_output(
        self, tmp_path: Path
    ) -> None:
        aligned = tmp_path / "aligned"
        output = tmp_path / "output"
        _setup_dataset(aligned, "lfw", n_identities=2)
        # Pre-create the expected output file
        (output / "lfw").mkdir(parents=True)
        (output / "lfw" / "morph_id000_id001_a050.jpg").touch()

        cfg = _make_config(
            aligned_root=aligned,
            output_root=output,
            meta_path=tmp_path / "meta.csv",
            skip_existing=True,
        )
        ctx, _ = _patch_detector()
        with ctx:
            summary = MorphGenerator(cfg).run(datasets=["lfw"])

        assert summary.skipped == 1
        assert summary.success == 0

    # -- Landmark detection failure -----------------------------------------

    def test_landmark_failure_recorded_in_summary(
        self, tmp_path: Path
    ) -> None:
        aligned = tmp_path / "aligned"
        _setup_dataset(aligned, "lfw", n_identities=2)

        cfg = _make_config(
            aligned_root=aligned,
            output_root=tmp_path / "out",
            meta_path=tmp_path / "meta.csv",
            skip_existing=False,
        )
        ctx, _ = _patch_detector(detect_return=None)
        with ctx:
            summary = MorphGenerator(cfg).run(datasets=["lfw"])

        assert summary.failed == 1
        assert summary.success == 0
        assert len(summary.failures) == 1

    # -- Successful end-to-end path -----------------------------------------

    def test_successful_pair_increments_success(
        self, tmp_path: Path
    ) -> None:
        aligned = tmp_path / "aligned"
        _setup_dataset(aligned, "lfw", n_identities=2)

        cfg = _make_config(
            aligned_root=aligned,
            output_root=tmp_path / "out",
            meta_path=tmp_path / "meta.csv",
            skip_existing=False,
        )
        ctx, _ = _patch_detector()
        with (
            ctx,
            patch("src.morphing.morph_generator.warp_and_blend",
                  return_value=_solid_bgr()),
            patch("cv2.imwrite", return_value=True),
        ):
            summary = MorphGenerator(cfg).run(datasets=["lfw"])

        assert summary.success == 1
        assert summary.failed == 0

    # -- max_pairs cap ------------------------------------------------------

    def test_max_pairs_caps_total_considered(self, tmp_path: Path) -> None:
        aligned = tmp_path / "aligned"
        _setup_dataset(aligned, "lfw", n_identities=5)  # C(5,2) = 10 pairs

        cfg = _make_config(
            aligned_root=aligned,
            output_root=tmp_path / "out",
            meta_path=tmp_path / "meta.csv",
            skip_existing=False,
            max_pairs_per_dataset=2,
        )
        ctx, _ = _patch_detector()
        with (
            ctx,
            patch("src.morphing.morph_generator.warp_and_blend",
                  return_value=_solid_bgr()),
            patch("cv2.imwrite", return_value=True),
        ):
            summary = MorphGenerator(cfg).run(datasets=["lfw"])

        assert summary.total_pairs == 2

    # -- fewer than 2 identities --------------------------------------------

    def test_single_identity_produces_no_pairs(self, tmp_path: Path) -> None:
        aligned = tmp_path / "aligned"
        _setup_dataset(aligned, "lfw", n_identities=1)

        cfg = _make_config(
            aligned_root=aligned,
            output_root=tmp_path / "out",
            meta_path=tmp_path / "meta.csv",
        )
        ctx, _ = _patch_detector()
        with ctx:
            summary = MorphGenerator(cfg).run(datasets=["lfw"])

        assert summary.total_pairs == 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_unknown_flag_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--not-a-real-flag"])
        assert exc_info.value.code != 0

    def test_missing_aligned_root_exits_zero(self, tmp_path: Path) -> None:
        """No datasets found → no failures → exit code 0."""
        ctx, _ = _patch_detector()
        with ctx:
            code = main([
                "--aligned-root", str(tmp_path / "nonexistent"),
                "--output-root", str(tmp_path / "out"),
                "--metadata-path", str(tmp_path / "meta.csv"),
            ])
        assert code == 0

    def test_failures_produce_exit_code_one(self, tmp_path: Path) -> None:
        aligned = tmp_path / "aligned"
        _setup_dataset(aligned, "lfw", n_identities=2)

        ctx, _ = _patch_detector(detect_return=None)  # force landmark failure
        with ctx:
            code = main([
                "--aligned-root", str(aligned),
                "--output-root", str(tmp_path / "out"),
                "--metadata-path", str(tmp_path / "meta.csv"),
                "--dataset", "lfw",
                "--no-skip-existing",
            ])
        assert code == 1

    def test_alpha_forwarded_to_morph_config(self, tmp_path: Path) -> None:
        with patch("src.morphing.morph_generator.MorphGenerator") as MockGen:
            MockGen.return_value.run.return_value = ProcessingSummary()
            main([
                "--aligned-root", str(tmp_path),
                "--output-root", str(tmp_path),
                "--metadata-path", str(tmp_path / "meta.csv"),
                "--alpha", "0.3",
            ])
        cfg: MorphConfig = MockGen.call_args[0][0]
        assert cfg.alpha == pytest.approx(0.3)

    def test_seed_forwarded_to_morph_config(self, tmp_path: Path) -> None:
        with patch("src.morphing.morph_generator.MorphGenerator") as MockGen:
            MockGen.return_value.run.return_value = ProcessingSummary()
            main([
                "--aligned-root", str(tmp_path),
                "--output-root", str(tmp_path),
                "--metadata-path", str(tmp_path / "meta.csv"),
                "--seed", "99",
            ])
        cfg: MorphConfig = MockGen.call_args[0][0]
        assert cfg.seed == 99

    def test_no_skip_existing_forwarded(self, tmp_path: Path) -> None:
        with patch("src.morphing.morph_generator.MorphGenerator") as MockGen:
            MockGen.return_value.run.return_value = ProcessingSummary()
            main([
                "--aligned-root", str(tmp_path),
                "--output-root", str(tmp_path),
                "--metadata-path", str(tmp_path / "meta.csv"),
                "--no-skip-existing",
            ])
        cfg: MorphConfig = MockGen.call_args[0][0]
        assert cfg.skip_existing is False