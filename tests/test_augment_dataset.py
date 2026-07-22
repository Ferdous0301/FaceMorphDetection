"""Unit tests for :mod:`src.augmentation.augment_dataset`.

All tests operate exclusively on temporary directories and synthetic,
in-memory NumPy images (encoded to disk with OpenCV); no external
datasets are used or required.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.augmentation.augment_dataset import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    DatasetAugmentationError,
    DatasetProcessingSummary,
    augment_dataset,
)
from src.augmentation.operators.base import AugmentationResult, BaseAugmentation
from src.augmentation.operators.brightness import BrightnessOperator
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_image(
    *, height: int = 16, width: int = 16, seed: int = 0
) -> np.ndarray:
    """Build a small, deterministic synthetic RGB image for testing.

    Args:
        height: The image height in pixels.
        width: The image width in pixels.
        seed: A seed controlling the pixel content, so that different
            calls can produce distinguishably different images.

    Returns:
        numpy.ndarray: An ``(height, width, 3)`` ``uint8`` array.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _write_image(path: Path, image_rgb: np.ndarray) -> None:
    """Write a synthetic RGB image to disk at ``path`` using OpenCV.

    Args:
        path: The destination file path; its parent directories are
            created automatically.
        image_rgb: The image to write, in RGB channel order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(path.suffix, image_bgr)
    assert success
    buffer.tofile(str(path))


def _write_corrupted_file(path: Path) -> None:
    """Write garbage bytes to ``path`` to simulate a corrupted image file.

    Args:
        path: The destination file path; its parent directories are
            created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this-is-not-a-valid-image-file")


class _AlwaysFailsOperator(BaseAugmentation):
    """A test-only operator whose ``_apply`` always raises an exception."""

    def __init__(self) -> None:
        super().__init__(operator_name="always_fails", probability=1.0, seed=0)

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        raise RuntimeError("synthetic operator failure")


class _AlwaysAppliesOperator(BaseAugmentation):
    """A test-only operator that deterministically inverts every pixel."""

    def __init__(self, operator_name: str = "always_applies") -> None:
        super().__init__(operator_name=operator_name, probability=1.0, seed=0)

    def _apply(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        inverted = (255 - image).astype(image.dtype, copy=False)
        return inverted, {}


# ---------------------------------------------------------------------------
# Discovery and supported/unsupported extensions
# ---------------------------------------------------------------------------


def test_recursive_image_discovery(tmp_path: Path) -> None:
    """Images nested several directories deep are all discovered."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "a.jpg", _make_synthetic_image(seed=1))
    _write_image(input_dir / "sub" / "b.png", _make_synthetic_image(seed=2))
    _write_image(input_dir / "sub" / "nested" / "c.bmp", _make_synthetic_image(seed=3))

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[], dry_run=True
    )

    assert summary.total_images == 3


def test_supported_extensions_are_discovered(tmp_path: Path) -> None:
    """Every documented supported extension is discovered."""
    input_dir = tmp_path / "input"
    for index, extension in enumerate(sorted(DEFAULT_SUPPORTED_EXTENSIONS)):
        _write_image(input_dir / f"image_{index}{extension}", _make_synthetic_image(seed=index))

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[], dry_run=True
    )

    assert summary.total_images == len(DEFAULT_SUPPORTED_EXTENSIONS)


def test_unsupported_extensions_are_ignored(tmp_path: Path) -> None:
    """Files with unsupported extensions are not discovered or counted."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "valid.jpg", _make_synthetic_image(seed=1))
    (input_dir / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (input_dir / "notes.txt").write_text("not an image")
    (input_dir / "data.csv").write_text("a,b,c")

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[], dry_run=True
    )

    assert summary.total_images == 1


# ---------------------------------------------------------------------------
# Corrupted images
# ---------------------------------------------------------------------------


def test_corrupted_image_is_skipped_and_counted_as_failed(tmp_path: Path) -> None:
    """A corrupted image is logged, skipped, and counted as failed."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "good.jpg", _make_synthetic_image(seed=1))
    _write_corrupted_file(input_dir / "corrupted.jpg")

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[HorizontalFlipOperator(probability=1.0)]
    )

    assert summary.total_images == 2
    assert summary.failed_images == 1
    assert summary.processed_images == 1
    assert summary.augmented_images == 1


def test_corrupted_image_does_not_halt_processing_of_others(tmp_path: Path) -> None:
    """Processing continues past a corrupted image to later files."""
    input_dir = tmp_path / "input"
    _write_corrupted_file(input_dir / "aaa_corrupted.jpg")
    _write_image(input_dir / "zzz_good.jpg", _make_synthetic_image(seed=7))

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[HorizontalFlipOperator(probability=1.0)]
    )

    assert summary.failed_images == 1
    assert summary.augmented_images == 1
    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1


# ---------------------------------------------------------------------------
# Deterministic filenames
# ---------------------------------------------------------------------------


def test_deterministic_filenames_across_runs(tmp_path: Path) -> None:
    """Two identically seeded runs over the same input produce identical filenames."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "person001" / "img01.jpg", _make_synthetic_image(seed=5))

    summary_one = augment_dataset(
        input_dir,
        tmp_path / "output_one",
        operators=[BrightnessOperator(probability=1.0, random_state=42)],
    )
    summary_two = augment_dataset(
        input_dir,
        tmp_path / "output_two",
        operators=[BrightnessOperator(probability=1.0, random_state=42)],
    )

    files_one = {p.name for p in (tmp_path / "output_one").rglob("*.jpg")}
    files_two = {p.name for p in (tmp_path / "output_two").rglob("*.jpg")}

    assert files_one == files_two
    assert summary_one.augmented_images == summary_two.augmented_images == 1


def test_filename_reflects_applied_operator_name(tmp_path: Path) -> None:
    """The output filename includes the name of the operator that applied."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=9))

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[HorizontalFlipOperator(probability=1.0)],
    )

    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1
    assert "horizontal_flip" in output_files[0].name
    assert output_files[0].name.startswith("img01_aug_")


def test_filename_uses_noop_token_when_no_operator_applies(tmp_path: Path) -> None:
    """When every operator's probability gate misses, the filename uses 'noop'."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=9))

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[HorizontalFlipOperator(probability=0.0)],
    )

    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1
    assert "noop" in output_files[0].name


# ---------------------------------------------------------------------------
# Directory structure preservation
# ---------------------------------------------------------------------------


def test_directory_hierarchy_is_preserved(tmp_path: Path) -> None:
    """The relative directory structure of the input is mirrored in the output."""
    input_dir = tmp_path / "input"
    _write_image(
        input_dir / "train" / "person001" / "img01.jpg",
        _make_synthetic_image(seed=3),
    )

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[HorizontalFlipOperator(probability=1.0)],
    )

    expected_dir = tmp_path / "output" / "train" / "person001"
    assert expected_dir.is_dir()
    assert len(list(expected_dir.glob("*.jpg"))) == 1


# ---------------------------------------------------------------------------
# overwrite behaviour
# ---------------------------------------------------------------------------


def test_overwrite_false_skips_existing_output(tmp_path: Path) -> None:
    """With overwrite=False, an existing output file is left untouched."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=4))

    operator = HorizontalFlipOperator(probability=1.0)
    augment_dataset(input_dir, output_dir, operators=[operator], overwrite=False)

    output_files = list(output_dir.rglob("*.jpg"))
    assert len(output_files) == 1
    original_mtime = output_files[0].stat().st_mtime_ns

    # Re-run with a fresh operator instance (same seed) against the same
    # output directory; the existing file should be left untouched.
    time.sleep(0.01)
    second_operator = HorizontalFlipOperator(probability=1.0)
    summary = augment_dataset(
        input_dir, output_dir, operators=[second_operator], overwrite=False
    )

    assert summary.skipped_images == 1
    assert summary.augmented_images == 0
    assert output_files[0].stat().st_mtime_ns == original_mtime


def test_overwrite_true_replaces_existing_output(tmp_path: Path) -> None:
    """With overwrite=True, an existing output file is replaced."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=4))

    augment_dataset(
        input_dir,
        output_dir,
        operators=[HorizontalFlipOperator(probability=1.0)],
        overwrite=True,
    )
    output_files = list(output_dir.rglob("*.jpg"))
    assert len(output_files) == 1
    original_mtime = output_files[0].stat().st_mtime_ns

    time.sleep(0.01)
    summary = augment_dataset(
        input_dir,
        output_dir,
        operators=[HorizontalFlipOperator(probability=1.0)],
        overwrite=True,
    )

    assert summary.augmented_images == 1
    assert summary.skipped_images == 0
    assert output_files[0].stat().st_mtime_ns > original_mtime


# ---------------------------------------------------------------------------
# dry_run mode
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_any_files(tmp_path: Path) -> None:
    """In dry-run mode, no output files or directories are created."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=1))

    summary = augment_dataset(
        input_dir,
        output_dir,
        operators=[HorizontalFlipOperator(probability=1.0)],
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.augmented_images == 1
    assert not output_dir.exists()


def test_dry_run_does_not_modify_input_dataset(tmp_path: Path) -> None:
    """Dry-run execution never touches the input dataset's files."""
    input_dir = tmp_path / "input"
    image_path = input_dir / "img01.jpg"
    _write_image(image_path, _make_synthetic_image(seed=1))
    original_bytes = image_path.read_bytes()

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[HorizontalFlipOperator(probability=1.0)],
        dry_run=True,
    )

    assert image_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Original dataset is never modified (non-dry-run too)
# ---------------------------------------------------------------------------


def test_input_dataset_is_never_modified(tmp_path: Path) -> None:
    """A real (non-dry-run) run never modifies files under the input root."""
    input_dir = tmp_path / "input"
    image_path = input_dir / "img01.jpg"
    _write_image(image_path, _make_synthetic_image(seed=2))
    original_bytes = image_path.read_bytes()

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[BrightnessOperator(probability=1.0, random_state=1)],
    )

    assert image_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Failed image / operator handling
# ---------------------------------------------------------------------------


def test_operator_failure_is_logged_and_processing_continues(tmp_path: Path) -> None:
    """An operator that raises is logged and skipped; the image still saves."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=6))

    summary = augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[_AlwaysFailsOperator(), HorizontalFlipOperator(probability=1.0)],
    )

    assert summary.failed_images == 0
    assert summary.augmented_images == 1
    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1
    assert "horizontal_flip" in output_files[0].name
    assert "always_fails" not in output_files[0].name


def test_unwritable_output_path_is_counted_as_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If saving fails, the image is counted as failed rather than raising."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=1))

    import src.augmentation.augment_dataset as module_under_test

    def _fake_save(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(module_under_test, "_save_image_from_rgb", _fake_save)

    summary = augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[HorizontalFlipOperator(probability=1.0)],
    )

    assert summary.failed_images == 1
    assert summary.augmented_images == 0


# ---------------------------------------------------------------------------
# Processing summary / statistics
# ---------------------------------------------------------------------------


def test_processing_summary_fields_are_populated(tmp_path: Path) -> None:
    """The returned summary is a complete, correctly populated dataclass."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "a.jpg", _make_synthetic_image(seed=1))
    _write_image(input_dir / "b.jpg", _make_synthetic_image(seed=2))
    _write_corrupted_file(input_dir / "c.jpg")

    summary = augment_dataset(
        input_dir, output_dir, operators=[HorizontalFlipOperator(probability=1.0)]
    )

    assert isinstance(summary, DatasetProcessingSummary)
    assert summary.total_images == 3
    assert summary.processed_images == 2
    assert summary.augmented_images == 2
    assert summary.failed_images == 1
    assert summary.skipped_images == 0
    assert summary.elapsed_time >= 0.0
    assert summary.output_directory == output_dir


def test_processing_summary_is_immutable(tmp_path: Path) -> None:
    """The processing summary dataclass cannot be mutated after creation."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "a.jpg", _make_synthetic_image(seed=1))

    summary = augment_dataset(
        input_dir, tmp_path / "output", operators=[HorizontalFlipOperator(probability=1.0)]
    )

    with pytest.raises(Exception):
        summary.total_images = 999  # type: ignore[misc]


def test_statistics_track_multiple_categories_simultaneously(tmp_path: Path) -> None:
    """Skipped, failed, and augmented counts are all tracked independently."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "existing.jpg", _make_synthetic_image(seed=1))
    _write_image(input_dir / "fresh.jpg", _make_synthetic_image(seed=2))
    _write_corrupted_file(input_dir / "bad.jpg")

    # Pre-populate the output directory so the second run skips existing files.
    augment_dataset(
        input_dir,
        output_dir,
        operators=[HorizontalFlipOperator(probability=1.0)],
    )

    summary = augment_dataset(
        input_dir,
        output_dir,
        operators=[HorizontalFlipOperator(probability=1.0)],
        overwrite=False,
    )

    assert summary.skipped_images == 2  # existing.jpg and fresh.jpg both now exist
    assert summary.failed_images == 1
    assert summary.augmented_images == 0


# ---------------------------------------------------------------------------
# Deterministic execution / integration with existing operators
# ---------------------------------------------------------------------------


def test_deterministic_execution_produces_identical_pixel_output(tmp_path: Path) -> None:
    """Two runs with identically seeded operators write byte-identical images."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=11))

    augment_dataset(
        input_dir,
        tmp_path / "output_one",
        operators=[BrightnessOperator(probability=1.0, random_state=99)],
        output_image_format="png",
    )
    augment_dataset(
        input_dir,
        tmp_path / "output_two",
        operators=[BrightnessOperator(probability=1.0, random_state=99)],
        output_image_format="png",
    )

    file_one = next((tmp_path / "output_one").rglob("*.png"))
    file_two = next((tmp_path / "output_two").rglob("*.png"))
    assert file_one.read_bytes() == file_two.read_bytes()


def test_integration_with_multiple_existing_operators(tmp_path: Path) -> None:
    """A chain of real operators from the existing framework runs successfully."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=13))

    operators = [
        BrightnessOperator(probability=1.0, random_state=1),
        HorizontalFlipOperator(probability=1.0, random_state=2),
    ]

    summary = augment_dataset(input_dir, tmp_path / "output", operators=operators)

    assert summary.augmented_images == 1
    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1
    assert "brightness" in output_files[0].name
    assert "horizontal_flip" in output_files[0].name


def test_applied_image_content_actually_differs_from_source(tmp_path: Path) -> None:
    """The saved augmented image content differs from the untouched source."""
    input_dir = tmp_path / "input"
    source_image = _make_synthetic_image(seed=21)
    _write_image(input_dir / "img01.jpg", source_image)

    augment_dataset(
        input_dir,
        tmp_path / "output",
        operators=[_AlwaysAppliesOperator()],
        output_image_format="png",
    )

    output_file = next((tmp_path / "output").rglob("*.png"))
    saved_bgr = cv2.imread(str(output_file), cv2.IMREAD_COLOR)
    saved_rgb = cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB)

    assert not np.array_equal(saved_rgb, source_image)


# ---------------------------------------------------------------------------
# Empty operator chain (copy-through) behaviour
# ---------------------------------------------------------------------------


def test_empty_operator_chain_still_saves_image(tmp_path: Path) -> None:
    """With no operators configured, the image is still written through unmodified."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=14))

    summary = augment_dataset(input_dir, tmp_path / "output", operators=[])

    assert summary.augmented_images == 1
    output_files = list((tmp_path / "output").rglob("*.jpg"))
    assert len(output_files) == 1
    assert "noop" in output_files[0].name


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_invalid_input_directory_raises(tmp_path: Path) -> None:
    """A nonexistent input directory raises a DatasetAugmentationError."""
    with pytest.raises(DatasetAugmentationError):
        augment_dataset(
            tmp_path / "does_not_exist", tmp_path / "output", operators=[]
        )


def test_invalid_jpeg_quality_raises(tmp_path: Path) -> None:
    """An out-of-range JPEG quality raises a DatasetAugmentationError."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=1))

    with pytest.raises(DatasetAugmentationError):
        augment_dataset(
            input_dir, tmp_path / "output", operators=[], jpeg_quality=101
        )


def test_invalid_output_image_format_raises(tmp_path: Path) -> None:
    """An unrecognised output image format raises a DatasetAugmentationError."""
    input_dir = tmp_path / "input"
    _write_image(input_dir / "img01.jpg", _make_synthetic_image(seed=1))

    with pytest.raises(DatasetAugmentationError):
        augment_dataset(
            input_dir,
            tmp_path / "output",
            operators=[],
            output_image_format="webp",
        )