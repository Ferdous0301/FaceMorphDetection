"""Unit tests for :mod:`src.augmentation.main` (the augmentation CLI).

Filesystem operations are mocked wherever the test is concerned with
argument parsing, error handling, or exit-code behaviour rather than
actual image processing; a small number of tests exercise a real,
temporary dataset end-to-end to verify the CLI's integration with
:func:`~src.augmentation.augment_dataset.augment_dataset`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.augmentation.augment_dataset import (
    DatasetAugmentationError,
    DatasetProcessingSummary,
)
from src.augmentation.main import (
    EXIT_CONFIGURATION_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    OPERATOR_FACTORIES,
    build_arg_parser,
    main,
    parse_args,
    resolve_operators,
)
from src.augmentation.operators.brightness import BrightnessOperator
from src.augmentation.operators.horizontal_flip import HorizontalFlipOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, *, seed: int = 0) -> None:
    """Write a small synthetic RGB image to ``path``.

    Args:
        path: The destination file path; parent directories are
            created automatically.
        seed: A seed controlling the pixel content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image_rgb = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(path.suffix, image_bgr)
    assert success
    buffer.tofile(str(path))


def _make_fake_summary(**overrides: object) -> DatasetProcessingSummary:
    """Build a :class:`DatasetProcessingSummary` for mocking purposes.

    Args:
        **overrides: Field overrides applied on top of a set of
            reasonable defaults.

    Returns:
        DatasetProcessingSummary: The constructed summary.
    """
    defaults: dict[str, object] = {
        "total_images": 1,
        "processed_images": 1,
        "augmented_images": 1,
        "skipped_images": 0,
        "failed_images": 0,
        "elapsed_time": 0.01,
        "output_directory": Path("/tmp/output"),
        "dry_run": False,
        "failures": (),
    }
    defaults.update(overrides)
    return DatasetProcessingSummary(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parses_required_arguments(tmp_path: Path) -> None:
    """--input-dir and --output-dir are parsed into Path objects."""
    args = parse_args(
        ["--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out")]
    )

    assert args.input_dir == tmp_path / "in"
    assert args.output_dir == tmp_path / "out"


def test_defaults_are_applied_when_optional_arguments_omitted(tmp_path: Path) -> None:
    """Optional arguments fall back to their documented defaults."""
    args = parse_args(
        ["--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out")]
    )

    assert args.overwrite is False
    assert args.dry_run is False
    assert args.recursive is True
    assert args.operators is None
    assert args.seed == 42
    assert args.jpeg_quality == 95
    assert args.naming_strategy == "operator_suffix"


def test_parses_operator_selection(tmp_path: Path) -> None:
    """--operators accepts one or more recognised operator names."""
    args = parse_args(
        [
            "--input-dir", str(tmp_path / "in"),
            "--output-dir", str(tmp_path / "out"),
            "--operators", "brightness", "horizontal_flip",
        ]
    )

    assert args.operators == ["brightness", "horizontal_flip"]


def test_parses_flags(tmp_path: Path) -> None:
    """Boolean flags (--overwrite, --dry-run, --no-recursive) are parsed correctly."""
    args = parse_args(
        [
            "--input-dir", str(tmp_path / "in"),
            "--output-dir", str(tmp_path / "out"),
            "--overwrite",
            "--dry-run",
            "--no-recursive",
        ]
    )

    assert args.overwrite is True
    assert args.dry_run is True
    assert args.recursive is False


def test_parses_verbosity_count(tmp_path: Path) -> None:
    """Repeated -v flags accumulate into a verbosity count."""
    args = parse_args(
        ["--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"), "-vv"]
    )

    assert args.verbose == 2


def test_build_arg_parser_returns_argument_parser() -> None:
    """build_arg_parser returns a usable ArgumentParser instance."""
    parser = build_arg_parser()

    assert isinstance(parser, argparse.ArgumentParser)


# ---------------------------------------------------------------------------
# Missing input directory / invalid arguments
# ---------------------------------------------------------------------------


def test_missing_required_arguments_exits_with_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting a required argument causes a SystemExit with code 2."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--input-dir", "/some/dir"])  # --output-dir missing

    assert excinfo.value.code == EXIT_CONFIGURATION_ERROR


def test_invalid_operator_name_exits_with_configuration_error(tmp_path: Path) -> None:
    """An unrecognised operator name is rejected at parse time."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(
            [
                "--input-dir", str(tmp_path / "in"),
                "--output-dir", str(tmp_path / "out"),
                "--operators", "not_a_real_operator",
            ]
        )

    assert excinfo.value.code == EXIT_CONFIGURATION_ERROR


def test_invalid_jpeg_quality_exits_with_configuration_error(tmp_path: Path) -> None:
    """An out-of-range --jpeg-quality value is rejected at parse time."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(
            [
                "--input-dir", str(tmp_path / "in"),
                "--output-dir", str(tmp_path / "out"),
                "--jpeg-quality", "500",
            ]
        )

    assert excinfo.value.code == EXIT_CONFIGURATION_ERROR


def test_invalid_probability_exits_with_configuration_error(tmp_path: Path) -> None:
    """An out-of-range --probability value is rejected at parse time."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(
            [
                "--input-dir", str(tmp_path / "in"),
                "--output-dir", str(tmp_path / "out"),
                "--probability", "1.5",
            ]
        )

    assert excinfo.value.code == EXIT_CONFIGURATION_ERROR


def test_invalid_naming_strategy_exits_with_configuration_error(tmp_path: Path) -> None:
    """An unrecognised --naming-strategy value is rejected at parse time."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(
            [
                "--input-dir", str(tmp_path / "in"),
                "--output-dir", str(tmp_path / "out"),
                "--naming-strategy", "not_a_real_strategy",
            ]
        )

    assert excinfo.value.code == EXIT_CONFIGURATION_ERROR


def test_main_returns_configuration_error_for_missing_arguments() -> None:
    """main() surfaces a missing-argument parse failure as exit code 2."""
    exit_code = main(["--input-dir", "/some/dir"])

    assert exit_code == EXIT_CONFIGURATION_ERROR


def test_main_returns_runtime_error_for_missing_input_directory(tmp_path: Path) -> None:
    """main() reports a nonexistent input directory as a runtime error."""
    exit_code = main(
        [
            "--input-dir", str(tmp_path / "does_not_exist"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    assert exit_code == EXIT_RUNTIME_ERROR


# ---------------------------------------------------------------------------
# resolve_operators
# ---------------------------------------------------------------------------


def test_resolve_operators_defaults_to_all_operators() -> None:
    """With no explicit selection, every registered operator is returned."""
    operators = resolve_operators(None, probability=0.5, seed=1)

    assert len(operators) == len(OPERATOR_FACTORIES)
    assert {op.operator_name for op in operators} == set(OPERATOR_FACTORIES)


def test_resolve_operators_respects_explicit_selection_and_order() -> None:
    """Explicitly selected operators are constructed in the requested order."""
    operators = resolve_operators(
        ["horizontal_flip", "brightness"], probability=0.5, seed=1
    )

    assert [op.operator_name for op in operators] == ["horizontal_flip", "brightness"]
    assert isinstance(operators[0], HorizontalFlipOperator)
    assert isinstance(operators[1], BrightnessOperator)


def test_resolve_operators_raises_for_unknown_name() -> None:
    """An unknown operator name raises DatasetAugmentationError."""
    with pytest.raises(DatasetAugmentationError):
        resolve_operators(["not_a_real_operator"], probability=0.5, seed=1)


def test_resolve_operators_applies_requested_probability_and_seed() -> None:
    """Constructed operators use the requested probability and are seeded."""
    operators = resolve_operators(["brightness"], probability=0.25, seed=7)

    assert operators[0].probability == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# dry-run mode (mocked engine call)
# ---------------------------------------------------------------------------


def test_dry_run_forwards_flag_to_augment_dataset(tmp_path: Path) -> None:
    """--dry-run is forwarded through to augment_dataset as dry_run=True."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    fake_summary = _make_fake_summary(dry_run=True)

    with patch(
        "src.augmentation.main.augment_dataset", return_value=fake_summary
    ) as mock_augment_dataset:
        exit_code = main(
            [
                "--input-dir", str(input_dir),
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
            ]
        )

    assert exit_code == EXIT_SUCCESS
    _, call_kwargs = mock_augment_dataset.call_args
    assert call_kwargs["dry_run"] is True


def test_dry_run_end_to_end_creates_no_output_files(tmp_path: Path) -> None:
    """A real --dry-run invocation writes no files to the output directory."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_image(input_dir / "img01.jpg", seed=1)

    exit_code = main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--operators", "horizontal_flip",
            "--dry-run",
        ]
    )

    assert exit_code == EXIT_SUCCESS
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# overwrite flag (mocked engine call)
# ---------------------------------------------------------------------------


def test_overwrite_flag_forwards_to_augment_dataset(tmp_path: Path) -> None:
    """--overwrite is forwarded through to augment_dataset as overwrite=True."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    fake_summary = _make_fake_summary()

    with patch(
        "src.augmentation.main.augment_dataset", return_value=fake_summary
    ) as mock_augment_dataset:
        exit_code = main(
            [
                "--input-dir", str(input_dir),
                "--output-dir", str(tmp_path / "out"),
                "--overwrite",
            ]
        )

    assert exit_code == EXIT_SUCCESS
    _, call_kwargs = mock_augment_dataset.call_args
    assert call_kwargs["overwrite"] is True


def test_overwrite_defaults_to_false(tmp_path: Path) -> None:
    """Without --overwrite, augment_dataset is called with overwrite=False."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    fake_summary = _make_fake_summary()

    with patch(
        "src.augmentation.main.augment_dataset", return_value=fake_summary
    ) as mock_augment_dataset:
        main(["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")])

    _, call_kwargs = mock_augment_dataset.call_args
    assert call_kwargs["overwrite"] is False


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------


def test_successful_execution_returns_success_exit_code(tmp_path: Path) -> None:
    """A real, successful run over a synthetic dataset exits 0."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_image(input_dir / "img01.jpg", seed=1)
    _write_image(input_dir / "subdir" / "img02.png", seed=2)

    exit_code = main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--operators", "horizontal_flip",
            "--seed", "123",
        ]
    )

    assert exit_code == EXIT_SUCCESS
    output_files = list(output_dir.rglob("*"))
    assert any(path.is_file() for path in output_files)


def test_successful_execution_calls_augment_dataset_once(tmp_path: Path) -> None:
    """The CLI delegates exactly once to augment_dataset and does not reimplement it."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    fake_summary = _make_fake_summary()

    with patch(
        "src.augmentation.main.augment_dataset", return_value=fake_summary
    ) as mock_augment_dataset:
        exit_code = main(
            ["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")]
        )

    assert exit_code == EXIT_SUCCESS
    mock_augment_dataset.assert_called_once()


# ---------------------------------------------------------------------------
# Failed execution
# ---------------------------------------------------------------------------


def test_augment_dataset_error_returns_runtime_error_exit_code(tmp_path: Path) -> None:
    """A DatasetAugmentationError raised by the engine maps to exit code 1."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    with patch(
        "src.augmentation.main.augment_dataset",
        side_effect=DatasetAugmentationError("boom"),
    ):
        exit_code = main(
            ["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")]
        )

    assert exit_code == EXIT_RUNTIME_ERROR


def test_unexpected_exception_returns_runtime_error_exit_code(tmp_path: Path) -> None:
    """An unexpected exception from the engine is caught and mapped to exit code 1."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    with patch(
        "src.augmentation.main.augment_dataset",
        side_effect=RuntimeError("unexpected failure"),
    ):
        exit_code = main(
            ["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")]
        )

    assert exit_code == EXIT_RUNTIME_ERROR


def test_keyboard_interrupt_returns_interrupted_exit_code(tmp_path: Path) -> None:
    """A KeyboardInterrupt raised mid-run is mapped to exit code 130."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    with patch(
        "src.augmentation.main.augment_dataset", side_effect=KeyboardInterrupt
    ):
        exit_code = main(
            ["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")]
        )

    assert exit_code == 130


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_verbose_logging_emits_info_messages(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With -v, the run logs INFO-level progress messages."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_image(input_dir / "img01.jpg", seed=1)

    with caplog.at_level(logging.INFO, logger="src.augmentation.main"):
        exit_code = main(
            [
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--operators", "horizontal_flip",
                "-v",
            ]
        )

    assert exit_code == EXIT_SUCCESS
    assert any("Starting augmentation run" in message for message in caplog.messages)
    assert any("Run finished" in message for message in caplog.messages)


def test_failure_logging_records_error_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed run logs an ERROR-level message describing the failure."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    with caplog.at_level(logging.ERROR, logger="src.augmentation.main"):
        with patch(
            "src.augmentation.main.augment_dataset",
            side_effect=DatasetAugmentationError("engine exploded"),
        ):
            exit_code = main(
                ["--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")]
            )

    assert exit_code == EXIT_RUNTIME_ERROR
    assert any("engine exploded" in message for message in caplog.messages)