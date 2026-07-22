"""Unit tests for the Dataset Split CLI.

These tests verify:
    * Argument parsing succeeds for valid inputs.
    * Invalid arguments (bad types, missing required flags) fail parsing.
    * Missing input/metadata paths return the correct exit code.
    * Successful execution writes manifests and returns exit code 0.
    * Verification mode passes and fails as expected, with correct codes.
    * Statistics mode prints a summary.
    * Exit codes match documented constants across scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_split.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_MISSING_PATH,
    EXIT_SUCCESS,
    EXIT_VERIFICATION_FAILED,
    build_parser,
    main,
    parse_args,
)


def _write_metadata(
    path: Path,
    bona_fide: list[dict[str, str]] | None = None,
    morphs: list[dict[str, str]] | None = None,
) -> None:
    """Write a metadata JSON file for test purposes."""
    payload = {
        "bona_fide": bona_fide or [],
        "morphs": morphs or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _basic_dataset(
    metadata_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a small, valid dataset and write its metadata file."""
    bona_fide = [
        {"image_id": "bf_a.png", "identity": "A"},
        {"image_id": "bf_b.png", "identity": "B"},
        {"image_id": "bf_c.png", "identity": "C"},
    ]
    morphs = [
        {"image_id": "morph_ab.png", "identity_a": "A", "identity_b": "B"},
    ]
    _write_metadata(metadata_path, bona_fide, morphs)
    return bona_fide, morphs


class TestArgumentParsing:
    """Tests for successful and structural argument parsing."""

    def test_parses_required_arguments(self, tmp_path: Path) -> None:
        """All required arguments are parsed into the namespace."""
        args = parse_args(
            [
                "--input",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "meta.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert args.input == str(tmp_path)
        assert args.metadata == str(tmp_path / "meta.json")
        assert args.output == str(tmp_path / "out")

    def test_default_values_applied(self, tmp_path: Path) -> None:
        """Optional arguments receive their documented default values."""
        args = parse_args(
            [
                "--input",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "meta.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert args.seed == 42
        assert args.train_ratio == 0.8
        assert args.val_ratio == 0.1
        assert args.test_ratio == 0.1
        assert args.verify is False
        assert args.statistics is False

    def test_custom_ratios_and_seed_parsed(self, tmp_path: Path) -> None:
        """Custom ratio and seed values are parsed correctly."""
        args = parse_args(
            [
                "--input",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "meta.json"),
                "--output",
                str(tmp_path / "out"),
                "--seed",
                "7",
                "--train-ratio",
                "0.6",
                "--val-ratio",
                "0.2",
                "--test-ratio",
                "0.2",
            ]
        )
        assert args.seed == 7
        assert args.train_ratio == 0.6
        assert args.val_ratio == 0.2
        assert args.test_ratio == 0.2

    def test_verify_and_statistics_flags_parsed(self, tmp_path: Path) -> None:
        """Boolean flags --verify and --statistics are parsed as True."""
        args = parse_args(
            [
                "--input",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "meta.json"),
                "--output",
                str(tmp_path / "out"),
                "--verify",
                "--statistics",
            ]
        )
        assert args.verify is True
        assert args.statistics is True

    def test_build_parser_returns_argument_parser(self) -> None:
        """build_parser() returns a usable ArgumentParser instance."""
        parser = build_parser()
        assert parser.prog == "dataset-split"


class TestInvalidArguments:
    """Tests for invalid or missing CLI arguments."""

    def test_missing_required_argument_exits_with_invalid_args_code(
        self, tmp_path: Path
    ) -> None:
        """Omitting a required argument causes argparse to exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(
                [
                    "--metadata",
                    str(tmp_path / "meta.json"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code == 2

    def test_non_numeric_ratio_exits_with_invalid_args_code(
        self, tmp_path: Path
    ) -> None:
        """A non-numeric --train-ratio value causes an argparse error."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(
                [
                    "--input",
                    str(tmp_path),
                    "--metadata",
                    str(tmp_path / "meta.json"),
                    "--output",
                    str(tmp_path / "out"),
                    "--train-ratio",
                    "not-a-number",
                ]
            )
        assert exc_info.value.code == 2

    def test_non_numeric_seed_exits_with_invalid_args_code(
        self, tmp_path: Path
    ) -> None:
        """A non-numeric --seed value causes an argparse error."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(
                [
                    "--input",
                    str(tmp_path),
                    "--metadata",
                    str(tmp_path / "meta.json"),
                    "--output",
                    str(tmp_path / "out"),
                    "--seed",
                    "not-an-int",
                ]
            )
        assert exc_info.value.code == 2


class TestMissingPaths:
    """Tests for missing input/metadata paths at execution time."""

    def test_missing_input_directory_returns_missing_path_code(
        self, tmp_path: Path
    ) -> None:
        """A non-existent --input directory returns EXIT_MISSING_PATH."""
        metadata_path = tmp_path / "meta.json"
        _write_metadata(metadata_path)
        exit_code = main(
            [
                "--input",
                str(tmp_path / "does_not_exist"),
                "--metadata",
                str(metadata_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == EXIT_MISSING_PATH

    def test_missing_metadata_file_returns_missing_path_code(
        self, tmp_path: Path
    ) -> None:
        """A non-existent --metadata file returns EXIT_MISSING_PATH."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(tmp_path / "missing_meta.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == EXIT_MISSING_PATH


class TestSuccessfulExecution:
    """Tests for a fully successful end-to-end CLI run."""

    def test_successful_run_returns_success_code(self, tmp_path: Path) -> None:
        """A valid run over a small dataset returns EXIT_SUCCESS."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
            ]
        )
        assert exit_code == EXIT_SUCCESS

    def test_successful_run_writes_manifests(self, tmp_path: Path) -> None:
        """A successful run writes train.csv, validation.csv, and test.csv."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
            ]
        )

        assert (output_dir / "train.csv").exists()
        assert (output_dir / "validation.csv").exists()
        assert (output_dir / "test.csv").exists()

    def test_successful_run_prints_manifest_location(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI prints where manifests were written on success."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr()
        assert "Manifests written to" in captured.out


class TestVerificationMode:
    """Tests for --verify mode."""

    def test_verification_passes_when_files_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verification passes when all referenced image files exist."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        bona_fide, morphs = _basic_dataset(metadata_path)

        for entry in bona_fide:
            (input_dir / entry["image_id"]).write_bytes(b"fake")
        for entry in morphs:
            (input_dir / entry["image_id"]).write_bytes(b"fake")

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--verify",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == EXIT_SUCCESS
        assert "Verification: PASSED" in captured.out

    def test_verification_fails_when_files_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verification fails when referenced image files are absent."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)
        # Intentionally do not create any files in input_dir.

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--verify",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == EXIT_VERIFICATION_FAILED
        assert "Verification: FAILED" in captured.err


class TestStatisticsMode:
    """Tests for --statistics mode."""

    def test_statistics_mode_prints_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Statistics mode prints a console summary including totals."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--statistics",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == EXIT_SUCCESS
        assert "Dataset Split Statistics" in captured.out
        assert "Total images:" in captured.out

    def test_statistics_and_verify_together(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--verify and --statistics can be combined in a single run."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        bona_fide, morphs = _basic_dataset(metadata_path)
        for entry in bona_fide + morphs:
            (input_dir / entry["image_id"]).write_bytes(b"fake")

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--verify",
                "--statistics",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == EXIT_SUCCESS
        assert "Verification: PASSED" in captured.out
        assert "Dataset Split Statistics" in captured.out


class TestExitCodes:
    """Tests exercising exit code correctness across scenarios."""

    def test_invalid_ratio_configuration_returns_config_error_code(
        self, tmp_path: Path
    ) -> None:
        """Ratios that don't sum to 1.0 return EXIT_CONFIG_ERROR."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--train-ratio",
                "0.5",
                "--val-ratio",
                "0.5",
                "--test-ratio",
                "0.5",
            ]
        )
        assert exit_code == EXIT_CONFIG_ERROR

    def test_negative_ratio_returns_config_error_code(
        self, tmp_path: Path
    ) -> None:
        """A negative ratio value returns EXIT_CONFIG_ERROR."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _basic_dataset(metadata_path)

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
                "--train-ratio",
                "-0.1",
                "--val-ratio",
                "0.6",
                "--test-ratio",
                "0.5",
            ]
        )
        assert exit_code == EXIT_CONFIG_ERROR

    def test_empty_dataset_still_succeeds(self, tmp_path: Path) -> None:
        """An empty (but valid) dataset still returns EXIT_SUCCESS."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        metadata_path = tmp_path / "meta.json"
        output_dir = tmp_path / "out"
        _write_metadata(metadata_path)

        exit_code = main(
            [
                "--input",
                str(input_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir),
            ]
        )
        assert exit_code == EXIT_SUCCESS