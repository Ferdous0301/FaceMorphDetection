"""Unit tests for vit.logging_utils.*"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vit.logging_utils.csv_logger import CSVLogger
from vit.logging_utils.experiment_logger import ExperimentLogger
from vit.logging_utils.tb_logger import TensorBoardLogger


class TestCSVLogger:
    def test_writes_header_and_row(self, tmp_path: Path) -> None:
        logger = CSVLogger(tmp_path / "log.csv")
        logger.log({"epoch": 0, "loss": 0.5})

        with (tmp_path / "log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows == [{"epoch": "0", "loss": "0.5"}]

    def test_appends_multiple_rows(self, tmp_path: Path) -> None:
        logger = CSVLogger(tmp_path / "log.csv")
        logger.log({"epoch": 0, "loss": 1.0})
        logger.log({"epoch": 1, "loss": 0.5})

        with (tmp_path / "log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[1]["epoch"] == "1"

    def test_empty_row_raises(self, tmp_path: Path) -> None:
        logger = CSVLogger(tmp_path / "log.csv")
        with pytest.raises(ValueError):
            logger.log({})

    def test_mismatched_keys_raise(self, tmp_path: Path) -> None:
        logger = CSVLogger(tmp_path / "log.csv")
        logger.log({"epoch": 0, "loss": 1.0})
        with pytest.raises(ValueError):
            logger.log({"epoch": 1, "accuracy": 0.9})

    def test_truncates_existing_file_on_construction(self, tmp_path: Path) -> None:
        path = tmp_path / "log.csv"
        path.write_text("stale content that should be wiped\n")
        CSVLogger(path)
        assert path.read_text() == ""

    def test_path_property(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "log.csv"
        logger = CSVLogger(path)
        assert logger.path == path
        assert path.parent.is_dir()


class TestTensorBoardLogger:
    def test_creates_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb"
        logger = TensorBoardLogger(log_dir)
        assert log_dir.is_dir()
        logger.close()

    def test_log_scalars_writes_event_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb"
        logger = TensorBoardLogger(log_dir)
        logger.log_scalars("train", {"loss": 0.5, "accuracy": 0.9}, step=0)
        logger.close()

        event_files = list(log_dir.glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_dir_property(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb"
        logger = TensorBoardLogger(log_dir)
        assert logger.log_dir == log_dir
        logger.close()


class TestExperimentLogger:
    def test_log_epoch_writes_to_csv_and_tensorboard(self, tmp_path: Path) -> None:
        csv_logger = CSVLogger(tmp_path / "train_log.csv")
        tb_logger = TensorBoardLogger(tmp_path / "tensorboard")
        logger = ExperimentLogger(csv_logger=csv_logger, tb_logger=tb_logger, name="test_exp")

        logger.log_epoch(split="train", epoch=0, metrics={"loss": 0.5, "accuracy": 0.8})

        with (tmp_path / "train_log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["split"] == "train"
        assert rows[0]["epoch"] == "0"

        logger.close()

    def test_log_hparams_does_not_raise(self, tmp_path: Path) -> None:
        csv_logger = CSVLogger(tmp_path / "log.csv")
        tb_logger = TensorBoardLogger(tmp_path / "tb")
        logger = ExperimentLogger(csv_logger=csv_logger, tb_logger=tb_logger, name="test_exp2")
        logger.log_hparams({"lr": 0.001, "epochs": 10})
        logger.close()

    def test_multiple_epochs_accumulate_rows(self, tmp_path: Path) -> None:
        csv_logger = CSVLogger(tmp_path / "log.csv")
        tb_logger = TensorBoardLogger(tmp_path / "tb")
        logger = ExperimentLogger(csv_logger=csv_logger, tb_logger=tb_logger, name="test_exp3")

        for epoch in range(3):
            logger.log_epoch(split="val", epoch=epoch, metrics={"loss": 1.0 - epoch * 0.1})

        with (tmp_path / "log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        logger.close()