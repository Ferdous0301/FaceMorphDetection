"""Unit tests for vit.engine.early_stopping.EarlyStopping."""

from __future__ import annotations

import pytest

from vit.engine.early_stopping import EarlyStopping


class TestConstruction:
    def test_rejects_negative_patience(self) -> None:
        with pytest.raises(ValueError):
            EarlyStopping(patience=-1, mode="min")

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            EarlyStopping(patience=1, mode="best")

    def test_rejects_negative_min_delta(self) -> None:
        with pytest.raises(ValueError):
            EarlyStopping(patience=1, mode="min", min_delta=-0.1)

    def test_initial_best_value_min_mode(self) -> None:
        stopper = EarlyStopping(patience=1, mode="min")
        assert stopper.best_value == float("inf")

    def test_initial_best_value_max_mode(self) -> None:
        stopper = EarlyStopping(patience=1, mode="max")
        assert stopper.best_value == float("-inf")


class TestMinMode:
    def test_improving_sequence_never_stops(self) -> None:
        stopper = EarlyStopping(patience=2, mode="min")
        assert stopper.step(1.0) is False
        assert stopper.step(0.9) is False
        assert stopper.step(0.8) is False
        assert stopper.counter == 0

    def test_stops_after_patience_exceeded(self) -> None:
        stopper = EarlyStopping(patience=2, mode="min")
        assert stopper.step(1.0) is False  # improvement, counter=0
        assert stopper.step(1.1) is False  # no improvement, counter=1
        assert stopper.step(1.2) is False  # no improvement, counter=2
        assert stopper.step(1.3) is True  # no improvement, counter=3 > patience

    def test_best_value_tracks_minimum(self) -> None:
        stopper = EarlyStopping(patience=5, mode="min")
        stopper.step(1.0)
        stopper.step(0.5)
        stopper.step(0.8)
        assert stopper.best_value == pytest.approx(0.5)


class TestMaxMode:
    def test_improving_sequence_never_stops(self) -> None:
        stopper = EarlyStopping(patience=2, mode="max")
        assert stopper.step(0.5) is False
        assert stopper.step(0.6) is False
        assert stopper.step(0.7) is False

    def test_stops_after_patience_exceeded(self) -> None:
        stopper = EarlyStopping(patience=1, mode="max")
        assert stopper.step(0.7) is False
        assert stopper.step(0.6) is False  # counter=1
        assert stopper.step(0.5) is True  # counter=2 > patience=1


class TestMinDelta:
    def test_small_improvement_within_min_delta_does_not_reset_counter(self) -> None:
        stopper = EarlyStopping(patience=1, mode="min", min_delta=0.1)
        stopper.step(1.0)
        # 0.95 is "better" but not by more than min_delta=0.1
        assert stopper.step(0.95) is False
        assert stopper.counter == 1
        assert stopper.step(0.94) is True


class TestPatienceZero:
    def test_stops_immediately_on_first_non_improvement(self) -> None:
        stopper = EarlyStopping(patience=0, mode="min")
        assert stopper.step(1.0) is False
        assert stopper.step(1.1) is True


class TestRepr:
    def test_repr_contains_key_fields(self) -> None:
        stopper = EarlyStopping(patience=3, mode="min")
        stopper.step(1.0)
        text = repr(stopper)
        assert "patience=3" in text
        assert "min" in text