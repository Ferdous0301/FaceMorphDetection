"""Unit tests for vit.metrics.metric_tracker.MetricTracker."""

from __future__ import annotations

import pytest

from vit.metrics.metric_tracker import MetricTracker


class TestUpdateAndAverage:
    def test_single_update(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 2.0, batch_size=4)
        assert tracker.average("loss") == pytest.approx(2.0)

    def test_weighted_average_across_uneven_batches(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0, batch_size=8)
        tracker.update("loss", 0.5, batch_size=2)
        # (1.0*8 + 0.5*2) / 10 = 9/10 = 0.9
        assert tracker.average("loss") == pytest.approx(0.9)

    def test_default_batch_size_is_one(self) -> None:
        tracker = MetricTracker()
        tracker.update("accuracy", 0.8)
        tracker.update("accuracy", 0.6)
        assert tracker.average("accuracy") == pytest.approx(0.7)

    def test_multiple_metrics_independent(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0, batch_size=1)
        tracker.update("accuracy", 0.9, batch_size=1)
        assert tracker.average("loss") == pytest.approx(1.0)
        assert tracker.average("accuracy") == pytest.approx(0.9)

    def test_unknown_metric_raises(self) -> None:
        tracker = MetricTracker()
        with pytest.raises(KeyError):
            tracker.average("nonexistent")

    def test_non_positive_batch_size_raises(self) -> None:
        tracker = MetricTracker()
        with pytest.raises(ValueError):
            tracker.update("loss", 1.0, batch_size=0)
        with pytest.raises(ValueError):
            tracker.update("loss", 1.0, batch_size=-1)


class TestReset:
    def test_reset_clears_all_metrics(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0, batch_size=4)
        tracker.reset()
        with pytest.raises(KeyError):
            tracker.average("loss")

    def test_reusable_after_reset(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 5.0, batch_size=1)
        tracker.reset()
        tracker.update("loss", 2.0, batch_size=1)
        assert tracker.average("loss") == pytest.approx(2.0)


class TestAsDict:
    def test_empty_tracker(self) -> None:
        tracker = MetricTracker()
        assert tracker.as_dict() == {}

    def test_returns_all_metrics(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0, batch_size=2)
        tracker.update("accuracy", 0.5, batch_size=2)
        result = tracker.as_dict()
        assert result == pytest.approx({"loss": 1.0, "accuracy": 0.5})


class TestRepr:
    def test_repr_contains_metric_names(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.2345, batch_size=1)
        text = repr(tracker)
        assert "loss" in text
        assert "1.2345" in text or "1.234" in text