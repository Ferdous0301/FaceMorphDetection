"""Unit tests for the StatisticsReporter class.

These tests verify:
    * Identity, bona fide, and morph counts are computed correctly.
    * Split percentages sum correctly and are computed relative to the
      full dataset.
    * Class balance proportions are computed correctly per split.
    * Component counts are included when an identity graph is supplied,
      and omitted (None) otherwise.
    * JSON output is valid and contains the expected fields.
    * CSV output has the expected header and row structure.
    * The formatted console summary contains key statistics.
    * Empty dataset handling.
"""

from __future__ import annotations

import csv
import io
import json

from src.dataset_split.identity_graph import IdentityGraph
from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment
from src.dataset_split.statistics_reporter import (
    StatisticsReporter,
)


def _assignment(
    image_id: str,
    split: str,
    identities: tuple[str, ...],
    label: str = "bona_fide",
) -> SplitAssignment:
    """Build a SplitAssignment for test purposes."""
    return SplitAssignment(
        image_id=image_id, split=split, label=label, identities=identities
    )


def _sample_result() -> DatasetSplitResult:
    """Build a representative DatasetSplitResult for testing."""
    return DatasetSplitResult(
        train=(
            _assignment("bf_a", "train", ("A",)),
            _assignment("bf_b", "train", ("B",)),
            _assignment("morph_ab", "train", ("A", "B"), label="morph"),
        ),
        val=(_assignment("bf_c", "val", ("C",)),),
        test=(
            _assignment("bf_d", "test", ("D",)),
            _assignment("bf_e", "test", ("E",)),
        ),
    )


class TestIdentityAndClassCounts:
    """Tests for identity, bona fide, and morph counts."""

    def test_total_identity_count(self) -> None:
        """Total identities across the dataset are counted correctly."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        assert report.total_identities == 5

    def test_total_bona_fide_count(self) -> None:
        """Total bona fide images are counted correctly."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        assert report.total_bona_fide == 4

    def test_total_morph_count(self) -> None:
        """Total morph images are counted correctly."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        assert report.total_morph == 1

    def test_total_images_matches_sum_of_splits(self) -> None:
        """Total image count equals the sum of per-split totals."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        assert report.total_images == sum(s.total_count for s in report.splits)
        assert report.total_images == 5

    def test_per_split_identity_counts(self) -> None:
        """Per-split identity counts reflect only that split's identities."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        train_stats = next(s for s in report.splits if s.split == "train")
        val_stats = next(s for s in report.splits if s.split == "val")
        assert train_stats.identity_count == 2
        assert val_stats.identity_count == 1


class TestSplitPercentages:
    """Tests for split percentage computation."""

    def test_split_percentages_sum_to_one_hundred(self) -> None:
        """Split percentages sum to (approximately) 100%."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        total_percentage = sum(s.percentage_of_dataset for s in report.splits)
        assert abs(total_percentage - 100.0) < 1e-6

    def test_split_percentage_reflects_relative_size(self) -> None:
        """A split with more images has a higher percentage of the dataset."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        train_stats = next(s for s in report.splits if s.split == "train")
        val_stats = next(s for s in report.splits if s.split == "val")
        assert train_stats.percentage_of_dataset > val_stats.percentage_of_dataset

    def test_train_percentage_is_correct(self) -> None:
        """Train split percentage matches 3 of 5 total images (60%)."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        train_stats = next(s for s in report.splits if s.split == "train")
        assert abs(train_stats.percentage_of_dataset - 60.0) < 1e-6


class TestClassBalance:
    """Tests for per-split class balance computation."""

    def test_train_class_balance_proportions(self) -> None:
        """Train split class balance reflects 2 bona_fide + 1 morph."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        train_stats = next(s for s in report.splits if s.split == "train")
        assert abs(train_stats.class_balance["bona_fide"] - (2 / 3)) < 1e-9
        assert abs(train_stats.class_balance["morph"] - (1 / 3)) < 1e-9

    def test_class_balance_proportions_sum_to_one(self) -> None:
        """Each split's class balance proportions sum to 1.0."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        for split in report.splits:
            if split.class_balance:
                assert abs(sum(split.class_balance.values()) - 1.0) < 1e-9

    def test_single_label_split_has_full_proportion(self) -> None:
        """A split with only one label has that label at proportion 1.0."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        test_stats = next(s for s in report.splits if s.split == "test")
        assert test_stats.class_balance == {"bona_fide": 1.0}


class TestComponentCounts:
    """Tests for connected identity component counting."""

    def test_component_count_included_when_graph_provided(self) -> None:
        """component_count reflects the number of connected components."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_identity("C")
        graph.add_identity("D")
        graph.add_identity("E")
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result(), identity_graph=graph)
        assert report.component_count == 4

    def test_component_count_none_without_graph(self) -> None:
        """component_count is None when no identity graph is supplied."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        assert report.component_count is None


class TestJsonOutput:
    """Tests for JSON rendering of the statistics report."""

    def test_json_output_is_valid_json(self) -> None:
        """to_json produces a string that parses as valid JSON."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        parsed = json.loads(reporter.to_json(report))
        assert isinstance(parsed, dict)

    def test_json_output_contains_expected_top_level_fields(self) -> None:
        """The JSON payload includes all required top-level statistics."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        parsed = json.loads(reporter.to_json(report))
        for field_name in (
            "total_images",
            "total_identities",
            "total_bona_fide",
            "total_morph",
            "component_count",
            "splits",
        ):
            assert field_name in parsed

    def test_json_output_contains_all_splits(self) -> None:
        """The JSON payload includes an entry for each of the 3 splits."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        parsed = json.loads(reporter.to_json(report))
        split_names = {s["split"] for s in parsed["splits"]}
        assert split_names == {"train", "val", "test"}

    def test_json_output_values_match_report(self) -> None:
        """JSON values match the underlying report's computed statistics."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        parsed = json.loads(reporter.to_json(report))
        assert parsed["total_bona_fide"] == report.total_bona_fide
        assert parsed["total_morph"] == report.total_morph


class TestCsvOutput:
    """Tests for CSV rendering of the statistics report."""

    def test_csv_output_has_correct_header(self) -> None:
        """The CSV header row matches the expected fieldnames."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        csv_text = reporter.to_csv(report)
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        assert header == [
            "split",
            "total_count",
            "bona_fide_count",
            "morph_count",
            "identity_count",
            "percentage_of_dataset",
            "class_balance",
        ]

    def test_csv_output_has_one_row_per_split_plus_overall(self) -> None:
        """The CSV has 3 split rows plus 1 overall summary row."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        csv_text = reporter.to_csv(report)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(rows) == 4
        split_names = {row["split"] for row in rows}
        assert split_names == {"train", "val", "test", "overall"}

    def test_csv_overall_row_matches_totals(self) -> None:
        """The overall CSV row reflects the report's dataset-wide totals."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        csv_text = reporter.to_csv(report)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        overall_row = next(row for row in rows if row["split"] == "overall")
        assert int(overall_row["total_count"]) == report.total_images
        assert int(overall_row["bona_fide_count"]) == report.total_bona_fide
        assert int(overall_row["morph_count"]) == report.total_morph

    def test_csv_class_balance_is_formatted(self) -> None:
        """Class balance cells are formatted as label:proportion pairs."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        csv_text = reporter.to_csv(report)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        train_row = next(row for row in rows if row["split"] == "train")
        assert "bona_fide:" in train_row["class_balance"]
        assert "morph:" in train_row["class_balance"]


class TestConsoleOutput:
    """Tests for the formatted console summary."""

    def test_console_output_contains_totals(self) -> None:
        """The console summary includes the dataset-wide totals."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        text = reporter.format_console(report)
        assert "Total images:      5" in text
        assert "Total identities:  5" in text
        assert "Bona fide images:  4" in text
        assert "Morph images:      1" in text

    def test_console_output_contains_each_split_section(self) -> None:
        """The console summary includes a section for every split."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        text = reporter.format_console(report)
        assert "[train]" in text
        assert "[val]" in text
        assert "[test]" in text

    def test_console_output_shows_na_for_missing_component_count(
        self,
    ) -> None:
        """The console summary shows N/A when no identity graph was used."""
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result())
        text = reporter.format_console(report)
        assert "Identity components: N/A" in text

    def test_console_output_shows_component_count_when_available(
        self,
    ) -> None:
        """The console summary shows the numeric component count when available."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_identity("C")
        graph.add_identity("D")
        graph.add_identity("E")
        reporter = StatisticsReporter()
        report = reporter.generate(_sample_result(), identity_graph=graph)
        text = reporter.format_console(report)
        assert "Identity components: 4" in text


class TestEmptyDataset:
    """Tests for statistics generation over an empty dataset."""

    def test_empty_result_has_zeroed_totals(self) -> None:
        """An empty DatasetSplitResult yields all-zero totals."""
        reporter = StatisticsReporter()
        report = reporter.generate(DatasetSplitResult())
        assert report.total_images == 0
        assert report.total_identities == 0
        assert report.total_bona_fide == 0
        assert report.total_morph == 0

    def test_empty_result_split_percentages_are_zero(self) -> None:
        """Empty splits report 0% of the dataset rather than raising."""
        reporter = StatisticsReporter()
        report = reporter.generate(DatasetSplitResult())
        for split in report.splits:
            assert split.percentage_of_dataset == 0.0
            assert split.class_balance == {}

    def test_empty_result_json_is_valid(self) -> None:
        """JSON rendering of an empty report does not raise and is valid."""
        reporter = StatisticsReporter()
        report = reporter.generate(DatasetSplitResult())
        parsed = json.loads(reporter.to_json(report))
        assert parsed["total_images"] == 0

    def test_empty_result_csv_still_has_header_and_overall_row(self) -> None:
        """CSV rendering of an empty report still has header and overall row."""
        reporter = StatisticsReporter()
        report = reporter.generate(DatasetSplitResult())
        csv_text = reporter.to_csv(report)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(rows) == 4
        overall_row = next(row for row in rows if row["split"] == "overall")
        assert overall_row["total_count"] == "0"

    def test_empty_result_console_output_does_not_raise(self) -> None:
        """Console rendering of an empty report succeeds without error."""
        reporter = StatisticsReporter()
        report = reporter.generate(DatasetSplitResult())
        text = reporter.format_console(report)
        assert "Total images:      0" in text
        assert "(empty split)" in text