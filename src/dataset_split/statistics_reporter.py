"""Dataset statistics reporting for the Dataset Split stage.

This module provides :class:`StatisticsReporter`, which computes summary
statistics over a :class:`~src.dataset_split.splitter.DatasetSplitResult`
and renders them as JSON, CSV, or a formatted console summary.

Statistics include, per split and overall:
    * Number of distinct identities.
    * Bona fide image count.
    * Morph image count.
    * Split percentages relative to the full dataset.
    * Class balance (per-label proportions within a split).
    * Connected identity-component counts (when an identity graph is
      supplied).

This module performs statistics computation and formatting only. It does
not perform splitting, manifest writing, or verification, and it writes
no files to disk.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

from src.dataset_split.identity_graph import IdentityGraph
from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment

#: Labels treated specially when reporting bona fide / morph counts.
BONA_FIDE_LABEL = "bona_fide"
MORPH_LABEL = "morph"

#: The canonical split names, in a fixed reporting order.
_SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitStatistics:
    """Statistics computed for a single dataset split.

    Attributes:
        split: The name of the split (``"train"``, ``"val"``, or
            ``"test"``).
        total_count: Total number of images assigned to this split.
        bona_fide_count: Number of bona fide images in this split.
        morph_count: Number of morph images in this split.
        identity_count: Number of distinct identities referenced by
            images in this split.
        percentage_of_dataset: This split's share of the overall
            dataset, expressed as a percentage (0-100).
        class_balance: Mapping from class label to its proportion
            (0.0-1.0) of the images within this split.
    """

    split: str
    total_count: int
    bona_fide_count: int
    morph_count: int
    identity_count: int
    percentage_of_dataset: float
    class_balance: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatisticsReport:
    """Aggregate statistics for an entire dataset split result.

    Attributes:
        total_images: Total number of images across all splits.
        total_identities: Total number of distinct identities across
            all splits.
        total_bona_fide: Total number of bona fide images.
        total_morph: Total number of morph images.
        component_count: Number of connected identity components, or
            ``None`` if no identity graph was supplied when the report
            was generated.
        splits: Per-split statistics, in a fixed reporting order
            (train, val, test).
    """

    total_images: int
    total_identities: int
    total_bona_fide: int
    total_morph: int
    component_count: int | None
    splits: tuple[SplitStatistics, ...]


class StatisticsReporter:
    """Computes and formats dataset split statistics.

    An instance of this class is stateless with respect to any single
    report: :meth:`generate` produces a fresh
    :class:`DatasetStatisticsReport` from its inputs, and the
    ``to_*``/``format_*`` methods render that report into different
    output formats without mutating it.
    """

    def generate(
        self,
        result: DatasetSplitResult,
        identity_graph: IdentityGraph | None = None,
    ) -> DatasetStatisticsReport:
        """Compute a full statistics report from a dataset split result.

        Args:
            result: The dataset split result to summarize.
            identity_graph: The identity graph the split was derived
                from. When provided, enables ``component_count`` in the
                resulting report.

        Returns:
            A populated :class:`DatasetStatisticsReport`.
        """
        assignments_by_split = {
            "train": result.train,
            "val": result.val,
            "test": result.test,
        }

        total_images = len(result.all_assignments())
        splits: list[SplitStatistics] = []

        for split_name in _SPLIT_NAMES:
            splits.append(
                self._compute_split_statistics(
                    split_name,
                    assignments_by_split[split_name],
                    total_images,
                )
            )

        total_identities = self._count_distinct_identities(
            result.all_assignments()
        )
        total_bona_fide = self._count_label(
            result.all_assignments(), BONA_FIDE_LABEL
        )
        total_morph = self._count_label(result.all_assignments(), MORPH_LABEL)

        component_count = (
            len(identity_graph.connected_components())
            if identity_graph is not None
            else None
        )

        return DatasetStatisticsReport(
            total_images=total_images,
            total_identities=total_identities,
            total_bona_fide=total_bona_fide,
            total_morph=total_morph,
            component_count=component_count,
            splits=tuple(splits),
        )

    def _compute_split_statistics(
        self,
        split_name: str,
        assignments: tuple[SplitAssignment, ...],
        total_images: int,
    ) -> SplitStatistics:
        """Compute statistics for a single split.

        Args:
            split_name: The name of the split being summarized.
            assignments: The assignments belonging to this split.
            total_images: Total images across the entire dataset, used
                to compute this split's percentage share.

        Returns:
            The computed :class:`SplitStatistics` for this split.
        """
        total_count = len(assignments)
        bona_fide_count = self._count_label(assignments, BONA_FIDE_LABEL)
        morph_count = self._count_label(assignments, MORPH_LABEL)
        identity_count = self._count_distinct_identities(assignments)
        percentage_of_dataset = (
            (total_count / total_images) * 100.0 if total_images > 0 else 0.0
        )
        class_balance = self._compute_class_balance(assignments)

        return SplitStatistics(
            split=split_name,
            total_count=total_count,
            bona_fide_count=bona_fide_count,
            morph_count=morph_count,
            identity_count=identity_count,
            percentage_of_dataset=percentage_of_dataset,
            class_balance=class_balance,
        )

    def _count_label(
        self, assignments: tuple[SplitAssignment, ...], label: str
    ) -> int:
        """Count assignments carrying a specific label.

        Args:
            assignments: The assignments to inspect.
            label: The label to count occurrences of.

        Returns:
            The number of assignments whose ``label`` equals ``label``.
        """
        return sum(1 for a in assignments if a.label == label)

    def _count_distinct_identities(
        self, assignments: tuple[SplitAssignment, ...]
    ) -> int:
        """Count the number of distinct identities referenced.

        Args:
            assignments: The assignments to inspect.

        Returns:
            The number of unique identities across all assignments'
            ``identities`` tuples.
        """
        identities: set[str] = set()
        for assignment in assignments:
            identities.update(assignment.identities)
        return len(identities)

    def _compute_class_balance(
        self, assignments: tuple[SplitAssignment, ...]
    ) -> dict[str, float]:
        """Compute the proportion of each label within a set of assignments.

        Args:
            assignments: The assignments to inspect.

        Returns:
            A mapping from label to its proportion (0.0-1.0) of the
            given assignments, sorted by label name. Empty if
            ``assignments`` is empty.
        """
        if not assignments:
            return {}

        counts: dict[str, int] = {}
        for assignment in assignments:
            counts[assignment.label] = counts.get(assignment.label, 0) + 1

        total = len(assignments)
        return {
            label: counts[label] / total for label in sorted(counts.keys())
        }

    def to_json(self, report: DatasetStatisticsReport, indent: int = 2) -> str:
        """Render a statistics report as a JSON string.

        Args:
            report: The statistics report to render.
            indent: The indentation level passed to ``json.dumps``.

        Returns:
            A JSON-formatted string representation of the report.
        """
        payload = {
            "total_images": report.total_images,
            "total_identities": report.total_identities,
            "total_bona_fide": report.total_bona_fide,
            "total_morph": report.total_morph,
            "component_count": report.component_count,
            "splits": [
                {
                    "split": split.split,
                    "total_count": split.total_count,
                    "bona_fide_count": split.bona_fide_count,
                    "morph_count": split.morph_count,
                    "identity_count": split.identity_count,
                    "percentage_of_dataset": split.percentage_of_dataset,
                    "class_balance": split.class_balance,
                }
                for split in report.splits
            ],
        }
        return json.dumps(payload, indent=indent, sort_keys=False)

    def to_csv(self, report: DatasetStatisticsReport) -> str:
        """Render a statistics report as a CSV string.

        One row is emitted per split, followed by an ``overall`` row
        summarizing the whole dataset. Class balance is flattened into
        a single ``label:proportion`` cell separated by semicolons.

        Args:
            report: The statistics report to render.

        Returns:
            A CSV-formatted string with a header row and one data row
            per split plus an overall summary row.
        """
        buffer = io.StringIO()
        fieldnames = [
            "split",
            "total_count",
            "bona_fide_count",
            "morph_count",
            "identity_count",
            "percentage_of_dataset",
            "class_balance",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()

        for split in report.splits:
            writer.writerow(
                {
                    "split": split.split,
                    "total_count": split.total_count,
                    "bona_fide_count": split.bona_fide_count,
                    "morph_count": split.morph_count,
                    "identity_count": split.identity_count,
                    "percentage_of_dataset": f"{split.percentage_of_dataset:.4f}",
                    "class_balance": self._format_class_balance(
                        split.class_balance
                    ),
                }
            )

        writer.writerow(
            {
                "split": "overall",
                "total_count": report.total_images,
                "bona_fide_count": report.total_bona_fide,
                "morph_count": report.total_morph,
                "identity_count": report.total_identities,
                "percentage_of_dataset": "100.0000",
                "class_balance": "",
            }
        )

        return buffer.getvalue()

    def _format_class_balance(self, class_balance: dict[str, float]) -> str:
        """Flatten a class balance mapping into a single CSV cell string.

        Args:
            class_balance: Mapping from label to proportion.

        Returns:
            A ``;``-separated string of ``label:proportion`` pairs,
            formatted to four decimal places.
        """
        return ";".join(
            f"{label}:{proportion:.4f}"
            for label, proportion in class_balance.items()
        )

    def format_console(self, report: DatasetStatisticsReport) -> str:
        """Render a statistics report as a human-readable console summary.

        Args:
            report: The statistics report to render.

        Returns:
            A multi-line, formatted plain-text summary suitable for
            printing to a terminal.
        """
        lines: list[str] = []
        lines.append("Dataset Split Statistics")
        lines.append("=" * 40)
        lines.append(f"Total images:      {report.total_images}")
        lines.append(f"Total identities:  {report.total_identities}")
        lines.append(f"Bona fide images:  {report.total_bona_fide}")
        lines.append(f"Morph images:      {report.total_morph}")
        component_display = (
            "N/A" if report.component_count is None else str(report.component_count)
        )
        lines.append(f"Identity components: {component_display}")
        lines.append("")

        for split in report.splits:
            lines.append(f"[{split.split}]")
            lines.append(
                f"  total: {split.total_count} "
                f"({split.percentage_of_dataset:.2f}% of dataset)"
            )
            lines.append(f"  bona fide: {split.bona_fide_count}")
            lines.append(f"  morph: {split.morph_count}")
            lines.append(f"  identities: {split.identity_count}")
            if split.class_balance:
                balance_str = ", ".join(
                    f"{label}={proportion:.2%}"
                    for label, proportion in split.class_balance.items()
                )
                lines.append(f"  class balance: {balance_str}")
            else:
                lines.append("  class balance: (empty split)")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"