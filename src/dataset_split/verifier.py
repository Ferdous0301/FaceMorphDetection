"""Verification of generated dataset split manifests.

This module provides :class:`ManifestVerifier`, which inspects a
:class:`~src.dataset_split.splitter.DatasetSplitResult` (and, optionally,
its originating identity graph and on-disk image files) for correctness
issues before the split is considered final. Verification produces a
structured :class:`VerificationReport` rather than raising exceptions,
so that callers can inspect, log, or act on every issue found in a
single pass.

Checks performed:
    * Identity leakage across splits.
    * Duplicate filenames (image IDs) within or across splits.
    * Missing image files on disk (when an image root is provided).
    * Invalid class labels.
    * Missing required metadata on individual assignments.
    * Connected-component leakage across splits.
    * Split completeness against an expected set of image IDs.
    * Internal CSV/record consistency (e.g. bucket/split field mismatch).

This module performs verification only; it does not perform splitting,
manifest writing, or any corrective action.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset_split.identity_graph import IdentityGraph
from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment

#: The set of split names a DatasetSplitResult is expected to expose.
_KNOWN_SPLITS: tuple[str, str, str] = ("train", "val", "test")

#: Default set of labels considered valid if none is explicitly supplied.
DEFAULT_VALID_LABELS: frozenset[str] = frozenset({"bona_fide", "morph"})

Severity = str  # "error" or "warning"


@dataclass(frozen=True)
class VerificationIssue:
    """A single verification finding.

    Attributes:
        check: Short machine-readable name of the check that produced
            this issue (e.g. ``"identity_leakage"``).
        severity: Either ``"error"`` (a correctness violation) or
            ``"warning"`` (a non-fatal concern worth surfacing).
        message: Human-readable description of the issue.
        context: Additional structured details relevant to the issue,
            such as the offending image ID or identity.
    """

    check: str
    severity: Severity
    message: str
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """The aggregated result of running all verification checks.

    Attributes:
        issues: Every issue discovered across all checks, in the order
            checks were executed.
    """

    issues: tuple[VerificationIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Whether the manifest passed verification.

        Returns:
            ``True`` if no issue with severity ``"error"`` was found;
            ``False`` otherwise. Warnings alone do not fail
            verification.
        """
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Return the number of error-severity issues found."""
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning-severity issues found."""
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def issues_for(self, check: str) -> tuple[VerificationIssue, ...]:
        """Return only the issues produced by a specific check.

        Args:
            check: The check name to filter by (e.g. ``"duplicate_image"``).

        Returns:
            A tuple of matching issues, preserving original order.
        """
        return tuple(issue for issue in self.issues if issue.check == check)


class ManifestVerifier:
    """Verifies dataset split manifests for correctness and consistency.

    An instance of this class can be reused across multiple
    verification runs. All checks are pure functions of their inputs:
    no state is mutated and no exceptions are raised for verification
    failures. Failures are instead reported as structured
    :class:`VerificationIssue` entries.
    """

    def __init__(
        self, valid_labels: frozenset[str] = DEFAULT_VALID_LABELS
    ) -> None:
        """Initialize the verifier.

        Args:
            valid_labels: The set of class labels considered valid.
                Defaults to :data:`DEFAULT_VALID_LABELS`.
        """
        self._valid_labels = valid_labels

    def verify(
        self,
        result: DatasetSplitResult,
        *,
        identity_graph: IdentityGraph | None = None,
        image_root: Path | str | None = None,
        expected_image_ids: set[str] | None = None,
    ) -> VerificationReport:
        """Run all verification checks against a dataset split result.

        Args:
            result: The dataset split result to verify.
            identity_graph: The identity graph the split was derived
                from. When provided, enables the component-leakage
                check, which verifies that no connected component was
                divided across multiple splits.
            image_root: Directory containing the actual image files.
                When provided, enables the missing-files check, which
                verifies that every referenced ``image_id`` exists as a
                file under this directory.
            expected_image_ids: The complete set of image IDs that were
                expected to be assigned. When provided, enables the
                split-completeness check, which verifies no image is
                missing from or extraneous to the split result.

        Returns:
            A :class:`VerificationReport` aggregating every issue found
            across all applicable checks.
        """
        issues: list[VerificationIssue] = []

        issues.extend(self._check_csv_consistency(result))
        issues.extend(self._check_missing_metadata(result))
        issues.extend(self._check_invalid_labels(result))
        issues.extend(self._check_duplicate_filenames(result))
        issues.extend(self._check_identity_leakage(result))

        if identity_graph is not None:
            issues.extend(
                self._check_component_leakage(result, identity_graph)
            )

        issues.extend(
            self._check_split_completeness(result, expected_image_ids)
        )

        if image_root is not None:
            issues.extend(
                self._check_missing_files(result, Path(image_root))
            )

        return VerificationReport(issues=tuple(issues))

    def _all_assignments_by_split(
        self, result: DatasetSplitResult
    ) -> dict[str, tuple[SplitAssignment, ...]]:
        """Return each split's assignments keyed by its declared split name."""
        return {"train": result.train, "val": result.val, "test": result.test}

    def _check_csv_consistency(
        self, result: DatasetSplitResult
    ) -> list[VerificationIssue]:
        """Verify each assignment's declared split matches its bucket.

        Args:
            result: The dataset split result to verify.

        Returns:
            One issue per assignment whose ``split`` field does not
            match the bucket (``train``/``val``/``test``) it was found
            in.
        """
        issues: list[VerificationIssue] = []
        for bucket_name, assignments in self._all_assignments_by_split(
            result
        ).items():
            for assignment in assignments:
                if assignment.split != bucket_name:
                    issues.append(
                        VerificationIssue(
                            check="csv_consistency",
                            severity="error",
                            message=(
                                f"Assignment for image {assignment.image_id!r} "
                                f"is stored in the {bucket_name!r} bucket but "
                                f"declares split {assignment.split!r}."
                            ),
                            context={
                                "image_id": assignment.image_id,
                                "bucket": bucket_name,
                                "declared_split": assignment.split,
                            },
                        )
                    )
        return issues

    def _check_missing_metadata(
        self, result: DatasetSplitResult
    ) -> list[VerificationIssue]:
        """Verify every assignment carries required, non-empty metadata.

        Args:
            result: The dataset split result to verify.

        Returns:
            One issue per assignment missing an ``image_id``, ``split``,
            ``label``, or non-empty ``identities`` tuple.
        """
        issues: list[VerificationIssue] = []
        for assignment in result.all_assignments():
            missing_fields = []
            if not assignment.image_id:
                missing_fields.append("image_id")
            if not assignment.split:
                missing_fields.append("split")
            if not assignment.label:
                missing_fields.append("label")
            if not assignment.identities:
                missing_fields.append("identities")

            if missing_fields:
                issues.append(
                    VerificationIssue(
                        check="missing_metadata",
                        severity="error",
                        message=(
                            f"Assignment {assignment.image_id or '<unknown>'!r} "
                            f"is missing required fields: {missing_fields}."
                        ),
                        context={
                            "image_id": assignment.image_id,
                            "missing_fields": ",".join(missing_fields),
                        },
                    )
                )
        return issues

    def _check_invalid_labels(
        self, result: DatasetSplitResult
    ) -> list[VerificationIssue]:
        """Verify every assignment's label is within the configured valid set.

        Args:
            result: The dataset split result to verify.

        Returns:
            One issue per assignment whose label is not recognized.
        """
        issues: list[VerificationIssue] = []
        for assignment in result.all_assignments():
            if assignment.label not in self._valid_labels:
                issues.append(
                    VerificationIssue(
                        check="invalid_label",
                        severity="error",
                        message=(
                            f"Assignment {assignment.image_id!r} has invalid "
                            f"label {assignment.label!r}; expected one of "
                            f"{sorted(self._valid_labels)}."
                        ),
                        context={
                            "image_id": assignment.image_id,
                            "label": assignment.label,
                        },
                    )
                )
        return issues

    def _check_duplicate_filenames(
        self, result: DatasetSplitResult
    ) -> list[VerificationIssue]:
        """Verify no ``image_id`` appears more than once across all splits.

        Args:
            result: The dataset split result to verify.

        Returns:
            One issue per ``image_id`` that appears more than once,
            listing the splits it was found in.
        """
        occurrences: dict[str, list[str]] = defaultdict(list)
        for assignment in result.all_assignments():
            occurrences[assignment.image_id].append(assignment.split)

        issues: list[VerificationIssue] = []
        for image_id, splits in occurrences.items():
            if len(splits) > 1:
                issues.append(
                    VerificationIssue(
                        check="duplicate_image",
                        severity="error",
                        message=(
                            f"Image {image_id!r} appears {len(splits)} times "
                            f"across splits {splits}."
                        ),
                        context={
                            "image_id": image_id,
                            "splits": ",".join(splits),
                        },
                    )
                )
        return issues

    def _check_identity_leakage(
        self, result: DatasetSplitResult
    ) -> list[VerificationIssue]:
        """Verify no identity's images are assigned to more than one split.

        Args:
            result: The dataset split result to verify.

        Returns:
            One issue per identity that appears in more than one split.
        """
        identity_to_splits: dict[str, set[str]] = defaultdict(set)
        for assignment in result.all_assignments():
            for identity in assignment.identities:
                identity_to_splits[identity].add(assignment.split)

        issues: list[VerificationIssue] = []
        for identity, splits in identity_to_splits.items():
            if len(splits) > 1:
                issues.append(
                    VerificationIssue(
                        check="identity_leakage",
                        severity="error",
                        message=(
                            f"Identity {identity!r} leaks across splits "
                            f"{sorted(splits)}."
                        ),
                        context={
                            "identity": identity,
                            "splits": ",".join(sorted(splits)),
                        },
                    )
                )
        return issues

    def _check_component_leakage(
        self, result: DatasetSplitResult, identity_graph: IdentityGraph
    ) -> list[VerificationIssue]:
        """Verify no connected identity component spans multiple splits.

        Args:
            result: The dataset split result to verify.
            identity_graph: The identity graph the split was derived
                from.

        Returns:
            One issue per connected component whose member identities
            resolve to more than one split.
        """
        identity_to_splits: dict[str, set[str]] = defaultdict(set)
        for assignment in result.all_assignments():
            for identity in assignment.identities:
                identity_to_splits[identity].add(assignment.split)

        issues: list[VerificationIssue] = []
        seen_components: set[frozenset[str]] = set()

        for identity in identity_graph.identities():
            component = frozenset(identity_graph.component_of(identity))
            if component in seen_components:
                continue
            seen_components.add(component)

            splits_touched: set[str] = set()
            for member in component:
                splits_touched.update(identity_to_splits.get(member, set()))

            if len(splits_touched) > 1:
                issues.append(
                    VerificationIssue(
                        check="component_leakage",
                        severity="error",
                        message=(
                            f"Connected component {sorted(component)} spans "
                            f"multiple splits: {sorted(splits_touched)}."
                        ),
                        context={
                            "component": ",".join(sorted(component)),
                            "splits": ",".join(sorted(splits_touched)),
                        },
                    )
                )
        return issues

    def _check_split_completeness(
        self,
        result: DatasetSplitResult,
        expected_image_ids: set[str] | None,
    ) -> list[VerificationIssue]:
        """Verify the split result covers exactly the expected image IDs.

        Args:
            result: The dataset split result to verify.
            expected_image_ids: The complete set of image IDs expected
                to appear across all splits, or ``None`` to skip this
                check.

        Returns:
            An issue describing any missing or unexpected extra image
            IDs. Returns an empty list if ``expected_image_ids`` is
            ``None``.
        """
        if expected_image_ids is None:
            return []

        actual_image_ids = {a.image_id for a in result.all_assignments()}
        missing = expected_image_ids - actual_image_ids
        extra = actual_image_ids - expected_image_ids

        issues: list[VerificationIssue] = []
        if missing:
            issues.append(
                VerificationIssue(
                    check="split_completeness",
                    severity="error",
                    message=(
                        f"{len(missing)} expected image(s) are missing from "
                        "the split result."
                    ),
                    context={"missing_image_ids": ",".join(sorted(missing))},
                )
            )
        if extra:
            issues.append(
                VerificationIssue(
                    check="split_completeness",
                    severity="error",
                    message=(
                        f"{len(extra)} unexpected image(s) appear in the "
                        "split result."
                    ),
                    context={"extra_image_ids": ",".join(sorted(extra))},
                )
            )
        return issues

    def _check_missing_files(
        self, result: DatasetSplitResult, image_root: Path
    ) -> list[VerificationIssue]:
        """Verify every referenced image file exists on disk.

        Args:
            result: The dataset split result to verify.
            image_root: Directory expected to contain each referenced
                image file, addressed by ``image_id`` as the filename.

        Returns:
            One issue per ``image_id`` with no corresponding file under
            ``image_root``.
        """
        issues: list[VerificationIssue] = []
        for assignment in result.all_assignments():
            candidate_path = image_root / assignment.image_id
            if not candidate_path.exists():
                issues.append(
                    VerificationIssue(
                        check="missing_file",
                        severity="error",
                        message=(
                            f"Image file for {assignment.image_id!r} was not "
                            f"found under {image_root}."
                        ),
                        context={
                            "image_id": assignment.image_id,
                            "expected_path": str(candidate_path),
                        },
                    )
                )
        return issues