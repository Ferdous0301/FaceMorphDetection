"""Unit tests for the ManifestVerifier class.

These tests verify:
    * Clean manifests pass verification with no issues.
    * Synthetic identity-leakage scenarios are detected.
    * Duplicate filenames are detected.
    * Missing files are detected (using a real temp directory).
    * Invalid labels are detected.
    * Missing metadata is detected.
    * Component leakage is detected using an IdentityGraph.
    * Split completeness (missing/extra images) is detected.
    * CSV/bucket consistency violations are detected.
"""

from __future__ import annotations

from pathlib import Path

from src.dataset_split.identity_graph import IdentityGraph
from src.dataset_split.splitter import DatasetSplitResult, SplitAssignment
from src.dataset_split.verifier import ManifestVerifier, VerificationReport


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


class TestCleanManifestPasses:
    """Tests verifying a well-formed manifest passes verification."""

    def test_clean_result_has_no_issues(self) -> None:
        """A correctly constructed split result produces zero issues."""
        result = DatasetSplitResult(
            train=(_assignment("bf_a", "train", ("A",)),),
            val=(_assignment("bf_b", "val", ("B",)),),
            test=(_assignment("bf_c", "test", ("C",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.passed
        assert report.issues == ()

    def test_clean_result_with_morphs_passes(self) -> None:
        """A clean result including morph samples passes verification."""
        result = DatasetSplitResult(
            train=(
                _assignment("bf_a", "train", ("A",)),
                _assignment("bf_b", "train", ("B",)),
                _assignment(
                    "morph_ab", "train", ("A", "B"), label="morph"
                ),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.passed


class TestIdentityLeakage:
    """Synthetic identity leakage scenarios."""

    def test_identity_split_across_train_and_test(self) -> None:
        """The same identity appearing in train and test is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("bf_a1", "train", ("A",)),),
            test=(_assignment("bf_a2", "test", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        assert not report.passed
        leakage_issues = report.issues_for("identity_leakage")
        assert len(leakage_issues) == 1
        assert leakage_issues[0].context["identity"] == "A"

    def test_morph_identity_leaks_into_different_split_than_bona_fide(
        self,
    ) -> None:
        """A morph referencing an identity already placed elsewhere leaks."""
        result = DatasetSplitResult(
            train=(_assignment("bf_a", "train", ("A",)),),
            val=(
                _assignment(
                    "morph_ab", "val", ("A", "B"), label="morph"
                ),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert not report.passed
        assert report.issues_for("identity_leakage")


class TestDuplicateFilenames:
    """Tests for duplicate image_id detection."""

    def test_duplicate_image_id_across_splits(self) -> None:
        """The same image_id in two splits is flagged as a duplicate."""
        result = DatasetSplitResult(
            train=(_assignment("dup1", "train", ("A",)),),
            val=(_assignment("dup1", "val", ("B",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        duplicate_issues = report.issues_for("duplicate_image")
        assert len(duplicate_issues) == 1
        assert duplicate_issues[0].context["image_id"] == "dup1"

    def test_duplicate_image_id_within_same_split(self) -> None:
        """Two entries with the same image_id in one split are flagged."""
        result = DatasetSplitResult(
            train=(
                _assignment("dup1", "train", ("A",)),
                _assignment("dup1", "train", ("A",)),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("duplicate_image")

    def test_no_duplicates_produces_no_issue(self) -> None:
        """Unique image IDs never trigger a duplicate_image issue."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",)),),
            val=(_assignment("img2", "val", ("B",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("duplicate_image") == ()


class TestMissingFiles:
    """Tests for the missing-files check, using real temp directories."""

    def test_missing_file_is_flagged(self, tmp_path: Path) -> None:
        """An image_id with no corresponding file on disk is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("missing.png", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result, image_root=tmp_path)

        assert not report.passed
        missing_issues = report.issues_for("missing_file")
        assert len(missing_issues) == 1
        assert missing_issues[0].context["image_id"] == "missing.png"

    def test_existing_file_passes(self, tmp_path: Path) -> None:
        """An image_id with a matching file on disk is not flagged."""
        (tmp_path / "present.png").write_bytes(b"fake image bytes")
        result = DatasetSplitResult(
            train=(_assignment("present.png", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result, image_root=tmp_path)
        assert report.issues_for("missing_file") == ()

    def test_missing_files_check_skipped_without_image_root(self) -> None:
        """Omitting image_root skips the missing-files check entirely."""
        result = DatasetSplitResult(
            train=(_assignment("whatever.png", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("missing_file") == ()


class TestInvalidLabels:
    """Tests for invalid class label detection."""

    def test_unrecognized_label_is_flagged(self) -> None:
        """A label outside the valid set is flagged as invalid."""
        result = DatasetSplitResult(
            train=(
                _assignment(
                    "img1", "train", ("A",), label="not_a_real_label"
                ),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        assert not report.passed
        invalid_issues = report.issues_for("invalid_label")
        assert len(invalid_issues) == 1
        assert invalid_issues[0].context["label"] == "not_a_real_label"

    def test_custom_valid_labels_accepted(self) -> None:
        """A custom valid_labels set is respected."""
        result = DatasetSplitResult(
            train=(
                _assignment(
                    "img1", "train", ("A",), label="synthetic_attack"
                ),
            ),
        )
        verifier = ManifestVerifier(
            valid_labels=frozenset({"synthetic_attack", "bona_fide"})
        )
        report = verifier.verify(result)
        assert report.issues_for("invalid_label") == ()


class TestMissingMetadata:
    """Tests for missing required metadata detection."""

    def test_empty_image_id_is_flagged(self) -> None:
        """An assignment with an empty image_id is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        metadata_issues = report.issues_for("missing_metadata")
        assert len(metadata_issues) == 1
        assert "image_id" in metadata_issues[0].context["missing_fields"]

    def test_empty_identities_is_flagged(self) -> None:
        """An assignment with no identities is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ()),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        metadata_issues = report.issues_for("missing_metadata")
        assert len(metadata_issues) == 1
        assert "identities" in metadata_issues[0].context["missing_fields"]

    def test_empty_label_is_flagged(self) -> None:
        """An assignment with an empty label is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",), label=""),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        metadata_issues = report.issues_for("missing_metadata")
        assert len(metadata_issues) == 1
        assert "label" in metadata_issues[0].context["missing_fields"]


class TestComponentLeakage:
    """Synthetic component leakage scenarios using an IdentityGraph."""

    def test_transitive_component_split_across_splits_is_flagged(
        self,
    ) -> None:
        """Morph(A,B) + Morph(B,C) with C in a different split is flagged.

        Even though the split assignments here do not directly mark A
        and C as leaking against each other via shared image rows, the
        identity graph shows they belong to the same connected
        component, so any split divergence must be caught.
        """
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")

        result = DatasetSplitResult(
            train=(
                _assignment("bf_a", "train", ("A",)),
                _assignment("bf_b", "train", ("B",)),
            ),
            test=(_assignment("bf_c", "test", ("C",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result, identity_graph=graph)

        assert not report.passed
        component_issues = report.issues_for("component_leakage")
        assert len(component_issues) == 1
        assert component_issues[0].context["component"] == "A,B,C"

    def test_intact_component_is_not_flagged(self) -> None:
        """A component fully contained within one split passes."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")

        result = DatasetSplitResult(
            train=(
                _assignment("bf_a", "train", ("A",)),
                _assignment("bf_b", "train", ("B",)),
                _assignment("bf_c", "train", ("C",)),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result, identity_graph=graph)
        assert report.issues_for("component_leakage") == ()

    def test_component_leakage_skipped_without_graph(self) -> None:
        """Omitting identity_graph skips the component-leakage check."""
        result = DatasetSplitResult(
            train=(_assignment("bf_a", "train", ("A",)),),
            test=(_assignment("bf_c", "test", ("C",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("component_leakage") == ()


class TestSplitCompleteness:
    """Tests for the split-completeness check."""

    def test_missing_expected_image_is_flagged(self) -> None:
        """An expected image absent from the result is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(
            result, expected_image_ids={"img1", "img2"}
        )

        completeness_issues = report.issues_for("split_completeness")
        assert len(completeness_issues) == 1
        assert "img2" in completeness_issues[0].context["missing_image_ids"]

    def test_extra_unexpected_image_is_flagged(self) -> None:
        """An image present in the result but not expected is flagged."""
        result = DatasetSplitResult(
            train=(
                _assignment("img1", "train", ("A",)),
                _assignment("img_extra", "train", ("B",)),
            ),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result, expected_image_ids={"img1"})

        completeness_issues = report.issues_for("split_completeness")
        assert len(completeness_issues) == 1
        assert "img_extra" in completeness_issues[0].context["extra_image_ids"]

    def test_matching_expected_ids_passes(self) -> None:
        """A result matching the expected ID set produces no issue."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",)),),
            val=(_assignment("img2", "val", ("B",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(
            result, expected_image_ids={"img1", "img2"}
        )
        assert report.issues_for("split_completeness") == ()

    def test_completeness_skipped_without_expected_ids(self) -> None:
        """Omitting expected_image_ids skips the completeness check."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("split_completeness") == ()


class TestCsvConsistency:
    """Tests for internal bucket/split field consistency."""

    def test_mismatched_split_field_is_flagged(self) -> None:
        """An assignment stored in 'train' but declaring 'test' is flagged."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "test", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        assert not report.passed
        consistency_issues = report.issues_for("csv_consistency")
        assert len(consistency_issues) == 1
        assert consistency_issues[0].context["bucket"] == "train"
        assert consistency_issues[0].context["declared_split"] == "test"

    def test_consistent_split_fields_pass(self) -> None:
        """Assignments whose split field matches their bucket pass."""
        result = DatasetSplitResult(
            train=(_assignment("img1", "train", ("A",)),),
            val=(_assignment("img2", "val", ("B",)),),
            test=(_assignment("img3", "test", ("C",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)
        assert report.issues_for("csv_consistency") == ()


class TestVerificationReport:
    """Tests for VerificationReport helper properties."""

    def test_error_and_warning_counts(self) -> None:
        """error_count and warning_count reflect issue severities."""
        result = DatasetSplitResult(
            train=(_assignment("dup", "train", ("A",)),),
            val=(_assignment("dup", "val", ("A",)),),
        )
        verifier = ManifestVerifier()
        real_report = verifier.verify(result)
        assert real_report.error_count >= 1
        assert real_report.warning_count == 0

    def test_multiple_issue_types_all_collected(self) -> None:
        """Multiple simultaneous violations all appear in the same report."""
        result = DatasetSplitResult(
            train=(
                _assignment("dup", "train", ("A",), label="bad_label"),
            ),
            val=(_assignment("dup", "val", ("A",)),),
        )
        verifier = ManifestVerifier()
        report = verifier.verify(result)

        checks_found = {issue.check for issue in report.issues}
        assert "duplicate_image" in checks_found
        assert "identity_leakage" in checks_found
        assert "invalid_label" in checks_found
        assert not report.passed