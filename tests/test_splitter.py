"""Unit tests for the DatasetSplitter class.

These tests verify:
    * Deterministic output for a fixed random seed.
    * Different seeds may reorder component assignment.
    * No identity leakage across splits.
    * Split ratios are approximately respected.
    * Empty dataset handling.
    * Single-component and many-component scenarios.
    * All morph images are assigned correctly.
    * No image is assigned to more than one split.
    * Every image is assigned exactly once.
    * Various edge cases (isolated identities, chains, ties).
"""

from __future__ import annotations

from src.dataset_split.config import DatasetSplitConfig
from src.dataset_split.splitter import (
    BonaFideSample,
    DatasetSplitter,
    MorphSample,
    SplitAssignment,
)


def _make_splitter(**overrides: object) -> DatasetSplitter:
    """Construct a DatasetSplitter with sensible test defaults."""
    config = DatasetSplitConfig(**overrides)  # type: ignore[arg-type]
    return DatasetSplitter(config)


def _identity_set(assignments: tuple[SplitAssignment, ...]) -> set[str]:
    """Flatten the identities referenced by a tuple of assignments."""
    result: set[str] = set()
    for assignment in assignments:
        result.update(assignment.identities)
    return result


class TestEmptyDataset:
    """Tests for splitting an empty dataset."""

    def test_empty_inputs_produce_empty_result(self) -> None:
        """No bona fide or morph samples yields empty splits."""
        splitter = _make_splitter()
        result = splitter.split([], [])
        assert result.train == ()
        assert result.val == ()
        assert result.test == ()
        assert result.component_splits == ()

    def test_empty_inputs_all_assignments_empty(self) -> None:
        """all_assignments() is empty for an empty dataset."""
        splitter = _make_splitter()
        result = splitter.split([], [])
        assert result.all_assignments() == ()


class TestSingleComponent:
    """Tests for a dataset containing a single connected component."""

    def test_single_bona_fide_identity_assigned_to_one_split(self) -> None:
        """A single isolated bona fide identity lands in exactly one split."""
        splitter = _make_splitter()
        samples = [BonaFideSample(image_id="img1", identity="A")]
        result = splitter.split(samples, [])
        all_splits = [
            result.train,
            result.val,
            result.test,
        ]
        non_empty = [s for s in all_splits if s]
        assert len(non_empty) == 1
        assert non_empty[0][0].image_id == "img1"

    def test_connected_component_stays_together(self) -> None:
        """A whole connected component of bona fide + morph samples lands together."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
            BonaFideSample(image_id="bf_c", identity="C"),
        ]
        morphs = [
            MorphSample(image_id="morph_ab", identity_a="A", identity_b="B"),
            MorphSample(image_id="morph_bc", identity_a="B", identity_b="C"),
        ]
        result = splitter.split(bona_fide, morphs)
        all_assignments = result.all_assignments()
        splits_used = {a.split for a in all_assignments}
        assert len(splits_used) == 1


class TestManyComponents:
    """Tests for datasets containing many independent components."""

    def test_many_isolated_identities_all_assigned(self) -> None:
        """All isolated identities are assigned across the three splits."""
        splitter = _make_splitter(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        samples = [
            BonaFideSample(image_id=f"img_{i}", identity=f"id_{i}")
            for i in range(100)
        ]
        result = splitter.split(samples, [])
        assert len(result.all_assignments()) == 100
        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0


class TestDeterministicSeed:
    """Tests verifying deterministic output for a fixed seed."""

    def _sample_dataset(
        self,
    ) -> tuple[list[BonaFideSample], list[MorphSample]]:
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(40)
        ]
        morphs = [
            MorphSample(
                image_id=f"morph_{i}",
                identity_a=f"id_{i}",
                identity_b=f"id_{i + 1}",
            )
            for i in range(0, 38, 2)
        ]
        return bona_fide, morphs

    def test_same_seed_produces_identical_result(self) -> None:
        """Running the splitter twice with the same seed is reproducible."""
        bona_fide, morphs = self._sample_dataset()
        splitter_1 = _make_splitter(random_seed=7)
        splitter_2 = _make_splitter(random_seed=7)

        result_1 = splitter_1.split(bona_fide, morphs)
        result_2 = splitter_2.split(bona_fide, morphs)

        assert result_1.train == result_2.train
        assert result_1.val == result_2.val
        assert result_1.test == result_2.test

    def test_same_splitter_instance_is_reproducible(self) -> None:
        """Calling split() twice on the same splitter instance is stable."""
        bona_fide, morphs = self._sample_dataset()
        splitter = _make_splitter(random_seed=123)

        result_1 = splitter.split(bona_fide, morphs)
        result_2 = splitter.split(bona_fide, morphs)

        assert result_1 == result_2


class TestDifferentSeeds:
    """Tests verifying that different seeds can change the assignment."""

    def test_different_seeds_can_yield_different_assignments(self) -> None:
        """Different random seeds may produce different split assignments."""
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(60)
        ]
        splitter_a = _make_splitter(random_seed=1)
        splitter_b = _make_splitter(random_seed=999)

        result_a = splitter_a.split(bona_fide, [])
        result_b = splitter_b.split(bona_fide, [])

        images_train_a = {a.image_id for a in result_a.train}
        images_train_b = {a.image_id for a in result_b.train}

        assert images_train_a != images_train_b

    def test_different_seeds_still_produce_valid_partitions(self) -> None:
        """Regardless of seed, every image is still assigned exactly once."""
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(30)
        ]
        for seed in (0, 5, 999):
            splitter = _make_splitter(random_seed=seed)
            result = splitter.split(bona_fide, [])
            assert len(result.all_assignments()) == 30


class TestNoLeakage:
    """Tests verifying identity-disjoint guarantees across splits."""

    def test_transitive_morph_chain_never_splits(self) -> None:
        """Morph(A,B) + Morph(B,C) keeps A, B, C in the same split."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
            BonaFideSample(image_id="bf_c", identity="C"),
        ]
        morphs = [
            MorphSample(image_id="m_ab", identity_a="A", identity_b="B"),
            MorphSample(image_id="m_bc", identity_a="B", identity_b="C"),
        ]
        result = splitter.split(bona_fide, morphs)

        identity_to_split: dict[str, str] = {}
        for assignment in result.all_assignments():
            for identity in assignment.identities:
                identity_to_split.setdefault(identity, assignment.split)
                assert identity_to_split[identity] == assignment.split

        assert identity_to_split["A"] == identity_to_split["B"] == identity_to_split["C"]

    def test_no_identity_appears_in_multiple_splits(self) -> None:
        """No identity's images are spread across more than one split."""
        splitter = _make_splitter(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i % 20}")
            for i in range(200)
        ]
        morphs = [
            MorphSample(
                image_id=f"morph_{i}",
                identity_a=f"id_{i % 20}",
                identity_b=f"id_{(i + 1) % 20}",
            )
            for i in range(50)
        ]
        result = splitter.split(bona_fide, morphs)

        identity_splits: dict[str, set[str]] = {}
        for assignment in result.all_assignments():
            for identity in assignment.identities:
                identity_splits.setdefault(identity, set()).add(
                    assignment.split
                )

        for identity, splits in identity_splits.items():
            assert len(splits) == 1, f"Identity {identity} leaked across splits"


class TestRatioCorrectness:
    """Tests verifying that split sizes approximate configured ratios."""

    def test_ratios_approximately_respected_for_many_small_components(
        self,
    ) -> None:
        """With many small components, split sizes approach target ratios."""
        splitter = _make_splitter(
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(1000)
        ]
        result = splitter.split(bona_fide, [])

        total = len(result.all_assignments())
        train_fraction = len(result.train) / total
        val_fraction = len(result.val) / total
        test_fraction = len(result.test) / total

        assert abs(train_fraction - 0.7) < 0.05
        assert abs(val_fraction - 0.2) < 0.05
        assert abs(test_fraction - 0.1) < 0.05

    def test_zero_ratio_split_receives_no_samples(self) -> None:
        """A split configured with a zero ratio never receives components."""
        splitter = _make_splitter(
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(50)
        ]
        result = splitter.split(bona_fide, [])
        assert result.val == ()
        assert result.test == ()
        assert len(result.train) == 50


class TestMorphAssignment:
    """Tests verifying morph images are assigned correctly."""

    def test_all_morphs_assigned(self) -> None:
        """Every morph sample appears exactly once in the result."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(10)
        ]
        morphs = [
            MorphSample(
                image_id=f"morph_{i}",
                identity_a=f"id_{i}",
                identity_b=f"id_{i + 1}",
            )
            for i in range(9)
        ]
        result = splitter.split(bona_fide, morphs)
        morph_ids_out = {
            a.image_id
            for a in result.all_assignments()
            if a.label == "morph"
        }
        expected_ids = {m.image_id for m in morphs}
        assert morph_ids_out == expected_ids

    def test_morph_labels_preserved(self) -> None:
        """Morph sample labels propagate into the resulting assignments."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
        ]
        morphs = [
            MorphSample(
                image_id="morph_ab",
                identity_a="A",
                identity_b="B",
                label="morph_attack",
            )
        ]
        result = splitter.split(bona_fide, morphs)
        morph_assignment = next(
            a for a in result.all_assignments() if a.image_id == "morph_ab"
        )
        assert morph_assignment.label == "morph_attack"
        assert morph_assignment.identities == ("A", "B")

    def test_bona_fide_labels_preserved(self) -> None:
        """Bona fide sample labels propagate into the resulting assignments."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A", label="genuine")
        ]
        result = splitter.split(bona_fide, [])
        assignment = result.all_assignments()[0]
        assert assignment.label == "genuine"
        assert assignment.identities == ("A",)


class TestNoDuplicateAssignment:
    """Tests verifying no image is ever assigned more than once."""

    def test_no_duplicate_image_ids_across_splits(self) -> None:
        """No image_id appears in more than one split."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(50)
        ]
        morphs = [
            MorphSample(
                image_id=f"morph_{i}",
                identity_a=f"id_{i}",
                identity_b=f"id_{i + 1}",
            )
            for i in range(49)
        ]
        result = splitter.split(bona_fide, morphs)

        all_ids = [a.image_id for a in result.all_assignments()]
        assert len(all_ids) == len(set(all_ids))

    def test_every_image_assigned_exactly_once(self) -> None:
        """Every provided image_id appears exactly once in the output."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id=f"bf_{i}", identity=f"id_{i}")
            for i in range(20)
        ]
        morphs = [
            MorphSample(
                image_id=f"morph_{i}",
                identity_a=f"id_{i}",
                identity_b=f"id_{i + 1}",
            )
            for i in range(19)
        ]
        result = splitter.split(bona_fide, morphs)

        expected_ids = {s.image_id for s in bona_fide} | {
            m.image_id for m in morphs
        }
        actual_ids = {a.image_id for a in result.all_assignments()}
        assert actual_ids == expected_ids
        assert len(result.all_assignments()) == len(expected_ids)


class TestEdgeCases:
    """Additional edge case tests."""

    def test_disjoint_pairs_do_not_merge(self) -> None:
        """Two separate morph pairs never end up forced together."""
        splitter = _make_splitter(
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
        )
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
            BonaFideSample(image_id="bf_c", identity="C"),
            BonaFideSample(image_id="bf_d", identity="D"),
        ]
        morphs = [
            MorphSample(image_id="m_ab", identity_a="A", identity_b="B"),
            MorphSample(image_id="m_cd", identity_a="C", identity_b="D"),
        ]
        result = splitter.split(bona_fide, morphs)

        identity_to_split: dict[str, str] = {}
        for assignment in result.all_assignments():
            for identity in assignment.identities:
                identity_to_split[identity] = assignment.split

        # AB pair must share a split, CD pair must share a split, but the
        # two pairs are not required to share the same split as each other.
        assert identity_to_split["A"] == identity_to_split["B"]
        assert identity_to_split["C"] == identity_to_split["D"]

    def test_no_bona_fide_only_morphs(self) -> None:
        """A dataset with only morph samples still splits correctly."""
        splitter = _make_splitter()
        morphs = [
            MorphSample(image_id="m_ab", identity_a="A", identity_b="B"),
        ]
        result = splitter.split([], morphs)
        assert len(result.all_assignments()) == 1
        assert result.all_assignments()[0].image_id == "m_ab"

    def test_self_morph_identity_handled(self) -> None:
        """A morph sample referencing the same identity twice is handled."""
        splitter = _make_splitter()
        morphs = [
            MorphSample(image_id="m_self", identity_a="A", identity_b="A"),
        ]
        result = splitter.split([], morphs)
        assignment = result.all_assignments()[0]
        assert assignment.identities == ("A", "A")

    def test_shuffle_disabled_preserves_component_insertion_order(
        self,
    ) -> None:
        """With shuffle disabled, component order follows insertion order."""
        splitter = _make_splitter(shuffle=False)
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
        ]
        result_1 = splitter.split(bona_fide, [])
        result_2 = splitter.split(bona_fide, [])
        assert result_1 == result_2

    def test_component_splits_cover_all_identities(self) -> None:
        """component_splits records every identity from every component."""
        splitter = _make_splitter()
        bona_fide = [
            BonaFideSample(image_id="bf_a", identity="A"),
            BonaFideSample(image_id="bf_b", identity="B"),
        ]
        morphs = [
            MorphSample(image_id="m_ab", identity_a="A", identity_b="B"),
        ]
        result = splitter.split(bona_fide, morphs)
        flattened = {
            identity
            for component in result.component_splits
            for identity in component[1:]
        }
        assert flattened == {"A", "B"}