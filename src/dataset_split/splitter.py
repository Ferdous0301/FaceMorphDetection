"""Deterministic, identity-disjoint dataset splitting.

This module implements :class:`DatasetSplitter`, which partitions a
dataset of bona fide and morph images into train / validation / test
splits while guaranteeing that no connected component of the identity
graph (see :mod:`src.dataset_split.identity_graph`) is ever divided
across splits. This prevents identity leakage between splits.

The splitter is fully deterministic for a given
:class:`~src.dataset_split.config.DatasetSplitConfig`: the same
configuration and inputs always produce the same split assignment. Only
the configured ``random_seed`` (and ``shuffle`` flag) influence the
order in which identity components are considered, which in turn
influences the greedy balancing of split sizes.

This module contains splitting logic only. It does not read or write
manifests, CSV files, or any other artifacts, and it exposes no CLI.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from src.dataset_split.config import DatasetSplitConfig
from src.dataset_split.exceptions import IdentityLeakageError
from src.dataset_split.identity_graph import IdentityGraph

#: The canonical names of the three dataset splits, in a fixed order.
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class BonaFideSample:
    """A single bona fide (genuine, non-morphed) image sample.

    Attributes:
        image_id: Unique identifier of the image.
        identity: Identifier of the subject depicted in the image.
        label: Class label associated with the sample. Defaults to
            ``"bona_fide"``.
    """

    image_id: str
    identity: str
    label: str = "bona_fide"


@dataclass(frozen=True)
class MorphSample:
    """A single morph image sample generated from two identities.

    Attributes:
        image_id: Unique identifier of the morph image.
        identity_a: The first contributing identity.
        identity_b: The second contributing identity.
        label: Class label associated with the sample. Defaults to
            ``"morph"``.
    """

    image_id: str
    identity_a: str
    identity_b: str
    label: str = "morph"


@dataclass(frozen=True)
class SplitAssignment:
    """The split assignment for a single image sample.

    Attributes:
        image_id: Unique identifier of the assigned image.
        split: The split the image was assigned to (``"train"``,
            ``"val"``, or ``"test"``).
        label: The class label preserved from the source sample.
        identities: The identity or identities associated with this
            image (one for bona fide samples, two for morph samples).
    """

    image_id: str
    split: str
    label: str
    identities: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplitResult:
    """The complete result of a dataset split operation.

    Attributes:
        train: Assignments belonging to the training split.
        val: Assignments belonging to the validation split.
        test: Assignments belonging to the test split.
        component_splits: Mapping from a representative identity of
            each connected component to the split it was assigned to,
            useful for auditing and verification.
    """

    train: tuple[SplitAssignment, ...] = field(default_factory=tuple)
    val: tuple[SplitAssignment, ...] = field(default_factory=tuple)
    test: tuple[SplitAssignment, ...] = field(default_factory=tuple)
    component_splits: tuple[tuple[str, ...], ...] = field(
        default_factory=tuple
    )

    def all_assignments(self) -> tuple[SplitAssignment, ...]:
        """Return every assignment across all three splits.

        Returns:
            A tuple of all :class:`SplitAssignment` instances, in the
            order train, then val, then test.
        """
        return self.train + self.val + self.test

    def split_of(self, split_name: str) -> tuple[SplitAssignment, ...]:
        """Return the assignments for a given split name.

        Args:
            split_name: One of ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            The tuple of assignments belonging to that split.

        Raises:
            ValueError: If ``split_name`` is not a recognized split.
        """
        mapping = {"train": self.train, "val": self.val, "test": self.test}
        if split_name not in mapping:
            raise ValueError(f"Unknown split name: {split_name!r}.")
        return mapping[split_name]


class DatasetSplitter:
    """Performs deterministic, identity-disjoint dataset splitting.

    The splitter builds an :class:`~src.dataset_split.identity_graph.IdentityGraph`
    from the provided bona fide and morph samples, computes its
    connected components, and assigns each component in its entirety to
    exactly one of the train/validation/test splits. Components are
    assigned using a deterministic, seed-controlled shuffle followed by
    a greedy balancing procedure that targets the configured split
    ratios as closely as possible without ever dividing a component.
    """

    def __init__(self, config: DatasetSplitConfig) -> None:
        """Initialize the splitter with a validated configuration.

        Args:
            config: The dataset split configuration, including split
                ratios, random seed, and shuffle behavior.
        """
        self._config = config

    def split(
        self,
        bona_fide_samples: list[BonaFideSample],
        morph_samples: list[MorphSample],
    ) -> DatasetSplitResult:
        """Split bona fide and morph samples into train/val/test sets.

        Args:
            bona_fide_samples: All bona fide image samples to assign.
            morph_samples: All morph image samples to assign.

        Returns:
            A :class:`DatasetSplitResult` containing every input image
            assigned to exactly one split, with class labels preserved
            and identity components kept fully intact within a single
            split.

        Raises:
            IdentityLeakageError: If, due to an internal invariant
                violation, a connected component would be split across
                more than one dataset split. This should never occur
                under normal operation and indicates a programming
                error rather than a user error.
        """
        graph = self._build_identity_graph(bona_fide_samples, morph_samples)
        components = graph.connected_components()

        if not components:
            return DatasetSplitResult()

        identity_to_component_index = self._map_identities_to_components(
            components
        )
        component_image_counts = self._count_images_per_component(
            components,
            identity_to_component_index,
            bona_fide_samples,
            morph_samples,
        )
        component_order = self._deterministic_component_order(
            len(components)
        )
        component_split_assignment = self._assign_components_to_splits(
            component_order, component_image_counts
        )

        self._verify_no_leakage(
            components, identity_to_component_index, component_split_assignment
        )

        return self._build_result(
            bona_fide_samples,
            morph_samples,
            identity_to_component_index,
            component_split_assignment,
            components,
        )

    def _build_identity_graph(
        self,
        bona_fide_samples: list[BonaFideSample],
        morph_samples: list[MorphSample],
    ) -> IdentityGraph:
        """Construct the identity graph from bona fide and morph samples.

        Args:
            bona_fide_samples: All bona fide image samples.
            morph_samples: All morph image samples.

        Returns:
            The populated :class:`IdentityGraph`.
        """
        graph = IdentityGraph()
        for sample in bona_fide_samples:
            graph.add_identity(sample.identity)
        for morph in morph_samples:
            graph.add_morph(morph.identity_a, morph.identity_b)
        return graph

    def _map_identities_to_components(
        self, components: list[list[str]]
    ) -> dict[str, int]:
        """Map each identity to the index of its connected component.

        Args:
            components: The list of connected components, as returned
                by :meth:`IdentityGraph.connected_components`.

        Returns:
            A mapping from identity to its owning component's index.
        """
        identity_to_component_index: dict[str, int] = {}
        for index, component in enumerate(components):
            for identity in component:
                identity_to_component_index[identity] = index
        return identity_to_component_index

    def _count_images_per_component(
        self,
        components: list[list[str]],
        identity_to_component_index: dict[str, int],
        bona_fide_samples: list[BonaFideSample],
        morph_samples: list[MorphSample],
    ) -> list[int]:
        """Count the total number of images belonging to each component.

        Args:
            components: The list of connected components.
            identity_to_component_index: Identity-to-component-index map.
            bona_fide_samples: All bona fide image samples.
            morph_samples: All morph image samples.

        Returns:
            A list where index ``i`` holds the total image count
            (bona fide plus morph) for component ``i``.
        """
        counts = [0] * len(components)
        for sample in bona_fide_samples:
            index = identity_to_component_index[sample.identity]
            counts[index] += 1
        for morph in morph_samples:
            index = identity_to_component_index[morph.identity_a]
            counts[index] += 1
        return counts

    def _deterministic_component_order(self, num_components: int) -> list[int]:
        """Compute a deterministic traversal order over component indices.

        Args:
            num_components: The total number of connected components.

        Returns:
            A list of component indices in the order they should be
            considered for split assignment. If ``config.shuffle`` is
            ``False``, this is simply insertion order. Otherwise, it is
            a seeded, reproducible shuffle of that order.
        """
        order = list(range(num_components))
        if self._config.shuffle:
            rng = random.Random(self._config.random_seed)
            rng.shuffle(order)
        return order

    def _assign_components_to_splits(
        self, component_order: list[int], component_image_counts: list[int]
    ) -> dict[int, str]:
        """Greedily assign components to splits to approximate target ratios.

        Each component is assigned, in ``component_order``, to whichever
        split currently has the largest unmet proportional need,
        computed as the ratio of images already assigned to that split
        relative to its target ratio. Splits with a target ratio of
        zero are never selected.

        Args:
            component_order: Deterministic order in which to consider
                components for assignment.
            component_image_counts: Total image count per component,
                indexed identically to ``component_order`` values.

        Returns:
            A mapping from component index to the split name it was
            assigned to.
        """
        ratios = {
            "train": self._config.train_ratio,
            "val": self._config.val_ratio,
            "test": self._config.test_ratio,
        }
        assigned_counts = {split: 0 for split in SPLIT_NAMES}
        component_split_assignment: dict[int, str] = {}

        for component_index in component_order:
            best_split = self._select_split_with_greatest_deficit(
                ratios, assigned_counts
            )
            component_split_assignment[component_index] = best_split
            assigned_counts[best_split] += component_image_counts[
                component_index
            ]

        return component_split_assignment

    def _select_split_with_greatest_deficit(
        self, ratios: dict[str, float], assigned_counts: dict[str, int]
    ) -> str:
        """Select the split furthest below its target proportional share.

        Args:
            ratios: Target ratio per split.
            assigned_counts: Images already assigned per split so far.

        Returns:
            The name of the split with the smallest
            ``assigned_count / target_ratio`` value, preferring earlier
            entries in :data:`SPLIT_NAMES` on ties. Splits with a
            target ratio of zero are treated as having infinite
            deficit-fill (i.e. never selected while any other split
            still has capacity).
        """
        best_split = SPLIT_NAMES[0]
        best_fill_level = float("inf")

        for split in SPLIT_NAMES:
            target_ratio = ratios[split]
            if target_ratio <= 0:
                continue
            fill_level = assigned_counts[split] / target_ratio
            if fill_level < best_fill_level:
                best_fill_level = fill_level
                best_split = split

        return best_split

    def _verify_no_leakage(
        self,
        components: list[list[str]],
        identity_to_component_index: dict[str, int],
        component_split_assignment: dict[int, str],
    ) -> None:
        """Verify that every identity's split matches its component's split.

        Args:
            components: The list of connected components.
            identity_to_component_index: Identity-to-component-index map.
            component_split_assignment: Component-index-to-split map.

        Raises:
            IdentityLeakageError: If any identity would resolve to a
                split different from its component's assigned split.
        """
        for index, component in enumerate(components):
            expected_split = component_split_assignment[index]
            for identity in component:
                actual_index = identity_to_component_index[identity]
                if component_split_assignment[actual_index] != expected_split:
                    raise IdentityLeakageError(
                        f"Identity {identity!r} resolved to a split "
                        "inconsistent with its connected component; "
                        "this indicates a component was divided across "
                        "splits."
                    )

    def _build_result(
        self,
        bona_fide_samples: list[BonaFideSample],
        morph_samples: list[MorphSample],
        identity_to_component_index: dict[str, int],
        component_split_assignment: dict[int, str],
        components: list[list[str]],
    ) -> DatasetSplitResult:
        """Assemble the final per-split assignment lists.

        Args:
            bona_fide_samples: All bona fide image samples.
            morph_samples: All morph image samples.
            identity_to_component_index: Identity-to-component-index map.
            component_split_assignment: Component-index-to-split map.
            components: The list of connected components.

        Returns:
            The final :class:`DatasetSplitResult`, preserving input
            order within each split.
        """
        buckets: dict[str, list[SplitAssignment]] = defaultdict(list)

        for sample in bona_fide_samples:
            index = identity_to_component_index[sample.identity]
            split = component_split_assignment[index]
            buckets[split].append(
                SplitAssignment(
                    image_id=sample.image_id,
                    split=split,
                    label=sample.label,
                    identities=(sample.identity,),
                )
            )

        for morph in morph_samples:
            index = identity_to_component_index[morph.identity_a]
            split = component_split_assignment[index]
            buckets[split].append(
                SplitAssignment(
                    image_id=morph.image_id,
                    split=split,
                    label=morph.label,
                    identities=(morph.identity_a, morph.identity_b),
                )
            )

        component_splits = tuple(
            (component_split_assignment[index], *components[index])
            for index in range(len(components))
        )

        return DatasetSplitResult(
            train=tuple(buckets["train"]),
            val=tuple(buckets["val"]),
            test=tuple(buckets["test"]),
            component_splits=component_splits,
        )