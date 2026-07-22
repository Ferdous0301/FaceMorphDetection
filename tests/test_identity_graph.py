"""Unit tests for the IdentityGraph class.

These tests verify:
    * Isolated identities, simple edges, chains, stars, and cycles.
    * Disconnected graphs produce the correct number of components.
    * Duplicate edges do not create duplicate adjacency entries.
    * Deterministic, stable traversal ordering.
    * Empty graph behavior.
    * Large synthetic graph correctness.
    * Component lookup via component_of.
    * The critical leakage-prevention case: Morph(A,B) + Morph(B,C) must
      yield a single connected component containing A, B, and C.
"""

from __future__ import annotations

import pytest

from src.dataset_split.identity_graph import IdentityGraph, UnknownIdentityError


class TestIsolatedIdentities:
    """Tests for identities with no morph relationships."""

    def test_single_isolated_identity(self) -> None:
        """A single added identity forms its own component."""
        graph = IdentityGraph()
        graph.add_identity("A")
        assert graph.connected_components() == [["A"]]

    def test_multiple_isolated_identities(self) -> None:
        """Multiple isolated identities each form their own component."""
        graph = IdentityGraph()
        graph.add_identity("A")
        graph.add_identity("B")
        graph.add_identity("C")
        components = graph.connected_components()
        assert components == [["A"], ["B"], ["C"]]

    def test_adding_same_identity_twice_is_noop(self) -> None:
        """Re-adding an identity does not duplicate it or reset adjacency."""
        graph = IdentityGraph()
        graph.add_identity("A")
        graph.add_morph("A", "B")
        graph.add_identity("A")
        assert graph.identities() == ["A", "B"]
        assert graph.component_of("A") == ["A", "B"]


class TestSimpleEdge:
    """Tests for a single morph relationship."""

    def test_simple_edge_forms_one_component(self) -> None:
        """A single morph edge joins two identities into one component."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        assert graph.connected_components() == [["A", "B"]]

    def test_neighbors_are_symmetric(self) -> None:
        """Morph edges are undirected: neighbors are symmetric."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        assert graph.neighbors("A") == ["B"]
        assert graph.neighbors("B") == ["A"]


class TestChainGraph:
    """Tests for a chain of morph relationships."""

    def test_chain_forms_single_component(self) -> None:
        """A chain A-B-C-D forms a single connected component."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")
        graph.add_morph("C", "D")
        components = graph.connected_components()
        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C", "D"}


class TestStarGraph:
    """Tests for a star-shaped morph relationship."""

    def test_star_forms_single_component(self) -> None:
        """A star graph with a central hub forms one component."""
        graph = IdentityGraph()
        graph.add_morph("HUB", "A")
        graph.add_morph("HUB", "B")
        graph.add_morph("HUB", "C")
        components = graph.connected_components()
        assert len(components) == 1
        assert set(components[0]) == {"HUB", "A", "B", "C"}

    def test_star_hub_neighbors(self) -> None:
        """The hub's neighbor list contains all spokes in insertion order."""
        graph = IdentityGraph()
        graph.add_morph("HUB", "A")
        graph.add_morph("HUB", "B")
        graph.add_morph("HUB", "C")
        assert graph.neighbors("HUB") == ["A", "B", "C"]


class TestDisconnectedGraphs:
    """Tests for graphs containing multiple disjoint components."""

    def test_two_disjoint_pairs(self) -> None:
        """Two separate morph pairs form two independent components."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("C", "D")
        components = graph.connected_components()
        assert len(components) == 2
        assert {"A", "B"} in (set(c) for c in components)
        assert {"C", "D"} in (set(c) for c in components)

    def test_mixed_isolated_and_connected(self) -> None:
        """Isolated identities and connected pairs coexist as separate components."""
        graph = IdentityGraph()
        graph.add_identity("SOLO")
        graph.add_morph("A", "B")
        components = graph.connected_components()
        assert len(components) == 2


class TestCycles:
    """Tests for cyclic morph relationships."""

    def test_triangle_cycle_forms_single_component(self) -> None:
        """A 3-cycle A-B-C-A forms a single connected component."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")
        graph.add_morph("C", "A")
        components = graph.connected_components()
        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C"}

    def test_self_morph_does_not_create_self_loop_edge(self) -> None:
        """Morphing an identity with itself adds no edge, only the node."""
        graph = IdentityGraph()
        graph.add_morph("A", "A")
        assert graph.identities() == ["A"]
        assert graph.neighbors("A") == []


class TestDuplicateEdges:
    """Tests for duplicate morph relationships."""

    def test_duplicate_edge_does_not_duplicate_adjacency(self) -> None:
        """Adding the same morph edge twice does not duplicate neighbors."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("A", "B")
        assert graph.neighbors("A") == ["B"]
        assert graph.neighbors("B") == ["A"]

    def test_reversed_duplicate_edge_does_not_duplicate_adjacency(self) -> None:
        """Adding a morph edge in reversed order does not duplicate it."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "A")
        assert graph.neighbors("A") == ["B"]
        assert graph.neighbors("B") == ["A"]


class TestDeterministicOrdering:
    """Tests verifying deterministic, stable traversal ordering."""

    def test_identities_preserve_insertion_order(self) -> None:
        """Identities are returned in first-seen insertion order."""
        graph = IdentityGraph()
        graph.add_identity("Z")
        graph.add_identity("A")
        graph.add_morph("M", "N")
        assert graph.identities() == ["Z", "A", "M", "N"]

    def test_repeated_runs_produce_identical_components(self) -> None:
        """Running connected_components() repeatedly yields identical output."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("C", "D")
        graph.add_identity("E")
        first = graph.connected_components()
        second = graph.connected_components()
        assert first == second

    def test_component_order_follows_insertion_order(self) -> None:
        """Components are ordered by the insertion order of their first node."""
        graph = IdentityGraph()
        graph.add_morph("B", "A")
        graph.add_identity("C")
        components = graph.connected_components()
        assert components[0][0] == "B"
        assert components[1][0] == "C"


class TestEmptyGraph:
    """Tests for an empty identity graph."""

    def test_empty_graph_has_no_components(self) -> None:
        """An empty graph produces an empty list of components."""
        graph = IdentityGraph()
        assert graph.connected_components() == []

    def test_empty_graph_has_no_identities(self) -> None:
        """An empty graph reports zero identities."""
        graph = IdentityGraph()
        assert graph.identities() == []
        assert len(graph) == 0

    def test_empty_graph_component_lookup_raises(self) -> None:
        """Looking up a component in an empty graph raises an error."""
        graph = IdentityGraph()
        with pytest.raises(UnknownIdentityError):
            graph.component_of("A")


class TestLargeSyntheticGraph:
    """Tests using a larger synthetic graph for scalability sanity checks."""

    def test_large_chain_forms_single_component(self) -> None:
        """A long chain of 1000 identities forms a single component."""
        graph = IdentityGraph()
        n = 1000
        for i in range(n - 1):
            graph.add_morph(f"id_{i}", f"id_{i + 1}")
        components = graph.connected_components()
        assert len(components) == 1
        assert len(components[0]) == n

    def test_large_set_of_disjoint_pairs(self) -> None:
        """500 disjoint morph pairs form 500 separate components."""
        graph = IdentityGraph()
        n_pairs = 500
        for i in range(n_pairs):
            graph.add_morph(f"a_{i}", f"b_{i}")
        components = graph.connected_components()
        assert len(components) == n_pairs
        assert all(len(c) == 2 for c in components)

    def test_large_graph_total_identity_count(self) -> None:
        """Total identities across all components matches graph size."""
        graph = IdentityGraph()
        for i in range(200):
            graph.add_morph(f"x_{i}", f"y_{i}")
            graph.add_identity(f"solo_{i}")
        total_in_components = sum(
            len(c) for c in graph.connected_components()
        )
        assert total_in_components == len(graph)


class TestComponentLookup:
    """Tests for the component_of method."""

    def test_component_of_returns_full_component(self) -> None:
        """component_of returns all identities reachable from the given one."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")
        assert set(graph.component_of("A")) == {"A", "B", "C"}
        assert set(graph.component_of("C")) == {"A", "B", "C"}

    def test_component_of_isolated_identity(self) -> None:
        """component_of an isolated identity returns just that identity."""
        graph = IdentityGraph()
        graph.add_identity("SOLO")
        assert graph.component_of("SOLO") == ["SOLO"]

    def test_component_of_unknown_identity_raises(self) -> None:
        """component_of an unregistered identity raises UnknownIdentityError."""
        graph = IdentityGraph()
        graph.add_identity("A")
        with pytest.raises(UnknownIdentityError):
            graph.component_of("UNKNOWN")

    def test_neighbors_of_unknown_identity_raises(self) -> None:
        """neighbors() on an unregistered identity raises UnknownIdentityError."""
        graph = IdentityGraph()
        with pytest.raises(UnknownIdentityError):
            graph.neighbors("UNKNOWN")


class TestIdentityLeakagePrevention:
    """Critical leakage-prevention tests for transitive morph chains."""

    def test_transitive_morph_chain_forms_single_component(self) -> None:
        """Morph(A,B) and Morph(B,C) must place A, B, and C together.

        This is the core identity-leakage-prevention guarantee: even
        though A and C were never directly morphed, B links them
        transitively, so all three must reside in the same dataset
        split.
        """
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")

        components = graph.connected_components()

        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C"}
        assert set(graph.component_of("A")) == {"A", "B", "C"}
        assert set(graph.component_of("B")) == {"A", "B", "C"}
        assert set(graph.component_of("C")) == {"A", "B", "C"}

    def test_multiple_transitive_chains_stay_isolated_from_each_other(
        self,
    ) -> None:
        """Two separate transitive chains do not merge into one component."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")
        graph.add_morph("X", "Y")
        graph.add_morph("Y", "Z")

        components = graph.connected_components()

        assert len(components) == 2
        component_sets = [set(c) for c in components]
        assert {"A", "B", "C"} in component_sets
        assert {"X", "Y", "Z"} in component_sets

    def test_membership_check_via_component(self) -> None:
        """__contains__ correctly reflects identities added via morphs."""
        graph = IdentityGraph()
        graph.add_morph("A", "B")
        graph.add_morph("B", "C")
        assert "A" in graph
        assert "C" in graph
        assert "D" not in graph