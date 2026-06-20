"""Deterministic path-metric math: node-F1, edge-F1, edit distance.

These operate on plain sequences (tool-name lists / node lists), independent of
the trace model, so they pin the metric *definitions* before any plumbing exists.
"""

import math

from plumbline.scorer.metrics import PRF, edge_f1, edit_distance, edit_similarity, node_f1


def _approx(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


# --- node_f1: tool selection over the node multiset (order-independent) ------


def test_node_f1_identical_is_perfect() -> None:
    s = node_f1(["Read", "Edit", "Bash"], ["Read", "Edit", "Bash"])
    assert s == PRF(1.0, 1.0, 1.0)


def test_node_f1_is_order_independent() -> None:
    assert node_f1(["Bash", "Read", "Edit"], ["Read", "Edit", "Bash"]) == PRF(1.0, 1.0, 1.0)


def test_node_f1_disjoint_is_zero() -> None:
    assert node_f1(["Read"], ["Write"]) == PRF(0.0, 0.0, 0.0)


def test_node_f1_partial_overlap() -> None:
    # pred has 2 of 3 right; ref expects 2 tools. matched=2 (Read, Edit).
    s = node_f1(["Read", "Edit", "Bash"], ["Read", "Edit"])
    assert _approx(s.precision, 2 / 3)
    assert _approx(s.recall, 1.0)
    assert _approx(s.f1, 2 * (2 / 3) / (2 / 3 + 1.0))


def test_node_f1_is_multiset_not_set() -> None:
    # pred fires Read twice; ref expects it once. Only one match counts.
    s = node_f1(["Read", "Read"], ["Read"])
    assert _approx(s.precision, 0.5)
    assert _approx(s.recall, 1.0)


def test_node_f1_both_empty_is_perfect() -> None:
    # Nothing expected, nothing done — vacuously correct.
    assert node_f1([], []) == PRF(1.0, 1.0, 1.0)


def test_node_f1_pred_empty_ref_nonempty_is_zero() -> None:
    assert node_f1([], ["Read"]) == PRF(0.0, 0.0, 0.0)


# --- edge_f1: order via the consecutive-pair (bigram) multiset --------------


def test_edge_f1_identical_is_perfect() -> None:
    assert edge_f1(["A", "B", "C"], ["A", "B", "C"]) == PRF(1.0, 1.0, 1.0)


def test_edge_f1_penalizes_wrong_order() -> None:
    # Same nodes (node_f1 would be 1.0) but reversed order shares no bigram.
    assert node_f1(["A", "B", "C"], ["C", "B", "A"]).f1 == 1.0
    assert edge_f1(["A", "B", "C"], ["C", "B", "A"]).f1 == 0.0


def test_edge_f1_single_node_has_no_edges_and_is_perfect() -> None:
    # A length-1 path has an empty edge set; matching empties is vacuously perfect.
    assert edge_f1(["A"], ["A"]) == PRF(1.0, 1.0, 1.0)


def test_edge_f1_different_single_nodes_is_vacuously_perfect() -> None:
    # No edges exist in either path, so ordering is vacuously perfect even though
    # the tools differ. Selection (node_f1) is what catches the wrong tool here;
    # the composite score must not let this vacuous 1.0 inflate the result.
    assert edge_f1(["A"], ["B"]) == PRF(1.0, 1.0, 1.0)
    assert node_f1(["A"], ["B"]) == PRF(0.0, 0.0, 0.0)


def test_edge_f1_missing_trailing_edge() -> None:
    # pred edges {(A,B)}; ref edges {(A,B),(B,C)}. recall = 1/2.
    s = edge_f1(["A", "B"], ["A", "B", "C"])
    assert _approx(s.precision, 1.0)
    assert _approx(s.recall, 0.5)


# --- edit distance / similarity: Levenshtein over the node sequence ---------


def test_edit_distance_identical_is_zero() -> None:
    assert edit_distance(["A", "B"], ["A", "B"]) == 0


def test_edit_distance_one_substitution() -> None:
    assert edit_distance(["A", "B", "C"], ["A", "X", "C"]) == 1


def test_edit_distance_insertion_and_deletion() -> None:
    assert edit_distance(["A", "C"], ["A", "B", "C"]) == 1
    assert edit_distance(["A", "B", "C"], ["A", "C"]) == 1


def test_edit_distance_both_empty_is_zero() -> None:
    assert edit_distance([], []) == 0


def test_edit_similarity_identical_is_one() -> None:
    assert _approx(edit_similarity(["A", "B"], ["A", "B"]), 1.0)


def test_edit_similarity_fully_different_same_length_is_zero() -> None:
    assert _approx(edit_similarity(["A", "B"], ["X", "Y"]), 0.0)


def test_edit_similarity_both_empty_is_one() -> None:
    assert _approx(edit_similarity([], []), 1.0)
