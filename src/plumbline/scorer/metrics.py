"""Deterministic decision-path metrics — pure functions over plain sequences.

These reimplement the well-established trajectory primitives (zero-dep) so the
scorer owns their definitions against the Plumbline trace shape:

* ``node_f1`` — *tool selection*: F1 over the multiset of nodes, order-ignored
  (T-eval's Node-F1 / TRAJECT-Bench's selection score).
* ``edge_f1`` — *ordering*: F1 over the multiset of consecutive-pair edges
  (T-eval's Edge-F1). Penalizes a right-tools-wrong-order path.
* ``edit_distance`` / ``edit_similarity`` — Levenshtein over the node sequence,
  a complementary order signal that's robust to a single insertion/deletion.

Inputs are plain sequences of hashable nodes (e.g. tool names, or (name, arg)
tuples), so the math is independent of the trace model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


@dataclass(frozen=True)
class PRF:
    """A precision / recall / F1 triple."""

    precision: float
    recall: float
    f1: float


def _prf(matched: int, pred_total: int, ref_total: int) -> PRF:
    """Build a PRF from match counts.

    Convention: empty-vs-empty is perfect (1,1,1) — nothing expected, nothing
    done. Either side empty while the other is non-empty is (0,0,0): an
    all-wrong / all-missed path. The two asymmetric cases are deliberately
    symmetric (both 0) so a mismatch never scores partial credit on an empty side.
    """
    if pred_total == 0 and ref_total == 0:
        return PRF(1.0, 1.0, 1.0)
    precision = matched / pred_total if pred_total else 0.0
    recall = matched / ref_total if ref_total else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    return PRF(precision, recall, f1)


def _multiset_matched(pred: Sequence[Hashable], ref: Sequence[Hashable]) -> int:
    """Count items present in both, respecting multiplicity (multiset ∩)."""
    return sum((Counter(pred) & Counter(ref)).values())


def node_f1(pred: Sequence[Hashable], ref: Sequence[Hashable]) -> PRF:
    """Tool-selection F1 over the node multiset (order-independent)."""
    return _prf(_multiset_matched(pred, ref), len(pred), len(ref))


def _edges(seq: Sequence[Hashable]) -> list[tuple[Hashable, Hashable]]:
    """Consecutive-pair (bigram) edges of a node sequence."""
    return list(pairwise(seq))


def edge_f1(pred: Sequence[Hashable], ref: Sequence[Hashable]) -> PRF:
    """Ordering F1 over the multiset of consecutive-pair edges."""
    pred_edges, ref_edges = _edges(pred), _edges(ref)
    return _prf(_multiset_matched(pred_edges, ref_edges), len(pred_edges), len(ref_edges))


def edit_distance(pred: Sequence[Hashable], ref: Sequence[Hashable]) -> int:
    """Levenshtein distance over the node sequence (insert/delete/substitute)."""
    m, n = len(pred), len(ref)
    if m == 0 or n == 0:
        return m or n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i, *([0] * n)]
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == ref[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def edit_similarity(pred: Sequence[Hashable], ref: Sequence[Hashable]) -> float:
    """Normalized edit distance in [0, 1]; 1.0 = identical, empty-vs-empty = 1.0."""
    longest = max(len(pred), len(ref))
    if longest == 0:
        return 1.0
    return 1.0 - edit_distance(pred, ref) / longest
