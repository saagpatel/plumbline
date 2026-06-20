"""Score a realized trace against a reference case -> a deterministic scorecard.

This is the deterministic core of the scorer. It compares the realized path
(node names) against the case's reference path along three orthogonal axes:

* **selection** — were the right tools chosen? (``node_f1``)
* **ordering** — in the right order? (``edge_f1``)
* **edit_similarity** — how close is the realized sequence overall? (Levenshtein)

Parameter sub-scores and the calibration judge layer on top of this; they are
not part of the deterministic composite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plumbline.scorer.metrics import PRF, edge_f1, edit_similarity, node_f1
from plumbline.scorer.trace import names

if TYPE_CHECKING:
    from plumbline.scorer.case import Case
    from plumbline.scorer.trace import Trace


@dataclass(frozen=True)
class Scorecard:
    """The deterministic scoring result for one trace against one case.

    ``overall`` is the unweighted mean of the *informative* axes. Ordering is
    informative only when at least one path has an edge (>= 2 nodes); a 1-node
    path's edge-F1 is vacuously 1.0, and counting it would inflate the score of a
    single wrong tool. ``ordering_informative`` records whether it was counted.
    """

    case_id: str
    selection: PRF
    ordering: PRF
    edit_similarity: float
    overall: float
    ordering_informative: bool


def _composite(
    selection: PRF, ordering: PRF, edit_sim: float, *, ordering_informative: bool
) -> float:
    components = [selection.f1, edit_sim]
    if ordering_informative:
        components.append(ordering.f1)
    return sum(components) / len(components)


def score(trace: Trace, case: Case, subagent_id: str | None = None) -> Scorecard:
    """Score a trace's realized path (default: main agent) against a case."""
    pred = names(trace.path(subagent_id))
    ref = [node.tool for node in case.reference_path]
    selection = node_f1(pred, ref)
    ordering = edge_f1(pred, ref)
    edit_sim = edit_similarity(pred, ref)
    ordering_informative = len(pred) >= 2 or len(ref) >= 2  # noqa: PLR2004 - an edge needs 2 nodes
    return Scorecard(
        case_id=case.case_id,
        selection=selection,
        ordering=ordering,
        edit_similarity=edit_sim,
        overall=_composite(
            selection, ordering, edit_sim, ordering_informative=ordering_informative
        ),
        ordering_informative=ordering_informative,
    )
