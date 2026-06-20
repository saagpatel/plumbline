"""End-to-end deterministic scoring: a trace + a reference case -> a scorecard."""

import math
from pathlib import Path

from plumbline.scorer.case import Case, RefNode
from plumbline.scorer.score import score
from plumbline.scorer.trace import Trace

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "example-run.plumbline.json"


def _approx(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def test_case_from_dict_round_trips() -> None:
    case = Case.from_dict(
        {
            "case_id": "c1",
            "description": "ideal rate-limit path",
            "reference_path": [
                {"tool": "Read"},
                {"tool": "Edit", "args": {"file_path": "/repo/config/middleware.py"}},
            ],
        }
    )
    assert case.case_id == "c1"
    assert [r.tool for r in case.reference_path] == ["Read", "Edit"]
    assert case.reference_path[1].args == {"file_path": "/repo/config/middleware.py"}
    assert case.reference_path[0].args is None


def test_perfect_match_scores_one() -> None:
    # Reference == the exact realized main path -> everything perfect.
    t = Trace.from_json_file(EXAMPLE)
    case = Case(
        case_id="exact",
        reference_path=tuple(
            RefNode(tool=name) for name in ["Read", "agent:code-reviewer", "Bash", "Edit", "Bash"]
        ),
    )
    card = score(t, case)
    assert card.selection.f1 == 1.0
    assert card.ordering.f1 == 1.0
    assert _approx(card.edit_similarity, 1.0)
    assert _approx(card.overall, 1.0)


def test_extra_blocked_call_lowers_precision() -> None:
    # The IDEAL path omits the guard-denied curl. The realized path attempted it,
    # so selection precision drops while recall stays perfect.
    t = Trace.from_json_file(EXAMPLE)
    ideal = Case(
        case_id="ideal",
        reference_path=tuple(
            RefNode(tool=name) for name in ["Read", "agent:code-reviewer", "Edit", "Bash"]
        ),
    )
    card = score(t, ideal)
    assert _approx(card.selection.precision, 0.8)  # 4 of 5 realized nodes are wanted
    assert _approx(card.selection.recall, 1.0)  # all 4 wanted nodes happened
    assert card.overall < 1.0


def test_score_can_target_a_subagent_context() -> None:
    t = Trace.from_json_file(EXAMPLE)
    case = Case(case_id="sub", reference_path=(RefNode(tool="Read"),))
    card = score(t, case, subagent_id="agent_rev1")
    assert card.selection.f1 == 1.0


def test_single_step_wrong_tool_is_not_inflated() -> None:
    # A 1-step path has no edges, so ordering is vacuously perfect. That 1.0 must
    # NOT prop up the composite: a single wrong tool should score 0, not 0.33.
    t = Trace.from_json_file(EXAMPLE)
    wrong = Case(case_id="wrong", reference_path=(RefNode(tool="Write"),))
    card = score(t, wrong, subagent_id="agent_rev1")
    assert card.ordering_informative is False
    assert _approx(card.overall, 0.0)
