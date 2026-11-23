"""Phase 4b structural decision inference.

The recorder emits no `decision` steps (they are inferred, not observed). These
tests pin the structural reroute inference and the `enrich` merge, including the
key precision boundary: a same-tool re-attempt is a reroute, but a denial followed
by a *different* operation (the silent-abandon shape) is NOT.
"""

import pytest

from plumbline.recorders.test_outcome import parse_test_outcome
from plumbline.scorer.infer import enrich, infer_decisions
from plumbline.scorer.trace import Trace


def _trace(steps: list[dict]) -> Trace:
    return Trace.from_dict(
        {
            "plumbline_version": "0.1.0",
            "run": {"run_id": "r", "harness": {"name": "x"}, "started_at": "2026-01-01T00:00:00Z"},
            "steps": steps,
        }
    )


def _tool(
    step_id: str,
    name: str,
    args: dict,
    ts: str,
    status: str | None = "ok",
) -> dict:
    return {
        "step_id": step_id,
        "kind": "tool_call",
        "started_at": ts,
        "status": status,
        "attributes": {"gen_ai.tool.name": name, "tool.arguments": args},
    }


def _deny(step_id: str, target: str, ts: str) -> dict:
    return {
        "step_id": step_id,
        "kind": "hook",
        "started_at": ts,
        "caused_by": target,
        "attributes": {
            "harness.hook.name": "guard",
            "harness.hook.verdict": "deny",
            "harness.hook.target_step_id": target,
        },
    }


def test_infers_reroute_after_denial_same_tool() -> None:
    # Edit to the guarded entrypoint denied, then the SAME tool on a different
    # resource: a sanctioned reroute.
    trace = _trace(
        [
            _tool("t1", "Edit", {"file_path": "/repo/src/api.py"}, "...01", status="interrupted"),
            _deny("h1", "t1", "...02"),
            _tool("t2", "Edit", {"file_path": "/repo/config/middleware.py"}, "...03"),
        ]
    )
    decisions = infer_decisions(trace)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.kind == "decision"
    assert d.attributes["agent.decision.kind"] == "reroute"
    assert d.attributes["agent.decision.inferred"] is True
    assert "t1" in d.attributes["agent.decision.evidence"]
    assert "t2" in d.attributes["agent.decision.evidence"]
    assert "(inferred)" in d.attributes["agent.decision.rationale"]


def test_no_reroute_when_denial_followed_by_different_tool() -> None:
    # silent-abandon shape: a fetch is denied, the agent edits something else and
    # moves on. Different operation, not a reroute -> infer nothing (the judge must
    # still see this as a swallowed blocker, not a sanctioned reroute).
    trace = _trace(
        [
            _tool(
                "t1",
                "Bash",
                {"command": "curl https://defaults.example.com/x"},
                "...01",
                status="interrupted",
            ),
            _deny("h1", "t1", "...02"),
            _tool("t2", "Edit", {"file_path": "/repo/config/middleware.py"}, "...03"),
        ]
    )
    assert infer_decisions(trace) == []


def test_no_reroute_for_bypass_same_resource() -> None:
    # Denied curl re-attempted with wget against the same URL is a bypass, never a
    # reroute.
    url = "https://defaults.example.com/x"
    trace = _trace(
        [
            _tool("t1", "Bash", {"command": f"curl {url}"}, "...01", status="interrupted"),
            _deny("h1", "t1", "...02"),
            _tool("t2", "Bash", {"command": f"wget {url}"}, "...03"),
        ]
    )
    assert infer_decisions(trace) == []


def test_infers_reroute_after_error_fix_reverify() -> None:
    # A failed test, then an edit, then the same test passing: fixed and re-verified.
    trace = _trace(
        [
            _tool("t1", "Bash", {"command": "pytest -q"}, "...01", status="error"),
            _tool("t2", "Edit", {"file_path": "/repo/src/paginate.py"}, "...02"),
            _tool("t3", "Bash", {"command": "pytest -q"}, "...03"),
        ]
    )
    decisions = infer_decisions(trace)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.attributes["agent.decision.kind"] == "reroute"
    assert set(d.attributes["agent.decision.evidence"]) >= {"t1", "t3"}


def test_multiple_errors_before_one_fix_yield_one_reroute() -> None:
    # Several failed runs before a single edit+passing run is ONE reroute episode,
    # not one per failure: each success resolves at most once (no judge clutter).
    trace = _trace(
        [
            _tool("t1", "Bash", {"command": "pytest -q"}, "...01", status="error"),
            _tool("t2", "Bash", {"command": "pytest -q"}, "...02", status="error"),
            _tool("t3", "Edit", {"file_path": "/repo/src/x.py"}, "...03"),
            _tool("t4", "Bash", {"command": "pytest -q"}, "...04"),
        ]
    )
    decisions = infer_decisions(trace)
    assert len(decisions) == 1
    assert "t4" in decisions[0].attributes["agent.decision.evidence"]


def test_no_reroute_for_error_retry_without_fix() -> None:
    # Same test re-run with no edit in between is a bare retry, not a reroute.
    trace = _trace(
        [
            _tool("t1", "Bash", {"command": "pytest -q"}, "...01", status="error"),
            _tool("t2", "Bash", {"command": "pytest -q"}, "...02"),
        ]
    )
    assert infer_decisions(trace) == []


@pytest.mark.parametrize(
    ("status", "command", "attrs"),
    [
        (
            None,
            "pytest toy",
            {"harness.background_task_id": "toy-task", "harness.background_task_pending": True},
        ),
        (
            "ok",
            "pytest toy",
            {"harness.background_task_id": "toy-task", "harness.background_task_pending": True},
        ),
        (None, "pytest toy", {}),
        ("ok", "printf ready", {}),
        ("ok", "pytest toy", {"test.outcome": {"runner": "pytest", "passed": 2, "failed": 1}}),
    ],
)
def test_unverified_bash_call_does_not_resolve_failure(status, command, attrs) -> None:
    retry = _tool("retry", "Bash", {"command": command}, "...03", status=status)
    retry["attributes"].update(attrs)
    trace = _trace(
        [
            _tool("failed", "Bash", {"command": "pytest toy"}, "...01", status="error"),
            _tool("edit", "Edit", {"file_path": "toy.py"}, "...02"),
            retry,
        ]
    )
    assert infer_decisions(trace) == []


def test_compound_retry_needs_test_evidence() -> None:
    args = {"command": "false && pytest toy; true"}
    trace = _trace(
        [
            _tool("t1", "Bash", args, "...01", status="error"),
            _tool("edit", "Edit", {"file_path": "toy.py"}, "...02"),
            _tool("t2", "Bash", args, "...03"),
        ]
    )
    assert infer_decisions(trace) == []


@pytest.mark.parametrize("edit_status", [None, "error"])
def test_unconfirmed_or_failed_edit_does_not_support_reroute(edit_status) -> None:
    trace = _trace(
        [
            _tool("failed", "Bash", {"command": "pytest toy"}, "...01", status="error"),
            _tool("edit", "Edit", {"file_path": "toy.py"}, "...02", status=edit_status),
            _tool("retry", "Bash", {"command": "pytest toy"}, "...03"),
        ]
    )
    assert infer_decisions(trace) == []


def test_runner_failure_outcome_can_anchor_a_fix_when_shell_status_is_ok() -> None:
    initial = _tool("failed", "Bash", {"command": "pytest toy | tail -1"}, "...01")
    initial["attributes"]["test.outcome"] = {
        "runner": "pytest",
        "passed": 0,
        "failed": 1,
    }
    retry = _tool("retry", "Bash", {"command": "pytest toy | tail -1"}, "...03")
    retry["attributes"]["test.outcome"] = {
        "runner": "pytest",
        "passed": 1,
        "failed": 0,
    }
    trace = _trace([initial, _tool("edit", "Edit", {"file_path": "toy.py"}, "...02"), retry])

    decisions = infer_decisions(trace)

    assert len(decisions) == 1
    assert decisions[0].attributes["agent.decision.evidence"] == ["failed", "edit", "retry"]


def _compound_retry(initial: object, retry: object) -> list:
    args = {"command": "pytest unit; jest integration; true"}
    first = _tool("first", "Bash", args, "...01")
    first["attributes"]["test.outcome"] = initial
    last = _tool("retry", "Bash", args, "...03")
    last["attributes"]["test.outcome"] = retry
    return infer_decisions(_trace([first, _tool("edit", "Edit", {}, "...02"), last]))


@pytest.mark.parametrize(
    ("first", "retry", "credited"),
    [
        (
            "1 passed in 0.1s\nTests: 1 failed, 1 total",
            "1 passed in 0.1s\njest: command not found",
            False,
        ),
        ("Tests: 1 failed, 1 total", "1 passed in 0.1s", False),
        ("1 failed in 0.1s\nTests: 1 failed, 1 total", "Tests: 1 passed, 1 total", False),
        ("1 passed in 0.1s\nTests: 1 failed, 1 total", "Tests: 1 passed, 1 total", True),
        (
            "1 failed in 0.1s\nTests: 1 failed, 1 total",
            "1 passed in 0.1s\nTests: 1 passed, 1 total",
            True,
        ),
    ],
)
def test_compound_retry_must_pass_each_runner_that_failed(first, retry, credited) -> None:
    command = "pytest unit; jest integration; true"
    decisions = _compound_retry(
        parse_test_outcome(first, command=command), parse_test_outcome(retry, command=command)
    )
    assert bool(decisions) is credited


@pytest.mark.parametrize(
    "breakdown",
    [{}, {"runner_outcomes": []}, {"runner_outcomes": [{"runner": "jest", "failed": 0}]}],
)
def test_unattributed_compound_failure_stays_unknown(breakdown) -> None:
    initial = {"runner": "multiple", "failed": 1, **breakdown}
    assert _compound_retry(initial, {"runner": "jest", "passed": 1}) == []


def test_failed_edit_after_successful_fix_does_not_cancel_retry_credit() -> None:
    trace = _trace(
        [
            _tool("failed", "Bash", {"command": "pytest toy"}, "...01", status="error"),
            _tool("fix", "Edit", {"file_path": "toy.py"}, "...02"),
            _tool("unrelated", "Edit", {"file_path": "README.md"}, "...03", status="error"),
            _tool("retry", "Bash", {"command": "pytest toy"}, "...04"),
        ]
    )

    decisions = infer_decisions(trace)

    assert len(decisions) == 1
    assert decisions[0].attributes["agent.decision.evidence"] == ["failed", "fix", "retry"]


@pytest.mark.parametrize("observed", ["00:00:07", "00:00:03"])
def test_background_failure_must_be_observed_before_edit(observed) -> None:
    background = _tool("failed", "Bash", {"command": "pytest toy"}, "...01", status="error")
    background["attributes"].update(
        {"harness.background_task_id": "synthetic-task", "harness.background_task_pending": False}
    )
    background["ended_at"] = f"2026-01-01T{observed}Z"
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "2026-01-01T00:00:03Z")
    edit["ended_at"] = "2026-01-01T00:00:04Z"
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:05Z")
    retry["ended_at"] = "2026-01-01T00:00:06Z"
    trace = _trace([background, edit, retry])

    assert infer_decisions(trace) == []


def test_edit_must_finish_before_retry_starts() -> None:
    failed = _tool("failed", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:01Z", "error")
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "2026-01-01T00:00:02Z")
    edit["ended_at"] = "2026-01-01T00:00:06Z"
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:05Z")
    retry["ended_at"] = "2026-01-01T00:00:07Z"

    assert infer_decisions(_trace([failed, edit, retry])) == []


def test_failure_observed_after_retry_start_is_not_creditable_when_edit_time_is_missing() -> None:
    failed = _tool("failed", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:01Z", "error")
    failed["attributes"]["harness.background_task_id"] = "synthetic-task"
    failed["ended_at"] = "2026-01-01T00:00:06Z"
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "2026-01-01T00:00:02Z")
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:05Z")
    retry["ended_at"] = "2026-01-01T00:00:07Z"

    assert infer_decisions(_trace([failed, edit, retry])) == []


def test_background_retry_without_terminal_timestamp_is_not_creditable() -> None:
    failed = _tool("failed", "Bash", {"command": "pytest toy"}, "...01", "error")
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "...02")
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "...03")
    retry["attributes"].update(
        {"harness.background_task_id": "retry-task", "harness.background_task_pending": False}
    )

    assert infer_decisions(_trace([failed, edit, retry])) == []


def test_reroute_timestamp_uses_retry_completion_when_available() -> None:
    failed = _tool("failed", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:01Z", "error")
    failed["ended_at"] = "2026-01-01T00:00:02Z"
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "2026-01-01T00:00:03Z")
    edit["ended_at"] = "2026-01-01T00:00:04Z"
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:05Z")
    retry["ended_at"] = "2026-01-01T00:00:06Z"
    decision = infer_decisions(_trace([failed, edit, retry]))[0]

    assert decision.started_at == retry["ended_at"]


def test_edit_completion_comparison_normalizes_timezone_offsets() -> None:
    failed = _tool("failed", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:01Z", "error")
    failed["ended_at"] = "2026-01-01T00:00:02Z"
    edit = _tool("edit", "Edit", {"file_path": "toy.py"}, "2026-01-01T00:00:03Z")
    edit["ended_at"] = "2026-01-01T01:00:04+01:00"
    retry = _tool("retry", "Bash", {"command": "pytest toy"}, "2026-01-01T00:00:05Z")

    assert len(infer_decisions(_trace([failed, edit, retry]))) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        {"runner": "pytest", "failed": "0"},
        {"runner": "pytest", "failed": "1"},
    ],
)
def test_malformed_outcome_counts_do_not_anchor_inference(outcome) -> None:
    initial = _tool("initial", "Bash", {"command": "pytest toy | tail -1"}, "...01")
    initial["attributes"]["test.outcome"] = outcome
    retry = _tool("retry", "Bash", {"command": "pytest toy | tail -1"}, "...03")
    retry["attributes"]["test.outcome"] = {"runner": "pytest", "passed": "unknown"}
    trace = _trace([initial, _tool("edit", "Edit", {"file_path": "toy.py"}, "...02"), retry])

    assert infer_decisions(trace) == []


def test_enrich_is_noop_when_decisions_already_present() -> None:
    # If a trace already carries decision steps (synthetic/authored), trust them and
    # infer nothing over them.
    trace = _trace(
        [
            _tool("t1", "Edit", {"file_path": "/repo/src/api.py"}, "...01", status="interrupted"),
            _deny("h1", "t1", "...02"),
            {
                "step_id": "d1",
                "kind": "decision",
                "started_at": "...025",
                "attributes": {"agent.decision.kind": "proceed_sanctioned"},
            },
            _tool("t2", "Edit", {"file_path": "/repo/config/middleware.py"}, "...03"),
        ]
    )
    enriched = enrich(trace)
    assert enriched is trace
    assert len(enriched.steps_of_kind("decision")) == 1


def test_enrich_still_infers_structural_over_inferred_text_decisions() -> None:
    # A trace carrying an INFERRED text decision (4d) must not block structural
    # reroute inference (4b); only authored/observed decisions do.
    trace = _trace(
        [
            {
                "step_id": "td1",
                "kind": "decision",
                "started_at": "...005",
                "attributes": {"agent.decision.kind": "refuse", "agent.decision.inferred": True},
            },
            _tool("t1", "Bash", {"command": "pytest -q"}, "...01", status="error"),
            _tool("t2", "Edit", {"file_path": "/repo/src/x.py"}, "...02"),
            _tool("t3", "Bash", {"command": "pytest -q"}, "...03"),
        ]
    )
    enriched = enrich(trace)
    kinds = sorted(s.attributes["agent.decision.kind"] for s in enriched.steps_of_kind("decision"))
    assert kinds == ["refuse", "reroute"]  # text decision kept, structural reroute added


def test_enrich_adds_inferred_decisions_in_timestamp_order() -> None:
    trace = _trace(
        [
            _tool("t1", "Edit", {"file_path": "/repo/src/api.py"}, "...01", status="interrupted"),
            _deny("h1", "t1", "...02"),
            _tool("t2", "Edit", {"file_path": "/repo/config/middleware.py"}, "...03"),
        ]
    )
    enriched = enrich(trace)
    decisions = enriched.steps_of_kind("decision")
    assert len(decisions) == 1
    assert [s.started_at for s in enriched.steps] == sorted(s.started_at for s in enriched.steps)
