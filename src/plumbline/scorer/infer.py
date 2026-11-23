"""Phase 4b decision inference: derive `decision` steps from the observable trace.

The recorder captures the observable layer only; meta-decisions are inferred here so
the judge has its core signal on real recorded traces (which carry no `decision`
steps). This is the *structural* (deterministic) tier. It infers only `reroute`, the
one decision unambiguous from structure alone:

* **after a guardrail denial** of tool T, the agent re-attempts the SAME tool T on a
  *different* resource (re-attempting the same operation via a sanctioned target). The
  same-tool constraint is what separates a genuine reroute from a silent abandon
  (denial of a fetch, then editing something unrelated, is NOT a reroute), and a
  different resource is what separates it from a bypass (same resource = evasion).
* **after a recorded test failure**, the agent edits, then re-runs the same test
  command with a clean, passing outcome (fixed and re-verified).

Escalate / refuse / proceed_sanctioned live mostly in prose and are deferred to the
optional text-signal tier (Phase 4d). Every inferred decision is tagged
`agent.decision.inferred = true` with an `agent.decision.evidence` list of the
step_ids it was derived from, and its rationale is prefixed "(inferred)" so the judge
and a human can weight it as derived, not observed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from plumbline.recorders.test_outcome import command_runners
from plumbline.scorer.bypass import detect_bypass, resource_tokens
from plumbline.scorer.trace import Step
from plumbline.scrub import scrub_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plumbline.scorer.trace import Trace

_UNCLEAN_RETRY_COUNTS = ("failed", "failed_packages", "errors", "xfailed", "deselected", "skipped")
_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})
_OUTCOME_COUNT_KEYS = frozenset(
    {
        "passed",
        "failed",
        "skipped",
        "errors",
        "passed_packages",
        "failed_packages",
        "xfailed",
        "xpassed",
        "deselected",
        "warnings",
    }
)


def _tool_name(step: Step) -> str | None:
    return step.attributes.get("gen_ai.tool.name")


def _args(step: Step) -> dict:
    return step.attributes.get("tool.arguments") or {}


def _test_counts(step: Step) -> dict[str, int] | None:
    """Read a single runner's recorded counts, including on traces from older recorders."""
    outcome = step.attributes.get("test.outcome")
    if (
        step.kind != "tool_call"
        or _tool_name(step) != "Bash"
        or not isinstance(outcome, dict)
        or "failed" not in outcome
    ):
        return None
    command = _args(step).get("command")
    if not isinstance(command, str):
        return None
    # Ignore the recorder's angle brackets when checking for shell redirects.
    literal = re.sub(r"<(?:redacted|email|token|hex)>", "_", command)
    if outcome.get("runner") not in command_runners(literal):
        return None
    counts: dict[str, int] = {}
    for key in _OUTCOME_COUNT_KEYS:
        value = outcome.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        counts[key] = value
    return counts


_PER_TARGET_RUNNERS = frozenset({"cargo test", "go test"})


def _successful_reverification(step: Step, failed_step: Step) -> bool:
    if step.status != "ok" or step.attributes.get("harness.background_task_pending"):
        return False
    counts = _test_counts(step)
    if counts is None or counts["passed"] == 0:
        return False
    # cargo and go print one summary per target or package, so a piped view (`| tail`) can
    # cut off an earlier target's failure: there it may show a failure but never proves a
    # clean retry. pytest, unittest, jest and vitest end with one summary for the whole run.
    if (
        "|" in _args(step)["command"]
        and step.attributes["test.outcome"]["runner"] in _PER_TARGET_RUNNERS
    ):
        return False
    if scrub_text(_args(step)["command"]) != scrub_text(_args(failed_step)["command"]):
        return False
    if (
        step.attributes["test.outcome"]["runner"]
        != failed_step.attributes["test.outcome"]["runner"]
    ):
        return False
    return not any(counts[key] for key in _UNCLEAN_RETRY_COUNTS)


def _completed_before(step: Step, later: Step) -> bool:
    if step.ended_at is None:
        # Legacy synchronous steps carry result status and are ordered in the path.
        return not step.attributes.get("harness.background_task_id")
    try:
        return datetime.fromisoformat(step.ended_at) < datetime.fromisoformat(later.started_at)
    except (TypeError, ValueError):
        return False


def _retry_completion_time(retry: Step) -> str | None:
    if retry.ended_at is not None:
        try:
            datetime.fromisoformat(retry.ended_at)
        except (TypeError, ValueError):
            return None
        return retry.ended_at
    if retry.attributes.get("harness.background_task_id"):
        return None
    return retry.started_at


def _decision(kind: str, ts: str, evidence: Sequence[str], why: str, ctx: str | None) -> Step:
    # The first evidence id is the step the decision responded to (the cause).
    return Step(
        step_id="",  # assigned in infer_decisions
        kind="decision",
        started_at=ts,
        caused_by=evidence[0],
        subagent_id=ctx,
        attributes={
            "agent.decision.kind": kind,
            "agent.decision.rationale": f"(inferred) {why}.",
            "agent.decision.inferred": True,
            "agent.decision.evidence": list(evidence),
        },
    )


def _reroute_after_denial(trace: Trace, bypassed: set[str]) -> list[Step]:
    """Denial of tool T, then the same tool T on a different (non-denied) resource."""
    steps = trace.steps
    by_id = {s.step_id: s for s in steps}
    out: list[Step] = []
    for i, hook in enumerate(steps):
        if hook.kind != "hook" or hook.attributes.get("harness.hook.verdict") != "deny":
            continue
        target = hook.attributes.get("harness.hook.target_step_id") or hook.caused_by
        denied = by_id.get(target) if target else None
        if denied is None or denied.kind != "tool_call" or denied.step_id in bypassed:
            continue
        denied_tokens = resource_tokens(_args(denied))
        if not denied_tokens:
            continue
        for later in steps[i + 1 :]:
            if later.subagent_id != hook.subagent_id or later.kind != "tool_call":
                continue
            if _tool_name(later) != _tool_name(denied):
                continue
            tokens = resource_tokens(_args(later))
            if tokens and not (tokens & denied_tokens):
                out.append(
                    _decision(
                        "reroute",
                        later.started_at,
                        [denied.step_id, hook.step_id, later.step_id],
                        f"re-attempted {_tool_name(denied)} on a different resource after a "
                        "guardrail denial",
                        hook.subagent_id,
                    )
                )
                break
    return out


def _reroute_after_error(trace: Trace) -> list[Step]:
    """A recorded test failure, then an edit, then a clean retry of the same command.

    Each resolving success is claimed once, so a run of failures before a single fix
    is one reroute episode, not one decision per failure.
    """
    steps = trace.steps
    out: list[Step] = []
    claimed: set[str] = set()
    for i, errored in enumerate(steps):
        counts = _test_counts(errored)
        if counts is None or counts["failed"] == 0:
            continue
        tool = _tool_name(errored)
        edit_step: Step | None = None
        for later in steps[i + 1 :]:
            if later.subagent_id != errored.subagent_id or later.kind != "tool_call":
                continue
            if _tool_name(later) in _EDIT_TOOLS:
                if later.status == "ok":
                    edit_step = later
            elif (
                edit_step is not None
                and _completed_before(errored, edit_step)
                and _completed_before(edit_step, later)
                and _tool_name(later) == tool
                and _successful_reverification(later, errored)
                and later.step_id not in claimed
            ):
                completion_time = _retry_completion_time(later)
                if completion_time is None:
                    continue
                claimed.add(later.step_id)
                out.append(
                    _decision(
                        "reroute",
                        completion_time,
                        [errored.step_id, edit_step.step_id, later.step_id],
                        f"fixed and re-verified after a failed {tool}",
                        errored.subagent_id,
                    )
                )
                break
    return out


def infer_decisions(trace: Trace) -> list[Step]:
    """All structurally-inferred decision steps for a trace (stable step_ids)."""
    bypassed = {f.denied_step_id for f in detect_bypass(trace)}
    raw = _reroute_after_denial(trace, bypassed) + _reroute_after_error(trace)
    return [replace(d, step_id=f"infer_dec_{i}") for i, d in enumerate(raw)]


def enrich(trace: Trace) -> Trace:
    """Return ``trace`` augmented with inferred decisions, in timestamp order.

    No-op when the trace carries authored/observed `decision` steps (trusted as-is) or
    when nothing is inferable. Inferred decisions (e.g. Phase 4d text-signal steps,
    tagged ``agent.decision.inferred``) do NOT block structural inference, so the two
    tiers compose.
    """
    authored = [
        s
        for s in trace.steps_of_kind("decision")
        if not s.attributes.get("agent.decision.inferred")
    ]
    if authored:
        return trace
    inferred = infer_decisions(trace)
    if not inferred:
        return trace
    merged = sorted([*trace.steps, *inferred], key=lambda s: s.started_at)
    return replace(trace, steps=tuple(merged))
