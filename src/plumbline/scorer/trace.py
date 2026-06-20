"""A thin, typed read-model over a Plumbline trace dict.

The recorder emits open ``dict`` traces (zero-dep by design). The scorer reads
them through this model so metric code never spelunks ``step["attributes"]["..."]``
by hand. Frozen dataclasses, pure stdlib.

The central output is the **realized path**: the ordered tool-bearing nodes for an
agent context. A node is a ``tool_call`` (named by its tool) or an ``agent``
dispatch (named ``agent:<type>``, since choosing to delegate is itself a tool
selection). Interrupted / errored steps are kept — a guard-denied call is part of
the decision path, and the bypass-pattern gate needs to see it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# Step kinds that constitute a realized action in the decision path.
_PATH_KINDS = frozenset({"tool_call", "agent"})


@dataclass(frozen=True)
class PlanItem:
    """One checklist entry of the plan anchor."""

    id: str
    text: str
    status: str | None = None


@dataclass(frozen=True)
class Plan:
    """The stated intent the decision path is scored against."""

    source: str
    statement: str
    items: tuple[PlanItem, ...] = ()


@dataclass(frozen=True)
class Run:
    """Run-level context plus the on-plan anchor."""

    run_id: str
    harness: str
    model: str | None = None
    plan: Plan | None = None
    outcome_status: str | None = None


@dataclass(frozen=True)
class Step:
    """A single node of the decision DAG (any kind)."""

    step_id: str
    kind: str
    started_at: str
    subagent_id: str | None = None
    parent_step_id: str | None = None
    caused_by: str | None = None
    status: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    attribution: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PathNode:
    """A realized action in the path: a tool call or an agent dispatch."""

    step_id: str
    kind: str
    name: str
    args: Mapping[str, Any]
    status: str | None
    subagent_id: str | None


def _plan(raw: Mapping[str, Any] | None) -> Plan | None:
    if not raw:
        return None
    items = tuple(
        PlanItem(id=item["id"], text=item.get("text", ""), status=item.get("status"))
        for item in (raw.get("items") or [])
    )
    return Plan(source=raw["source"], statement=raw["statement"], items=items)


# Strictness convention: run-level metadata is *best-effort* — it mirrors the
# recorder, which itself emits "unknown" fallbacks when a session lacks an id, so
# a degraded trace still scores. Step *structural* fields (step_id/kind/started_at)
# are required and fail fast: a step missing them is corrupt, not merely degraded.
def _run(raw: Mapping[str, Any]) -> Run:
    outcome = raw.get("outcome") or {}
    return Run(
        run_id=raw.get("run_id", "unknown"),
        harness=(raw.get("harness") or {}).get("name", "unknown"),
        model=raw.get("model"),
        plan=_plan(raw.get("plan")),
        outcome_status=outcome.get("status"),
    )


def _step(raw: Mapping[str, Any]) -> Step:
    return Step(
        step_id=raw["step_id"],
        kind=raw["kind"],
        started_at=raw["started_at"],
        subagent_id=raw.get("subagent_id"),
        parent_step_id=raw.get("parent_step_id"),
        caused_by=raw.get("caused_by"),
        status=raw.get("status"),
        attributes=raw.get("attributes") or {},
        attribution=raw.get("attribution"),
    )


def _path_node(step: Step) -> PathNode:
    if step.kind == "agent":
        name = f"agent:{step.attributes.get('agent.type', 'unknown')}"
        args = {k: v for k, v in step.attributes.items() if k.startswith("agent.")}
    else:  # tool_call
        name = step.attributes.get("gen_ai.tool.name", "unknown")
        args = step.attributes.get("tool.arguments") or {}
    return PathNode(
        step_id=step.step_id,
        kind=step.kind,
        name=name,
        args=args,
        status=step.status,
        subagent_id=step.subagent_id,
    )


@dataclass(frozen=True)
class Trace:
    """A parsed Plumbline trace: run context + the ordered step list."""

    run: Run
    steps: tuple[Step, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trace:
        return cls(
            run=_run(data.get("run") or {}),
            steps=tuple(_step(s) for s in data.get("steps") or []),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> Trace:
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def steps_of_kind(self, kind: str) -> list[Step]:
        """All steps of a given kind, in trace order."""
        return [s for s in self.steps if s.kind == kind]

    def subagent_ids(self) -> list[str]:
        """Distinct subagent context ids present in the trace, in first-seen order."""
        seen: dict[str, None] = {}
        for s in self.steps:
            if s.subagent_id is not None:
                seen.setdefault(s.subagent_id, None)
        return list(seen)

    def path(self, subagent_id: str | None = None) -> list[PathNode]:
        """The realized tool/agent path for one agent context (default: main)."""
        return [
            _path_node(s)
            for s in self.steps
            if s.kind in _PATH_KINDS and s.subagent_id == subagent_id
        ]


def names(path: Sequence[PathNode]) -> list[str]:
    """Project a path to its node names — the input to selection/order metrics."""
    return [node.name for node in path]
