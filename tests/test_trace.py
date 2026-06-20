"""Typed read-model over a Plumbline trace + realized-path extraction.

Tested against the committed worked example so the model tracks the real shape.
"""

from pathlib import Path

from plumbline.scorer.trace import Trace, names

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "example-run.plumbline.json"


def test_loads_run_metadata() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert t.run.run_id == "run_synthetic_0001"
    assert t.run.harness == "claude-code"
    assert t.run.model == "claude-opus-4-8"
    assert t.run.outcome_status == "completed"


def test_loads_plan_anchor() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert t.run.plan is not None
    assert t.run.plan.source == "approved_plan"
    assert t.run.plan.statement.startswith("Add a rate-limit guard")
    assert [item.id for item in t.run.plan.items] == ["p1", "p2", "p3"]


def test_main_path_is_tool_and_agent_steps_in_order() -> None:
    # Main-agent realized path: Read, the code-reviewer dispatch, the (blocked)
    # curl, the Edit, the test run. Subagent-internal steps are excluded.
    t = Trace.from_json_file(EXAMPLE)
    assert [n.name for n in t.path()] == [
        "Read",
        "agent:code-reviewer",
        "Bash",
        "Edit",
        "Bash",
    ]


def test_main_path_excludes_subagent_steps() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert all(n.subagent_id is None for n in t.path())


def test_subagent_path_is_scoped_to_that_context() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert [n.name for n in t.path("agent_rev1")] == ["Read"]


def test_path_node_keeps_interrupted_status_and_args() -> None:
    # The guard-denied curl is part of the decision path — it must survive with
    # its interrupted status so the bypass-pattern gate can see it later.
    t = Trace.from_json_file(EXAMPLE)
    curl = next(n for n in t.path() if "curl" in str(n.args.get("command", "")))
    assert curl.status == "interrupted"
    assert curl.kind == "tool_call"


def test_steps_of_kind_finds_the_hook_deny() -> None:
    t = Trace.from_json_file(EXAMPLE)
    hooks = t.steps_of_kind("hook")
    assert len(hooks) == 1
    assert hooks[0].attributes["harness.hook.verdict"] == "deny"


def test_subagent_ids_enumerates_contexts() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert t.subagent_ids() == ["agent_rev1"]


def test_names_projects_path_to_metric_input() -> None:
    t = Trace.from_json_file(EXAMPLE)
    assert names(t.path("agent_rev1")) == ["Read"]


def test_from_dict_with_no_plan_is_tolerated() -> None:
    t = Trace.from_dict(
        {
            "plumbline_version": "0.1.0",
            "run": {"run_id": "r", "harness": {"name": "x"}, "started_at": "2026-01-01T00:00:00Z"},
            "steps": [],
        }
    )
    assert t.run.plan is None
    assert t.path() == []
