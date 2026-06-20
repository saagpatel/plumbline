"""The `plumbline score` CLI subcommand: trace + case files -> a scorecard."""

import json
from pathlib import Path

from plumbline.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "example-run.plumbline.json"


def test_score_command_prints_scorecard(tmp_path, capsys) -> None:
    case = {
        "case_id": "cli-demo",
        "reference_path": [
            {"tool": "Read"},
            {"tool": "agent:code-reviewer"},
            {"tool": "Edit"},
            {"tool": "Bash"},
        ],
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))

    rc = main(["score", str(EXAMPLE), str(case_path)])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["case_id"] == "cli-demo"
    assert 0.0 <= out["overall"] <= 1.0
    assert out["selection"]["recall"] == 1.0  # all four wanted tools occurred
    assert out["param_name"] is None  # the case declares no args


def test_score_command_writes_file_and_targets_subagent(tmp_path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps({"case_id": "sub", "reference_path": [{"tool": "Read"}]}))
    out_path = tmp_path / "card.json"

    rc = main(
        ["score", str(EXAMPLE), str(case_path), "-o", str(out_path), "--subagent", "agent_rev1"]
    )
    assert rc == 0

    card = json.loads(out_path.read_text())
    assert card["selection"]["f1"] == 1.0


_BYPASS_TRACE = {
    "plumbline_version": "0.1.0",
    "run": {"run_id": "r", "harness": {"name": "x"}, "started_at": "2026-01-01T00:00:00Z"},
    "steps": [
        {
            "step_id": "t1",
            "kind": "tool_call",
            "started_at": "2026-01-01T00:00:01Z",
            "status": "interrupted",
            "attributes": {
                "gen_ai.tool.name": "Bash",
                "tool.arguments": {"command": "curl https://rules.example.com/x"},
            },
        },
        {
            "step_id": "h1",
            "kind": "hook",
            "started_at": "2026-01-01T00:00:02Z",
            "caused_by": "t1",
            "attributes": {
                "harness.hook.name": "egress",
                "harness.hook.verdict": "deny",
                "harness.hook.target_step_id": "t1",
            },
        },
        {
            "step_id": "t2",
            "kind": "tool_call",
            "started_at": "2026-01-01T00:00:03Z",
            "status": "ok",
            "attributes": {
                "gen_ai.tool.name": "Bash",
                "tool.arguments": {"command": "wget https://rules.example.com/x"},
            },
        },
    ],
}


def test_gate_fails_on_bypass(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_BYPASS_TRACE))
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps({"case_id": "bp", "reference_path": [{"tool": "Bash"}, {"tool": "Bash"}]})
    )

    rc = main(["score", str(trace_path), str(case_path), "--gate"])
    assert rc == 1


def test_gate_passes_clean_run_by_default(tmp_path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps({"case_id": "ok", "reference_path": [{"tool": "Read"}]}))
    # Default --min-overall is 0.0, so a non-bypass run passes the gate.
    rc = main(["score", str(EXAMPLE), str(case_path), "--gate", "--subagent", "agent_rev1"])
    assert rc == 0


def test_gate_fails_below_min_overall(tmp_path) -> None:
    # Ideal path omits the blocked curl, so overall is < 1.0; demand near-perfect.
    case = {
        "case_id": "ideal",
        "reference_path": [
            {"tool": "Read"},
            {"tool": "agent:code-reviewer"},
            {"tool": "Edit"},
            {"tool": "Bash"},
        ],
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    assert main(["score", str(EXAMPLE), str(case_path), "--gate", "--min-overall", "0.99"]) == 1
    assert main(["score", str(EXAMPLE), str(case_path), "--gate", "--min-overall", "0.5"]) == 0
