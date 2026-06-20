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
