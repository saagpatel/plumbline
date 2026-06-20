"""Phase 1 recorder behavior, driven test-first against a synthetic CC fixture."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from plumbline.cli import _validate, main
from plumbline.recorders.claude_code import record_session

FIXTURE = Path(__file__).parent / "fixtures" / "cc_session" / "sess.jsonl"
SCHEMA = Path(__file__).parents[1] / "schema" / "plumbline-trace.schema.json"


@pytest.fixture(scope="module")
def trace() -> dict:
    return record_session(FIXTURE)


def steps_of_kind(trace: dict, kind: str) -> list[dict]:
    return [s for s in trace["steps"] if s["kind"] == kind]


def test_run_metadata(trace: dict) -> None:
    run = trace["run"]
    assert run["run_id"] == "sess-1"
    assert run["harness"]["name"] == "claude-code"
    assert run["harness"]["version"] == "2.0.0"
    assert run["model"] == "claude-opus-4-8"
    assert run["plan"]["source"] == "user_prompt"
    assert "rate-limit guard" in run["plan"]["statement"]


def test_llm_steps_have_model_and_usage(trace: dict) -> None:
    llm = steps_of_kind(trace, "llm")
    assert len(llm) >= 5
    first = next(s for s in llm if s["step_id"] == "a1")
    assert first["attributes"]["gen_ai.request.model"] == "claude-opus-4-8"
    assert first["attributes"]["gen_ai.usage.input_tokens"] == 1800
    assert first["attributes"]["agent.reasoning"] is True


def test_tool_call_read(trace: dict) -> None:
    read = next(
        s
        for s in steps_of_kind(trace, "tool_call")
        if s["attributes"]["gen_ai.tool.name"] == "Read" and s["subagent_id"] is None
    )
    assert read["attributes"]["gen_ai.tool.call.id"] == "tu_read1"
    assert read["status"] == "ok"
    assert "/Users/example" not in json.dumps(read["attributes"]["tool.arguments"])


def test_tool_call_blocked_is_error(trace: dict) -> None:
    bash = next(
        s
        for s in steps_of_kind(trace, "tool_call")
        if s["attributes"]["gen_ai.tool.call.id"] == "tu_bash1"
    )
    assert bash["status"] == "interrupted"


def test_agent_and_subagent_merge(trace: dict) -> None:
    agents = steps_of_kind(trace, "agent")
    assert len(agents) == 1
    ag = agents[0]
    assert ag["attributes"]["agent.type"] == "code-reviewer"
    assert ag["attributes"]["agent.spawns_subagent_id"] == "arev1"

    sub_steps = [s for s in trace["steps"] if s["subagent_id"] == "arev1"]
    assert len(sub_steps) == 3
    sub_llm = next(s for s in sub_steps if s["kind"] == "llm")
    assert sub_llm["attribution"]["agent"] == "code-reviewer"


def test_hook_step(trace: dict) -> None:
    hooks = steps_of_kind(trace, "hook")
    assert len(hooks) == 1
    h = hooks[0]["attributes"]
    assert h["harness.hook.name"] == "bash-egress-guard"
    assert h["harness.hook.verdict"] == "deny"
    assert h["harness.hook.prevented_continuation"] is True
    assert h["harness.hook.target_step_id"] == "tu_bash1"


def test_mode_change(trace: dict) -> None:
    tos = [m["attributes"]["harness.mode.to"] for m in steps_of_kind(trace, "mode_change")]
    assert "acceptEdits" in tos


def test_compaction(trace: dict) -> None:
    # Shape verified against real Claude Code transcripts: compaction is a
    # `type:system, subtype:compact_boundary` event whose compactMetadata carries
    # trigger (auto|manual|refusal) + preTokens + postTokens.
    comp = steps_of_kind(trace, "compaction")
    assert len(comp) == 1
    attrs = comp[0]["attributes"]
    assert attrs["harness.compaction.reason"] == "auto"
    assert attrs["harness.compaction.tokens_before"] == 140000
    assert attrs["harness.compaction.tokens_after"] == 38000


def test_no_pii_anywhere(trace: dict) -> None:
    blob = json.dumps(trace)
    assert "/Users/example" not in blob
    assert "dev@example.com" not in blob


def test_output_validates_against_schema(trace: dict) -> None:
    schema = json.loads(SCHEMA.read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(trace),
        key=lambda e: [str(p) for p in e.path],
    )
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def test_unscrubbed_keeps_raw() -> None:
    raw = record_session(FIXTURE, scrub=False)
    assert "/Users/example" in json.dumps(raw)


def test_string_tool_use_result_does_not_crash(tmp_path: Path) -> None:
    # Real CC sometimes emits toolUseResult as a bare string, not an object.
    sess = tmp_path / "s.jsonl"
    lines = [
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": None,
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:00Z",
            "message": {
                "model": "m",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x"}}
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "a1",
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:01Z",
            "toolUseResult": "plain string result",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False}]
            },
        },
    ]
    sess.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    trace = record_session(sess)
    tool_call = next(s for s in trace["steps"] if s["kind"] == "tool_call")
    assert tool_call["status"] == "ok"


def test_cli_validate_returns_nonzero_on_invalid_trace() -> None:
    # Mixed-level errors (run-level + multiple step-level) must not crash the sort.
    bad = {
        "plumbline_version": "0.1.0",
        "run": {"harness": {"name": "x"}, "started_at": "2026-01-01T00:00:00Z"},
        "steps": [
            {
                "step_id": "s1",
                "kind": "decision",
                "started_at": "2026-01-01T00:00:00Z",
                "attributes": {},
            },
            {
                "step_id": "s2",
                "kind": "tool_call",
                "started_at": "2026-01-01T00:00:00Z",
                "attributes": {},
            },
        ],
    }
    assert _validate(bad, str(SCHEMA)) == 1


def test_cli_record_writes_validated_output(tmp_path: Path) -> None:
    out = tmp_path / "trace.json"
    rc = main(["record", str(FIXTURE), "-o", str(out), "--validate", "--schema", str(SCHEMA)])
    assert rc == 0
    text = out.read_text()
    assert json.loads(text)["plumbline_version"] == "0.1.0"
    assert "/Users/example" not in text
