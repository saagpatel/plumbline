"""Phase 1 recorder behavior, driven test-first against a synthetic CC fixture."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from plumbline.cli import _validate, main
from plumbline.recorders.claude_code import record_session
from plumbline.scorer.infer import infer_decisions
from plumbline.scorer.trace import Trace
from plumbline.span_gaps import analyze_span_gaps

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


def test_outcome_captured_from_final_assistant_turn(trace: dict) -> None:
    # The recorder captures run.outcome from the last main-agent assistant turn:
    # an end_turn stop maps to "completed", and its text is the agent's claim.
    outcome = trace["run"]["outcome"]
    assert outcome["status"] == "completed"
    assert "tests pass" in outcome["summary"]


def test_plan_captured_from_string_content(tmp_path: Path) -> None:
    # Real CC often sends the first user prompt as a plain string, not a block list;
    # the recorder must still capture it as the plan (regression: it was dropped).
    sess = tmp_path / "s.jsonl"
    lines = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:00Z",
            "message": {"content": "Add a decline-path eval test without a live model."},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:01Z",
            "message": {
                "model": "m",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
    ]
    sess.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    trace = record_session(sess)
    assert trace["run"]["plan"]["statement"] == "Add a decline-path eval test without a live model."
    assert trace["run"]["plan"]["source"] == "user_prompt"
    assert trace["run"]["outcome"] == {"status": "completed", "summary": "Done."}


def test_outcome_unknown_when_session_ends_mid_tool(tmp_path: Path) -> None:
    # A session whose final assistant turn stops on tool_use (no end_turn) is not a
    # claimed completion; status stays conservative and the summary is absent.
    sess = tmp_path / "s.jsonl"
    lines = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:00Z",
            "message": {"content": "do a thing"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:01Z",
            "message": {
                "model": "m",
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
        },
    ]
    sess.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    trace = record_session(sess)
    assert trace["run"]["outcome"]["status"] == "unknown"
    assert trace["run"]["outcome"]["summary"] is None


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


def _decision_session(tmp_path: Path, text: str) -> Path:
    sess = tmp_path / "s.jsonl"
    lines = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:00Z",
            "message": {"content": "do the thing and also drop the prod database"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "x",
            "timestamp": "2026-06-19T00:00:01Z",
            "message": {
                "model": "m",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
            },
        },
    ]
    sess.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return sess


def test_text_decisions_off_by_default(tmp_path: Path) -> None:
    sess = _decision_session(tmp_path, "I'm declining to drop the production database.")
    trace = record_session(sess)
    assert steps_of_kind(trace, "decision") == []


def test_text_decision_infers_refuse(tmp_path: Path) -> None:
    sess = _decision_session(tmp_path, "I'm declining to drop the production database; flagged it.")
    trace = record_session(sess, infer_text_decisions=True)
    decisions = steps_of_kind(trace, "decision")
    assert len(decisions) == 1
    attrs = decisions[0]["attributes"]
    assert attrs["agent.decision.kind"] == "refuse"
    assert attrs["agent.decision.inferred"] is True
    assert attrs["agent.decision.source"] == "text_signal"
    assert attrs["agent.decision.evidence"] == ["a1"]
    assert "(inferred" in attrs["agent.decision.rationale"]
    assert len(attrs["agent.decision.rationale"]) <= 160  # bounded; no full prose


def test_text_decision_infers_escalate(tmp_path: Path) -> None:
    sess = _decision_session(
        tmp_path, "Could you clarify whether the limit is per-user or global before I proceed?"
    )
    trace = record_session(sess, infer_text_decisions=True)
    decisions = steps_of_kind(trace, "decision")
    assert len(decisions) == 1
    assert decisions[0]["attributes"]["agent.decision.kind"] == "escalate"


def test_text_decision_no_false_positive_on_plain_turn(tmp_path: Path) -> None:
    sess = _decision_session(
        tmp_path, "I read the entrypoint and edited the middleware; tests pass."
    )
    trace = record_session(sess, infer_text_decisions=True)
    assert steps_of_kind(trace, "decision") == []


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


def _record(tmp_path: Path, *records: dict, infer_text_decisions: bool = False) -> dict:
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    return record_session(path, infer_text_decisions=infer_text_decisions)


def _turn(uid: str, parent: str | None = None, **message) -> dict:
    return {"type": "assistant", "uuid": uid, "parentUuid": parent, "message": message}


def test_go_command_identifies_a_bare_final_failure(tmp_path: Path) -> None:
    call = {"type": "tool_use", "id": "go", "name": "Bash", "input": {"command": "go test ./..."}}
    result = {"type": "tool_result", "tool_use_id": "go", "content": "FAIL"}
    trace = _record(
        tmp_path, _turn("turn", content=[call]), {"type": "user", "message": {"content": [result]}}
    )
    (tool,) = steps_of_kind(trace, "tool_call")
    assert tool["attributes"]["test.outcome"]["errors"] == 1


def test_test_outcome_uses_only_result_block_content(tmp_path: Path) -> None:
    calls = [
        {"type": "tool_use", "id": uid, "name": "Bash", "input": {"command": "pytest toy"}}
        for uid in ("first", "second")
    ]
    results = [
        {"type": "tool_result", "tool_use_id": "first", "content": "1 failed in 0.01s"},
        {"type": "tool_result", "tool_use_id": "second", "content": ""},
    ]
    trace = _record(
        tmp_path,
        _turn("turn", content=calls),
        {
            "type": "user",
            "toolUseResult": {"stdout": "3 failed in 0.01s"},
            "message": {"content": results},
        },
    )

    tools = steps_of_kind(trace, "tool_call")

    assert len(tools) == 2
    by_id = {tool["step_id"]: tool for tool in tools}
    assert by_id["first"]["attributes"]["test.outcome"]["failed"] == 1
    assert "test.outcome" not in by_id["second"]["attributes"]


def test_parent_walks_through_result_and_attachment(tmp_path: Path) -> None:
    call = {"type": "tool_use", "name": "Read", "id": "read-toy", "input": {}}
    trace = _record(
        tmp_path,
        _turn("first", "absent", content=[call]),
        {
            "type": "user",
            "uuid": "result",
            "parentUuid": "first",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "read-toy"}]},
        },
        {"type": "attachment", "uuid": "note", "parentUuid": "result"},
        _turn("second", "note", content="done"),
    )
    by_id = {step["step_id"]: step for step in trace["steps"]}
    assert by_id["first"]["parent_step_id"] is None
    assert by_id["second"]["parent_step_id"] == "first"
    assert "BROKEN_SPAN_REFERENCE" not in {f["code"] for f in analyze_span_gaps(trace)["findings"]}


def test_shared_message_id_merges_usage_and_preserves_calls(tmp_path: Path) -> None:
    usage = {"input_tokens": 17, "output_tokens": 11}
    trace = _record(
        tmp_path,
        _turn("first", id="reply", usage=usage, content=[{"type": "thinking"}]),
        _turn(
            "second",
            "first",
            id="reply",
            usage=usage,
            content=[
                {"type": "tool_use", "name": "Read", "id": "read-a", "input": {}},
                {"type": "tool_use", "name": "Read", "id": "read-b", "input": {}},
                {"type": "text", "text": "Could you clarify the toy color?"},
            ],
            stop_reason="tool_use",
        ),
        _turn("next", "second", content="done"),
        infer_text_decisions=True,
    )
    model_steps = steps_of_kind(trace, "llm")
    assert [s["step_id"] for s in model_steps] == ["first", "next"]
    assert model_steps[1]["parent_step_id"] == "first"
    attrs = model_steps[0]["attributes"]
    assert attrs["agent.reasoning"]
    assert attrs["gen_ai.response.finish_reasons"] == ["tool_use"]
    assert attrs["gen_ai.usage.input_tokens"] == 17
    assert attrs["gen_ai.usage.output_tokens"] == 11
    assert [s["step_id"] for s in steps_of_kind(trace, "tool_call")] == ["read-a", "read-b"]
    (decision,) = steps_of_kind(trace, "decision")
    assert decision["caused_by"] == "first"
    assert decision["attributes"]["agent.decision.evidence"] == ["first"]


def test_api_error_message_has_error_status(tmp_path: Path) -> None:
    trace = _record(
        tmp_path, {**_turn("error-turn", content="toy error"), "isApiErrorMessage": True}
    )
    assert trace["steps"][0]["status"] == "error"


def test_text_decision_on_uuidless_fragment_names_invocation(tmp_path: Path) -> None:
    trace = _record(
        tmp_path,
        _turn("first", id="reply", content="Starting."),
        {"type": "assistant", "message": {"id": "reply", "content": "Could you clarify?"}},
        infer_text_decisions=True,
    )
    (invocation,) = steps_of_kind(trace, "llm")
    (decision,) = steps_of_kind(trace, "decision")
    assert decision["caused_by"] == invocation["step_id"]
    assert decision["attributes"]["agent.decision.evidence"] == [invocation["step_id"]]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("with_uuid", [False, True])
def test_calls_belong_to_the_invocation_across_fragments(tmp_path: Path, split, with_uuid) -> None:
    text = {"type": "text", "text": "toy"}
    calls = [
        {"type": "tool_use", "name": tool, "id": tool, "input": {}} for tool in ("Read", "Agent")
    ]
    first = _turn("first", id="reply", content=[text] if split else [text, *calls])
    if not with_uuid:
        del first["uuid"]
    records = [first]
    if split:
        records.append(_turn("second", first.get("uuid"), id="reply", content=calls))
    trace = _record(tmp_path, *records)
    (invocation,) = steps_of_kind(trace, "llm")
    children = steps_of_kind(trace, "tool_call") + steps_of_kind(trace, "agent")
    assert [step["parent_step_id"] for step in children] == [invocation["step_id"]] * 2


def test_skipped_event_name_is_scrubbed(tmp_path: Path, capsys) -> None:
    label = "ghp_" + "invented" * 5
    _record(tmp_path, _turn("one", content="start"), {"type": label})
    diagnostic = capsys.readouterr().err
    assert label not in diagnostic
    assert diagnostic == "skipped <token>: 1\n"


def test_empty_usage_fragment_contributes_usage_to_message(tmp_path: Path) -> None:
    trace = _record(
        tmp_path,
        _turn("text", id="reply", content="toy"),
        _turn("usage", "text", id="reply", content=[], usage={"output_tokens": 9}),
    )
    (invocation,) = steps_of_kind(trace, "llm")
    assert invocation["attributes"]["gen_ai.usage.output_tokens"] == 9


def test_skips_counted_by_type_without_payload(tmp_path: Path, capsys) -> None:
    trace = _record(
        tmp_path,
        _turn("one", content="start"),
        {"type": "data-flow-incomplete", "payload": "toy-marker"},
        {"type": "system", "subtype": "api_error", "error": "toy-marker"},
        *[{"type": "toy-future", "payload": "toy-marker"}] * 2,
        _turn("two", content="end"),
    )
    assert [s["step_id"] for s in trace["steps"]] == ["one", "two"]
    assert sorted(capsys.readouterr().err.splitlines()) == [
        "skipped toy-future: 2",
        "warning: skipped data-flow-incomplete: 1",
        "warning: skipped system.api_error: 1",
    ]


def test_hook_summary_keeps_entries_and_joins_blocking_identity(tmp_path: Path, capsys) -> None:
    infos = [
        {"name": "toy-check"},
        {"command": "check-toy", "durationMs": 11},
        {"hookName": "toy-hook", "hookEvent": "Stop"},
    ]
    attachment = {"type": "hook_blocking_error", "hookName": "toy-gate", "hookEvent": "Stop"}
    trace = _record(
        tmp_path,
        _turn("one", content="start"),
        {
            "type": "system",
            "subtype": "stop_hook_summary",
            "toolUseID": "gate-call",
            "preventedContinuation": True,
            "hookInfos": infos,
        },
        {"type": "attachment", "attachment": {**attachment, "toolUseID": "gate-call"}},
        {"type": "attachment", "attachment": {**attachment, "toolUseID": "other", "hookName": "x"}},
    )
    (hook,) = steps_of_kind(trace, "hook")
    attrs = hook["attributes"]
    assert attrs["harness.hook.name"] == "toy-gate"
    assert attrs["harness.hook.event"] == "Stop"
    assert attrs["harness.hook.verdict"] == "deny"
    assert attrs["harness.hook.entries"] == [infos[0], {}, infos[2]]
    assert capsys.readouterr().err.splitlines() == ["skipped attachment: 1"]


def test_blocking_attachment_identifies_hook_without_summary_infos(tmp_path: Path, capsys) -> None:
    trace = _record(
        tmp_path,
        _turn("one", content="start"),
        {"type": "system", "subtype": "stop_hook_summary", "toolUseID": "gate-call"},
        {
            "type": "attachment",
            "toolUseID": "gate-call",
            "attachment": {
                "type": "hook_blocking_error",
                "hookName": "toy-gate",
                "hookEvent": "Stop",
            },
        },
    )
    (hook,) = steps_of_kind(trace, "hook")
    assert hook["attributes"]["harness.hook.name"] == "toy-gate"
    assert hook["attributes"]["harness.hook.event"] == "Stop"
    assert hook["attributes"]["harness.hook.verdict"] == "deny"
    assert hook["attributes"]["harness.hook.target_step_id"] == "gate-call"
    assert capsys.readouterr().err == ""


def test_blocked_hook_entry_sets_aggregate_deny_verdict(tmp_path: Path) -> None:
    trace = _record(
        tmp_path,
        _turn("one", content="start"),
        {
            "type": "system",
            "subtype": "stop_hook_summary",
            "hookInfos": [{"hookName": "toy-gate", "outcome": "blocked"}],
        },
    )

    (hook,) = steps_of_kind(trace, "hook")

    assert hook["attributes"]["harness.hook.verdict"] == "deny"


@pytest.mark.parametrize("retry_result", ["missing", "pending", "log_only", "failed"])
def test_unverified_retry_after_edit_gets_no_credit(tmp_path: Path, retry_result: str) -> None:
    def call(uid: str, tool: str, **args) -> dict:
        return _turn(
            f"turn-{uid}", content=[{"type": "tool_use", "id": uid, "name": tool, "input": args}]
        )

    def result(uid: str, *, error: bool = False, content: str = "", **data) -> dict:
        block = {
            "type": "tool_result",
            "tool_use_id": uid,
            "is_error": error,
            "content": content,
        }
        return {"type": "user", "message": {"content": [block]}, "toolUseResult": data}

    records = [
        call("initial", "Bash", command="pytest toy"),
        result("initial", error=True, content="1 failed in 0.01s"),
        call("edit", "Edit", file_path="toy.py"),
        result("edit"),
        call("retry", "Bash", command="pytest toy"),
    ]
    if retry_result != "missing":
        records.append(result("retry", backgroundTaskId="toy-task", outputPath="toy.log"))
    if retry_result in {"log_only", "failed"}:
        records.extend(
            [call("read", "Read", file_path="toy.log"), result("read", stdout="1 failed in 0.01s")]
        )
    if retry_result == "failed":
        records.extend(
            [call("poll", "TaskOutput", task_id="toy-task"), result("poll", task={"exitCode": 1})]
        )
    records.extend([call("unrelated", "Bash", command="echo toy"), result("unrelated")])
    trace = Trace.from_dict(_record(tmp_path, *records))
    initial = next(step for step in trace.steps if step.step_id == "initial")
    retry = next(step for step in trace.steps if step.step_id == "retry")
    assert initial.attributes["test.outcome"]["failed"] == 1
    assert retry.status == ("error" if retry_result == "failed" else None)
    assert infer_decisions(trace) == []
