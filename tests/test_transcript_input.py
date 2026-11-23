"""Synthetic input-format and malformed-record regressions."""

import json
from pathlib import Path

import pytest

from plumbline.cli import main
from plumbline.recorders.claude_code import record_session


def _write(path: Path, records: list[object]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


@pytest.mark.parametrize(
    "record",
    [
        {"type": "response_item", "payload": {"call_id": "call-1"}},
        {"role": "assistant", "message": {"content": [{"type": "tool_use"}]}},
        {"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "exec"}]},
        {
            "type": "metric",
            "sessionId": "synthetic-session",
            "timestamp": "2026-01-01T00:00:00Z",
            "value": 1,
        },
        {"type": []},
    ],
)
def test_cli_rejects_other_jsonl_formats(tmp_path: Path, capsys, record: object) -> None:
    source = _write(tmp_path / "other.jsonl", [record])

    assert main(["record", str(source)]) != 0
    captured = capsys.readouterr()
    assert "unsupported transcript format" in captured.err.lower()
    assert captured.out == ""


def test_cli_rejects_binary_input_with_format_diagnostic(tmp_path: Path, capsys) -> None:
    source = tmp_path / "binary.jsonl"
    source.write_bytes(b"\xff\xfe\x00\x81")

    assert main(["record", str(source)]) != 0
    captured = capsys.readouterr()
    assert "unsupported transcript format" in captured.err.lower()
    assert "UnicodeDecodeError" not in captured.err
    assert captured.out == ""


def test_unrecognized_records_are_bounded_by_sniff_limit(tmp_path: Path) -> None:
    source = tmp_path / "large-other.jsonl"
    unknown = [{"type": "future", "index": i} for i in range(100)]
    valid = {
        "type": "assistant",
        "uuid": "after-limit",
        "message": {"content": "synthetic turn"},
    }
    _write(source, [*unknown, valid])

    with pytest.raises(ValueError, match="unsupported transcript format"):
        record_session(source)


def test_claude_record_within_sniff_limit_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "inside-limit.jsonl"
    unknown = [{"type": "future", "index": i} for i in range(99)]
    valid = {
        "type": "assistant",
        "uuid": "at-limit",
        "message": {"content": "synthetic turn"},
    }
    _write(source, [*unknown, valid])

    trace = record_session(source)

    assert [step["step_id"] for step in trace["steps"] if step["kind"] == "llm"] == ["at-limit"]


def test_malformed_records_do_not_use_sniff_limit(tmp_path: Path) -> None:
    source = tmp_path / "malformed-prefix.jsonl"
    malformed = [{"type": "assistant", "message": {}} for _ in range(101)]
    valid = {
        "type": "assistant",
        "uuid": "valid-turn",
        "message": {"content": [{"type": "text", "text": "done"}]},
    }
    _write(source, [*malformed, valid])

    trace = record_session(source)

    assert [step["step_id"] for step in trace["steps"] if step["kind"] == "llm"] == ["valid-turn"]


@pytest.mark.parametrize(
    ("bad_line", "reason"),
    [
        (b'{"type":"assistant"', "json"),
        (b"\xff", "utf-8"),
        (b"42", "record"),
        (b'{"type":"attachment","uuid":["synthetic"]}', "uuid"),
        (
            b'{"type":"assistant","isSidechain":true,"agentId":[],'
            b'"message":{"id":"m","content":"synthetic"}}',
            "agentId",
        ),
        (b'{"type":"assistant","agentId":null,"message":{"content":"synthetic"}}', "agentId"),
        (b'{"type":"assistant","message":{"id":42,"content":"synthetic"}}', "message.id"),
        (
            b'{"type":"assistant","message":{"content":[{"type":"tool_use",'
            b'"name":["synthetic"]}]}}',
            "tool_use.name",
        ),
        (
            b'{"type":"attachment","attachment":{"type":"hook_blocking_error",'
            b'"toolUseID":["synthetic"]}}',
            "tooluseid",
        ),
        (
            b'{"type":"assistant","message":{"content":[{"type":"tool_use",'
            b'"id":["synthetic"],"name":"Read","input":{}}]}}',
            "tool_use.id",
        ),
        (
            b'{"type":"user","toolUseResult":{"stdout":"1 failed in 0.1s"},'
            b'"message":{"content":[{"type":"tool_result","tool_use_id":["t1"]}]}}',
            "tool_result.tool_use_id",
        ),
        (b'{"type":"system","subtype":"stop_hook_summary","toolUseID":["synthetic"]}', "toolUseID"),
        (b'{"type":"user","timestamp":null,"message":{"content":"bad plan"}}', "timestamp"),
        (b'{"type":"user","timestamp":17,"message":{"content":"bad plan"}}', "timestamp"),
        (
            b'{"type":"assistant","timestamp":17,"message":{"content":"bad outcome",'
            b'"stop_reason":"end_turn"}}',
            "timestamp",
        ),
        (
            b'{"type":"assistant","message":{"content":[{"type":"tool_use",'
            b'"name":"Agent","input":["synthetic"]}]}}',
            "input",
        ),
        (b'{"type":"assistant","message":{"content":[42]}}', "content"),
        (b'{"type":"assistant","message":42}', "message"),
        (b'{"type":"assistant","message":{"content":[]}}', "content"),
        (b'{"type":"assistant","message":{}}', "content"),
        (
            b'{"type":"assistant","parentUuid":[],"message":{"content":"bad parent"}}',
            "parentuuid",
        ),
        (
            b'{"type":"mode","uuid":"bad-mode","parentUuid":[],"mode":"acceptEdits"}',
            "parentuuid",
        ),
        (
            b'{"type":"assistant","message":{"content":[{"type":"text","text":{"bad":true}}]}}',
            "text",
        ),
    ],
)
def test_malformed_lines_preserve_valid_events_and_report_line(
    tmp_path: Path, capsys, bad_line: bytes, reason: str
) -> None:
    source = tmp_path / "mixed.jsonl"
    first = {
        "type": "assistant",
        "uuid": "a1",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest toy"}}
            ]
        },
    }
    result = {
        "type": "user",
        "uuid": "u1",
        "timestamp": "2026-01-01T00:00:01Z",
        "toolUseResult": {"stdout": "complete"},
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    }
    later = {
        "type": "assistant",
        "uuid": "a2",
        "timestamp": "2026-01-01T00:00:02Z",
        "message": {"content": [{"type": "text", "text": "done"}]},
    }
    source.write_bytes(
        b"\n".join(
            [
                json.dumps(first).encode(),
                bad_line,
                json.dumps(result).encode(),
                json.dumps(later).encode(),
            ]
        )
        + b"\n"
    )

    trace = record_session(source)

    tools = [step for step in trace["steps"] if step["kind"] == "tool_call"]
    assert len(tools) == 1
    assert tools[0]["status"] == "ok"
    assert len([step for step in trace["steps"] if step["kind"] == "llm"]) == 2
    assert "plan" not in trace["run"]
    assert trace["run"]["outcome"]["summary"] == "done"
    assert "test.outcome" not in tools[0]["attributes"]
    diagnostics = capsys.readouterr().err.lower()
    assert diagnostics == f"record line 2 skipped: {reason}\n".lower()
    assert "synthetic" not in diagnostics
    assert "complete" not in diagnostics


def test_truncated_final_line_preserves_complete_tool_result(tmp_path: Path, capsys) -> None:
    source = tmp_path / "truncated.jsonl"
    assistant = {
        "type": "assistant",
        "uuid": "call-turn",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {
            "content": [{"type": "tool_use", "id": "read-toy", "name": "Read", "input": {}}]
        },
    }
    result = {
        "type": "user",
        "uuid": "result-turn",
        "timestamp": "2026-01-01T00:00:01Z",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "read-toy"}]},
    }
    source.write_bytes(
        json.dumps(assistant).encode()
        + b"\n"
        + json.dumps(result).encode()
        + b'\n{"type":"assistant","message":'
    )

    trace = record_session(source)

    tool = next(step for step in trace["steps"] if step["kind"] == "tool_call")
    assert tool["status"] == "ok"
    assert "line 3 skipped: json" in capsys.readouterr().err


def test_malformed_optional_sidecar_keeps_main_records(tmp_path: Path, capsys) -> None:
    source = tmp_path / "main.jsonl"
    main_event = {
        "type": "assistant",
        "uuid": "main-turn",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"content": [{"type": "text", "text": "done"}]},
    }
    _write(source, [main_event])
    sidecar = source.with_suffix("") / "subagents" / "agent-toy.jsonl"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b'{"type":"assistant"')

    trace = record_session(source)

    assert [step["step_id"] for step in trace["steps"] if step["kind"] == "llm"] == ["main-turn"]
    assert "line 1 skipped: json" in capsys.readouterr().err
