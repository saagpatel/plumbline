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
    source.write_text("\n".join(json.dumps({"type": "future", "index": i}) for i in range(101)))

    with pytest.raises(ValueError, match="unsupported transcript format"):
        record_session(source)


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
        (b'{"type":"assistant","message":{"content":[42]}}', "content"),
        (b'{"type":"assistant","message":42}', "message"),
        (b'{"type":"assistant","message":{"content":[]}}', "content"),
        (b'{"type":"assistant","message":{}}', "content"),
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
        "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
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
    diagnostics = capsys.readouterr().err.lower()
    assert "line 2" in diagnostics
    assert reason in diagnostics
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
