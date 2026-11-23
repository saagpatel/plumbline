"""Record background launches and join only matching terminal output."""

import json

import pytest

from plumbline.recorders.claude_code import record_session

_PATH = "toy/tasks/retry.out"
_ID = "task-toy"
_TS = "2026-01-01T00:00:04Z"


def _call(uid: str, name: str, args: dict) -> dict:
    block = {"type": "tool_use", "id": uid, "name": name, "input": args}
    return {"type": "assistant", "message": {"content": [block]}}


def _result(uid: str, data: dict, text: str, *, error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": uid, "content": text, "is_error": error}
    return {"type": "user", "toolUseResult": data, "message": {"content": [block]}}


@pytest.mark.parametrize(
    "case",
    [
        (None, {}, {}, "", False, None),
        ("TaskOutput", {"task_id": _ID}, {}, "Process exited with code 0", False, "ok"),
        ("Read", {"file_path": _PATH}, {}, "Process exited with code 1", False, "error"),
        ("Read", {"file_path": _PATH + ".old"}, {}, "Process exited with code 0", False, None),
        ("TaskOutput", {"task_id": _ID + "2"}, {}, "Process exited with code 0", False, None),
        (
            "TaskOutput",
            {"task_id": _ID},
            {"task": {"status": "running"}},
            "1 passed in 0.1s",
            False,
            None,
        ),
        (
            "TaskOutput",
            {"task_id": _ID},
            {"task": {"exitCode": 0}},
            "1 failed in 0.1s",
            False,
            "ok",
        ),
        (
            "TaskOutput",
            {"task_id": _ID},
            {"task": {"id": "different", "exitCode": 0}},
            "",
            False,
            None,
        ),
        ("Read", {"file_path": _PATH}, {}, "Process exited with code 1", True, None),
        ("Bash", {"command": f"printf {_ID}"}, {}, "Process exited with code 0", False, None),
        (
            "Bash",
            {"command": f"cat {_PATH} && echo toy"},
            {},
            "Process exited with code 0",
            False,
            None,
        ),
        ("Bash", {"command": f"cat {_PATH} 2"}, {}, "Process exited with code 0", False, None),
        ("Bash", {"command": f"tail -1 '{_PATH}'"}, {}, "1 failed in 0.1s", False, "error"),
    ],
)
def test_only_matching_terminal_output_resolves_launch(tmp_path, case) -> None:
    reader, args, data, text, error, expected = case
    records = [
        _call("launch", "Bash", {"command": "pytest toy"}),
        _result(
            "launch",
            {"backgroundTaskId": _ID},
            f"Running with ID: {_ID}. Output is being written to: {_PATH}",
        ),
    ]
    if reader:
        records.extend(
            [
                _call("lookup", reader, args),
                {**_result("lookup", data, text, error=error), "timestamp": _TS},
            ]
        )
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")
    attrs = launch["attributes"]
    assert attrs["harness.background_task_id"] == _ID
    assert attrs["harness.background_output_path"] == _PATH
    assert attrs["harness.background_task_pending"] is (expected is None)
    assert launch["status"] == expected
    if expected is None:
        assert "ended_at" not in launch
        assert "test.outcome" not in attrs
    else:
        assert launch["ended_at"] == _TS
        if "1 failed" in text:
            assert attrs["test.outcome"]["failed"] == 1


def test_output_read_started_before_launch_cannot_resolve_it(tmp_path) -> None:
    records = [
        _call("prior", "Read", {"file_path": _PATH}),
        _call("launch", "Bash", {"command": "pytest toy"}),
        _result("launch", {"backgroundTaskId": _ID, "outputPath": _PATH}, ""),
        _result("prior", {}, "1 passed in 0.01s"),
    ]
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")
    assert launch["status"] is None
    assert "ended_at" not in launch
