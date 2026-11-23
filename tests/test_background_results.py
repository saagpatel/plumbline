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
        ("TaskOutput", {"task_id": _ID}, "Process exited with code 0", "", False, "ok"),
        ("TaskOutput", {"task_id": _ID}, {"task": {"status": "completed"}}, "", False, "ok"),
        (
            "TaskOutput",
            {"task_id": _ID},
            {"task": {"status": "completed", "exitCode": 1}},
            "",
            False,
            "error",
        ),
        ("TaskOutput", {"task_id": _ID}, {}, "1 passed in 0.1s", False, None),
        ("TaskOutput", {"task_id": _ID}, {}, "TaskOutput failed unexpectedly", True, None),
        ("TaskOutput", {"task_id": _ID}, {"task": {"status": "failed"}}, "", True, "error"),
        ("TaskOutput", {"task_id": _ID}, {"task_id": _ID, "exitCode": 0}, "", False, "ok"),
        ("TaskOutput", {"task_id": _ID}, {"task_id": "other", "exitCode": 0}, "", False, None),
        ("TaskOutput", {"task_id": _ID}, {}, "Process exited with code 0", True, None),
        ("Read", {"file_path": _PATH}, {}, "1 passed in 0.1s", False, None),
        (
            "Read",
            {"file_path": _PATH},
            {},
            "FAIL example/setup [setup failed]",
            False,
            None,
        ),
        (
            "Read",
            {"file_path": _PATH},
            {},
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out",
            False,
            None,
        ),
        ("Read", {"file_path": _PATH}, {}, "Process exited with code 1", False, None),
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
            {"task": {"exitCode": 0}, "stdout": "4 passed in 0.1s"},
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
        ("Bash", {"command": f"tail -1 '{_PATH}'"}, {}, "1 failed in 0.1s", False, None),
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


@pytest.mark.parametrize(
    "metadata",
    [
        {"task": {"status": "completed", "exitCode": 0}},
        {"task": {"id": "task-b", "exitCode": 0}},
        {"stdout": "Process exited with code 0"},
    ],
)
def test_batched_task_outputs_settle_only_from_their_own_blocks(tmp_path, metadata) -> None:
    records = []
    for task in ("task-a", "task-b"):
        records.extend(
            [
                _call(task, "Bash", {"command": "pytest toy"}),
                _result(task, {"backgroundTaskId": task}, ""),
                _call(f"lookup-{task}", "TaskOutput", {"task_id": task}),
            ]
        )
    blocks = [
        {"type": "tool_result", "tool_use_id": "lookup-task-a", "content": "Exited with code 1"},
        {"type": "tool_result", "tool_use_id": "lookup-task-b", "content": "Still running"},
    ]
    records.append(
        {
            "type": "user",
            "timestamp": _TS,
            "toolUseResult": metadata,
            "message": {"content": blocks},
        }
    )
    path = tmp_path / "batch.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))

    steps = {step["step_id"]: step for step in record_session(path)["steps"]}
    assert steps["task-a"]["status"] == "error"
    assert steps["task-a"]["ended_at"] == _TS
    assert steps["task-b"]["status"] is None
    assert steps["task-b"]["attributes"]["harness.background_task_pending"] is True
    assert "ended_at" not in steps["task-b"]


def test_task_output_started_before_launch_cannot_resolve_it(tmp_path) -> None:
    records = [
        _call("prior", "TaskOutput", {"task_id": _ID}),
        _call("launch", "Bash", {"command": "pytest toy"}),
        _result("launch", {"backgroundTaskId": _ID, "outputPath": _PATH}, ""),
        _result("prior", {"task": {"exitCode": 0}}, "Process exited with code 0"),
    ]
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")
    assert launch["status"] is None
    assert "ended_at" not in launch


def test_intermediate_log_summary_does_not_resolve_launch(tmp_path) -> None:
    records = [
        _call("launch", "Bash", {"command": "go test ./..."}),
        _result(
            "launch",
            {"backgroundTaskId": _ID, "outputPath": _PATH},
            f"Running with ID: {_ID}",
        ),
        _call("read-log", "Read", {"file_path": _PATH}),
        {
            **_result("read-log", {}, "ok example/unit 0.01s"),
            "timestamp": "2026-01-01T00:00:03Z",
        },
        _call("finish", "TaskOutput", {"task_id": _ID}),
        {
            **_result(
                "finish",
                {"task": {"status": "failed", "exitCode": 1}},
                "Process exited with code 1",
            ),
            "timestamp": _TS,
        },
    ]
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))

    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")

    assert launch["status"] == "error"
    assert launch["ended_at"] == _TS
    assert launch["attributes"]["harness.background_task_pending"] is False


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("make check", "1 passed in 0.1s"),
        ("pytest unit; pytest integration", "1 passed in 0.1s"),
        ("pytest unit", "1 passed in 0.1s\nmore work in 0.1s"),
    ],
)
def test_log_output_never_settles_the_launch(tmp_path, command, output) -> None:
    records = [
        _call("launch", "Bash", {"command": command}),
        _result(
            "launch",
            {"backgroundTaskId": _ID, "outputPath": _PATH},
            f"Running with ID: {_ID}",
        ),
        _call("read-log", "Read", {"file_path": _PATH}),
        {
            **_result("read-log", {}, output),
            "timestamp": _TS,
        },
    ]
    path = tmp_path / "compound.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))

    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")

    assert launch["status"] is None
    assert "ended_at" not in launch
    assert launch["attributes"]["harness.background_task_pending"] is True


def test_reader_task_metadata_is_not_authoritative(tmp_path) -> None:
    records = [
        _call("launch", "Bash", {"command": "pytest unit"}),
        _result(
            "launch",
            {"backgroundTaskId": _ID, "outputPath": _PATH},
            f"Running with ID: {_ID}",
        ),
        _call("read-log", "Read", {"file_path": _PATH}),
        {
            **_result("read-log", {"task": {"status": "completed", "exitCode": 0}}, ""),
            "timestamp": _TS,
        },
    ]
    path = tmp_path / "reader-task.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))

    launch = next(s for s in record_session(path)["steps"] if s["step_id"] == "launch")

    assert launch["status"] is None
    assert "ended_at" not in launch
