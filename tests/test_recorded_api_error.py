import json

import pytest

from plumbline.recorders.claude_code import record_session


@pytest.mark.parametrize(
    ("flag", "expected"), [(True, "error"), (False, "ok"), (None, "ok"), ("true", "ok")]
)
def test_api_error_marks_its_own_model_step(tmp_path, flag, expected):
    message = {
        "type": "assistant",
        "sessionId": "fixture",
        "uuid": "first",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "fixture"}]},
    }
    other = {**message, "uuid": "second"}
    if flag is not None:
        message["isApiErrorMessage"] = flag
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(map(json.dumps, [message, other])))
    trace = record_session(path)
    assert [(s["step_id"], s["status"]) for s in trace["steps"] if s["kind"] == "llm"] == [
        ("first", expected),
        ("second", "ok"),
    ]
