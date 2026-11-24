import json

from plumbline.recorders.claude_code import record_session


def test_blocking_hook_is_named_by_its_own_attachment(tmp_path):
    path = tmp_path / "fixture.jsonl"
    summary = {
        "type": "system",
        "sessionId": "s",
        "subtype": "stop_hook_summary",
        "uuid": "summary",
        "toolUseID": "call",
        "preventedContinuation": True,
    }

    def attachment(target, name):
        return {
            "type": "attachment",
            "sessionId": "s",
            "attachment": {
                "type": "hook_blocking_error",
                "toolUseID": target,
                "hookName": name,
                "hookEvent": "Stop",
                "command": "unrelated-private-fixture",
            },
        }

    first = attachment("call", "observed")
    for records, expected in (
        ([summary, first, attachment("other", "different")], "observed"),
        ([first, summary], "observed"),
        ([summary, first, attachment("call", "ambiguous")], "unknown"),
        ([summary, {**first, "sessionId": "other"}], "unknown"),
        ([summary, {**first, "agentId": "other"}], "unknown"),
    ):
        path.write_text("\n".join(map(json.dumps, records)))
        trace = record_session(path)
        (attrs,) = [s["attributes"] for s in trace["steps"] if s["kind"] == "hook"]
        assert attrs["harness.hook.name"] == expected
        assert attrs["harness.hook.verdict"] == "deny"
        assert "unrelated-private-fixture" not in json.dumps(trace)

    without_prevention = {
        key: value for key, value in summary.items() if key != "preventedContinuation"
    }
    path.write_text("\n".join(map(json.dumps, [without_prevention, first])))
    trace = record_session(path)
    (hook,) = [step for step in trace["steps"] if step["kind"] == "hook"]
    assert hook["attributes"]["harness.hook.verdict"] == "deny"
    assert "harness.hook.prevented_continuation" not in hook["attributes"]
