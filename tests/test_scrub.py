"""PII scrubber behavior."""

import json
import shlex

import pytest

from plumbline.recorders.claude_code import record_session
from plumbline.scrub import scrub_obj, scrub_text


def test_scrub_home_path() -> None:
    assert "/Users/alice" not in scrub_text("see /Users/alice/secret.txt")


def test_scrub_email() -> None:
    assert "a@b.com" not in scrub_text("mail a@b.com now")


def test_scrub_long_hex_token() -> None:
    secret = "deadbeef" * 5  # 40 hex chars
    assert secret not in scrub_text(f"token {secret}")


def test_scrub_prefixed_token() -> None:
    assert "sk-ABCD1234EFGH5678IJKL" not in scrub_text("token sk-ABCD1234EFGH5678IJKL")


def test_short_command_preview_scrubs_aws_secret_assignment() -> None:
    command = "AWS_SECRET_ACCESS_KEY=example echo toy"
    assert scrub_text(command) == "AWS_SECRET_ACCESS_KEY=<redacted> echo toy"


def test_scrub_whole_quoted_environment_assignments() -> None:
    for quote, value in [
        ("'", "two words"),
        ('"', "toy dog's word"),
        ('"', r"toy \"quoted\" word"),
        ("'", "synthetic first line\nfake second line"),
    ]:
        command = f"env {quote}DB_PASSWORD={value}{quote} cmd"
        assert scrub_text(command) == f"env {quote}DB_PASSWORD=<redacted>{quote} cmd"


def test_scrub_assignment_truncated_inside_recorded_summary() -> None:
    assert scrub_text('env "DB_PASSWORD=two fake words') == 'env "DB_PASSWORD=<redacted>'


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("'--password=first last'", "'--password=<redacted>'"),
        ('"--token=a b"', '"--token=<redacted>"'),
        (shlex.quote("--token=fake dog's token"), "'--token=<redacted>'"),
        (shlex.quote("DB_PASSWORD=fake dog's password"), "'DB_PASSWORD=<redacted>'"),
    ],
)
def test_scrub_whole_credential_argument(argument: str, expected: str) -> None:
    assert scrub_text(f"toy {argument} --verbose") == f"toy {expected} --verbose"


def test_quoted_environment_assignment_is_scrubbed_from_recorded_trace(tmp_path) -> None:
    path = tmp_path / "synthetic.jsonl"
    command = 'env "DB_PASSWORD=toy dog\'s word" cmd FAKE_TRAILING_MARKER'
    call = {"type": "tool_use", "id": "call", "name": "Bash", "input": {"command": command}}
    path.write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {"type": "user", "message": {"content": "toy"}},
                    {"type": "assistant", "message": {"content": [call]}},
                ],
            )
        )
    )
    trace = record_session(path)
    (step,) = [step for step in trace["steps"] if step["kind"] == "tool_call"]
    assert step["attributes"]["tool.arguments"]["command"] == (
        'env "DB_PASSWORD=<redacted>" cmd FAKE_TRAILING_MARKER'
    )


def test_scrub_keeps_ordinary_identifiers() -> None:
    # Must NOT over-redact: these look prefix-ish but are not secrets.
    text = "skip_empty_lines patient-monitoring pattern_match_score ghost_mode"
    assert scrub_text(text) == text


def test_scrub_obj_is_deep() -> None:
    obj = {"a": ["/Users/bob/x", {"b": "c@d.com"}], "n": 3, "ok": True}
    out = scrub_obj(obj)
    blob = str(out)
    assert "/Users/bob" not in blob
    assert "c@d.com" not in blob
    # non-strings pass through untouched
    assert out["n"] == 3
    assert out["ok"] is True


def test_hook_command_credentials_are_scrubbed_from_default_trace(tmp_path) -> None:
    path = tmp_path / "synthetic.jsonl"
    events = [
        {"type": "user", "message": {"content": "toy"}},
        {
            "type": "system",
            "subtype": "stop_hook_summary",
            "hookInfos": [
                {
                    "name": "toy-hook",
                    "event": "Stop",
                    "outcome": "blocked",
                    "command": "EXAMPLE_PASSWORD=not-a-secret AWS_SECRET_ACCESS_KEY=example toy",
                }
            ],
        },
    ]
    path.write_text("\n".join(map(json.dumps, events)))
    trace = record_session(path)
    assert "not-a-secret" not in json.dumps(trace)
    assert "AWS_SECRET_ACCESS_KEY=example" not in json.dumps(trace)
    (entry,) = trace["steps"][0]["attributes"]["harness.hook.entries"]
    assert entry == {"name": "toy-hook", "event": "Stop", "outcome": "blocked"}
