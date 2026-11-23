"""Synthetic summaries for recognized test runners."""

import json

import pytest

from plumbline.recorders.claude_code import record_session
from plumbline.recorders.test_outcome import parse_test_outcome


def _command_for(expected: dict[str, int | str]) -> str:
    name = str(expected["runner"])
    return {"unittest": "python -m unittest", "jest": "npm test"}.get(name, name)


def _record(tmp_path, command: str, stdout: str):
    transcript = tmp_path / "toy.jsonl"
    call = {"type": "tool_use", "id": "call", "name": "Bash", "input": {"command": command}}
    result = {"type": "tool_result", "tool_use_id": "call", "is_error": False, "content": stdout}
    events = [
        {"type": "assistant", "message": {"content": [call]}},
        {"type": "user", "message": {"content": [result]}},
    ]
    transcript.write_text("\n".join(map(json.dumps, events)))
    return record_session(transcript, scrub=False)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "2 failed, 5 passed, 1 skipped in 0.12s",
            {"runner": "pytest", "passed": 5, "failed": 2, "skipped": 1, "errors": 0},
        ),
        (
            "Ran 4 tests in 0.01s\n\nFAILED (failures=1, errors=1)",
            {"runner": "unittest", "passed": 2, "failed": 1, "skipped": 0, "errors": 1},
        ),
        (
            "Tests  1 failed | 3 passed (4)",
            {"runner": "vitest", "passed": 3, "failed": 1, "skipped": 0, "errors": 0},
        ),
        (
            "Tests: 1 failed, 4 passed, 5 total",
            {"runner": "jest", "passed": 4, "failed": 1, "skipped": 0, "errors": 0},
        ),
        (
            "test result: FAILED. 3 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out",
            {"runner": "cargo test", "passed": 3, "failed": 2, "skipped": 1, "errors": 0},
        ),
        (
            "ok example.invalid/package 0.02s",
            {"runner": "go test", "passed_packages": 1, "failed_packages": 0},
        ),
    ],
)
def test_parses_recognizable_runner_summary(output: str, expected: dict[str, int | str]) -> None:
    assert parse_test_outcome(output, command=_command_for(expected)) == expected


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("pytest", "ordinary command output"),
        ("false && pytest toy; true", "false && pytest toy; true"),
        ("pytest toy", "pytest toy completed without a summary"),
        ("jest", "Tests: unknown, 3 total"),
        ("python -m unittest", "Ran 3 tests in 0.01s\n\nFAILED"),
    ],
)
def test_requires_test_runner_result_summary(command: str, output: str) -> None:
    assert parse_test_outcome(output, command=command) is None


def test_footer_must_match_the_runner_in_the_command() -> None:
    assert parse_test_outcome("1 failed in 0.01s", command="jest") is None


def test_runner_command_not_found_invalidates_other_summary_for_that_runner() -> None:
    output = "Tests: 1 passed, 1 total\nsh: jest: command not found"
    assert parse_test_outcome(output, command="jest") is None


def test_quoted_shell_separator_does_not_start_a_runner_command() -> None:
    assert parse_test_outcome("1 passed in 0.01s", command="echo ';' pytest") is None


@pytest.mark.parametrize(
    "command",
    [
        "echo start\npytest toy",
        "false && pytest unit; cat",
        "pytest unit; pytest integration; true",
        "pytest unit && true",
        "false || pytest unit",
        "pytest unit &",
        "pytest unit | cat",
        "pytest unit | head",
        "pytest unit | head -n 2",
        "poetry run pytest toy 2>&1 | head -n 4",
        "pytest unit | tail -1 saved-summary",
        "pytest unit | head < saved-summary",
        "pytest unit | tail -1 | cat",
        "'not an assignment'=x pytest unit",
    ],
)
def test_other_shell_stages_preclude_outcome(tmp_path, command: str) -> None:
    trace = _record(tmp_path, command, "1 passed in 0.01s")
    (call,) = [step for step in trace["steps"] if step["kind"] == "tool_call"]
    assert "test.outcome" not in call["attributes"]


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest",
        "uv run --frozen pytest tests/test_toy.py | tail -1",
        "python -m pytest toy 2>&1",
        "poetry run pytest toy 2>&1 | tail -n 4",
    ],
)
def test_pytest_launchers_record_a_single_filtered_summary(tmp_path, command: str) -> None:
    trace = _record(tmp_path, command, "1 failed in 0.01s")
    (call,) = [step for step in trace["steps"] if step["kind"] == "tool_call"]
    assert call["attributes"]["test.outcome"]["failed"] == 1


def test_go_verbose_counts_observed_tests() -> None:
    outcome = parse_test_outcome(
        "--- PASS: TestToy (0.00s)\n--- FAIL: TestOtherToy (0.00s)\n"
        "FAIL\nFAIL\texample.invalid/toy\t0.01s",
        command="go test ./...",
    )
    assert outcome is not None
    assert (outcome["passed"], outcome["failed"]) == (1, 1)


def test_unattributed_final_failure_is_unknown() -> None:
    assert parse_test_outcome("FAIL", command="pytest") is None


def test_saved_log_output_is_not_attributed_to_a_test_invocation(tmp_path) -> None:
    trace = _record(tmp_path, "cat previous-test.log", "1 failed in 0.01s")
    (call,) = [step for step in trace["steps"] if step["kind"] == "tool_call"]
    assert "test.outcome" not in call["attributes"]


def test_test_summary_requires_a_recognized_runner_command() -> None:
    assert parse_test_outcome("1 failed in 0.01s", command="printf '1 failed in 0.01s'") is None


def test_test_summary_requires_an_explicit_runner_command() -> None:
    assert parse_test_outcome("1 passed in 0.01s") is None


@pytest.mark.parametrize(
    "command",
    [
        "echo pytest",
        "cat pytest",
        "echo 'x; pytest'",
        "printf x # pytest",
        "cat <<EOF\npytest\nEOF",
        r"echo \; pytest",
        "cat toy.log 2>&pytest",
        "TOY=/unused/pytest",
        "</unused/pytest",
    ],
)
def test_runner_names_outside_command_position_do_not_count(command: str) -> None:
    assert parse_test_outcome("1 passed in 0.01s", command=command) is None


@pytest.mark.parametrize(
    ("output", "runner"),
    [
        (
            "Test Suites: 1 failed, 2 passed, 3 total\nTests:       4 passed, 4 total",
            "jest",
        ),
        (
            "Test Files  1 failed | 2 passed (3)\nTests       4 passed (4)",
            "vitest",
        ),
    ],
)
def test_failed_suite_or_file_prevents_clean_runner_result(output: str, runner: str) -> None:
    outcome = parse_test_outcome(output, command=runner)
    assert outcome is not None
    assert outcome["runner"] == runner
    assert isinstance(outcome["errors"], int)
    assert outcome["errors"] > 0


@pytest.mark.parametrize(
    ("output", "runner"),
    [
        ("Test Suites: 1 failed, 1 total", "jest"),
        ("Test Suites: 1 failed, 1 of 2 total\nTests: 1 passed, 1 total", "jest"),
        ("Test Files  1 failed (1)", "vitest"),
    ],
)
def test_suite_failure_is_recorded_independently_of_test_counts(output: str, runner: str) -> None:
    outcome = parse_test_outcome(output, command=runner)
    assert outcome is not None
    assert outcome["errors"] == 1


def test_go_package_headlines_require_go_test_command() -> None:
    outcome = parse_test_outcome("FAIL ./sample.test.js\nTests: 1 passed, 1 total", command="jest")
    assert outcome is not None
    assert outcome["runner"] == "jest"
    assert "failed_packages" not in outcome
    assert parse_test_outcome("ok toy 0.1s", command="go ./test") is None


def test_mixed_go_and_jest_output_is_unattributed() -> None:
    assert (
        parse_test_outcome(
            "ok toy 0.1s\nFAIL ./sample.test.js\nTests: 1 failed, 1 total", command="go test; jest"
        )
        is None
    )


def test_go_aggregate_failure_does_not_invent_a_failed_test_or_package() -> None:
    for output in ("FAIL", "--- PASS: TestToy (0.00s)\nok toy (cached)\nFAIL"):
        outcome = parse_test_outcome(output, command="go test ./...")
        assert outcome is not None
        assert outcome["runner"] == "go test"
        assert outcome["errors"] == 1
        assert outcome.get("failed", 0) == outcome.get("failed_packages", 0) == 0


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "============================= 2 passed in 0.01s =============================",
            {"runner": "pytest", "passed": 2, "failed": 0, "skipped": 0, "errors": 0},
        ),
        (
            "2 passed, 1 warning in 0.01s",
            {"runner": "pytest", "passed": 2, "failed": 0, "skipped": 0, "errors": 0},
        ),
        (
            "\x1b[32m  Test Files  1 failed | 9 passed (10)\x1b[0m\n"
            "      Tests  3 failed | 2 passed | 1 skipped (6)",
            {"runner": "vitest", "passed": 2, "failed": 3, "skipped": 1, "errors": 1},
        ),
        (
            "Ran 3 tests in 0.01s\n\nOK (skipped=1)",
            {"runner": "unittest", "passed": 2, "failed": 0, "skipped": 1, "errors": 0},
        ),
        (
            "Ran 4 tests in 0.01s\n\nFAILED (failures=1, skipped=1)",
            {"runner": "unittest", "passed": 2, "failed": 1, "skipped": 1, "errors": 0},
        ),
        (
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out",
            {"runner": "cargo test", "passed": 3, "failed": 0, "skipped": 0, "errors": 0},
        ),
    ],
)
def test_runner_summary_uses_its_own_counts(output: str, expected: dict[str, int | str]) -> None:
    assert parse_test_outcome(output, command=_command_for(expected)) == expected


def test_summary_counts_come_from_matched_line_only() -> None:
    output = "1 failed, 2 passed in 0.01s\nnoise: 900 passed, 0 failed"
    assert parse_test_outcome(output, command="pytest") == {
        "runner": "pytest",
        "passed": 2,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
    }


def test_pytest_summaries_from_compound_command_are_summed() -> None:
    assert parse_test_outcome(
        "1 failed in 0.01s\n"
        "======================== short test summary ========================\n"
        "3 passed in 0.02s",
        command="pytest",
    ) == {
        "runner": "pytest",
        "passed": 3,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
    }


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "Tests  1 failed | 2 passed (3)\nTests  3 passed (3)",
            {"runner": "vitest", "passed": 5, "failed": 1, "skipped": 0, "errors": 0},
        ),
        (
            "Tests: 1 failed, 2 passed, 3 total\nTests: 3 passed, 3 total",
            {"runner": "jest", "passed": 5, "failed": 1, "skipped": 0, "errors": 0},
        ),
        (
            "Ran 2 tests in 0.01s\n\nFAILED (failures=1)\nRan 3 tests in 0.02s\n\nOK",
            {"runner": "unittest", "passed": 4, "failed": 1, "skipped": 0, "errors": 0},
        ),
    ],
)
def test_compound_runner_summaries_sum_failures_and_passes(
    output: str, expected: dict[str, int | str]
) -> None:
    assert parse_test_outcome(output, command=_command_for(expected)) == expected


@pytest.mark.parametrize("jest_output", ["Tests: 3 passed, 3 total", "sh: jest: command not found"])
def test_several_runners_have_no_attributable_outcome(jest_output: str) -> None:
    output = "1 failed in 0.01s\n" + jest_output
    assert parse_test_outcome(output, command="pytest; jest; true") is None


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "ok example.invalid/fast 0.02s\nok example.invalid/cached (cached)",
            {"runner": "go test", "passed_packages": 2, "failed_packages": 0},
        ),
        (
            "ok example.invalid/fast 0.02s\nFAIL example.invalid/setup [setup failed]\nFAIL",
            {"runner": "go test", "passed_packages": 1, "failed_packages": 1},
        ),
        (
            "ok example.invalid/fast 0.02s\nFAIL",
            {
                "runner": "go test",
                "passed_packages": 1,
                "failed_packages": 0,
                "errors": 1,
            },
        ),
    ],
)
def test_go_aggregate_results_include_cached_passes_and_setup_failures(
    output: str, expected: dict[str, int | str]
) -> None:
    assert parse_test_outcome(output, command="go test ./...") == expected


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            "1 failed, 2 deselected in 0.16s",
            {
                "runner": "pytest",
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "errors": 0,
                "deselected": 2,
            },
        ),
        (
            "3 passed in 120.31s (0:02:00)",
            {"runner": "pytest", "passed": 3, "failed": 0, "skipped": 0, "errors": 0},
        ),
        (
            "1 xfailed, 1 xpassed, 1 warning, 1 error in 0.16s",
            {
                "runner": "pytest",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 1,
                "xfailed": 1,
                "xpassed": 1,
            },
        ),
    ],
)
def test_pytest_extended_summary_counts_parse(summary: str, expected: dict[str, int | str]) -> None:
    assert parse_test_outcome(summary, command="pytest") == expected


def test_malformed_count_line_does_not_become_success() -> None:
    assert parse_test_outcome("summary: 2 passed, 0 failed in 0.01s", command="pytest") is None


def test_recorder_distinguishes_piped_test_results_and_ignores_unrun_command(tmp_path) -> None:
    command = "python -m pytest toy/test_widget.py | tail -1"
    failed = _record(tmp_path, command, "1 failed, 0 passed in 0.01s")
    passed = _record(tmp_path, command, "1 passed in 0.01s")
    unrun = _record(tmp_path, "false && pytest toy; true", "ordinary shell output")

    failed_tool = next(step for step in failed["steps"] if step["kind"] == "tool_call")
    passed_tool = next(step for step in passed["steps"] if step["kind"] == "tool_call")
    unrun_tool = next(step for step in unrun["steps"] if step["kind"] == "tool_call")
    assert failed_tool["status"] == passed_tool["status"] == "ok"
    assert failed_tool["attributes"]["test.outcome"]["failed"] == 1
    assert passed_tool["attributes"]["test.outcome"]["passed"] == 1
    assert "test.outcome" not in unrun_tool["attributes"]
