"""Track background tool calls until their terminal output is read."""

from __future__ import annotations

import re
import shlex
from typing import Any

from plumbline.recorders.test_outcome import parse_test_outcome

_EXIT_CODE = re.compile(r"^(?:process )?exited with code (-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_OUTPUT_PATH = re.compile(r"\bOutput is being written to:[ \t]*([^\r\n]+)")
_UNITTEST_FINAL = re.compile(r"(?:OK|FAILED)(?: \([^)]+\))?")


def mark_pending(step: dict[str, Any], result: dict[str, Any], ack_text: str = "") -> bool:
    """Mark a launched background call as unresolved and return whether it matched."""
    task_id = result.get("backgroundTaskId")
    if not isinstance(task_id, str) or not task_id:
        return False

    attrs = step["attributes"]
    attrs["harness.background_task_id"] = task_id
    attrs["harness.background_task_pending"] = True
    output_path = next(
        (
            result[key]
            for key in ("outputPath", "output_path", "outputFile", "output_file")
            if isinstance(result.get(key), str) and result[key]
        ),
        None,
    )
    if output_path is None and (match := _OUTPUT_PATH.search(ack_text)):
        output_path = match.group(1).strip()
    if output_path:
        attrs["harness.background_output_path"] = output_path
    step["status"] = None
    step.pop("ended_at", None)
    return True


def resolve_from_output(
    launch: dict[str, Any], reader: dict[str, Any], result: dict[str, Any], output: str
) -> bool:
    attrs = launch["attributes"]
    if reader["subagent_id"] != launch["subagent_id"] or not _reads_task_output(attrs, reader):
        return False
    is_task_output = reader["attributes"].get("gen_ai.tool.name") == "TaskOutput"
    task_result = result.get("task") if is_task_output else None
    task = task_result if isinstance(task_result, dict) else {}
    reported_id = task.get("task_id") or task.get("id")
    if reported_id and reported_id != attrs["harness.background_task_id"]:
        return False
    if task.get("status") in {"running", "pending", "queued"}:
        return False
    arguments = attrs.get("tool.arguments")
    command = arguments.get("command") if isinstance(arguments, dict) else None
    outcome = parse_test_outcome(output, command=command)
    status = _terminal_status(task, outcome, output, command)
    if status is None:
        return False
    if outcome:
        attrs["test.outcome"] = outcome
    launch["status"] = status
    launch["ended_at"] = reader["ended_at"]
    attrs["harness.background_task_pending"] = False
    return True


def _reads_task_output(pending: dict[str, Any], reader: dict[str, Any]) -> bool:
    attrs = reader["attributes"]
    args = attrs.get("tool.arguments") or {}
    name = attrs.get("gen_ai.tool.name")
    if name == "TaskOutput":
        return (args.get("task_id") or args.get("taskId")) == pending["harness.background_task_id"]
    path = pending.get("harness.background_output_path")
    if not path:
        return False
    if name == "Read":
        return args.get("file_path") == path
    if name != "Bash" or not isinstance(command := args.get("command"), str):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return _single_file_read(words, path)


def _single_file_read(words: list[str], path: str) -> bool:
    if not words[1:] or words[-1] != path:
        return False
    options = words[1:-1]
    if options[-1:] == ["--"]:
        options = options[:-1]
    if words[0] == "cat":
        return all(re.fullmatch(r"-[benstuvAET]+", option) for option in options)
    if words[0] == "tail":
        return not options or bool(re.fullmatch(r"(?:-\d+|-n ?\d+|--lines=\d+)", " ".join(options)))
    return False


def _terminal_status(
    task: dict[str, Any], outcome: dict[str, Any] | None, output: str, command: object
) -> str | None:
    final_line = output.rstrip().rpartition("\n")[2].strip()
    status = None
    if task.get("status") in {"failed", "error"}:
        status = "error"
    elif type(code := task.get("exitCode", task.get("exit_code"))) is int:
        status = "ok" if code == 0 else "error"
    elif match := _EXIT_CODE.fullmatch(final_line):
        status = "ok" if int(match.group(1)) == 0 else "error"
    else:
        completed = task.get("status") == "completed"
        if completed or (outcome and _has_final_runner_summary(outcome, final_line, command)):
            if outcome and any(outcome.get(k) for k in ("failed", "errors", "failed_packages")):
                status = "error"
            elif completed or (outcome and outcome.get("passed")):
                status = "ok"
    return status


def _has_final_runner_summary(outcome: dict[str, Any], final_line: str, command: object) -> bool:
    """Recognize a complete runner summary at EOF, not an intermediate package line."""
    # A single suite's footer cannot establish that an entire shell pipeline
    # or command list finished. Wait for an explicit exit in those cases.
    if not isinstance(command, str) or re.search(r"[;&|\n`$()]", command):
        return False
    runner = outcome.get("runner")
    if runner == "go test":
        return final_line == "FAIL"
    if runner == "cargo test":
        return False
    if runner == "unittest":
        return bool(_UNITTEST_FINAL.fullmatch(final_line))
    final_outcome = parse_test_outcome(final_line)
    return bool(final_outcome and final_outcome.get("runner") == runner)
