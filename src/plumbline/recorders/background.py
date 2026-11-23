"""Track background tool calls until their terminal output is read."""

from __future__ import annotations

import re
import shlex
from typing import Any

from plumbline.recorders.test_outcome import parse_test_outcome

_EXIT_CODE = re.compile(r"^(?:process )?exited with code (-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_OUTPUT_PATH = re.compile(r"\bOutput is being written to:[ \t]*([^\r\n]+)")


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
    task = result.get("task")
    task = task if isinstance(task, dict) else {}
    reported_id = task.get("task_id") or task.get("id")
    if reported_id and reported_id != attrs["harness.background_task_id"]:
        return False
    if task.get("status") in {"running", "pending", "queued"}:
        return False
    outcome = parse_test_outcome(output)
    status = _terminal_status(task, outcome, output)
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
    task: dict[str, Any], outcome: dict[str, Any] | None, output: str
) -> str | None:
    if task.get("status") in {"failed", "error"}:
        return "error"
    code = task.get("exitCode", task.get("exit_code"))
    if type(code) is int:
        return "ok" if code == 0 else "error"
    if match := _EXIT_CODE.search(output):
        return "ok" if int(match.group(1)) == 0 else "error"
    if outcome and any(outcome.get(key) for key in ("failed", "errors", "failed_packages")):
        return "error"
    if task.get("status") == "completed" or (
        outcome and (outcome.get("passed") or outcome.get("passed_packages"))
    ):
        return "ok"
    return None
