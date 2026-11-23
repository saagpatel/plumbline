"""Track background tool calls until a matching TaskOutput reports completion."""

from __future__ import annotations

import re
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
    if reader["subagent_id"] != launch["subagent_id"] or not _matches_task_output(attrs, reader):
        return False
    task_result = result.get("task")
    task = task_result if isinstance(task_result, dict) else result
    task_id = attrs["harness.background_task_id"]
    for metadata in (task, result):
        reported_id = metadata.get("task_id") or metadata.get("id")
        if reported_id and reported_id != task_id:
            return False
    status = _terminal_status(task, output if reader["status"] == "ok" else "")
    if status is None:
        return False
    command = attrs["tool.arguments"].get("command")
    if outcome := parse_test_outcome(output, command=command):
        attrs["test.outcome"] = outcome
    launch["status"] = status
    launch["ended_at"] = reader["ended_at"]
    attrs["harness.background_task_pending"] = False
    return True


def _matches_task_output(pending: dict[str, Any], reader: dict[str, Any]) -> bool:
    attrs = reader["attributes"]
    args = attrs.get("tool.arguments") or {}
    return (
        attrs.get("gen_ai.tool.name") == "TaskOutput"
        and (args.get("task_id") or args.get("taskId")) == pending["harness.background_task_id"]
    )


def _terminal_status(task: dict[str, Any], output: str) -> str | None:
    status = task.get("status")
    if status in ("running", "pending", "queued"):
        return None
    if status in ("failed", "error"):
        return "error"
    code = task.get("exitCode", task.get("exit_code"))
    if type(code) is int:
        return "ok" if code == 0 else "error"
    if match := _EXIT_CODE.fullmatch(output.rstrip().rpartition("\n")[2].strip()):
        return "ok" if int(match.group(1)) == 0 else "error"
    return "ok" if status == "completed" else None
