"""Read Claude Code JSONL while accounting for malformed and unsupported input."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

JsonObj = dict[str, Any]
_SNIFF_LIMIT = 100


class UnsupportedTranscriptFormatError(ValueError):
    """The input does not contain recognizable Claude Code transcript records."""

    def __str__(self) -> str:
        return "unsupported transcript format"


def _message_problem(record: JsonObj) -> str | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return "message"
    content = message.get("content")
    if isinstance(content, str) and content:
        return None
    if (
        isinstance(content, list)
        and all(isinstance(block, dict) for block in content)
        and (
            content
            or message.get("usage")
            or message.get("stop_reason")
            or record.get("isApiErrorMessage")
        )
    ):
        return None
    return "content"


def _is_claude_record(record: JsonObj) -> bool:
    kind = record.get("type")
    if kind in ("assistant", "user"):
        return True  # Message shape was checked before format detection.
    control_field = {"system": "subtype", "mode": "mode", "permission-mode": "permissionMode"}
    if isinstance(kind, str) and kind in control_field:
        return isinstance(record.get(control_field[kind]), str)
    return (
        isinstance(kind, str)
        and isinstance(record.get("sessionId"), str)
        and ("uuid" in record or "timestamp" in record)
    )


def load_jsonl(path: Path) -> list[JsonObj]:
    """Load valid Claude records, reporting malformed lines without their content."""
    events: list[JsonObj] = []
    recognized = False
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                decoded = stripped.decode("utf-8")
            except UnicodeDecodeError:
                sys.stderr.write(f"record line {line_number} skipped: utf-8\n")
                continue
            try:
                value = json.loads(decoded)
            except json.JSONDecodeError:
                sys.stderr.write(f"record line {line_number} skipped: json\n")
                continue
            if not isinstance(value, dict):
                sys.stderr.write(f"record line {line_number} skipped: record\n")
                continue
            if value.get("type") in ("assistant", "user") and (reason := _message_problem(value)):
                sys.stderr.write(f"record line {line_number} skipped: {reason}\n")
                continue
            if not recognized:
                recognized = _is_claude_record(value)
                if len(events) + 1 == _SNIFF_LIMIT and not recognized:
                    raise UnsupportedTranscriptFormatError
            events.append(value)
    if not recognized:
        raise UnsupportedTranscriptFormatError
    return events
