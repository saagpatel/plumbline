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
        if any(
            block.get("type") == "text"
            and "text" in block
            and not isinstance(block.get("text"), str)
            for block in content
        ):
            return "text"
        if any(
            block.get("type") == "tool_use"
            and "input" in block
            and not isinstance(block["input"], dict)
            for block in content
        ):
            return "input"
        return None
    return "content"


def _record_problem(record: JsonObj) -> str | None:
    uuid = record.get("uuid")
    if uuid is not None and not isinstance(uuid, str):
        return "uuid"
    parent = record.get("parentUuid")
    if parent is not None and not isinstance(parent, str):
        return "parentUuid"
    if record.get("type") in ("assistant", "user"):
        return _message_problem(record)
    return None


def _is_claude_record(record: JsonObj) -> bool:
    kind = record.get("type")
    return kind in ("assistant", "user")


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
            if reason := _record_problem(value):
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
