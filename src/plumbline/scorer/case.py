"""The reference-case format: the intended path a trace is scored against.

A case is the rubric. ``reference_path`` is the ideal sequence of tool/agent
nodes (agent dispatches named ``agent:<type>``). A node may carry ``args`` — the
expected key parameters — which the parameter sub-scores consume; name/order
metrics read only ``tool``.

Cases are plain data (authored as JSON/dict), so a corpus is a directory of
files with no code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RefNode:
    """One expected action in the reference path."""

    tool: str
    args: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Case:
    """A scoring case: the reference path a realized trace is graded against."""

    case_id: str
    reference_path: tuple[RefNode, ...]
    description: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Case:
        if not isinstance(data, Mapping):
            msg = "case must be an object"
            raise TypeError(msg)
        if not isinstance(data.get("case_id"), str) or not data["case_id"].strip():
            msg = "case 'case_id' is required"
            raise ValueError(msg)
        if "reference_path" not in data or not isinstance(data["reference_path"], list):
            msg = "case 'reference_path' must be an array"
            raise ValueError(msg)
        for index, node in enumerate(data["reference_path"]):
            tool = node.get("tool") if isinstance(node, Mapping) else None
            if not isinstance(node, Mapping) or not isinstance(tool, str) or not tool.strip():
                msg = f"case reference_path[{index}].tool is required"
                raise ValueError(msg)
        path = tuple(
            RefNode(tool=node["tool"], args=node.get("args")) for node in data["reference_path"]
        )
        return cls(
            case_id=data["case_id"],
            reference_path=path,
            description=data.get("description"),
        )
