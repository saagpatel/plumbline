"""The reference-case format: the intended path a trace is scored against.

A case is the rubric. ``reference_path`` is the ideal sequence of tool/agent
nodes (agent dispatches named ``agent:<type>``). A node may carry ``args`` — the
expected key parameters — which the parameter sub-scores consume; name/order
metrics read only ``tool``.

Cases are plain data (authored as JSON/dict), so a corpus is a directory of
files with no code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        path = tuple(
            RefNode(tool=node["tool"], args=node.get("args"))
            for node in data.get("reference_path") or []
        )
        return cls(
            case_id=data["case_id"],
            reference_path=path,
            description=data.get("description"),
        )
