"""Deterministic ancestry and workflow-gap analysis for agent traces."""

# Dynamic JSON validation and stable contract errors are intentional at this boundary.
# ruff: noqa: ANN401, C901, EM101, EM102, PLR0912, PLR0913, RET504, TRY003

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from plumbline.scrub import scrub_text

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

REPORT_VERSION: Final = "PlanToolSpanGapReportV1"
MAPPING_VERSION: Final = "PlanToolSpanRoleMapV1"
STANDARDS_AS_OF: Final = "2026-08-11"
OTEL_GENAI_COMMIT: Final = "8d3e4a0f3c34a46f6edb9c71e8666e02e6bf3958"
OTEL_CORE_VERSION: Final = "1.44.0"

_ROLES = {
    "create_agent",
    "invoke_agent",
    "plan",
    "invoke_workflow",
    "execute_tool",
    "tool_result",
    "outcome",
    "compaction",
}
_EXECUTION_ROLES = {"invoke_agent", "execute_tool", "tool_result", "outcome"}
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_AGENT_ID_KEYS = ("gen_ai.agent.id", "agent.id", "agent_id")
_TOOL_CALL_ID_KEYS = ("gen_ai.tool.call.id", "tool.call.id", "tool_call_id")
_WORKFLOW_ID_KEYS = ("gen_ai.workflow.id", "workflow.id", "workflow_id")
_PLAN_REF_KEYS = ("agent.decision.plan_ref", "plan.step.id", "plan_step_id")
_CAPTURE_KEYS = {"spans", "events", "agent_lifecycle", "plans", "outcomes"}
_CAPTURE_VALUES = {"complete", "partial", "unknown"}
_MAX_RAW_TEXT = 1024


class SpanGapContractError(ValueError):
    """Stable fail-closed error for invalid trace or mapping input."""


@dataclass(frozen=True)
class MappingRule:
    """One configurable raw-signal to semantic-role mapping."""

    rule_id: str
    role: str
    span_name_prefixes: tuple[str, ...] = ()
    event_name_prefixes: tuple[str, ...] = ()
    plumbline_kinds: tuple[str, ...] = ()
    attributes: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class Node:
    """Normalized graph node with raw provenance retained by pointer."""

    node_id: str
    trace_id: str | None
    parent_ids: tuple[str, ...]
    name: str
    roles: tuple[str, ...]
    mapping_ids: tuple[str, ...]
    start_ns: int | None
    end_ns: int | None
    attributes: dict[str, Any]
    source_path: str
    source_index: int
    source_format: str
    is_event: bool = False
    virtual_kind: str | None = None


@dataclass(frozen=True)
class FindingDraft:
    """Finding before deterministic IDs and evidence references are assigned."""

    code: str
    classification: str
    severity: str
    confidence: str
    message: str
    nodes: tuple[Node, ...]


def _default_rules() -> tuple[MappingRule, ...]:
    operation = "gen_ai.operation.name"
    return (
        MappingRule(
            "otel.create_agent",
            "create_agent",
            ("create_agent",),
            attributes=((operation, ("create_agent",)),),
        ),
        MappingRule(
            "otel.invoke_agent",
            "invoke_agent",
            ("invoke_agent",),
            attributes=((operation, ("invoke_agent",)),),
            plumbline_kinds=("agent",),
        ),
        MappingRule("otel.plan", "plan", ("plan",), attributes=((operation, ("plan",)),)),
        MappingRule(
            "otel.invoke_workflow",
            "invoke_workflow",
            ("invoke_workflow",),
            attributes=((operation, ("invoke_workflow",)),),
        ),
        MappingRule(
            "otel.execute_tool",
            "execute_tool",
            ("execute_tool",),
            attributes=((operation, ("execute_tool",)),),
            plumbline_kinds=("tool_call",),
        ),
        MappingRule(
            "event.tool_result",
            "tool_result",
            span_name_prefixes=("tool_result", "tool.result"),
            event_name_prefixes=("tool_result", "tool.result", "gen_ai.tool.message"),
        ),
        MappingRule(
            "event.outcome",
            "outcome",
            ("outcome",),
            event_name_prefixes=("outcome", "agent.outcome", "workflow.outcome"),
        ),
        MappingRule("plumbline.compaction", "compaction", plumbline_kinds=("compaction",)),
    )


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpanGapContractError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpanGapContractError(f"{path} must be an array")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpanGapContractError(f"{path} must be a non-empty string")
    result = value.strip()
    if result != value or len(result) > _MAX_RAW_TEXT or "\n" in result or "\r" in result:
        raise SpanGapContractError(f"{path} must be bounded single-line text")
    return result


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    result = tuple(_text(item, f"{path}[]").lower() for item in _array(value, path))
    if len(result) != len(set(result)):
        raise SpanGapContractError(f"{path} must not contain duplicates")
    return result


def _parse_rule(value: Any, index: int) -> MappingRule:
    raw = _object(value, f"mapping.mappings[{index}]")
    allowed = {"id", "role", "match"}
    extra = sorted(raw.keys() - allowed)
    if extra:
        raise SpanGapContractError(
            f"mapping.mappings[{index}] has unsupported keys: {', '.join(extra)}"
        )
    rule_id = _text(raw.get("id"), f"mapping.mappings[{index}].id")
    role = _text(raw.get("role"), f"mapping.mappings[{index}].role")
    if role not in _ROLES:
        raise SpanGapContractError(f"mapping.mappings[{index}].role is unsupported")
    match = _object(raw.get("match"), f"mapping.mappings[{index}].match")
    match_allowed = {"span_name_prefixes", "event_name_prefixes", "plumbline_kinds", "attributes"}
    match_extra = sorted(match.keys() - match_allowed)
    if match_extra:
        raise SpanGapContractError(
            f"mapping.mappings[{index}].match has unsupported keys: {', '.join(match_extra)}"
        )
    attr_rules: list[tuple[str, tuple[str, ...]]] = []
    for key, values in sorted(_object(match.get("attributes", {}), "mapping attributes").items()):
        attr_rules.append(
            (_text(key, "mapping attribute key"), _string_tuple(values, "mapping attribute values"))
        )
    rule = MappingRule(
        rule_id=rule_id,
        role=role,
        span_name_prefixes=_string_tuple(match.get("span_name_prefixes"), "span_name_prefixes"),
        event_name_prefixes=_string_tuple(match.get("event_name_prefixes"), "event_name_prefixes"),
        plumbline_kinds=_string_tuple(match.get("plumbline_kinds"), "plumbline_kinds"),
        attributes=tuple(attr_rules),
    )
    if not any(
        (rule.span_name_prefixes, rule.event_name_prefixes, rule.plumbline_kinds, rule.attributes)
    ):
        raise SpanGapContractError(f"mapping rule {rule_id} must declare at least one matcher")
    return rule


def load_mapping(value: Any | None) -> tuple[tuple[MappingRule, ...], dict[str, Any]]:
    """Validate and normalize a mapping document, extending defaults by default."""
    if value is None:
        rules = _default_rules()
        return rules, {"schema_version": MAPPING_VERSION, "mode": "default", "mappings": []}
    root = _object(value, "mapping")
    allowed = {"schema_version", "mode", "mappings"}
    extra = sorted(root.keys() - allowed)
    if extra:
        raise SpanGapContractError(f"mapping has unsupported keys: {', '.join(extra)}")
    if root.get("schema_version") != MAPPING_VERSION:
        raise SpanGapContractError(f"mapping.schema_version must be {MAPPING_VERSION}")
    mode = root.get("mode", "extend")
    if mode not in {"extend", "replace"}:
        raise SpanGapContractError("mapping.mode must be extend or replace")
    custom = tuple(
        _parse_rule(item, index)
        for index, item in enumerate(_array(root.get("mappings"), "mapping.mappings"))
    )
    rule_ids = [rule.rule_id for rule in custom]
    if len(rule_ids) != len(set(rule_ids)):
        raise SpanGapContractError("mapping rule ids must be unique")
    rules = (*(_default_rules() if mode == "extend" else ()), *custom)
    return tuple(rules), root


def _matches(
    rule: MappingRule, *, name: str, kind: str, attributes: dict[str, Any], is_event: bool
) -> bool:
    lowered_name = name.lower()
    if is_event and any(lowered_name.startswith(prefix) for prefix in rule.event_name_prefixes):
        return True
    if not is_event and any(lowered_name.startswith(prefix) for prefix in rule.span_name_prefixes):
        return True
    if kind.lower() in rule.plumbline_kinds:
        return True
    return any(str(attributes.get(key, "")).lower() in values for key, values in rule.attributes)


def _roles(
    rules: Iterable[MappingRule],
    *,
    name: str,
    kind: str,
    attributes: dict[str, Any],
    is_event: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    matched = tuple(
        rule
        for rule in rules
        if _matches(rule, name=name, kind=kind, attributes=attributes, is_event=is_event)
    )
    return tuple(sorted({rule.role for rule in matched})), tuple(
        sorted(rule.rule_id for rule in matched)
    )


def _timestamp_ns(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SpanGapContractError(f"{path} must be a timestamp")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SpanGapContractError(f"{path} must be ISO 8601 or integer nanoseconds") from exc
        if parsed.tzinfo is None:
            raise SpanGapContractError(f"{path} must include a timezone")
        return int(parsed.astimezone(UTC).timestamp() * 1_000_000_000)
    raise SpanGapContractError(f"{path} must be ISO 8601 or integer nanoseconds")


def _attrs(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for index, item in enumerate(value):
            attr = _object(item, f"{path}[{index}]")
            key = _text(attr.get("key"), f"{path}[{index}].key")
            raw_value = attr.get("value")
            if isinstance(raw_value, dict) and len(raw_value) == 1:
                raw_value = next(iter(raw_value.values()))
            result[key] = raw_value
        return result
    raise SpanGapContractError(f"{path} must be an object or OTLP attribute array")


def _parent_ids(raw: dict[str, Any], path: str) -> tuple[str, ...]:
    if "parent_span_ids" in raw:
        return tuple(
            _text(item, f"{path}.parent_span_ids[]")
            for item in _array(raw["parent_span_ids"], f"{path}.parent_span_ids")
        )
    value = raw.get("parent_span_id", raw.get("parentSpanId"))
    if value in {None, ""}:
        return ()
    return (_text(value, f"{path}.parent_span_id"),)


def _flatten_otlp(root: dict[str, Any]) -> list[dict[str, Any]]:
    if "spans" in root:
        return [
            _object(item, f"spans[{index}]")
            for index, item in enumerate(_array(root["spans"], "spans"))
        ]
    spans: list[dict[str, Any]] = []
    for resource_index, resource in enumerate(_array(root.get("resourceSpans"), "resourceSpans")):
        resource_obj = _object(resource, f"resourceSpans[{resource_index}]")
        scopes = resource_obj.get("scopeSpans", resource_obj.get("instrumentationLibrarySpans", []))
        for scope_index, scope in enumerate(_array(scopes, "resourceSpans[].scopeSpans")):
            scope_obj = _object(scope, f"resourceSpans[{resource_index}].scopeSpans[{scope_index}]")
            spans.extend(
                _object(item, "resourceSpans[].scopeSpans[].spans[]")
                for item in _array(scope_obj.get("spans"), "scope spans")
            )
    return spans


def _normalize_plumbline(root: dict[str, Any], rules: tuple[MappingRule, ...]) -> list[Node]:
    nodes: list[Node] = []
    run = _object(root.get("run"), "run")
    run_id = _text(run.get("run_id"), "run.run_id")
    for index, item in enumerate(_array(root.get("steps"), "steps")):
        raw = _object(item, f"steps[{index}]")
        node_id = _text(raw.get("step_id"), f"steps[{index}].step_id")
        kind = _text(raw.get("kind"), f"steps[{index}].kind")
        attributes = _attrs(raw.get("attributes"), f"steps[{index}].attributes")
        name = str(attributes.get("gen_ai.operation.name", kind))
        roles, mapping_ids = _roles(
            rules, name=name, kind=kind, attributes=attributes, is_event=False
        )
        parent = raw.get("parent_step_id")
        nodes.append(
            Node(
                node_id,
                run_id,
                () if parent is None else (_text(parent, "parent_step_id"),),
                name,
                roles,
                mapping_ids,
                _timestamp_ns(raw.get("started_at"), "started_at"),
                _timestamp_ns(raw.get("ended_at"), "ended_at"),
                attributes,
                f"steps[{index}]",
                index,
                "plumbline",
            )
        )
    plan = run.get("plan")
    if isinstance(plan, dict):
        for item_index, item in enumerate(_array(plan.get("items", []), "run.plan.items")):
            plan_item = _object(item, f"run.plan.items[{item_index}]")
            plan_id = _text(plan_item.get("id"), f"run.plan.items[{item_index}].id")
            nodes.append(
                Node(
                    f"plan:{plan_id}",
                    run_id,
                    (),
                    "plan item",
                    ("plan",),
                    ("plumbline.plan_item",),
                    _timestamp_ns(run.get("started_at"), "run.started_at"),
                    None,
                    {"plan.step.id": plan_id, "plan.status": plan_item.get("status")},
                    f"run.plan.items[{item_index}]",
                    len(nodes),
                    "plumbline",
                    virtual_kind="plan_item",
                )
            )
    outcome = run.get("outcome")
    if isinstance(outcome, dict):
        nodes.append(
            Node(
                f"outcome:{run_id}",
                run_id,
                (),
                "run outcome",
                ("outcome",),
                ("plumbline.run_outcome",),
                _timestamp_ns(run.get("ended_at"), "run.ended_at"),
                None,
                {
                    "outcome.status": outcome.get("status"),
                    "outcome.summary": outcome.get("summary"),
                },
                "run.outcome",
                len(nodes),
                "plumbline",
                virtual_kind="run_outcome",
            )
        )
    return nodes


def _normalize_otel(root: dict[str, Any], rules: tuple[MappingRule, ...]) -> list[Node]:
    nodes: list[Node] = []
    for index, raw in enumerate(_flatten_otlp(root)):
        path = f"spans[{index}]"
        node_id = _text(raw.get("span_id", raw.get("spanId")), f"{path}.span_id").lower()
        trace_id = _text(raw.get("trace_id", raw.get("traceId")), f"{path}.trace_id").lower()
        name = _text(raw.get("name"), f"{path}.name")
        attributes = _attrs(raw.get("attributes"), f"{path}.attributes")
        roles, mapping_ids = _roles(
            rules, name=name, kind="", attributes=attributes, is_event=False
        )
        node = Node(
            node_id,
            trace_id,
            _parent_ids(raw, path),
            name,
            roles,
            mapping_ids,
            _timestamp_ns(
                raw.get(
                    "start_time_unix_nano", raw.get("startTimeUnixNano", raw.get("start_time"))
                ),
                f"{path}.start",
            ),
            _timestamp_ns(
                raw.get("end_time_unix_nano", raw.get("endTimeUnixNano", raw.get("end_time"))),
                f"{path}.end",
            ),
            attributes,
            path,
            len(nodes),
            "otel",
        )
        nodes.append(node)
        for event_index, event_value in enumerate(_array(raw.get("events", []), f"{path}.events")):
            event = _object(event_value, f"{path}.events[{event_index}]")
            event_name = _text(event.get("name"), f"{path}.events[{event_index}].name")
            event_attrs = _attrs(
                event.get("attributes"), f"{path}.events[{event_index}].attributes"
            )
            event_roles, event_mapping_ids = _roles(
                rules, name=event_name, kind="", attributes=event_attrs, is_event=True
            )
            nodes.append(
                Node(
                    f"{node_id}:event:{event_index}",
                    trace_id,
                    (node_id,),
                    event_name,
                    event_roles,
                    event_mapping_ids,
                    _timestamp_ns(
                        event.get(
                            "time_unix_nano", event.get("timeUnixNano", event.get("timestamp"))
                        ),
                        "event timestamp",
                    ),
                    None,
                    event_attrs,
                    f"{path}.events[{event_index}]",
                    len(nodes),
                    "otel",
                    is_event=True,
                )
            )
    return nodes


def normalize_trace(
    value: Any, rules: tuple[MappingRule, ...]
) -> tuple[list[Node], dict[str, str], str]:
    """Normalize Plumbline or OTLP-shaped JSON without discarding provenance."""
    root = _object(value, "trace")
    capture_raw = _object(root.get("capture_scope", {}), "capture_scope")
    extra_capture = sorted(capture_raw.keys() - _CAPTURE_KEYS)
    if extra_capture:
        raise SpanGapContractError(
            f"capture_scope has unsupported keys: {', '.join(extra_capture)}"
        )
    capture = {key: str(capture_raw.get(key, "unknown")) for key in sorted(_CAPTURE_KEYS)}
    if any(value not in _CAPTURE_VALUES for value in capture.values()):
        raise SpanGapContractError("capture_scope values must be complete, partial, or unknown")
    if "plumbline_version" in root:
        return _normalize_plumbline(root, rules), capture, "plumbline"
    if "spans" in root or "resourceSpans" in root:
        return _normalize_otel(root, rules), capture, "otel"
    raise SpanGapContractError("trace must be Plumbline JSON or contain spans/resourceSpans")


def _attribute(node: Node, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = node.attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _unknown_or_violation(capture: dict[str, str], key: str) -> tuple[str, str, str]:
    if capture[key] == "complete":
        return "VIOLATION", "error", "certain"
    return "UNKNOWN", "warning", "low"


def _ancestors(node: Node, unique: dict[str, Node]) -> list[Node]:
    result: list[Node] = []
    pending = list(node.parent_ids)
    seen: set[str] = set()
    while pending:
        parent_id = pending.pop()
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent = unique.get(parent_id)
        if parent is not None:
            result.append(parent)
            pending.extend(parent.parent_ids)
    return result


def _descendants(node: Node, children: dict[str, list[Node]]) -> list[Node]:
    result: list[Node] = []
    pending = list(children.get(node.node_id, []))
    seen: set[str] = set()
    while pending:
        child = pending.pop()
        if child.node_id in seen:
            continue
        seen.add(child.node_id)
        result.append(child)
        pending.extend(children.get(child.node_id, []))
    return result


def _finding(
    code: str,
    message: str,
    nodes: Iterable[Node],
    *,
    classification: str = "VIOLATION",
    severity: str = "error",
    confidence: str = "certain",
) -> FindingDraft:
    distinct = {(item.source_path, item.node_id): item for item in nodes}
    ordered = tuple(sorted(distinct.values(), key=lambda item: (item.source_index, item.node_id)))
    return FindingDraft(code, classification, severity, confidence, message, ordered)


def _detect_structure(
    nodes: list[Node], capture: dict[str, str]
) -> tuple[list[FindingDraft], dict[str, Node], dict[str, list[Node]]]:
    findings: list[FindingDraft] = []
    by_id: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        by_id[node.node_id].append(node)
    for node_id, duplicates in sorted(by_id.items()):
        if len(duplicates) > 1:
            findings.append(
                _finding(
                    "DUPLICATE_ID",
                    f"node id {scrub_text(node_id)} occurs {len(duplicates)} times",
                    duplicates,
                )
            )
    unique = {node_id: values[0] for node_id, values in by_id.items() if len(values) == 1}
    children: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if len(node.parent_ids) > 1:
            findings.append(
                _finding(
                    "MULTIPLE_PARENTS",
                    f"node {scrub_text(node.node_id)} declares multiple semantic parents",
                    [node],
                )
            )
        for parent_id in node.parent_ids:
            parents = by_id.get(parent_id, [])
            if not parents:
                findings.append(
                    _finding(
                        "BROKEN_SPAN_REFERENCE",
                        f"node {scrub_text(node.node_id)} references missing parent "
                        f"{scrub_text(parent_id)}",
                        [node],
                    )
                )
                continue
            parent = parents[0]
            children[parent_id].append(node)
            if node.trace_id and parent.trace_id and node.trace_id != parent.trace_id:
                findings.append(
                    _finding(
                        "BROKEN_TRACE_REFERENCE",
                        f"node {scrub_text(node.node_id)} and its parent use different trace ids",
                        [parent, node],
                    )
                )
        if node.source_format == "otel" and not node.is_event:
            if (
                node.trace_id is None
                or not _TRACE_ID_RE.fullmatch(node.trace_id)
                or node.trace_id == "0" * 32
            ):
                findings.append(
                    _finding(
                        "INVALID_TRACE_ID",
                        f"node {scrub_text(node.node_id)} has an invalid OTel trace id",
                        [node],
                    )
                )
            if not _SPAN_ID_RE.fullmatch(node.node_id) or node.node_id == "0" * 16:
                findings.append(
                    _finding(
                        "INVALID_SPAN_ID",
                        f"node {scrub_text(node.node_id)} has an invalid OTel span id",
                        [node],
                    )
                )
        if len(node.roles) > 1:
            findings.append(
                _finding(
                    "AMBIGUOUS_MAPPING",
                    f"node {scrub_text(node.node_id)} maps to multiple semantic roles: "
                    f"{', '.join(node.roles)}",
                    [node],
                    severity="warning",
                )
            )
        if node.end_ns is not None and node.start_ns is not None and node.end_ns < node.start_ns:
            findings.append(
                _finding(
                    "IMPOSSIBLE_ORDERING",
                    f"node {scrub_text(node.node_id)} ends before it starts",
                    [node],
                )
            )
    for node in nodes:
        for parent_id in node.parent_ids:
            parent = unique.get(parent_id)
            if parent is None:
                continue
            if (
                node.start_ns is not None
                and parent.start_ns is not None
                and node.start_ns < parent.start_ns
            ):
                findings.append(
                    _finding(
                        "IMPOSSIBLE_ORDERING",
                        f"child {scrub_text(node.node_id)} starts before parent "
                        f"{scrub_text(parent.node_id)}",
                        [parent, node],
                    )
                )
    semantic_children = {"execute_tool", "tool_result"}
    for node in nodes:
        if semantic_children.intersection(node.roles) and not node.parent_ids:
            classification, severity, confidence = _unknown_or_violation(capture, "spans")
            findings.append(
                _finding(
                    "MISSING_PARENT",
                    f"semantic child {scrub_text(node.node_id)} has no parent reference",
                    [node],
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    return findings, unique, children


def _detect_cycles(unique: dict[str, Node]) -> list[FindingDraft]:
    findings: list[FindingDraft] = []
    color: dict[str, int] = {}
    stack: list[str] = []
    reported: set[tuple[str, ...]] = set()

    def visit(node_id: str) -> None:
        color[node_id] = 1
        stack.append(node_id)
        node = unique[node_id]
        for parent_id in sorted(node.parent_ids):
            if parent_id not in unique:
                continue
            if color.get(parent_id, 0) == 0:
                visit(parent_id)
            elif color.get(parent_id) == 1:
                start = stack.index(parent_id)
                cycle_ids = tuple(sorted(stack[start:]))
                if cycle_ids not in reported:
                    reported.add(cycle_ids)
                    findings.append(
                        _finding(
                            "CYCLE",
                            f"parent graph contains a cycle across {len(cycle_ids)} nodes",
                            [unique[item] for item in cycle_ids],
                        )
                    )
        stack.pop()
        color[node_id] = 2

    for node_id in sorted(unique):
        if color.get(node_id, 0) == 0:
            visit(node_id)
    return findings


def _detect_agent_lifecycle(nodes: list[Node], capture: dict[str, str]) -> list[FindingDraft]:
    findings: list[FindingDraft] = []
    creates: dict[str, list[Node]] = defaultdict(list)
    invokes: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        agent_id = _attribute(node, _AGENT_ID_KEYS)
        if agent_id and "create_agent" in node.roles:
            creates[agent_id].append(node)
        if agent_id and "invoke_agent" in node.roles:
            invokes[agent_id].append(node)
    classification, severity, confidence = _unknown_or_violation(capture, "agent_lifecycle")
    for agent_id, create_nodes in sorted(creates.items()):
        later = [
            invoke
            for invoke in invokes.get(agent_id, [])
            if create_nodes[0].start_ns is None
            or invoke.start_ns is None
            or invoke.start_ns >= create_nodes[0].start_ns
        ]
        if not later:
            findings.append(
                _finding(
                    "CREATE_WITHOUT_INVOKE",
                    f"created agent {scrub_text(agent_id)} has no later invocation evidence",
                    create_nodes,
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    for agent_id, invoke_nodes in sorted(invokes.items()):
        earlier = [
            create
            for create in creates.get(agent_id, [])
            if invoke_nodes[0].start_ns is None
            or create.start_ns is None
            or create.start_ns <= invoke_nodes[0].start_ns
        ]
        if not earlier:
            findings.append(
                _finding(
                    "INVOKE_WITHOUT_CREATED_AGENT",
                    f"invoked agent {scrub_text(agent_id)} has no earlier creation evidence",
                    invoke_nodes,
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    return findings


def _detect_tools(
    nodes: list[Node], unique: dict[str, Node], capture: dict[str, str]
) -> list[FindingDraft]:
    findings: list[FindingDraft] = []
    invocations: dict[str, list[Node]] = defaultdict(list)
    results: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        call_id = _attribute(node, _TOOL_CALL_ID_KEYS)
        if call_id and "execute_tool" in node.roles:
            invocations[call_id].append(node)
        if call_id and "tool_result" in node.roles:
            results[call_id].append(node)
        if call_id is None and "tool_result" in node.roles:
            findings.append(
                _finding(
                    "TOOL_RESULT_WITHOUT_INVOCATION",
                    f"tool result {scrub_text(node.node_id)} has no invocation join key",
                    [node],
                    classification="UNKNOWN",
                    severity="warning",
                    confidence="low",
                )
            )
        if "execute_tool" in node.roles:
            ancestors = _ancestors(node, unique)
            if not any(
                {"invoke_agent", "invoke_workflow"}.intersection(parent.roles)
                for parent in ancestors
            ):
                classification, severity, confidence = _unknown_or_violation(capture, "spans")
                findings.append(
                    _finding(
                        "ORPHAN_TOOL_EXECUTION",
                        f"tool execution {scrub_text(node.node_id)} has no agent "
                        "or workflow ancestor",
                        [node],
                        classification=classification,
                        severity=severity,
                        confidence=confidence,
                    )
                )
    classification, severity, confidence = _unknown_or_violation(capture, "events")
    for call_id, result_nodes in sorted(results.items()):
        prior = [
            invocation
            for invocation in invocations.get(call_id, [])
            if result_nodes[0].start_ns is None
            or invocation.start_ns is None
            or invocation.start_ns <= result_nodes[0].start_ns
        ]
        if not prior:
            findings.append(
                _finding(
                    "TOOL_RESULT_WITHOUT_INVOCATION",
                    f"tool result {scrub_text(call_id)} has no prior invocation evidence",
                    result_nodes,
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    return findings


def _detect_plans_and_outcomes(
    nodes: list[Node],
    unique: dict[str, Node],
    children: dict[str, list[Node]],
    capture: dict[str, str],
) -> list[FindingDraft]:
    findings: list[FindingDraft] = []
    plan_refs = {
        _attribute(node, _PLAN_REF_KEYS) for node in nodes if _attribute(node, _PLAN_REF_KEYS)
    }
    for node in nodes:
        if "plan" not in node.roles:
            continue
        descendants = _descendants(node, children)
        plan_id = _attribute(node, ("plan.step.id",))
        executed = any(_EXECUTION_ROLES.intersection(child.roles) for child in descendants) or (
            plan_id is not None and plan_id in plan_refs
        )
        if not executed:
            classification, severity, confidence = _unknown_or_violation(capture, "plans")
            findings.append(
                _finding(
                    "PLAN_STEP_WITHOUT_EXECUTION",
                    f"plan node {scrub_text(node.node_id)} has no execution evidence",
                    [node],
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    executable_nodes = [
        node
        for node in nodes
        if {"invoke_agent", "execute_tool", "tool_result"}.intersection(node.roles)
    ]
    for node in nodes:
        if "outcome" not in node.roles:
            continue
        connected = node.virtual_kind == "run_outcome" and bool(executable_nodes)
        if not connected:
            connected = bool(node.parent_ids) and any(
                _EXECUTION_ROLES.intersection(parent.roles) for parent in _ancestors(node, unique)
            )
        if not connected:
            classification, severity, confidence = _unknown_or_violation(capture, "outcomes")
            findings.append(
                _finding(
                    "DISCONNECTED_OUTCOME",
                    f"outcome {scrub_text(node.node_id)} is disconnected from execution ancestry",
                    [node],
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                )
            )
    return findings


def _detect_workflow_escape(nodes: list[Node], unique: dict[str, Node]) -> list[FindingDraft]:
    findings: list[FindingDraft] = []
    workflow_nodes: dict[str, Node] = {}
    for node in nodes:
        if "invoke_workflow" in node.roles:
            workflow_id = _attribute(node, _WORKFLOW_ID_KEYS) or node.node_id
            workflow_nodes[workflow_id] = node
    for node in nodes:
        workflow_id = _attribute(node, _WORKFLOW_ID_KEYS)
        if workflow_id is None or "invoke_workflow" in node.roles:
            continue
        expected = workflow_nodes.get(workflow_id)
        if expected is not None and expected.node_id not in {
            parent.node_id for parent in _ancestors(node, unique)
        }:
            findings.append(
                _finding(
                    "WORKFLOW_SUBTREE_ESCAPE",
                    f"node {scrub_text(node.node_id)} claims workflow "
                    f"{scrub_text(workflow_id)} outside that workflow subtree",
                    [expected, node],
                )
            )
    return findings


def _materialize(findings: list[FindingDraft]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_findings = {
        (
            item.code,
            item.classification,
            item.message,
            tuple(node.source_path for node in item.nodes),
        ): item
        for item in findings
    }
    ordered = sorted(
        unique_findings.values(),
        key=lambda item: (item.code, tuple(node.source_path for node in item.nodes), item.message),
    )
    evidence_by_key = {
        (node.source_path, node.node_id): node for finding in ordered for node in finding.nodes
    }
    evidence_nodes = sorted(
        evidence_by_key.values(),
        key=lambda node: (node.source_index, node.source_path, node.node_id),
    )
    evidence_ids = {
        (node.source_path, node.node_id): f"E{index:04d}"
        for index, node in enumerate(evidence_nodes, 1)
    }
    evidence = [
        {
            "evidence_id": evidence_ids[(node.source_path, node.node_id)],
            "source_path": node.source_path,
            "source_format": node.source_format,
            "node_id": scrub_text(node.node_id),
            "trace_id": scrub_text(node.trace_id) if node.trace_id else None,
            "raw_name": scrub_text(node.name),
            "semantic_roles": list(node.roles),
            "mapping_ids": list(dict.fromkeys(scrub_text(value) for value in node.mapping_ids)),
        }
        for node in evidence_nodes
    ]
    rendered = [
        {
            "finding_id": f"G{index:04d}",
            "code": finding.code,
            "classification": finding.classification,
            "severity": finding.severity,
            "confidence": finding.confidence,
            "message": finding.message,
            "node_ids": list(dict.fromkeys(scrub_text(node.node_id) for node in finding.nodes)),
            "evidence_refs": [
                evidence_ids[(node.source_path, node.node_id)] for node in finding.nodes
            ],
        }
        for index, finding in enumerate(ordered, 1)
    ]
    return rendered, evidence


def analyze_span_gaps(trace: Any, mapping: Any | None = None) -> dict[str, Any]:
    """Return a deterministic, scrubbed, versioned ancestry-gap report."""
    rules, mapping_document = load_mapping(mapping)
    nodes, capture, source_format = normalize_trace(trace, rules)
    findings, unique, children = _detect_structure(nodes, capture)
    findings.extend(_detect_cycles(unique))
    findings.extend(_detect_agent_lifecycle(nodes, capture))
    findings.extend(_detect_tools(nodes, unique, capture))
    findings.extend(_detect_plans_and_outcomes(nodes, unique, children, capture))
    findings.extend(_detect_workflow_escape(nodes, unique))
    rendered, evidence = _materialize(findings)
    violations = sum(item["classification"] == "VIOLATION" for item in rendered)
    unknowns = sum(item["classification"] == "UNKNOWN" for item in rendered)
    disposition = "FAIL" if violations else ("UNKNOWN" if unknowns else "PASS")
    mapped = sum(bool(node.roles) for node in nodes)
    report = {
        "schema_version": REPORT_VERSION,
        "standards_snapshot": {
            "as_of": STANDARDS_AS_OF,
            "otel_genai_commit": OTEL_GENAI_COMMIT,
            "otel_genai_status": "Development",
            "otel_core_semconv_version": OTEL_CORE_VERSION,
            "interpretation": (
                "Configured role mappings are local design inferences; raw names remain provenance."
            ),
        },
        "input": {
            "format": source_format,
            "digest": _json_digest(trace),
            "capture_scope": capture,
            "node_count": len(nodes),
        },
        "mapping": {
            "schema_version": MAPPING_VERSION,
            "digest": _json_digest(mapping_document),
            "rule_count": len(rules),
            "mapped_node_count": mapped,
            "unmapped_node_count": len(nodes) - mapped,
        },
        "disposition": disposition,
        "summary": {
            "finding_count": len(rendered),
            "violation_count": violations,
            "unknown_count": unknowns,
            "error_count": sum(item["severity"] == "error" for item in rendered),
            "warning_count": sum(item["severity"] == "warning" for item in rendered),
        },
        "findings": rendered,
        "evidence_index": evidence,
        "claim_ceiling": (
            "Deterministic structural analysis of the supplied capture only; missing telemetry "
            "and semantic mapping uncertainty remain explicit UNKNOWNs."
        ),
    }
    return report


def load_json_file(path: Path) -> Any:
    """Load JSON with a stable, user-facing contract error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpanGapContractError(f"cannot read JSON from {path}: {exc}") from exc


def format_span_gap_report(report: dict[str, Any]) -> str:
    """Render a concise deterministic human report."""
    summary = _object(report["summary"], "report.summary")
    lines = [
        f"Plan/tool span gaps: {report['disposition']}",
        f"Findings: {summary['finding_count']} "
        f"({summary['violation_count']} violations, {summary['unknown_count']} unknowns)",
    ]
    for finding in _array(report["findings"], "report.findings"):
        item = _object(finding, "report.findings[]")
        lines.append(
            f"- [{item['severity'].upper()}] {item['code']} "
            f"({item['classification']}, {item['confidence']}): {item['message']} "
            f"[{', '.join(item['evidence_refs'])}]"
        )
    if not report["findings"]:
        lines.append("- No ancestry gaps detected in the supplied capture.")
    lines.append(f"Claim ceiling: {report['claim_ceiling']}")
    return "\n".join(lines)
