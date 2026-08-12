"""Deterministic, inert span-subtree reduction for local replay tests.

The generator deliberately accepts JSON-shaped traces and emits JSON-shaped data.
It never imports a provider SDK or executes a captured tool payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

REQUEST_VERSION = "SpanToTestGenerationRequestV1"
FIXTURE_VERSION = "SanitizedReplayFixtureV1"
RECEIPT_VERSION = "SpanToTestReductionReceiptV1"
GENERATOR_VERSION = "1.0.0"

_MAX_TRACE_BYTES = 16 * 1024 * 1024
_MAX_STEPS = 10_000
_MAX_IDENTIFIER_LENGTH = 512
_MAX_ATTRIBUTE_TEXT_LENGTH = 4096
_MAX_SAFE_TEXT_LENGTH = 128
_CONTROL_CHARACTER_LIMIT = 32
_FAILURE_STATUSES = frozenset({"error", "interrupted"})
_FAILURE_OUTCOMES = frozenset({"failed", "aborted"})
_FINDING_KEYS = frozenset({"finding.id", "finding_id", "plumbline.finding.id"})
_ASSERTION_KINDS = frozenset(
    {
        "span_failure",
        "span_status",
        "hook_verdict",
        "error_class",
        "finding",
        "run_outcome",
    }
)
_ASSERTION_FIELDS = frozenset({"kind", "target_span_id", "operator", "expected"})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,127}$")
_OPAQUE_ID = re.compile(r"^[a-z]+_[0-9a-f]{20}$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"(?i)\b(?:https?|wss?|file)://[^\s\"'<>]+")
_HOME_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\[^\\\s]+|/(?:Users|home)/[^/\s\"']+)")
_TOKEN = re.compile(
    r"(?i)(?:\bbearer\s+|\b(?:sk|gh[pousr]|xox[baprs])[-_])"
    r"[A-Za-z0-9._=-]{12,}|\b[A-Fa-f0-9]{32,}\b"
)
_ENV_KEY = re.compile(
    r"(?i)(?:^|[._-])(?:env|environment|api_key|token|secret|password)(?:$|[._-])"
)
_PROMPT_KEY = re.compile(
    r"(?i)(?:prompt|completion|message|content|instruction|rationale|summary|exception\.message)"
)
_TOOL_PAYLOAD_KEY = re.compile(r"(?i)(?:arguments?|results?|command|stdout|stderr|payload|body)")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]{128,}={0,2}$")


class SpanToTestContractError(ValueError):
    """Stable fail-closed generation or fixture-validation error."""


def _canonical_bytes(value: Any) -> bytes:  # noqa: ANN401 - JSON-shaped public contract
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:  # noqa: ANN401 - JSON-shaped public contract
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pseudonym(namespace: str, value: str) -> str:
    material = f"plumbline-span-to-test-v1\0{namespace}\0{value}".encode()
    return f"{namespace}_{hashlib.sha256(material).hexdigest()[:20]}"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:  # noqa: ANN401
    if not isinstance(value, Mapping):
        raise SpanToTestContractError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:  # noqa: ANN401
    if not isinstance(value, list):
        raise SpanToTestContractError(f"{path} must be an array")
    return value


def _text(value: Any, path: str, *, maximum: int = _MAX_IDENTIFIER_LENGTH) -> str:  # noqa: ANN401
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SpanToTestContractError(
            f"{path} must be a non-empty string of at most {maximum} chars"
        )
    return value


def _safe_name(value: Any, fallback: str) -> str:  # noqa: ANN401
    if not isinstance(value, str):
        return fallback
    normalized = unicodedata.normalize("NFC", value).strip()
    if _SAFE_NAME.fullmatch(normalized) and not _sensitive_text_category(normalized):
        return normalized
    return _pseudonym(fallback, normalized)


def _argument_key_identity(value: str) -> str:
    """Return the comparison identity used to avoid portable key collisions."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _sensitive_text_category(value: str) -> str | None:
    if _EMAIL.search(value):
        return "email"
    if _URL.search(value):
        return "url"
    if _HOME_PATH.search(value):
        return "home_or_environment_path"
    if _TOKEN.search(value):
        return "token"
    if _BASE64.fullmatch(value) or any(
        ord(character) < _CONTROL_CHARACTER_LIMIT for character in value
    ):
        return "binary_data"
    if len(value) > _MAX_SAFE_TEXT_LENGTH:
        return "oversized_content"
    return None


def _field_id(step_id: str, path: str) -> str:
    return _pseudonym("field", f"{step_id}\0{path}")


def _classify_sensitive(path: str, value: Any) -> str:  # noqa: ANN401
    lowered = path.lower()
    if _ENV_KEY.search(lowered):
        return "environment_or_secret"
    if isinstance(value, (bytes, bytearray)):
        return "binary_data"
    if isinstance(value, str):
        if _EMAIL.search(value):
            return "email"
        if _URL.search(value):
            return "url"
        if _HOME_PATH.search(value):
            return "home_or_environment_path"
        if _TOKEN.search(value):
            return "token"
        if _BASE64.fullmatch(value) or any(ord(character) == 0 for character in value):
            return "binary_data"
        if len(value) > _MAX_ATTRIBUTE_TEXT_LENGTH:
            return "oversized_content"
    if _PROMPT_KEY.search(lowered):
        return "model_content_or_exception_text"
    if _TOOL_PAYLOAD_KEY.search(lowered):
        return "tool_arguments_or_results"
    return "non_whitelisted_attribute"


def _leaf_values(value: Any, path: str = "") -> list[tuple[str, Any]]:  # noqa: ANN401
    leaves: list[tuple[str, Any]] = []
    stack: list[tuple[str, Any]] = [(path, value)]
    while stack:
        current_path, current = stack.pop()
        if isinstance(current, Mapping):
            if not current:
                leaves.append((current_path, current))
            else:
                for key in sorted(current, reverse=True):
                    child = f"{current_path}/{key}" if current_path else str(key)
                    stack.append((child, current[key]))
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if not current:
                leaves.append((current_path, current))
            else:
                stack.extend(
                    (f"{current_path}/{index}", current[index])
                    for index in range(len(current) - 1, -1, -1)
                )
        else:
            leaves.append((current_path, current))
    return leaves


def _record_removed(
    transformations: list[dict[str, str]],
    *,
    source_step_id: str,
    path: str,
    value: Any,  # noqa: ANN401
) -> None:
    leaves = _leaf_values(value, path)
    for leaf_path, leaf in leaves:
        transformations.append(
            {
                "span_id": _pseudonym("span", source_step_id),
                "field_id": _field_id(source_step_id, leaf_path),
                "category": _classify_sensitive(leaf_path, leaf),
                "action": "removed",
            }
        )


def _record_identity(
    transformations: list[dict[str, str]], *, source_step_id: str, path: str
) -> None:
    transformations.append(
        {
            "span_id": _pseudonym("span", source_step_id),
            "field_id": _field_id(source_step_id, path),
            "category": "identifier",
            "action": "pseudonymized",
        }
    )


def _record_timestamp(
    transformations: list[dict[str, str]], *, source_step_id: str, path: str
) -> None:
    transformations.append(
        {
            "span_id": _pseudonym("span", source_step_id),
            "field_id": _field_id(source_step_id, path),
            "category": "timestamp",
            "action": "replaced_with_sequence",
        }
    )


def _normalize_argument_keys(
    arguments: Mapping[Any, Any],
    transformations: list[dict[str, str]],
    *,
    source_step_id: str,
) -> list[str]:
    """Project argument keys to unique deterministic names without losing key receipts."""
    source_keys = sorted(str(key) for key in arguments)
    collision_counts: dict[str, int] = defaultdict(int)
    for source_key in source_keys:
        collision_counts[_argument_key_identity(source_key)] += 1

    projected: list[str] = []
    used_identities: set[str] = set()
    for source_key in source_keys:
        _record_identity(
            transformations,
            source_step_id=source_step_id,
            path=(f"attributes/tool.arguments/@key/{_pseudonym('argkey', source_key)}"),
        )
        collision_identity = _argument_key_identity(source_key)
        candidate = (
            _pseudonym("arg", source_key)
            if collision_counts[collision_identity] > 1
            else _safe_name(source_key, "arg")
        )
        candidate_identity = _argument_key_identity(candidate)
        disambiguator = 0
        while candidate_identity in used_identities:
            disambiguator += 1
            candidate = _pseudonym("arg", f"{source_key}\0{disambiguator}")
            candidate_identity = _argument_key_identity(candidate)
        used_identities.add(candidate_identity)
        projected.append(candidate)
    return sorted(projected)


def _normalize_bool(value: Any) -> bool | None:  # noqa: ANN401
    return value if isinstance(value, bool) else None


def _safe_attributes(
    step: Mapping[str, Any], transformations: list[dict[str, str]]
) -> dict[str, Any]:
    source_id = str(step["step_id"])
    kind = str(step["kind"])
    raw = _mapping(step.get("attributes") or {}, f"step {source_id} attributes")
    safe: dict[str, Any] = {}
    consumed: set[str] = set()

    def take_name(key: str, fallback: str) -> str | None:
        if key not in raw:
            return None
        consumed.add(key)
        return _safe_name(raw[key], fallback)

    if kind == "tool_call":
        tool_name = take_name("gen_ai.tool.name", "tool") or "tool_unknown"
        operation = take_name("gen_ai.operation.name", "operation") or "execute_tool"
        argument_keys: list[str] = []
        if "tool.arguments" in raw:
            consumed.add("tool.arguments")
            arguments = raw["tool.arguments"]
            if isinstance(arguments, Mapping):
                argument_keys = _normalize_argument_keys(
                    arguments,
                    transformations,
                    source_step_id=source_id,
                )
            _record_removed(
                transformations,
                source_step_id=source_id,
                path="attributes/tool.arguments",
                value=arguments,
            )
        result_kind = take_name("tool.result.kind", "result")
        for key in raw:
            if key.startswith("tool.result.") and key not in consumed:
                consumed.add(key)
                _record_removed(
                    transformations,
                    source_step_id=source_id,
                    path=f"attributes/{key}",
                    value=raw[key],
                )
        safe["tool"] = {
            "operation": operation,
            "name": tool_name,
            "argument_keys": argument_keys,
            "result_kind": result_kind,
            "descriptor_mode": "inert",
        }
    elif kind == "hook":
        for key in ("harness.hook.name", "harness.hook.event", "harness.hook.verdict"):
            value = take_name(key, "hook")
            if value is not None:
                safe[key] = value
        prevented = _normalize_bool(raw.get("harness.hook.prevented_continuation"))
        if prevented is not None:
            consumed.add("harness.hook.prevented_continuation")
            safe["harness.hook.prevented_continuation"] = prevented
        target = raw.get("harness.hook.target_step_id")
        if isinstance(target, str):
            consumed.add("harness.hook.target_step_id")
            safe["harness.hook.target_span_id"] = _pseudonym("span", target)
    elif kind == "decision":
        decision = take_name("agent.decision.kind", "decision")
        if decision is not None:
            safe["agent.decision.kind"] = decision
        on_plan = _normalize_bool(raw.get("agent.decision.on_plan"))
        if on_plan is not None:
            consumed.add("agent.decision.on_plan")
            safe["agent.decision.on_plan"] = on_plan
    elif kind == "agent":
        agent_type = take_name("agent.type", "agent")
        if agent_type is not None:
            safe["agent.type"] = agent_type
        spawned = raw.get("agent.spawns_subagent_id")
        if isinstance(spawned, str):
            consumed.add("agent.spawns_subagent_id")
            safe["agent.spawns_subagent_id"] = _pseudonym("agent", spawned)
    elif kind == "llm":
        operation = take_name("gen_ai.operation.name", "operation")
        if operation is not None:
            safe["gen_ai.operation.name"] = operation
        finish = raw.get("gen_ai.response.finish_reasons")
        if isinstance(finish, list):
            consumed.add("gen_ai.response.finish_reasons")
            safe["gen_ai.response.finish_reasons"] = [
                _safe_name(item, "finish") for item in finish if isinstance(item, str)
            ]
    elif kind == "memory":
        operation = take_name("harness.memory.op", "memory")
        if operation is not None:
            safe["harness.memory.op"] = operation
        scope = take_name("harness.memory.scope", "scope")
        if scope is not None:
            safe["harness.memory.scope"] = scope
    elif kind == "compaction":
        reason = take_name("harness.compaction.reason", "reason")
        if reason is not None:
            safe["harness.compaction.reason"] = reason
    elif kind == "mode_change":
        for key in ("harness.mode.kind", "harness.mode.from", "harness.mode.to"):
            value = take_name(key, "mode")
            if value is not None:
                safe[key] = value

    for key in sorted(set(raw) - consumed):
        _record_removed(
            transformations,
            source_step_id=source_id,
            path=f"attributes/{key}",
            value=raw[key],
        )
    return safe


def _validate_request(request: Mapping[str, Any]) -> tuple[str, str]:
    if set(request) != {"schema_version", "selector"}:
        raise SpanToTestContractError("request must contain only schema_version and selector")
    if request["schema_version"] != REQUEST_VERSION:
        raise SpanToTestContractError(f"request.schema_version must be {REQUEST_VERSION}")
    selector = _mapping(request["selector"], "request.selector")
    if set(selector) != {"kind", "value"}:
        raise SpanToTestContractError("request.selector must contain only kind and value")
    kind = _text(selector["kind"], "request.selector.kind")
    if kind not in {"trace", "span", "finding", "outcome"}:
        raise SpanToTestContractError(
            "request.selector.kind must be trace, span, finding, or outcome"
        )
    return kind, _text(selector["value"], "request.selector.value")


def _validate_trace(trace: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    run = _mapping(trace.get("run"), "trace.run")
    _text(run.get("run_id"), "trace.run.run_id")
    raw_steps = _list(trace.get("steps"), "trace.steps")
    if not raw_steps or len(raw_steps) > _MAX_STEPS:
        raise SpanToTestContractError(f"trace.steps must contain 1..{_MAX_STEPS} steps")
    steps = [_mapping(step, f"trace.steps[{index}]") for index, step in enumerate(raw_steps)]
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, step in enumerate(steps):
        step_id = _text(step.get("step_id"), f"trace.steps[{index}].step_id")
        _text(step.get("kind"), f"trace.steps[{index}].kind")
        if step_id in by_id:
            raise SpanToTestContractError(f"duplicate step_id: {step_id}")
        by_id[step_id] = step
    for step in steps:
        step_id = str(step["step_id"])
        for relation in ("parent_step_id", "caused_by"):
            reference = step.get(relation)
            if reference is not None and reference not in by_id:
                raise SpanToTestContractError(f"step {step_id} has missing {relation}: {reference}")
        attributes = step.get("attributes") or {}
        if isinstance(attributes, Mapping):
            target = attributes.get("harness.hook.target_step_id")
            if target is not None and target not in by_id:
                raise SpanToTestContractError(
                    f"step {step_id} has missing harness.hook.target_step_id: {target}"
                )
    _reject_cycles(steps, by_id)
    return run, steps


def _reject_cycles(steps: list[Mapping[str, Any]], by_id: Mapping[str, Mapping[str, Any]]) -> None:
    edges: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        step_id = str(step["step_id"])
        for relation in ("parent_step_id", "caused_by"):
            reference = step.get(relation)
            if isinstance(reference, str):
                edges[step_id].append(reference)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise SpanToTestContractError(f"trace contains a dependency cycle at step {step_id}")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in edges.get(step_id, []):
            if dependency in by_id:
                visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in by_id:
        visit(step_id)


def _has_failure(step: Mapping[str, Any]) -> bool:
    if step.get("status") in _FAILURE_STATUSES:
        return True
    attributes = step.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        return False
    return bool(
        attributes.get("harness.hook.verdict") == "deny"
        or attributes.get("error.type")
        or attributes.get("exception.type")
        or any(key in attributes for key in _FINDING_KEYS)
    )


def _failure_assertion(step: Mapping[str, Any]) -> tuple[str, str] | None:
    status = step.get("status")
    if status in _FAILURE_STATUSES:
        return "span_status", str(status)
    attributes = step.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        return None
    if attributes.get("harness.hook.verdict") == "deny":
        return "hook_verdict", "deny"
    error_class = attributes.get("error.type") or attributes.get("exception.type")
    if error_class is not None:
        return "error_class", _safe_name(error_class, "error")
    if any(key in attributes for key in _FINDING_KEYS):
        return "finding", "present"
    return None


def _first_failing_descendant(
    steps: list[Mapping[str, Any]], selected_id: str
) -> tuple[Mapping[str, Any], str, str] | None:
    children: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        parent = step.get("parent_step_id")
        if isinstance(parent, str):
            children[parent].append(str(step["step_id"]))

    descendants: set[str] = set()
    stack = list(reversed(children.get(selected_id, [])))
    while stack:
        current = stack.pop()
        if current in descendants:
            continue
        descendants.add(current)
        stack.extend(reversed(children.get(current, [])))

    for step in steps:
        if step["step_id"] not in descendants:
            continue
        assertion = _failure_assertion(step)
        if assertion is not None:
            return step, *assertion
    return None


def _select_step(
    run: Mapping[str, Any], steps: list[Mapping[str, Any]], selector_kind: str, selector_value: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str]:
    if selector_kind == "trace":
        if selector_value != run["run_id"]:
            raise SpanToTestContractError("trace selector does not match trace.run.run_id")
        for step in steps:
            assertion = _failure_assertion(step)
            if assertion is not None:
                return step, step, *assertion
        outcome = _mapping(run.get("outcome") or {}, "trace.run.outcome")
        if outcome.get("status") in _FAILURE_OUTCOMES:
            return steps[-1], steps[-1], "run_outcome", str(outcome["status"])
        raise SpanToTestContractError("selected trace has no supported failure signal")
    if selector_kind == "span":
        for step in steps:
            if step["step_id"] == selector_value:
                assertion = _failure_assertion(step)
                if assertion is None:
                    descendant = _first_failing_descendant(steps, str(step["step_id"]))
                    if descendant is None:
                        raise SpanToTestContractError(
                            "selected span subtree has no supported failure signal"
                        )
                    target, assertion_kind, expected = descendant
                    return step, target, assertion_kind, expected
                return step, step, *assertion
        raise SpanToTestContractError("span selector did not match a step")
    if selector_kind == "finding":
        for step in steps:
            attributes = step.get("attributes") or {}
            if isinstance(attributes, Mapping) and any(
                attributes.get(key) == selector_value for key in _FINDING_KEYS
            ):
                return step, step, "finding", "present"
        raise SpanToTestContractError("finding selector did not match a step")
    outcome = _mapping(run.get("outcome") or {}, "trace.run.outcome")
    if outcome.get("status") == selector_value:
        selected = next((step for step in steps if _has_failure(step)), steps[-1])
        return selected, selected, "run_outcome", selector_value
    for step in steps:
        if step.get("status") == selector_value:
            return step, step, "span_status", selector_value
    raise SpanToTestContractError("outcome selector did not match run outcome or span status")


def _closure(steps: list[Mapping[str, Any]], selected_id: str) -> set[str]:
    by_id = {str(step["step_id"]): step for step in steps}
    children: dict[str, list[str]] = defaultdict(list)
    caused_dependents: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        step_id = str(step["step_id"])
        parent = step.get("parent_step_id")
        if isinstance(parent, str):
            children[parent].append(step_id)
        caused_by = step.get("caused_by")
        if isinstance(caused_by, str):
            caused_dependents[caused_by].append(step_id)
        attributes = step.get("attributes") or {}
        if isinstance(attributes, Mapping):
            target = attributes.get("harness.hook.target_step_id")
            if isinstance(target, str):
                caused_dependents[target].append(step_id)

    included = {selected_id}
    stack = [selected_id]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child not in included:
                included.add(child)
                stack.append(child)

    changed = True
    while changed:
        changed = False
        for step_id in tuple(included):
            step = by_id[step_id]
            for relation in ("parent_step_id", "caused_by"):
                reference = step.get(relation)
                if isinstance(reference, str) and reference not in included:
                    included.add(reference)
                    changed = True
            attributes = step.get("attributes") or {}
            if isinstance(attributes, Mapping):
                hook_target = attributes.get("harness.hook.target_step_id")
                if isinstance(hook_target, str) and hook_target not in included:
                    included.add(hook_target)
                    changed = True
            for dependent in caused_dependents.get(step_id, []):
                dependent_step = by_id[dependent]
                is_control_evidence = dependent_step.get("kind") in {"hook", "decision"}
                if dependent not in included and (
                    is_control_evidence or _has_failure(dependent_step)
                ):
                    included.add(dependent)
                    changed = True
    return included


def _error_class(step: Mapping[str, Any]) -> str | None:
    attributes = step.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        return None
    value = attributes.get("error.type") or attributes.get("exception.type")
    return _safe_name(value, "error") if value is not None else None


def generate_span_to_test(
    trace: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce one failing span subtree to deterministic inert fixture data."""
    source = deepcopy(trace)
    selector_kind, selector_value = _validate_request(request)
    run, steps = _validate_trace(source)
    selected, assertion_target, assertion_kind, expected = _select_step(
        run, steps, selector_kind, selector_value
    )
    selected_id = str(selected["step_id"])
    assertion_target_id = str(assertion_target["step_id"])
    included_ids = _closure(steps, assertion_target_id)
    selected_subtree_has_failure = any(
        _has_failure(step) for step in steps if step["step_id"] in included_ids
    )
    run_outcome = _mapping(run.get("outcome") or {}, "trace.run.outcome")
    run_outcome_is_assertion = assertion_kind == "run_outcome"
    if not selected_subtree_has_failure and not (
        run_outcome_is_assertion and run_outcome.get("status") in _FAILURE_OUTCOMES
    ):
        raise SpanToTestContractError("selected span closure has no supported failure signal")

    transformations: list[dict[str, str]] = []
    _record_identity(transformations, source_step_id=selected_id, path="run/run_id")
    for timestamp_key in ("started_at", "ended_at"):
        if timestamp_key in run:
            _record_timestamp(
                transformations,
                source_step_id=selected_id,
                path=f"run/{timestamp_key}",
            )
    for key in ("model", "workspace", "plan"):
        if key in run:
            _record_removed(
                transformations,
                source_step_id=selected_id,
                path=f"run/{key}",
                value=run[key],
            )
    if "outcome" in run and isinstance(run["outcome"], Mapping):
        for key, value in sorted(run["outcome"].items()):
            if key != "status":
                _record_removed(
                    transformations,
                    source_step_id=selected_id,
                    path=f"run/outcome/{key}",
                    value=value,
                )
    reduced_spans: list[dict[str, Any]] = []
    for sequence, step in enumerate(steps):
        source_id = str(step["step_id"])
        if source_id not in included_ids:
            continue
        _record_identity(transformations, source_step_id=source_id, path="step_id")
        for identity_key in ("parent_step_id", "caused_by", "subagent_id"):
            if isinstance(step.get(identity_key), str):
                _record_identity(
                    transformations,
                    source_step_id=source_id,
                    path=identity_key,
                )
        for timestamp_key in ("started_at", "ended_at"):
            if timestamp_key in step:
                _record_timestamp(
                    transformations,
                    source_step_id=source_id,
                    path=timestamp_key,
                )
        span: dict[str, Any] = {
            "span_id": _pseudonym("span", source_id),
            "sequence": sequence,
            "kind": _safe_name(step["kind"], "kind"),
            "status": step.get("status"),
            "attributes": _safe_attributes(step, transformations),
        }
        parent = step.get("parent_step_id")
        caused_by = step.get("caused_by")
        subagent = step.get("subagent_id")
        span["parent_span_id"] = _pseudonym("span", parent) if isinstance(parent, str) else None
        span["caused_by_span_id"] = (
            _pseudonym("span", caused_by) if isinstance(caused_by, str) else None
        )
        span["agent_context_id"] = (
            _pseudonym("agent", subagent) if isinstance(subagent, str) else None
        )
        error_class = _error_class(step)
        if error_class is not None:
            span["error_class"] = error_class
        reduced_spans.append(span)

    source_digest = _digest(source)
    request_digest = _digest(request)
    fixture_id = _pseudonym("fixture", f"{source_digest}\0{request_digest}")
    fixture = {
        "schema_version": FIXTURE_VERSION,
        "fixture_id": fixture_id,
        "source": {
            "trace_id": _pseudonym("trace", str(run["run_id"])),
            "selected_span_id": _pseudonym("span", selected_id),
            "selector_kind": selector_kind,
        },
        "resource": {
            "service.name": "plumbline.synthetic-replay",
            "plumbline.source.version": _safe_name(source.get("plumbline_version"), "version"),
            "harness.name": _safe_name(
                _mapping(run.get("harness") or {}, "trace.run.harness").get("name"),
                "harness",
            ),
        },
        "spans": reduced_spans,
        "expected_assertion": {
            "kind": assertion_kind,
            "target_span_id": _pseudonym("span", assertion_target_id),
            "operator": "equals",
            "expected": _safe_name(expected, "expected"),
        },
        "safety": {
            "artifact_mode": "inert_data_only",
            "contains_executable_payloads": False,
            "prohibited_capabilities": [
                "credential_access",
                "filesystem_mutation",
                "network",
                "provider_call",
                "shell",
            ],
        },
    }
    transformations.sort(
        key=lambda item: (item["span_id"], item["field_id"], item["category"], item["action"])
    )
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "fixture_id": fixture_id,
        "source": {
            "trace_id": _pseudonym("trace", str(run["run_id"])),
            "selected_span_id": _pseudonym("span", selected_id),
            "source_fingerprint": _pseudonym("source", source_digest),
            "request_fingerprint": _pseudonym("request", request_digest),
        },
        "reduction": {
            "source_span_count": len(steps),
            "retained_span_count": len(reduced_spans),
            "removed_span_count": len(steps) - len(reduced_spans),
            "algorithm": "selected_subtree_plus_ancestry_and_causal_closure_v1",
        },
        "transformations": transformations,
        "claim_ceiling": (
            "Structural reproduction is not proof of provider or production reproduction."
        ),
    }
    validate_replay_fixture(fixture)
    return fixture, receipt


def validate_replay_fixture(fixture: Mapping[str, Any]) -> None:
    """Fail closed if a fixture is malformed or loses its inert-data boundary."""
    if fixture.get("schema_version") != FIXTURE_VERSION:
        raise SpanToTestContractError(f"fixture.schema_version must be {FIXTURE_VERSION}")
    safety = _mapping(fixture.get("safety"), "fixture.safety")
    if safety.get("artifact_mode") != "inert_data_only":
        raise SpanToTestContractError("fixture must declare inert_data_only artifact mode")
    if safety.get("contains_executable_payloads") is not False:
        raise SpanToTestContractError("fixture must exclude executable payloads")
    spans = _list(fixture.get("spans"), "fixture.spans")
    if not spans:
        raise SpanToTestContractError("fixture.spans must not be empty")
    ids: set[str] = set()
    previous_sequence = -1
    forbidden_keys = re.compile(
        r"(?i)(?:command|arguments?|results?|prompt|completion|payload|body)"
    )
    for index, raw_span in enumerate(spans):
        span = _mapping(raw_span, f"fixture.spans[{index}]")
        span_id = _text(span.get("span_id"), f"fixture.spans[{index}].span_id")
        if span_id in ids:
            raise SpanToTestContractError(f"duplicate fixture span_id: {span_id}")
        ids.add(span_id)
        sequence = span.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= previous_sequence
        ):
            raise SpanToTestContractError("fixture span sequence must be strictly increasing")
        previous_sequence = sequence
        for path, _value in _leaf_values(span):
            if forbidden_keys.search(path) and not any(
                allowed in path for allowed in ("argument_keys", "result_kind")
            ):
                raise SpanToTestContractError(f"fixture contains forbidden payload field: {path}")
    for raw_span in spans:
        span = _mapping(raw_span, "fixture.spans[]")
        for relation in ("parent_span_id", "caused_by_span_id"):
            reference = span.get(relation)
            if reference is not None and reference not in ids:
                raise SpanToTestContractError(f"fixture has missing {relation}: {reference}")
        attributes = _mapping(span.get("attributes"), "fixture.spans[].attributes")
        raw_tool = attributes.get("tool")
        if raw_tool is not None:
            tool = _mapping(raw_tool, "fixture.spans[].attributes.tool")
            argument_keys = _list(
                tool.get("argument_keys"),
                "fixture.spans[].attributes.tool.argument_keys",
            )
            validated_keys = [
                _text(
                    key,
                    f"fixture.spans[].attributes.tool.argument_keys[{index}]",
                    maximum=_MAX_SAFE_TEXT_LENGTH,
                )
                for index, key in enumerate(argument_keys)
            ]
            if len(set(validated_keys)) != len(validated_keys):
                raise SpanToTestContractError("fixture tool argument_keys must be unique")
            normalized_identities = {_argument_key_identity(key) for key in validated_keys}
            if len(normalized_identities) != len(validated_keys):
                raise SpanToTestContractError(
                    "fixture tool argument_keys contain a normalization collision"
                )
            if any(not _SAFE_NAME.fullmatch(key) for key in validated_keys):
                raise SpanToTestContractError(
                    "fixture tool argument_keys must be bounded structural text"
                )
        hook_target = attributes.get("harness.hook.target_span_id")
        if hook_target is not None:
            target_id = _text(
                hook_target,
                "fixture.spans[].attributes.harness.hook.target_span_id",
            )
            if not _OPAQUE_ID.fullmatch(target_id):
                raise SpanToTestContractError("fixture hook target_span_id must be an opaque ID")
            if target_id not in ids:
                raise SpanToTestContractError(
                    f"fixture has missing hook target_span_id: {target_id}"
                )
    assertion = _mapping(fixture.get("expected_assertion"), "fixture.expected_assertion")
    if set(assertion) != _ASSERTION_FIELDS:
        raise SpanToTestContractError(
            "fixture.expected_assertion must contain exactly kind, target_span_id, "
            "operator, and expected"
        )
    assertion_kind = _text(assertion.get("kind"), "fixture.expected_assertion.kind")
    if assertion_kind not in _ASSERTION_KINDS:
        raise SpanToTestContractError(
            "fixture.expected_assertion.kind must be one of span_failure, span_status, "
            "hook_verdict, error_class, finding, or run_outcome"
        )
    assertion_target = _text(
        assertion.get("target_span_id"),
        "fixture.expected_assertion.target_span_id",
    )
    if not _OPAQUE_ID.fullmatch(assertion_target):
        raise SpanToTestContractError(
            "fixture.expected_assertion.target_span_id must be an opaque ID"
        )
    if assertion_target not in ids:
        raise SpanToTestContractError("fixture assertion target is missing")
    if assertion.get("operator") != "equals":
        raise SpanToTestContractError("fixture.expected_assertion.operator must be equals")
    expected = _text(
        assertion.get("expected"),
        "fixture.expected_assertion.expected",
        maximum=_MAX_SAFE_TEXT_LENGTH,
    )
    if not _SAFE_NAME.fullmatch(expected):
        raise SpanToTestContractError(
            "fixture.expected_assertion.expected must be bounded structural text"
        )
    for path, value in _leaf_values(fixture):
        if isinstance(value, str) and not _OPAQUE_ID.fullmatch(value):
            category = _sensitive_text_category(value)
            if category is not None:
                raise SpanToTestContractError(
                    f"fixture contains unsafe structural text at {path}: {category}"
                )


def evaluate_expected_assertion(fixture: Mapping[str, Any]) -> bool:
    """Evaluate only the fixture's local structural assertion."""
    validate_replay_fixture(fixture)
    assertion = _mapping(fixture["expected_assertion"], "fixture.expected_assertion")
    target = next(
        span for span in fixture["spans"] if span["span_id"] == assertion["target_span_id"]
    )
    kind = assertion["kind"]
    expected = assertion["expected"]
    if kind in {"span_failure", "span_status"}:
        if expected == "failure":
            return bool(target.get("status") in _FAILURE_STATUSES or target.get("error_class"))
        return target.get("status") == expected
    if kind == "hook_verdict":
        return target.get("attributes", {}).get("harness.hook.verdict") == expected
    if kind == "error_class":
        return target.get("error_class") == expected
    if kind == "finding":
        return expected == "present"
    if kind == "run_outcome":
        return expected in _FAILURE_OUTCOMES
    return False


def load_trace(path: Path) -> dict[str, Any]:
    """Load one bounded JSON trace without following a non-file source."""
    try:
        if path.is_symlink() or not path.is_file():
            raise SpanToTestContractError("source trace must be a regular non-symlink file")
        if path.stat().st_size > _MAX_TRACE_BYTES:
            raise SpanToTestContractError(
                f"source trace exceeds the {_MAX_TRACE_BYTES}-byte safety limit"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpanToTestContractError(f"cannot read source trace: {exc}") from exc
    return dict(_mapping(value, "trace"))


def load_replay_fixture(path: Path) -> dict[str, Any]:
    """Load and validate an inert replay fixture."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpanToTestContractError(f"cannot read replay fixture: {exc}") from exc
    fixture = dict(_mapping(value, "fixture"))
    validate_replay_fixture(fixture)
    return fixture


def render_pytest_skeleton(fixture_filename: str) -> str:
    """Render a local-only regression test for a sibling fixture file."""
    safe_filename = Path(fixture_filename).name
    if safe_filename != fixture_filename or not safe_filename.endswith(".json"):
        raise SpanToTestContractError("pytest skeleton requires a sibling JSON fixture filename")
    return (
        '"""Generated by Plumbline from inert synthetic fixture data."""\n\n'
        "from pathlib import Path\n\n"
        "from plumbline.span_to_test import evaluate_expected_assertion, load_replay_fixture\n\n\n"
        f'FIXTURE = Path(__file__).with_name("{safe_filename}")\n\n\n'
        "def test_sanitized_replay_preserves_failure_signal() -> None:\n"
        "    fixture = load_replay_fixture(FIXTURE)\n"
        "    assert evaluate_expected_assertion(fixture)\n"
    )


def preflight_output_paths(
    source: Path, outputs: Sequence[Path], *, overwrite: bool = False
) -> list[Path]:
    """Resolve exact explicit outputs and reject aliases, symlinks, and implicit parents."""
    if not outputs:
        raise SpanToTestContractError("at least one explicit output path is required")
    source_resolved = source.resolve(strict=True)
    resolved: list[Path] = []
    for requested_output in outputs:
        output = (
            requested_output if requested_output.is_absolute() else Path.cwd() / requested_output
        )
        if output.exists() and output.is_symlink():
            raise SpanToTestContractError(f"refusing symlink output path: {output}")
        parent = output.parent.resolve(strict=True)
        if not parent.is_dir():
            raise SpanToTestContractError(f"output parent is not a directory: {parent}")
        target = parent / output.name
        if target == source_resolved or (target.exists() and target.samefile(source_resolved)):
            raise SpanToTestContractError("output path must not overwrite the source trace")
        aliases_existing_output = any(
            target.exists() and prior.exists() and target.samefile(prior) for prior in resolved
        )
        if target in resolved or aliases_existing_output:
            raise SpanToTestContractError("output paths must be distinct")
        if target.exists() and not overwrite:
            raise SpanToTestContractError(f"output already exists (use --force): {target}")
        if target.exists() and not target.is_file():
            raise SpanToTestContractError(f"output must be a regular file: {target}")
        resolved.append(target)
    return resolved


def write_json_output(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    """Write one preflighted JSON output without following a final symlink."""
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def write_text_output(path: Path, value: str, *, overwrite: bool = False) -> None:
    """Write one preflighted UTF-8 text output without following a final symlink."""
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
