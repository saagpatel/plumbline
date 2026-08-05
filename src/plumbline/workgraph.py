# ruff: noqa: ANN401, C901, EM101, EM102, FURB162, PERF401, PLR0912, PLR0915, SIM114, TRY003
"""Passive WorkGraphV1 shadow-runtime reconciliation.

This adapter observes a preregistered compiled plan.  It never dispatches,
leases, retries, rewrites a graph, or mutates an external control plane.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from plumbline.trajectory import canonical_json_bytes, sha256_file

if TYPE_CHECKING:
    from pathlib import Path

EVENTS_VERSION = "WorkGraphObservedEventsV1"
REPORT_VERSION = "WorkGraphShadowTraceV1"
REGISTRATION_VERSION = "WorkGraphShadowPilotRegistrationV1"
_START = "started"
_TERMINAL = {"blocked", "completed", "failed"}
_SHA256_LENGTH = 71
_MAX_TEXT_LENGTH = 512
_CONTROL_CHARACTER_LIMIT = 32
_METADATA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")
_METADATA_REF_RE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9._:/#@+-]{0,255}$")


class WorkGraphContractError(ValueError):
    """Stable fail-closed WorkGraph shadow validation error."""


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkGraphContractError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorkGraphContractError(f"{path} must be an array")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkGraphContractError(f"{path} must be a non-empty string")
    result = value.strip()
    if (
        result != value
        or len(result) > _MAX_TEXT_LENGTH
        or any(ord(character) < _CONTROL_CHARACTER_LIMIT for character in result)
    ):
        raise WorkGraphContractError(f"{path} must be bounded single-line metadata")
    return result


def _metadata_id(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _METADATA_ID_RE.fullmatch(result):
        raise WorkGraphContractError(f"{path} must be an opaque metadata identifier")
    return result


def _metadata_ref(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _METADATA_REF_RE.fullmatch(result):
        raise WorkGraphContractError(f"{path} must be an opaque metadata reference")
    return result


def _metadata_id_array(value: Any, path: str) -> list[str]:
    result = [
        _metadata_id(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    ]
    if len(result) != len(set(result)):
        raise WorkGraphContractError(f"{path} must not contain duplicates")
    return result


def _metadata_ref_array(value: Any, path: str) -> list[str]:
    result = [
        _metadata_ref(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    ]
    if len(result) != len(set(result)):
        raise WorkGraphContractError(f"{path} must not contain duplicates")
    return result


def _timestamp(value: Any, path: str) -> datetime:
    raw = _text(value, path)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkGraphContractError(f"{path} must be an ISO 8601 timestamp") from exc
    if result.tzinfo is None:
        raise WorkGraphContractError(f"{path} must include a timezone")
    return result.astimezone(UTC)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkGraphContractError(f"{path} must be a non-negative integer")
    return value


def _number_or_none(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise WorkGraphContractError(f"{path} must be null or a non-negative number")
    return float(value)


def _digest(value: Any, path: str) -> str:
    result = _text(value, path)
    if len(result) != _SHA256_LENGTH or not result.startswith("sha256:"):
        raise WorkGraphContractError(f"{path} must be sha256:<64 lowercase hex>")
    if any(character not in "0123456789abcdef" for character in result[7:]):
        raise WorkGraphContractError(f"{path} must be sha256:<64 lowercase hex>")
    return result


def _keys(value: dict[str, Any], path: str, *, required: set[str]) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing:
        raise WorkGraphContractError(f"{path} missing key(s): {', '.join(missing)}")
    if extra:
        raise WorkGraphContractError(f"{path} has unsupported key(s): {', '.join(extra)}")


def _compiled_lanes(compiled: Any) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    root = _object(compiled, "compiled_plan")
    if root.get("status") != "compiled":
        raise WorkGraphContractError("compiled_plan.status must be compiled")
    _metadata_id(root.get("graph_id"), "compiled_plan.graph_id")
    waves = _array(root.get("waves"), "compiled_plan.waves")
    lanes: dict[str, dict[str, Any]] = {}
    wave_index: dict[str, int] = {}
    for raw_wave in waves:
        wave = _object(raw_wave, "compiled_plan.waves[]")
        index = wave.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise WorkGraphContractError(
                "compiled_plan.waves[].index must be a non-negative integer"
            )
        for raw_lane in _array(wave.get("lanes"), "compiled_plan.waves[].lanes"):
            lane = _object(raw_lane, "compiled_plan.waves[].lanes[]")
            lane_id = _metadata_id(lane.get("id"), "compiled_plan.waves[].lanes[].id")
            if lane_id in lanes:
                raise WorkGraphContractError(f"compiled plan contains duplicate lane: {lane_id}")
            dependencies = _metadata_id_array(
                lane.get("depends_on"), "compiled_plan.waves[].lanes[].depends_on"
            )
            mode = lane.get("mutation_mode")
            if mode not in {"strict-read-only", "report-only", "repair-approved"}:
                raise WorkGraphContractError(f"compiled lane {lane_id} has invalid mutation_mode")
            lanes[lane_id] = {**lane, "id": lane_id, "depends_on": dependencies}
            wave_index[lane_id] = index
    for lane_id, lane in lanes.items():
        unknown = sorted(set(lane["depends_on"]) - lanes.keys())
        if unknown:
            raise WorkGraphContractError(
                f"compiled lane {lane_id} depends on unknown lane(s): {', '.join(unknown)}"
            )
    if not lanes:
        raise WorkGraphContractError("compiled plan must contain at least one lane")
    return lanes, wave_index


def _registration_bindings(
    registration: Any,
    lanes: dict[str, dict[str, Any]],
    wave_index: dict[str, int],
    *,
    graph_id: str,
    compiled_plan_digest: str,
) -> tuple[dict[str, str], datetime]:
    root = _object(registration, "registration")
    _keys(
        root,
        "registration",
        required={
            "schema_version",
            "registered_at",
            "graph_id",
            "source_graph_digest",
            "compiled_plan_digest",
            "compiler_version",
            "lane_bindings",
            "metric_contract",
            "admission",
            "claim_ceiling",
        },
    )
    if root["schema_version"] != REGISTRATION_VERSION:
        raise WorkGraphContractError(f"registration.schema_version must be {REGISTRATION_VERSION}")
    registered_at = _timestamp(root["registered_at"], "registration.registered_at")
    if _metadata_id(root["graph_id"], "registration.graph_id") != graph_id:
        raise WorkGraphContractError("registration.graph_id does not match compiled plan")
    _digest(root["source_graph_digest"], "registration.source_graph_digest")
    if (
        _digest(root["compiled_plan_digest"], "registration.compiled_plan_digest")
        != compiled_plan_digest
    ):
        raise WorkGraphContractError("registration does not bind the exact compiled plan")
    _metadata_id(root["compiler_version"], "registration.compiler_version")
    if root["admission"] != "compiled_and_digest_bound_before_dispatch":
        raise WorkGraphContractError("registration.admission is not prospective")
    _text(root["claim_ceiling"], "registration.claim_ceiling")
    metric_contract = _array(root["metric_contract"], "registration.metric_contract")
    if not metric_contract:
        raise WorkGraphContractError("registration.metric_contract must contain text entries")
    normalized_metrics = [
        _text(item, f"registration.metric_contract[{index}]")
        for index, item in enumerate(metric_contract)
    ]
    if len(normalized_metrics) != len(set(normalized_metrics)):
        raise WorkGraphContractError("registration.metric_contract must not contain duplicates")

    bindings: dict[str, str] = {}
    for index, raw_binding in enumerate(
        _array(root["lane_bindings"], "registration.lane_bindings")
    ):
        path = f"registration.lane_bindings[{index}]"
        binding = _object(raw_binding, path)
        _keys(
            binding,
            path,
            required={"lane_id", "wave", "agent_path", "depends_on", "mutation_mode"},
        )
        lane_id = _metadata_id(binding["lane_id"], f"{path}.lane_id")
        if lane_id in bindings:
            raise WorkGraphContractError(f"registration contains duplicate lane: {lane_id}")
        if lane_id not in lanes:
            raise WorkGraphContractError(f"registration contains unknown lane: {lane_id}")
        if binding["wave"] != wave_index[lane_id]:
            raise WorkGraphContractError(f"registration wave mismatch for lane: {lane_id}")
        dependencies = _metadata_id_array(binding["depends_on"], f"{path}.depends_on")
        if dependencies != lanes[lane_id]["depends_on"]:
            raise WorkGraphContractError(f"registration dependency mismatch for lane: {lane_id}")
        if binding["mutation_mode"] != lanes[lane_id]["mutation_mode"]:
            raise WorkGraphContractError(f"registration mutation mode mismatch for lane: {lane_id}")
        bindings[lane_id] = _metadata_ref(binding["agent_path"], f"{path}.agent_path")
    if set(bindings) != set(lanes):
        missing = sorted(set(lanes) - set(bindings))
        raise WorkGraphContractError(
            f"registration missing compiled lane binding(s): {', '.join(missing)}"
        )
    return bindings, registered_at


def load_json(path: Path) -> Any:
    """Load one JSON document with a stable contract error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkGraphContractError(f"cannot read JSON from {path}: {exc}") from exc


def evaluate_workgraph_shadow(
    compiled_plan: Any,
    registration: Any,
    observed_events: Any,
    *,
    compiled_plan_digest: str,
    registration_digest: str,
) -> dict[str, Any]:
    """Compare observed metadata-only transitions with a frozen compiled plan."""
    compiled_plan_digest = _digest(compiled_plan_digest, "compiled_plan_digest")
    lanes, wave_index = _compiled_lanes(compiled_plan)
    compiled = _object(compiled_plan, "compiled_plan")
    graph_id = _metadata_id(compiled["graph_id"], "compiled_plan.graph_id")
    expected_agents, registered_at = _registration_bindings(
        registration,
        lanes,
        wave_index,
        graph_id=graph_id,
        compiled_plan_digest=compiled_plan_digest,
    )
    _digest(registration_digest, "registration_digest")
    observed = _object(observed_events, "observed_events")
    _keys(
        observed,
        "observed_events",
        required={
            "schema_version",
            "graph_id",
            "compiled_plan_digest",
            "registration_digest",
            "events",
        },
    )
    if observed["schema_version"] != EVENTS_VERSION:
        raise WorkGraphContractError(f"observed_events.schema_version must be {EVENTS_VERSION}")
    observed_graph_id = _metadata_id(observed["graph_id"], "observed_events.graph_id")
    if observed_graph_id != graph_id:
        raise WorkGraphContractError("observed_events.graph_id does not match compiled plan")
    if (
        _digest(observed["compiled_plan_digest"], "observed_events.compiled_plan_digest")
        != compiled_plan_digest
    ):
        raise WorkGraphContractError(
            "observed_events.compiled_plan_digest does not match the exact compiled plan"
        )
    if (
        _digest(observed["registration_digest"], "observed_events.registration_digest")
        != registration_digest
    ):
        raise WorkGraphContractError(
            "observed_events.registration_digest does not match the exact registration"
        )

    events_by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ids: set[str] = set()
    violations: list[dict[str, str]] = []
    all_times: list[datetime] = []
    total_tokens = 0
    total_cost = 0.0
    cost_complete = True
    token_complete = True
    retries = 0
    terminal_event_count = 0
    terminal_evidence_count = 0
    unresolved_unknown_lanes: set[str] = set()
    mapping_drift_lanes: set[str] = set()
    for index, raw_event in enumerate(_array(observed["events"], "observed_events.events")):
        path = f"observed_events.events[{index}]"
        event = _object(raw_event, path)
        _keys(
            event,
            path,
            required={
                "event_id",
                "lane_id",
                "agent_path",
                "event_type",
                "observed_at",
                "evidence_refs",
                "mutation_observed",
                "duplicate_signature",
                "tokens",
                "cost_usd",
                "retry_count",
                "unknown_reason",
            },
        )
        event_id = _metadata_id(event["event_id"], f"{path}.event_id")
        if event_id in event_ids:
            raise WorkGraphContractError(f"duplicate observed event_id: {event_id}")
        event_ids.add(event_id)
        lane_id = _metadata_id(event["lane_id"], f"{path}.lane_id")
        agent_path = _metadata_ref(event["agent_path"], f"{path}.agent_path")
        if event["event_type"] not in {_START, *_TERMINAL}:
            raise WorkGraphContractError(f"{path}.event_type is unsupported")
        timestamp = _timestamp(event["observed_at"], f"{path}.observed_at")
        refs = _metadata_ref_array(event["evidence_refs"], f"{path}.evidence_refs")
        if not isinstance(event["mutation_observed"], bool):
            raise WorkGraphContractError(f"{path}.mutation_observed must be a boolean")
        if event["duplicate_signature"] is not None:
            _metadata_ref(event["duplicate_signature"], f"{path}.duplicate_signature")
        tokens = event["tokens"]
        if tokens is not None:
            _integer(tokens, f"{path}.tokens")
        cost = _number_or_none(event["cost_usd"], f"{path}.cost_usd")
        retry_count = _integer(event["retry_count"], f"{path}.retry_count")
        if event["unknown_reason"] is not None:
            _text(event["unknown_reason"], f"{path}.unknown_reason")
        if lane_id not in lanes:
            violations.append(
                {
                    "code": "UNREGISTERED_LANE",
                    "lane_id": lane_id,
                    "detail": "observed event is not declared in the compiled plan",
                }
            )
            continue

        all_times.append(timestamp)
        if timestamp < registered_at:
            violations.append(
                {
                    "code": "EVENT_BEFORE_REGISTRATION",
                    "lane_id": lane_id,
                    "detail": "observed event timestamp precedes the frozen registration",
                }
            )
        if agent_path != expected_agents[lane_id]:
            mapping_drift_lanes.add(lane_id)
        if event["event_type"] in _TERMINAL:
            terminal_event_count += 1
            if refs:
                terminal_evidence_count += 1
        if tokens is None:
            token_complete = False
        else:
            total_tokens += tokens
        if cost is None:
            cost_complete = False
        else:
            total_cost += cost
        retries += retry_count
        if event["unknown_reason"] is not None:
            unresolved_unknown_lanes.add(lane_id)
        events_by_lane[lane_id].append({**event, "_timestamp": timestamp})

    for lane_id in sorted(mapping_drift_lanes):
        violations.append(
            {
                "code": "LANE_AGENT_MAPPING_DRIFT",
                "lane_id": lane_id,
                "detail": "observed agent does not match the frozen registration",
            }
        )
    for lane_id in sorted(unresolved_unknown_lanes):
        violations.append(
            {
                "code": "UNRESOLVED_UNKNOWN",
                "lane_id": lane_id,
                "detail": "an observed event retains an unresolved unknown reason",
            }
        )

    terminal_times: dict[str, datetime] = {}
    start_times: dict[str, datetime] = {}
    incomplete_lanes: list[str] = []
    failed_lanes: list[str] = []
    blocked_lanes: list[str] = []
    lane_agents: dict[str, str] = dict(expected_agents)
    for lane_id, lane in lanes.items():
        lane_events = events_by_lane[lane_id]
        starts = [event for event in lane_events if event["event_type"] == _START]
        terminals = [event for event in lane_events if event["event_type"] in _TERMINAL]
        if len(starts) != 1:
            code = "MISSING_START" if not starts else "DUPLICATE_START"
            violations.append(
                {"code": code, "lane_id": lane_id, "detail": f"observed {len(starts)} start events"}
            )
        if len(terminals) != 1:
            incomplete_lanes.append(lane_id)
            if len(terminals) > 1:
                violations.append(
                    {
                        "code": "DUPLICATE_TERMINAL",
                        "lane_id": lane_id,
                        "detail": f"observed {len(terminals)} terminal events",
                    }
                )
        if starts:
            start_times[lane_id] = starts[0]["_timestamp"]
        if terminals:
            terminal = terminals[0]
            terminal_times[lane_id] = terminal["_timestamp"]
            if terminal["event_type"] == "failed":
                failed_lanes.append(lane_id)
            if terminal["event_type"] == "blocked":
                blocked_lanes.append(lane_id)
            if not terminal["evidence_refs"]:
                violations.append(
                    {
                        "code": "TERMINAL_EVIDENCE_MISSING",
                        "lane_id": lane_id,
                        "detail": "terminal event has no evidence reference",
                    }
                )
        if starts and terminals and terminals[0]["_timestamp"] < starts[0]["_timestamp"]:
            violations.append(
                {
                    "code": "TERMINAL_BEFORE_START",
                    "lane_id": lane_id,
                    "detail": "terminal timestamp precedes start timestamp",
                }
            )
        if lane["mutation_mode"] == "strict-read-only" and any(
            event["mutation_observed"] for event in lane_events
        ):
            violations.append(
                {
                    "code": "UNDECLARED_MUTATION",
                    "lane_id": lane_id,
                    "detail": "strict-read-only lane observed a mutation",
                }
            )
    for lane_id, start in start_times.items():
        for dependency in lanes[lane_id]["depends_on"]:
            dependency_terminal = terminal_times.get(dependency)
            if dependency_terminal is None:
                continue
            if start < dependency_terminal:
                violations.append(
                    {
                        "code": "DEPENDENCY_VIOLATION",
                        "lane_id": lane_id,
                        "detail": f"started before dependency {dependency} terminated",
                    }
                )

    signatures: dict[str, set[str]] = defaultdict(set)
    for lane_id, lane_events in events_by_lane.items():
        signatures_found = {
            event["duplicate_signature"]
            for event in lane_events
            if event["duplicate_signature"] is not None
        }
        for signature in signatures_found:
            signatures[signature].add(lane_id)
    duplicate_groups = [sorted(lane_ids) for lane_ids in signatures.values() if len(lane_ids) > 1]
    for group in sorted(duplicate_groups):
        violations.append(
            {
                "code": "DUPLICATE_WORK_SIGNATURE",
                "lane_id": ",".join(group),
                "detail": "multiple lanes reported the same duplicate signature",
            }
        )

    intervals_by_wave: dict[int, list[tuple[datetime, datetime]]] = defaultdict(list)
    for lane_id, start in start_times.items():
        end = terminal_times.get(lane_id)
        if end is not None:
            intervals_by_wave[wave_index[lane_id]].append((start, end))
    serialized_pairs = 0
    for intervals in intervals_by_wave.values():
        active_ends: list[datetime] = []
        ended_count = 0
        for start, end in sorted(intervals):
            while active_ends and active_ends[0] <= start:
                heapq.heappop(active_ends)
                ended_count += 1
            serialized_pairs += ended_count
            heapq.heappush(active_ends, end)

    unsafe_codes = {
        "DEPENDENCY_VIOLATION",
        "DUPLICATE_START",
        "DUPLICATE_TERMINAL",
        "EVENT_BEFORE_REGISTRATION",
        "LANE_AGENT_MAPPING_DRIFT",
        "TERMINAL_BEFORE_START",
        "UNDECLARED_MUTATION",
        "UNREGISTERED_LANE",
    }
    unknown_codes = {"MISSING_START", "TERMINAL_EVIDENCE_MISSING", "UNRESOLVED_UNKNOWN"}
    codes = {violation["code"] for violation in violations}
    if incomplete_lanes or codes & unknown_codes:
        disposition = "UNKNOWN"
    elif failed_lanes or blocked_lanes or codes & unsafe_codes:
        disposition = "NO_GO"
    elif duplicate_groups:
        disposition = "NO_GO"
    else:
        disposition = "GO"
    lane_count = len(lanes)
    terminal_count = len(terminal_times)
    wall_time_ms = None
    if all_times:
        wall_time_ms = int((max(all_times) - min(all_times)).total_seconds() * 1000)
    event_before_registration = "EVENT_BEFORE_REGISTRATION" in codes
    return {
        "schema_version": REPORT_VERSION,
        "graph_id": graph_id,
        "compiled_plan_digest": compiled_plan_digest,
        "registration_digest": registration_digest,
        "disposition": disposition,
        "coverage": terminal_count / lane_count if lane_count else 0.0,
        "evidence_completeness": (
            terminal_evidence_count / terminal_event_count if terminal_event_count else 0.0
        ),
        "metrics": {
            "declared_lanes": lane_count,
            "terminal_lanes": terminal_count,
            "failed_lanes": sorted(failed_lanes),
            "blocked_lanes": sorted(blocked_lanes),
            "incomplete_lanes": sorted(incomplete_lanes),
            "wall_time_ms": wall_time_ms,
            "observed_serialized_pairs": serialized_pairs,
            "duplicate_groups": sorted(duplicate_groups),
            "retry_count": retries,
            "tokens": total_tokens if token_complete else None,
            "cost_usd": total_cost if cost_complete else None,
        },
        "lane_agent_mapping": dict(sorted(lane_agents.items())),
        "prospectivity": {
            "registered_at": registered_at.isoformat().replace("+00:00", "Z"),
            "first_event_at": (
                min(all_times).isoformat().replace("+00:00", "Z") if all_times else None
            ),
            "digest_bound": True,
            "chronological": not event_before_registration,
            "external_time_authority": False,
        },
        "violations": sorted(
            violations, key=lambda item: (item["code"], item["lane_id"], item["detail"])
        ),
        "observed_events_digest": "sha256:"
        + hashlib.sha256(canonical_json_bytes(observed)).hexdigest(),
        "claim_ceiling": (
            "One passive shadow reconciliation with exact local registration and timestamp "
            "ordering. No external time notarization, dispatch, lease, runtime enforcement, "
            "optimality, causal-improvement, or production-readiness claim."
        ),
    }


def evaluate_workgraph_shadow_files(
    compiled_path: Path, registration_path: Path, events_path: Path
) -> dict[str, Any]:
    """Load exact files and bind the report to the compiled plan and registration."""
    return evaluate_workgraph_shadow(
        load_json(compiled_path),
        load_json(registration_path),
        load_json(events_path),
        compiled_plan_digest=sha256_file(compiled_path),
        registration_digest=sha256_file(registration_path),
    )
