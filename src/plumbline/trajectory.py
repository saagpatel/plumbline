# ruff: noqa: ANN401, C901, EM101, EM102, FURB162, PLR0912, PLR0915, PLR2004, TRY003
"""Privacy-safe outcome bindings for Plumbline traces.

``OutcomeBoundTrajectoryV1`` is deliberately a companion envelope.  A
Plumbline trace remains the portable decision DAG; this module binds its
digest to capability observations and independently attributable outcomes
without copying prompts, tool payloads, or outcome prose into telemetry.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "OutcomeBoundTrajectoryV1"
AGGREGATE_VERSION = "OutcomeBoundTrajectoryAggregateV1"
DECISION_VERSION = "OutcomeBoundTrajectoryDecisionV1"

_DIGEST_PREFIX = "sha256:"
_CAPABILITY_KINDS = {"agent", "harness", "mcp", "role", "skill", "tool"}
_CAPABILITY_STATES = {"available", "exposed", "adopted"}
_OUTCOME_STATUSES = {"aborted", "failed", "passed", "unknown"}
_CLAIM_STATES = {"observation", "correlation", "experiment", "causality"}
_METADATA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")
_METADATA_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#@+-]{0,255}$")


class TrajectoryContractError(ValueError):
    """Stable fail-closed validation error."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def sha256_value(value: Any) -> str:
    """Digest a JSON-compatible value using its canonical encoding."""
    return _DIGEST_PREFIX + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Digest one exact file without loading it into the envelope."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return _DIGEST_PREFIX + digest.hexdigest()


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryContractError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrajectoryContractError(f"{path} must be an array")
    return value


def _keys(
    value: dict[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required - optional)
    if missing:
        raise TrajectoryContractError(f"{path} missing key(s): {', '.join(missing)}")
    if extra:
        raise TrajectoryContractError(f"{path} has unsupported key(s): {', '.join(extra)}")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryContractError(f"{path} must be a non-empty string")
    result = value.strip()
    if len(result) > 512 or any(ord(character) < 32 for character in result):
        raise TrajectoryContractError(f"{path} must be bounded single-line metadata")
    return result


def _metadata_id(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _METADATA_ID_RE.fullmatch(result):
        raise TrajectoryContractError(f"{path} must be an opaque metadata identifier")
    return result


def _metadata_ref(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _METADATA_REF_RE.fullmatch(result):
        raise TrajectoryContractError(f"{path} must be an opaque metadata reference")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrajectoryContractError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryContractError(f"{path} must be a number")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise TrajectoryContractError(f"{path} must be >= {minimum}{suffix}")
    return result


def _timestamp(value: Any, path: str) -> datetime:
    raw = _text(value, path)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrajectoryContractError(f"{path} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise TrajectoryContractError(f"{path} must include a timezone")
    return parsed.astimezone(UTC)


def _digest(value: Any, path: str) -> str:
    raw = _text(value, path)
    if len(raw) != 71 or not raw.startswith(_DIGEST_PREFIX):
        raise TrajectoryContractError(f"{path} must be sha256:<64 lowercase hex>")
    hex_part = raw.removeprefix(_DIGEST_PREFIX)
    if any(character not in "0123456789abcdef" for character in hex_part):
        raise TrajectoryContractError(f"{path} must be sha256:<64 lowercase hex>")
    return raw


def _text_array(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    result = [_text(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path))]
    if not allow_empty and not result:
        raise TrajectoryContractError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise TrajectoryContractError(f"{path} must not contain duplicates")
    return result


def _reference_array(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    result = [
        _metadata_ref(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    ]
    if not allow_empty and not result:
        raise TrajectoryContractError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise TrajectoryContractError(f"{path} must not contain duplicates")
    return result


def validate_outcome_bound_trajectory(document: Any) -> dict[str, Any]:
    """Validate one ``OutcomeBoundTrajectoryV1`` document.

    Unknown keys fail closed.  The function returns the original document so
    callers can compose validation with deterministic aggregation.
    """
    root = _object(document, "trajectory")
    required = {
        "schema_version",
        "trajectory_id",
        "created_at",
        "task",
        "trace",
        "capability_usage_snapshot",
        "capabilities",
        "outcomes",
        "indicators",
        "privacy",
        "lifecycle",
        "schema_evolution",
        "claim",
    }
    _keys(root, "trajectory", required=required)
    if root["schema_version"] != SCHEMA_VERSION:
        raise TrajectoryContractError(f"trajectory.schema_version must be {SCHEMA_VERSION}")
    _metadata_id(root["trajectory_id"], "trajectory.trajectory_id")
    created_at = _timestamp(root["created_at"], "trajectory.created_at")

    task = _object(root["task"], "trajectory.task")
    _keys(task, "trajectory.task", required={"task_id", "identity_kind", "execution_id"})
    _metadata_id(task["task_id"], "trajectory.task.task_id")
    if task["identity_kind"] != "pseudonymous":
        raise TrajectoryContractError("trajectory.task.identity_kind must be pseudonymous")
    _metadata_id(task["execution_id"], "trajectory.task.execution_id")

    trace = _object(root["trace"], "trajectory.trace")
    _keys(trace, "trajectory.trace", required={"run_id", "digest", "schema_version"})
    _metadata_id(trace["run_id"], "trajectory.trace.run_id")
    _digest(trace["digest"], "trajectory.trace.digest")
    _metadata_id(trace["schema_version"], "trajectory.trace.schema_version")

    usage = _object(root["capability_usage_snapshot"], "trajectory.capability_usage_snapshot")
    _keys(
        usage,
        "trajectory.capability_usage_snapshot",
        required={"schema", "digest", "generated_at", "as_of", "claim_ceiling"},
    )
    if usage["schema"] != "CodexCapabilityUsageSnapshotV1":
        raise TrajectoryContractError(
            "trajectory.capability_usage_snapshot.schema must be CodexCapabilityUsageSnapshotV1"
        )
    _digest(usage["digest"], "trajectory.capability_usage_snapshot.digest")
    _timestamp(usage["generated_at"], "trajectory.capability_usage_snapshot.generated_at")
    _timestamp(usage["as_of"], "trajectory.capability_usage_snapshot.as_of")
    _text(usage["claim_ceiling"], "trajectory.capability_usage_snapshot.claim_ceiling")

    outcomes = _array(root["outcomes"], "trajectory.outcomes")
    outcome_ids: set[str] = set()
    outcome_by_id: dict[str, dict[str, Any]] = {}
    independent_outcomes = 0
    for index, raw_outcome in enumerate(outcomes):
        path = f"trajectory.outcomes[{index}]"
        outcome = _object(raw_outcome, path)
        _keys(
            outcome,
            path,
            required={
                "outcome_id",
                "label",
                "status",
                "source_type",
                "source_digest",
                "source_ref",
                "authority",
                "observed_at",
                "confidence",
                "independent",
            },
        )
        outcome_id = _metadata_id(outcome["outcome_id"], f"{path}.outcome_id")
        if outcome_id in outcome_ids:
            raise TrajectoryContractError(f"duplicate outcome_id: {outcome_id}")
        outcome_ids.add(outcome_id)
        outcome_by_id[outcome_id] = outcome
        _metadata_id(outcome["label"], f"{path}.label")
        if outcome["status"] not in _OUTCOME_STATUSES:
            raise TrajectoryContractError(
                f"{path}.status must be one of: {', '.join(sorted(_OUTCOME_STATUSES))}"
            )
        _metadata_id(outcome["source_type"], f"{path}.source_type")
        _digest(outcome["source_digest"], f"{path}.source_digest")
        _metadata_ref(outcome["source_ref"], f"{path}.source_ref")
        _metadata_id(outcome["authority"], f"{path}.authority")
        _timestamp(outcome["observed_at"], f"{path}.observed_at")
        _number(outcome["confidence"], f"{path}.confidence", maximum=1.0)
        if not isinstance(outcome["independent"], bool):
            raise TrajectoryContractError(f"{path}.independent must be a boolean")
        independent_outcomes += int(outcome["independent"])

    capabilities = _array(root["capabilities"], "trajectory.capabilities")
    if not capabilities:
        raise TrajectoryContractError("trajectory.capabilities must not be empty")
    capability_keys: set[tuple[str, str]] = set()
    capability_outcome_refs: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for index, raw_capability in enumerate(capabilities):
        path = f"trajectory.capabilities[{index}]"
        capability = _object(raw_capability, path)
        _keys(
            capability,
            path,
            required={
                "capability_id",
                "kind",
                "state",
                "observed_at",
                "evidence_refs",
                "span_refs",
                "outcome_refs",
                "provenance",
                "confidence",
            },
        )
        capability_id = _metadata_id(capability["capability_id"], f"{path}.capability_id")
        if capability["kind"] not in _CAPABILITY_KINDS:
            raise TrajectoryContractError(
                f"{path}.kind must be one of: {', '.join(sorted(_CAPABILITY_KINDS))}"
            )
        state = capability["state"]
        if state not in _CAPABILITY_STATES:
            raise TrajectoryContractError(
                f"{path}.state must be one of: {', '.join(sorted(_CAPABILITY_STATES))}"
            )
        key = (capability_id, state)
        if key in capability_keys:
            raise TrajectoryContractError(
                f"duplicate capability observation: {capability_id}/{state}"
            )
        capability_keys.add(key)
        _timestamp(capability["observed_at"], f"{path}.observed_at")
        evidence_refs = _reference_array(capability["evidence_refs"], f"{path}.evidence_refs")
        span_refs = _reference_array(capability["span_refs"], f"{path}.span_refs")
        refs = _reference_array(capability["outcome_refs"], f"{path}.outcome_refs")
        unknown_refs = sorted(set(refs) - outcome_ids)
        if unknown_refs:
            raise TrajectoryContractError(
                f"{path}.outcome_refs references unknown outcome(s): {', '.join(unknown_refs)}"
            )
        capability_outcome_refs[capability_id][state].update(refs)
        _metadata_id(capability["provenance"], f"{path}.provenance")
        _number(capability["confidence"], f"{path}.confidence", maximum=1.0)
        if state == "available" and not evidence_refs:
            raise TrajectoryContractError(f"{path}.evidence_refs required for available state")
        if state == "exposed" and not span_refs:
            raise TrajectoryContractError(f"{path}.span_refs required for exposed state")
        if state == "adopted" and (not span_refs or not refs):
            raise TrajectoryContractError(
                f"{path} adopted state requires span_refs and outcome_refs"
            )

    for capability_id, refs_by_state in capability_outcome_refs.items():
        available_refs = refs_by_state.get("available", set())
        exposed_states = {"exposed", "adopted"} & refs_by_state.keys()
        if available_refs and exposed_states:
            raise TrajectoryContractError(
                f"capability {capability_id} has ambiguous cohort attribution: "
                "available outcome_refs cannot be combined with exposed or adopted observations"
            )
        refs = set().union(*refs_by_state.values())
        terminal_labels = {
            "passed" if outcome_by_id[ref]["status"] == "passed" else "failed"
            for ref in refs
            if outcome_by_id[ref]["independent"]
            and outcome_by_id[ref]["status"] != "unknown"
        }
        if len(terminal_labels) > 1:
            raise TrajectoryContractError(
                f"capability {capability_id} has conflicting independent terminal outcomes"
            )

    indicators = _object(root["indicators"], "trajectory.indicators")
    _keys(
        indicators,
        "trajectory.indicators",
        required={
            "duration_ms",
            "rework_count",
            "error_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
        },
    )
    for key in ("duration_ms", "rework_count", "error_count"):
        _integer(indicators[key], f"trajectory.indicators.{key}")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if indicators[key] is not None:
            _integer(indicators[key], f"trajectory.indicators.{key}")
    if indicators["cost_usd"] is not None:
        _number(indicators["cost_usd"], "trajectory.indicators.cost_usd")
    if (
        all(
            indicators[key] is not None for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        and indicators["input_tokens"] + indicators["output_tokens"] != indicators["total_tokens"]
    ):
        raise TrajectoryContractError(
            "trajectory.indicators.total_tokens must equal input_tokens + output_tokens"
        )

    privacy = _object(root["privacy"], "trajectory.privacy")
    _keys(
        privacy,
        "trajectory.privacy",
        required={
            "profile",
            "raw_prompts_included",
            "full_tool_payloads_included",
            "secret_material_included",
            "redaction_state",
            "authorization_audience",
            "verification",
        },
    )
    if privacy["profile"] != "metadata-only-v1":
        raise TrajectoryContractError("trajectory.privacy.profile must be metadata-only-v1")
    for key in ("raw_prompts_included", "full_tool_payloads_included", "secret_material_included"):
        if privacy[key] is not False:
            raise TrajectoryContractError(f"trajectory.privacy.{key} must be false")
    if privacy["redaction_state"] not in {"not-required", "scrubbed", "verified-synthetic"}:
        raise TrajectoryContractError(
            "trajectory.privacy.redaction_state must be not-required, scrubbed, "
            "or verified-synthetic"
        )
    _reference_array(
        privacy["authorization_audience"],
        "trajectory.privacy.authorization_audience",
        allow_empty=False,
    )
    verification = _object(privacy["verification"], "trajectory.privacy.verification")
    _keys(
        verification,
        "trajectory.privacy.verification",
        required={"verifier", "ruleset", "receipt_ref", "receipt_digest", "verified_at"},
    )
    _metadata_id(verification["verifier"], "trajectory.privacy.verification.verifier")
    _metadata_id(verification["ruleset"], "trajectory.privacy.verification.ruleset")
    _metadata_ref(verification["receipt_ref"], "trajectory.privacy.verification.receipt_ref")
    _digest(verification["receipt_digest"], "trajectory.privacy.verification.receipt_digest")
    _timestamp(verification["verified_at"], "trajectory.privacy.verification.verified_at")

    lifecycle = _object(root["lifecycle"], "trajectory.lifecycle")
    _keys(
        lifecycle,
        "trajectory.lifecycle",
        required={"retention_days", "expires_at", "deletion_mode"},
    )
    retention_days = _integer(
        lifecycle["retention_days"], "trajectory.lifecycle.retention_days", minimum=1
    )
    expires_at = _timestamp(lifecycle["expires_at"], "trajectory.lifecycle.expires_at")
    if expires_at <= created_at:
        raise TrajectoryContractError("trajectory.lifecycle.expires_at must be after created_at")
    if (expires_at - created_at).total_seconds() > (retention_days + 1) * 86_400:
        raise TrajectoryContractError(
            "trajectory.lifecycle.expires_at exceeds the declared retention window"
        )
    if lifecycle["deletion_mode"] not in {"delete-envelope", "delete-envelope-and-private-source"}:
        raise TrajectoryContractError("trajectory.lifecycle.deletion_mode is unsupported")

    evolution = _object(root["schema_evolution"], "trajectory.schema_evolution")
    _keys(
        evolution,
        "trajectory.schema_evolution",
        required={"producer", "producer_version", "compatible_with"},
    )
    _metadata_id(evolution["producer"], "trajectory.schema_evolution.producer")
    _metadata_id(evolution["producer_version"], "trajectory.schema_evolution.producer_version")
    _reference_array(evolution["compatible_with"], "trajectory.schema_evolution.compatible_with")

    claim = _object(root["claim"], "trajectory.claim")
    _keys(
        claim,
        "trajectory.claim",
        required={"state", "ceiling", "basis_refs"},
        optional={"experimental_design_ref"},
    )
    if claim["state"] not in _CLAIM_STATES:
        raise TrajectoryContractError(
            f"trajectory.claim.state must be one of: {', '.join(sorted(_CLAIM_STATES))}"
        )
    _text(claim["ceiling"], "trajectory.claim.ceiling")
    _reference_array(claim["basis_refs"], "trajectory.claim.basis_refs", allow_empty=False)
    if claim["state"] in {"correlation", "experiment", "causality"} and not independent_outcomes:
        raise TrajectoryContractError(
            f"trajectory.claim.state={claim['state']} requires an independent outcome"
        )
    if claim["state"] in {"experiment", "causality"}:
        _metadata_ref(
            claim.get("experimental_design_ref"), "trajectory.claim.experimental_design_ref"
        )
    return root


def load_outcome_bound_trajectory(path: Path) -> dict[str, Any]:
    """Load a trajectory and resolve its exact local privacy-review receipt bytes."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryContractError(f"cannot read trajectory from {path}: {exc}") from exc
    validated = validate_outcome_bound_trajectory(document)
    receipt_ref = validated["privacy"]["verification"]["receipt_ref"]
    relative = Path(receipt_ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrajectoryContractError(
            "trajectory.privacy.verification.receipt_ref must be a contained relative path"
        )
    receipt = path.parent / relative
    if receipt.is_symlink() or not receipt.is_file():
        raise TrajectoryContractError(
            "trajectory.privacy.verification.receipt_ref must resolve to a regular non-symlink file"
        )
    base = path.parent.resolve()
    resolved = receipt.resolve()
    if resolved != base and base not in resolved.parents:
        raise TrajectoryContractError("trajectory privacy receipt escapes the envelope directory")
    expected = validated["privacy"]["verification"]["receipt_digest"]
    actual = sha256_file(resolved)
    if actual != expected:
        raise TrajectoryContractError(
            f"trajectory privacy receipt digest mismatch: expected {expected}, got {actual}"
        )
    return validated


def aggregate_outcome_bound_trajectories(documents: list[Any]) -> dict[str, Any]:
    """Aggregate descriptive capability/outcome counts without causal inflation."""
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "observed_runs": 0,
            "available_runs": 0,
            "exposed_runs": 0,
            "adopted_runs": 0,
            "independent_labeled_runs": 0,
            "passed_exposed_runs": 0,
            "failed_exposed_runs": 0,
            "passed_available_only_runs": 0,
            "failed_available_only_runs": 0,
            "duration_ms_sum": 0,
            "rework_count_sum": 0,
            "error_count_sum": 0,
            "indicator_samples": 0,
        }
    )
    trajectory_ids: set[str] = set()
    independent_labeled_count = 0
    source_hash = hashlib.sha256()
    source_hash.update(b"[")
    for document_index, raw_document in enumerate(documents):
        document = validate_outcome_bound_trajectory(raw_document)
        if document_index:
            source_hash.update(b",")
        source_hash.update(canonical_json_bytes(document))
        trajectory_id = document["trajectory_id"]
        if trajectory_id in trajectory_ids:
            raise TrajectoryContractError(f"duplicate trajectory_id: {trajectory_id}")
        trajectory_ids.add(trajectory_id)
        outcomes = {outcome["outcome_id"]: outcome for outcome in document["outcomes"]}
        per_capability: dict[str, set[str]] = defaultdict(set)
        per_capability_refs: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for observation in document["capabilities"]:
            capability_id = observation["capability_id"]
            per_capability[capability_id].add(observation["state"])
            per_capability_refs[capability_id][observation["state"]].update(
                observation["outcome_refs"]
            )
        trajectory_has_bound_label = False
        for capability_id, states in per_capability.items():
            row = rows[capability_id]
            row["observed_runs"] += 1
            if "available" in states:
                row["available_runs"] += 1
            exposed = bool(states & {"exposed", "adopted"})
            if exposed:
                row["exposed_runs"] += 1
            if "adopted" in states:
                row["adopted_runs"] += 1
            cohort_states = {"exposed", "adopted"} if exposed else {"available"}
            cohort_refs = set().union(
                *(per_capability_refs[capability_id].get(state, set()) for state in cohort_states)
            )
            bound_statuses = {
                "passed" if outcomes[ref]["status"] == "passed" else "failed"
                for ref in cohort_refs
                if outcomes[ref]["independent"] and outcomes[ref]["status"] != "unknown"
            }
            if len(bound_statuses) > 1:  # defensive: validation already rejects this
                raise TrajectoryContractError(
                    f"capability {capability_id} has conflicting independent terminal outcomes"
                )
            if bound_statuses:
                trajectory_has_bound_label = True
                row["independent_labeled_runs"] += 1
                bucket = "exposed" if exposed else "available_only"
                status = next(iter(bound_statuses))
                row[f"{status}_{bucket}_runs"] += 1
            row["duration_ms_sum"] += document["indicators"]["duration_ms"]
            row["rework_count_sum"] += document["indicators"]["rework_count"]
            row["error_count_sum"] += document["indicators"]["error_count"]
            row["indicator_samples"] += 1
        independent_labeled_count += int(trajectory_has_bound_label)
    source_hash.update(b"]")

    capability_rows = []
    for capability_id in sorted(rows):
        row = rows[capability_id]
        samples = row["indicator_samples"]
        capability_rows.append(
            {
                "capability_id": capability_id,
                "observed_runs": row["observed_runs"],
                "available_runs": row["available_runs"],
                "exposed_runs": row["exposed_runs"],
                "adopted_runs": row["adopted_runs"],
                "independent_labeled_runs": row["independent_labeled_runs"],
                "independent_label_coverage": (
                    row["independent_labeled_runs"] / row["observed_runs"]
                    if row["observed_runs"]
                    else 0.0
                ),
                "passed_exposed_runs": row["passed_exposed_runs"],
                "failed_exposed_runs": row["failed_exposed_runs"],
                "passed_available_only_runs": row["passed_available_only_runs"],
                "failed_available_only_runs": row["failed_available_only_runs"],
                "mean_duration_ms": row["duration_ms_sum"] / samples if samples else None,
                "mean_rework_count": row["rework_count_sum"] / samples if samples else None,
                "mean_error_count": row["error_count_sum"] / samples if samples else None,
            }
        )
    total = len(trajectory_ids)
    return {
        "schema_version": AGGREGATE_VERSION,
        "trajectory_count": total,
        "independent_labeled_count": independent_labeled_count,
        "independent_label_coverage": independent_labeled_count / total if total else 0.0,
        "capabilities": capability_rows,
        "source_digest": _DIGEST_PREFIX + source_hash.hexdigest(),
        "claim_state": "observation",
        "claim_ceiling": (
            "Descriptive availability, exposure, adoption, and independently labeled "
            "outcome counts. "
            "No causal effect, optimality, or portfolio-scale adoption claim."
        ),
    }


def query_capability_decision(
    aggregate: Any,
    capability_id: str,
    *,
    minimum_labeled_per_cohort: int = 3,
    minimum_label_coverage: float = 0.8,
    material_rate_delta: float = 0.1,
) -> dict[str, Any]:
    """Return a bounded scale/stop decision from a validated descriptive aggregate.

    The query can recommend more measurement or review of a correlation.  It
    never promotes an observed association to a causal scaling decision.
    """
    root = _object(aggregate, "aggregate")
    if root.get("schema_version") != AGGREGATE_VERSION:
        raise TrajectoryContractError(f"aggregate.schema_version must be {AGGREGATE_VERSION}")
    _integer(minimum_labeled_per_cohort, "minimum_labeled_per_cohort", minimum=1)
    _number(minimum_label_coverage, "minimum_label_coverage", maximum=1.0)
    _number(material_rate_delta, "material_rate_delta", maximum=1.0)
    rows = [
        row
        for row in _array(root.get("capabilities"), "aggregate.capabilities")
        if row.get("capability_id") == capability_id
    ]
    if len(rows) != 1:
        raise TrajectoryContractError(
            f"aggregate needs exactly one row for capability: {capability_id}"
        )
    row = rows[0]
    exposed_n = row["passed_exposed_runs"] + row["failed_exposed_runs"]
    control_n = row["passed_available_only_runs"] + row["failed_available_only_runs"]
    coverage = float(row.get("independent_label_coverage", 0.0))
    if coverage < minimum_label_coverage:
        decision = "STOP_LOW_LABEL_COVERAGE"
        reason = f"independent label coverage {coverage:.3f} is below {minimum_label_coverage:.3f}"
        delta = None
    elif exposed_n < minimum_labeled_per_cohort or control_n < minimum_labeled_per_cohort:
        decision = "HOLD_INSUFFICIENT_COHORTS"
        reason = (
            f"need {minimum_labeled_per_cohort} independently labeled runs in both exposed "
            f"and available-only cohorts; observed {exposed_n} and {control_n}"
        )
        delta = None
    else:
        exposed_rate = row["passed_exposed_runs"] / exposed_n
        control_rate = row["passed_available_only_runs"] / control_n
        delta = exposed_rate - control_rate
        if abs(delta) < material_rate_delta:
            decision = "DO_NOT_SCALE_NO_DECISION_CHANGE"
            reason = f"absolute pass-rate delta {abs(delta):.3f} is below {material_rate_delta:.3f}"
        else:
            decision = "REVIEW_CORRELATION"
            reason = (
                "a material descriptive association exists, but experiment or stronger "
                "identification "
                "is required before a causal scale decision"
            )
    return {
        "schema_version": DECISION_VERSION,
        "capability_id": capability_id,
        "decision": decision,
        "reason": reason,
        "exposed_labeled_runs": exposed_n,
        "available_only_labeled_runs": control_n,
        "independent_label_coverage": coverage,
        "pass_rate_delta": delta,
        "scale_authorized": False,
        "kill_criterion_triggered": decision.startswith(("STOP_", "DO_NOT_SCALE_")),
        "claim_ceiling": (
            "Bounded measurement decision only. REVIEW_CORRELATION is not a causal or "
            "scaling claim."
        ),
    }
