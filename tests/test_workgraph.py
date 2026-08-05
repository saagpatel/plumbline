from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from plumbline.workgraph import WorkGraphContractError, evaluate_workgraph_shadow

PLAN_DIGEST = "sha256:" + "a" * 64
REGISTRATION_DIGEST = "sha256:" + "c" * 64


def schema(name: str) -> dict:
    path = Path(__file__).parents[1] / "schema" / name
    return json.loads(path.read_text(encoding="utf-8"))


def compiled_plan() -> dict:
    return {
        "result_version": "WorkGraphCompileResultV1",
        "status": "compiled",
        "graph_id": "prospective-audit",
        "input_semantic_sha256": "b" * 64,
        "waves": [
            {
                "index": 1,
                "lanes": [
                    {
                        "id": "discovery",
                        "depends_on": [],
                        "mutation_mode": "strict-read-only",
                    }
                ],
            },
            {
                "index": 2,
                "lanes": [
                    {
                        "id": "synthesis",
                        "depends_on": ["discovery"],
                        "mutation_mode": "report-only",
                    }
                ],
            },
        ],
    }


def registration() -> dict:
    return {
        "schema_version": "WorkGraphShadowPilotRegistrationV1",
        "registered_at": "2026-08-05T04:59:00Z",
        "graph_id": "prospective-audit",
        "source_graph_digest": "sha256:" + "b" * 64,
        "compiled_plan_digest": PLAN_DIGEST,
        "compiler_version": "1.0.0",
        "lane_bindings": [
            {
                "lane_id": "discovery",
                "wave": 1,
                "agent_path": "/root/discovery",
                "depends_on": [],
                "mutation_mode": "strict-read-only",
            },
            {
                "lane_id": "synthesis",
                "wave": 2,
                "agent_path": "/root/synthesis",
                "depends_on": ["discovery"],
                "mutation_mode": "report-only",
            },
        ],
        "metric_contract": ["coverage", "terminal evidence"],
        "admission": "compiled_and_digest_bound_before_dispatch",
        "claim_ceiling": "Local prospective shadow registration only.",
    }


def event(event_id: str, lane: str, kind: str, timestamp: str, **overrides: object) -> dict:
    result = {
        "event_id": event_id,
        "lane_id": lane,
        "agent_path": f"/root/{lane}",
        "event_type": kind,
        "observed_at": timestamp,
        "evidence_refs": [f"receipt:{event_id}"] if kind != "started" else [],
        "mutation_observed": False,
        "duplicate_signature": None,
        "tokens": 10,
        "cost_usd": 0.01,
        "retry_count": 0,
        "unknown_reason": None,
    }
    result.update(overrides)
    return result


def observed() -> dict:
    return {
        "schema_version": "WorkGraphObservedEventsV1",
        "graph_id": "prospective-audit",
        "compiled_plan_digest": PLAN_DIGEST,
        "registration_digest": REGISTRATION_DIGEST,
        "events": [
            event("e1", "discovery", "started", "2026-08-05T05:00:00Z"),
            event("e2", "discovery", "completed", "2026-08-05T05:01:00Z"),
            event("e3", "synthesis", "started", "2026-08-05T05:01:01Z"),
            event("e4", "synthesis", "completed", "2026-08-05T05:02:00Z"),
        ],
    }


def evaluate(events: dict) -> dict:
    return evaluate_workgraph_shadow(
        compiled_plan(),
        registration(),
        events,
        compiled_plan_digest=PLAN_DIGEST,
        registration_digest=REGISTRATION_DIGEST,
    )


def test_complete_shadow_is_go_with_recomputed_metrics() -> None:
    report = evaluate(observed())
    assert report["disposition"] == "GO"
    assert report["coverage"] == 1.0
    assert report["evidence_completeness"] == 1.0
    assert report["metrics"]["tokens"] == 40
    assert report["metrics"]["cost_usd"] == pytest.approx(0.04)
    assert report["metrics"]["wall_time_ms"] == 120000
    assert report["prospectivity"] == {
        "registered_at": "2026-08-05T04:59:00Z",
        "first_event_at": "2026-08-05T05:00:00Z",
        "digest_bound": True,
        "chronological": True,
        "external_time_authority": False,
    }


def test_missing_terminal_is_unknown_not_green() -> None:
    events = observed()
    events["events"] = events["events"][:-1]
    report = evaluate(events)
    assert report["disposition"] == "UNKNOWN"
    assert report["metrics"]["incomplete_lanes"] == ["synthesis"]


def test_dependency_violation_is_no_go() -> None:
    events = observed()
    events["events"][2]["observed_at"] = "2026-08-05T05:00:30Z"
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert "DEPENDENCY_VIOLATION" in {item["code"] for item in report["violations"]}


def test_read_only_mutation_and_duplicate_signature_are_no_go() -> None:
    events = observed()
    events["events"][1]["mutation_observed"] = True
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert "UNDECLARED_MUTATION" in {item["code"] for item in report["violations"]}

    events = observed()
    events["events"][1]["duplicate_signature"] = "same-work"
    events["events"][3]["duplicate_signature"] = "same-work"
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert report["metrics"]["duplicate_groups"] == [["discovery", "synthesis"]]


def test_compiled_and_registration_digest_bindings_fail_closed() -> None:
    events = observed()
    events["compiled_plan_digest"] = "sha256:" + "d" * 64
    with pytest.raises(WorkGraphContractError, match="exact compiled plan"):
        evaluate(events)

    events = observed()
    events["registration_digest"] = "sha256:" + "d" * 64
    with pytest.raises(WorkGraphContractError, match="exact registration"):
        evaluate(events)

    events = observed()
    events["graph_id"] = "other"
    with pytest.raises(WorkGraphContractError, match="graph_id"):
        evaluate(events)


def test_frozen_agent_mapping_is_enforced_even_when_events_agree() -> None:
    events = observed()
    events["events"][0]["agent_path"] = "/root/unregistered"
    events["events"][1]["agent_path"] = "/root/unregistered"
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert "LANE_AGENT_MAPPING_DRIFT" in {item["code"] for item in report["violations"]}


def test_event_before_registration_is_no_go_and_unknown_reason_is_unknown() -> None:
    events = observed()
    events["events"][0]["observed_at"] = "2026-08-05T04:58:59Z"
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert report["prospectivity"]["chronological"] is False

    events = observed()
    events["events"][1]["unknown_reason"] = "receipt provenance unresolved"
    report = evaluate(events)
    assert report["disposition"] == "UNKNOWN"
    assert "UNRESOLVED_UNKNOWN" in {item["code"] for item in report["violations"]}


def test_event_metadata_rejects_payload_shaped_and_unbounded_values() -> None:
    events = observed()
    events["events"][1]["evidence_refs"] = ["receipt\nraw-payload"]
    with pytest.raises(WorkGraphContractError, match="bounded single-line"):
        evaluate(events)

    events = observed()
    events["events"][1]["evidence_refs"] = ["freeform payload text"]
    with pytest.raises(WorkGraphContractError, match="opaque metadata reference"):
        evaluate(events)

    events = observed()
    events["events"][1]["unknown_reason"] = "x" * 513
    with pytest.raises(WorkGraphContractError, match="bounded single-line"):
        evaluate(events)

    events = observed()
    events["events"][1]["evidence_refs"] = ["receipt:e2", "receipt:e2"]
    with pytest.raises(WorkGraphContractError, match="must not contain duplicates"):
        evaluate(events)


def test_registration_metadata_rejects_multiline_agent_path() -> None:
    current_registration = registration()
    current_registration["lane_bindings"][0]["agent_path"] = "/root/discovery\nraw"
    with pytest.raises(WorkGraphContractError, match="bounded single-line"):
        evaluate_workgraph_shadow(
            compiled_plan(),
            current_registration,
            observed(),
            compiled_plan_digest=PLAN_DIGEST,
            registration_digest=REGISTRATION_DIGEST,
        )


def test_metadata_boundary_is_enforced_by_input_schemas() -> None:
    events_validator = Draft202012Validator(schema("workgraph-observed-events.schema.json"))
    registration_validator = Draft202012Validator(
        schema("workgraph-pilot-registration.schema.json")
    )
    assert not list(events_validator.iter_errors(observed()))
    assert not list(registration_validator.iter_errors(registration()))

    events = observed()
    events["events"][1]["evidence_refs"] = ["receipt\nraw-payload"]
    assert list(events_validator.iter_errors(events))

    events = observed()
    events["events"][1]["evidence_refs"] = ["receipt:e2", "receipt:e2"]
    assert list(events_validator.iter_errors(events))

    current_registration = registration()
    current_registration["lane_bindings"][0]["agent_path"] = "/root/discovery\nraw"
    assert list(registration_validator.iter_errors(current_registration))


def test_unregistered_events_do_not_contaminate_declared_metrics() -> None:
    events = observed()
    events["events"].extend(
        [
            event(
                "x1",
                "intruder",
                "started",
                "2000-01-01T00:00:00Z",
                agent_path="/root/intruder",
                tokens=10**12,
                cost_usd=10**9,
                retry_count=10**6,
            ),
            event(
                "x2",
                "intruder",
                "completed",
                "2100-01-01T00:00:00Z",
                agent_path="/root/intruder",
                tokens=10**12,
                cost_usd=10**9,
                retry_count=10**6,
            ),
        ]
    )
    report = evaluate(events)
    assert report["disposition"] == "NO_GO"
    assert report["metrics"]["tokens"] == 40
    assert report["metrics"]["cost_usd"] == pytest.approx(0.04)
    assert report["metrics"]["retry_count"] == 0
    assert report["metrics"]["wall_time_ms"] == 120000


def test_empty_compiled_graph_is_invalid() -> None:
    plan = compiled_plan()
    plan["waves"] = []
    with pytest.raises(WorkGraphContractError, match="at least one lane"):
        evaluate_workgraph_shadow(
            plan,
            registration(),
            observed(),
            compiled_plan_digest=PLAN_DIGEST,
            registration_digest=REGISTRATION_DIGEST,
        )


def test_same_wave_serialization_count_handles_large_lane_set() -> None:
    lane_count = 1000
    lanes = [
        {"id": f"lane-{index}", "depends_on": [], "mutation_mode": "strict-read-only"}
        for index in range(lane_count)
    ]
    plan = {
        "status": "compiled",
        "graph_id": "large-wave",
        "waves": [{"index": 1, "lanes": lanes}],
    }
    reg = {
        "schema_version": "WorkGraphShadowPilotRegistrationV1",
        "registered_at": "2026-08-05T00:00:00Z",
        "graph_id": "large-wave",
        "source_graph_digest": "sha256:" + "b" * 64,
        "compiled_plan_digest": PLAN_DIGEST,
        "compiler_version": "1.0.0",
        "lane_bindings": [
            {
                "lane_id": f"lane-{index}",
                "wave": 1,
                "agent_path": f"/root/lane-{index}",
                "depends_on": [],
                "mutation_mode": "strict-read-only",
            }
            for index in range(lane_count)
        ],
        "metric_contract": ["serialization"],
        "admission": "compiled_and_digest_bound_before_dispatch",
        "claim_ceiling": "Synthetic scale fixture.",
    }
    events = {
        "schema_version": "WorkGraphObservedEventsV1",
        "graph_id": "large-wave",
        "compiled_plan_digest": PLAN_DIGEST,
        "registration_digest": REGISTRATION_DIGEST,
        "events": [],
    }
    for index in range(lane_count):
        start = f"2026-08-05T00:{index // 60 + 1:02d}:{index % 60:02d}Z"
        end = f"2026-08-05T00:{index // 60 + 1:02d}:{index % 60:02d}.500000Z"
        events["events"].append(event(f"s-{index}", f"lane-{index}", "started", start))
        events["events"].append(event(f"t-{index}", f"lane-{index}", "completed", end))
    report = evaluate_workgraph_shadow(
        plan,
        reg,
        events,
        compiled_plan_digest=PLAN_DIGEST,
        registration_digest=REGISTRATION_DIGEST,
    )
    assert report["disposition"] == "GO"
    assert report["metrics"]["observed_serialized_pairs"] == lane_count * (lane_count - 1) // 2
