from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from plumbline.trajectory import (
    TrajectoryContractError,
    aggregate_outcome_bound_trajectories,
    load_outcome_bound_trajectory,
    query_capability_decision,
    validate_outcome_bound_trajectory,
)


def example() -> dict:
    path = Path(__file__).parents[1] / "examples" / "outcome-bound-trajectory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def with_id(document: dict, index: int, *, exposed: bool, passed: bool | None) -> dict:
    result = copy.deepcopy(document)
    result["trajectory_id"] = f"trajectory-{index}"
    result["task"]["task_id"] = f"task-{index}"
    result["task"]["execution_id"] = f"execution-{index}"
    if not exposed:
        result["capabilities"] = [result["capabilities"][0]]
        if passed is not None:
            result["capabilities"][0]["outcome_refs"] = [result["outcomes"][0]["outcome_id"]]
    if passed is None:
        result["outcomes"] = []
        for capability in result["capabilities"]:
            capability["outcome_refs"] = []
        result["capabilities"] = [
            capability for capability in result["capabilities"] if capability["state"] != "adopted"
        ]
    else:
        result["outcomes"][0]["status"] = "passed" if passed else "failed"
    return result


def test_example_is_valid() -> None:
    assert validate_outcome_bound_trajectory(example())["trajectory_id"] == "obt-synthetic-001"


def test_file_loader_resolves_exact_privacy_review_receipt() -> None:
    path = Path(__file__).parents[1] / "examples" / "outcome-bound-trajectory.json"
    assert load_outcome_bound_trajectory(path)["trajectory_id"] == "obt-synthetic-001"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "outcome-bound-trajectory.json").write_text(
            json.dumps(example()), encoding="utf-8"
        )
        (root / "outcome-privacy-review-receipt.json").write_text(
            "tampered", encoding="utf-8"
        )
        with pytest.raises(TrajectoryContractError, match="privacy receipt digest mismatch"):
            load_outcome_bound_trajectory(root / "outcome-bound-trajectory.json")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("privacy", "raw_prompts_included"), True, "must be false"),
        (("privacy", "full_tool_payloads_included"), True, "must be false"),
        (("privacy", "secret_material_included"), True, "must be false"),
        (("task", "identity_kind"), "email", "must be pseudonymous"),
        (("trace", "digest"), "missing", "sha256"),
    ],
)
def test_privacy_and_identity_fail_closed(
    path: tuple[str, str], value: object, message: str
) -> None:
    document = example()
    document[path[0]][path[1]] = value
    with pytest.raises(TrajectoryContractError, match=message):
        validate_outcome_bound_trajectory(document)


def test_adoption_requires_span_and_outcome_binding() -> None:
    document = example()
    document["capabilities"][2]["outcome_refs"] = []
    with pytest.raises(TrajectoryContractError, match="adopted state requires"):
        validate_outcome_bound_trajectory(document)


def test_unknown_keys_and_outcome_refs_fail_closed() -> None:
    document = example()
    document["raw_prompt"] = "should never fit the contract"
    with pytest.raises(TrajectoryContractError, match="unsupported key"):
        validate_outcome_bound_trajectory(document)

    document = example()
    document["capabilities"][2]["outcome_refs"] = ["missing"]
    with pytest.raises(TrajectoryContractError, match="unknown outcome"):
        validate_outcome_bound_trajectory(document)


def test_metadata_fields_exclude_freeform_payload_content() -> None:
    document = example()
    document["capabilities"][0]["provenance"] = "freeform payload text is forbidden"
    with pytest.raises(TrajectoryContractError, match="opaque metadata identifier"):
        validate_outcome_bound_trajectory(document)

    document = example()
    document["outcomes"][0]["source_ref"] = "receipt\nsecret-like-payload"
    with pytest.raises(TrajectoryContractError, match="bounded single-line"):
        validate_outcome_bound_trajectory(document)


def test_privacy_profile_requires_digest_bound_verification() -> None:
    document = example()
    del document["privacy"]["verification"]
    with pytest.raises(TrajectoryContractError, match="missing key"):
        validate_outcome_bound_trajectory(document)

def test_causal_claim_requires_independent_outcome_and_design() -> None:
    document = with_id(example(), 1, exposed=True, passed=None)
    document["claim"]["state"] = "causality"
    with pytest.raises(TrajectoryContractError, match="requires an independent outcome"):
        validate_outcome_bound_trajectory(document)

    document = example()
    document["claim"]["state"] = "causality"
    with pytest.raises(TrajectoryContractError, match="experimental_design_ref"):
        validate_outcome_bound_trajectory(document)


def test_aggregate_keeps_availability_exposure_and_adoption_distinct() -> None:
    documents = [
        with_id(example(), 1, exposed=True, passed=True),
        with_id(example(), 2, exposed=True, passed=True),
        with_id(example(), 3, exposed=True, passed=True),
        with_id(example(), 4, exposed=False, passed=True),
        with_id(example(), 5, exposed=False, passed=False),
        with_id(example(), 6, exposed=False, passed=False),
    ]
    aggregate = aggregate_outcome_bound_trajectories(documents)
    row = aggregate["capabilities"][0]
    assert row["available_runs"] == 6
    assert row["exposed_runs"] == 3
    assert row["adopted_runs"] == 3
    assert aggregate["independent_label_coverage"] == 1.0

    decision = query_capability_decision(aggregate, "quality-gatekeeper")
    assert decision["decision"] == "REVIEW_CORRELATION"
    assert decision["pass_rate_delta"] == pytest.approx(2 / 3)
    assert decision["scale_authorized"] is False
    assert decision["kill_criterion_triggered"] is False


def test_decision_kill_criteria_block_scale_when_labels_are_incomplete() -> None:
    documents = [
        with_id(example(), 1, exposed=True, passed=True),
        with_id(example(), 2, exposed=False, passed=None),
    ]
    aggregate = aggregate_outcome_bound_trajectories(documents)
    decision = query_capability_decision(aggregate, "quality-gatekeeper")
    assert decision["decision"] == "STOP_LOW_LABEL_COVERAGE"
    assert decision["kill_criterion_triggered"] is True


def test_duplicate_trajectory_ids_fail_closed() -> None:
    document = example()
    with pytest.raises(TrajectoryContractError, match="duplicate trajectory_id"):
        aggregate_outcome_bound_trajectories([document, copy.deepcopy(document)])


def test_conflicting_independent_labels_for_one_capability_fail_closed() -> None:
    document = example()
    failed = copy.deepcopy(document["outcomes"][0])
    failed["outcome_id"] = "outcome-focused-tests-failed"
    failed["status"] = "failed"
    document["outcomes"].append(failed)
    document["capabilities"][2]["outcome_refs"].append(failed["outcome_id"])
    with pytest.raises(TrajectoryContractError, match="conflicting independent terminal"):
        validate_outcome_bound_trajectory(document)


def test_available_outcome_cannot_be_promoted_into_an_exposed_cohort() -> None:
    document = example()
    document["capabilities"][0]["outcome_refs"] = [document["outcomes"][0]["outcome_id"]]
    with pytest.raises(TrajectoryContractError, match="ambiguous cohort attribution"):
        validate_outcome_bound_trajectory(document)


def test_only_capability_bound_known_labels_enter_cohorts() -> None:
    unbound = example()
    unbound["trajectory_id"] = "unbound-trajectory"
    unbound["capabilities"] = unbound["capabilities"][:2]
    aggregate = aggregate_outcome_bound_trajectories([unbound])
    row = aggregate["capabilities"][0]
    assert row["independent_labeled_runs"] == 0
    assert row["passed_exposed_runs"] == 0
    assert row["independent_label_coverage"] == 0.0

    unknown = example()
    unknown["trajectory_id"] = "unknown-trajectory"
    unknown["outcomes"][0]["status"] = "unknown"
    aggregate = aggregate_outcome_bound_trajectories([unknown])
    row = aggregate["capabilities"][0]
    assert row["independent_labeled_runs"] == 0
    assert row["failed_exposed_runs"] == 0
    assert row["independent_label_coverage"] == 0.0
