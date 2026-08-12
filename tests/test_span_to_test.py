"""Span-to-test reduction, privacy, safety, and CLI behavior."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from plumbline.cli import main
from plumbline.span_to_test import (
    FIXTURE_VERSION,
    RECEIPT_VERSION,
    REQUEST_VERSION,
    SpanToTestContractError,
    evaluate_expected_assertion,
    generate_span_to_test,
    preflight_output_paths,
    render_pytest_skeleton,
    validate_replay_fixture,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "tests" / "fixtures" / "span_to_test" / "failing-trace.json"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "span_to_test" / "failing-fixture.golden.json"
SCHEMA_DIR = ROOT / "schema"
SENTINELS = (
    "P10-RAW-SECRET",
    "alice@example.com",
    "private.example",
    "/Users/alice",
    "sk-ABCDEFGHIJKLMNOPQRSTUV",
    "Bearer abcdefghijklmnop",
)


def _source() -> dict:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def _request(kind: str = "span", value: str = "failing-tool") -> dict:
    return {
        "schema_version": REQUEST_VERSION,
        "selector": {"kind": kind, "value": value},
    }


def _generated() -> tuple[dict, dict]:
    return generate_span_to_test(_source(), _request())


def _serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_generation_is_deterministic_and_source_is_immutable() -> None:
    source = _source()
    before = deepcopy(source)
    first = generate_span_to_test(source, _request())
    second = generate_span_to_test(source, _request())
    assert first == second
    assert source == before


def test_minimum_dependency_closure_excludes_irrelevant_siblings() -> None:
    fixture, receipt = _generated()
    assert [span["kind"] for span in fixture["spans"]] == [
        "llm",
        "tool_call",
        "hook",
        "decision",
    ]
    assert receipt["reduction"] == {
        "source_span_count": 6,
        "retained_span_count": 4,
        "removed_span_count": 2,
        "algorithm": "selected_subtree_plus_ancestry_and_causal_closure_v1",
    }
    rendered = _serialized(fixture)
    assert "UnrelatedRead" not in rendered
    assert "UnrelatedWrite" not in rendered


def test_parent_ancestor_does_not_pull_unrelated_causal_sibling() -> None:
    source = _source()
    source["steps"].append(
        {
            "step_id": "root-caused-but-unrelated",
            "parent_step_id": "root-llm",
            "caused_by": "root-llm",
            "kind": "tool_call",
            "started_at": "2026-08-11T12:00:06Z",
            "status": "ok",
            "attributes": {"gen_ai.tool.name": "ExtraTool"},
        }
    )
    fixture, _receipt = generate_span_to_test(source, _request())
    assert "ExtraTool" not in _serialized(fixture)


def test_topology_order_status_error_and_assertion_are_preserved() -> None:
    fixture, _receipt = _generated()
    root, failing, hook, decision = fixture["spans"]
    assert [span["sequence"] for span in fixture["spans"]] == [0, 2, 3, 4]
    assert failing["parent_span_id"] == root["span_id"]
    assert hook["parent_span_id"] == root["span_id"]
    assert hook["caused_by_span_id"] == failing["span_id"]
    assert decision["caused_by_span_id"] == hook["span_id"]
    assert failing["status"] == "error"
    assert failing["error_class"] == "TimeoutError"
    assert fixture["expected_assertion"]["target_span_id"] == failing["span_id"]
    assert evaluate_expected_assertion(fixture)


def test_tool_payload_is_replaced_by_inert_descriptor() -> None:
    fixture, _receipt = _generated()
    failing = next(span for span in fixture["spans"] if span["kind"] == "tool_call")
    descriptor = failing["attributes"]["tool"]
    assert descriptor["descriptor_mode"] == "inert"
    assert descriptor["name"] == "Shell"
    assert descriptor["operation"] == "execute_tool"
    assert descriptor["argument_keys"] == [
        "binary",
        "command",
        "email",
        "environment",
        "home_path",
        "unicode",
    ]
    assert descriptor["result_kind"] == "shell"


def test_no_raw_secrets_and_each_sensitive_class_is_receipted() -> None:
    fixture, receipt = _generated()
    rendered = _serialized({"fixture": fixture, "receipt": receipt})
    for sentinel in SENTINELS:
        assert sentinel not in rendered
    categories = {item["category"] for item in receipt["transformations"]}
    assert {
        "binary_data",
        "email",
        "environment_or_secret",
        "home_or_environment_path",
        "identifier",
        "model_content_or_exception_text",
        "timestamp",
        "token",
        "tool_arguments_or_results",
        "url",
    } <= categories


def test_oversized_and_nested_unicode_content_is_removed_deterministically() -> None:
    source = _source()
    source["steps"][2]["attributes"]["arbitrary"] = {
        "nested": [{"large": "é" * 5000, "control": "a\u0000b"}]
    }
    fixture, receipt = generate_span_to_test(source, _request())
    assert "é" not in _serialized(fixture)
    categories = {item["category"] for item in receipt["transformations"]}
    assert "oversized_content" in categories
    assert "binary_data" in categories


def test_pseudonyms_are_stable_when_sensitive_content_changes() -> None:
    first_source = _source()
    second_source = _source()
    second_source["steps"][2]["attributes"]["tool.arguments"]["command"] = "different secret"
    first, _first_receipt = generate_span_to_test(first_source, _request())
    second, _second_receipt = generate_span_to_test(second_source, _request())
    assert first["source"]["trace_id"] == second["source"]["trace_id"]
    assert first["source"]["selected_span_id"] == second["source"]["selected_span_id"]
    assert [span["span_id"] for span in first["spans"]] == [
        span["span_id"] for span in second["spans"]
    ]
    assert first["fixture_id"] != second["fixture_id"]


@pytest.mark.parametrize(
    ("kind", "value"),
    [("trace", "trace-secret-source-42"), ("outcome", "failed"), ("outcome", "error")],
)
def test_trace_and_outcome_selectors(kind: str, value: str) -> None:
    fixture, _receipt = generate_span_to_test(_source(), _request(kind, value))
    assert fixture["source"]["selector_kind"] == kind
    assert evaluate_expected_assertion(fixture)


def test_finding_selector() -> None:
    source = _source()
    source["steps"][2]["attributes"]["finding.id"] = "finding-42"
    fixture, _receipt = generate_span_to_test(source, _request("finding", "finding-42"))
    assert fixture["expected_assertion"]["kind"] == "finding"
    assert "finding-42" not in _serialized(fixture)
    assert evaluate_expected_assertion(fixture)


def test_denied_hook_asserts_verdict_not_ok_status() -> None:
    fixture, _receipt = generate_span_to_test(_source(), _request("span", "denial-hook"))
    assert fixture["expected_assertion"]["kind"] == "hook_verdict"
    assert fixture["expected_assertion"]["expected"] == "deny"
    assert evaluate_expected_assertion(fixture)


def test_error_class_failure_assertion() -> None:
    source = _source()
    selected = source["steps"][2]
    selected["status"] = "ok"
    source["run"]["outcome"] = {"status": "completed"}
    fixture, _receipt = generate_span_to_test(source, _request())
    assert fixture["expected_assertion"]["kind"] == "error_class"
    assert fixture["expected_assertion"]["expected"] == "TimeoutError"
    assert evaluate_expected_assertion(fixture)


def test_span_without_failure_signal_is_rejected() -> None:
    source = _source()
    source["run"]["outcome"] = {"status": "completed"}
    with pytest.raises(SpanToTestContractError, match="no supported failure signal"):
        generate_span_to_test(source, _request("span", "irrelevant-before"))


def test_missing_parent_is_rejected() -> None:
    source = _source()
    source["steps"][2]["parent_step_id"] = "missing"
    with pytest.raises(SpanToTestContractError, match="missing parent_step_id"):
        generate_span_to_test(source, _request())


def test_dependency_cycle_is_rejected() -> None:
    source = _source()
    source["steps"][0]["parent_step_id"] = "failing-tool"
    with pytest.raises(SpanToTestContractError, match="dependency cycle"):
        generate_span_to_test(source, _request())


def test_duplicate_span_id_is_rejected() -> None:
    source = _source()
    source["steps"][1]["step_id"] = "failing-tool"
    with pytest.raises(SpanToTestContractError, match="duplicate step_id"):
        generate_span_to_test(source, _request())


def test_fixture_validator_rejects_executable_payload_field() -> None:
    fixture, _receipt = _generated()
    fixture["spans"][1]["attributes"]["command"] = "echo unsafe"
    with pytest.raises(SpanToTestContractError, match="forbidden payload"):
        validate_replay_fixture(fixture)


def test_fixture_validator_rejects_sensitive_structural_text() -> None:
    fixture, _receipt = _generated()
    fixture["spans"][2]["attributes"]["harness.hook.name"] = "alice@example.com"
    with pytest.raises(SpanToTestContractError, match="unsafe structural text"):
        validate_replay_fixture(fixture)


def test_sensitive_tool_name_is_pseudonymized() -> None:
    source = _source()
    source["steps"][2]["attributes"]["gen_ai.tool.name"] = "https://private.example"
    fixture, _receipt = generate_span_to_test(source, _request())
    rendered = _serialized(fixture)
    assert "private.example" not in rendered
    tool_name = fixture["spans"][1]["attributes"]["tool"]["name"]
    assert tool_name.startswith("tool_")


def test_fixture_validator_rejects_missing_topology_reference() -> None:
    fixture, _receipt = _generated()
    fixture["spans"][1]["parent_span_id"] = "span_00000000000000000000"
    with pytest.raises(SpanToTestContractError, match="missing parent_span_id"):
        validate_replay_fixture(fixture)


def test_machine_schemas_validate_request_fixture_and_receipt() -> None:
    fixture, receipt = _generated()
    documents = (
        (
            _request(),
            SCHEMA_DIR / "span-to-test-generation-request.schema.json",
        ),
        (fixture, SCHEMA_DIR / "sanitized-replay-fixture.schema.json"),
        (receipt, SCHEMA_DIR / "span-to-test-reduction-receipt.schema.json"),
    )
    for document, schema_path in documents:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
    assert fixture["schema_version"] == FIXTURE_VERSION
    assert receipt["schema_version"] == RECEIPT_VERSION


def test_golden_fixture() -> None:
    fixture, _receipt = _generated()
    assert fixture == json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_output_preflight_refuses_source_alias_existing_file_and_symlink(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(existing)
    with pytest.raises(SpanToTestContractError, match="source trace"):
        preflight_output_paths(source, [source])
    with pytest.raises(SpanToTestContractError, match="already exists"):
        preflight_output_paths(source, [existing])
    with pytest.raises(SpanToTestContractError, match="symlink"):
        preflight_output_paths(source, [link], overwrite=True)


def test_output_preflight_refuses_hardlink_aliases(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    source_alias = tmp_path / "source-alias.json"
    source_alias.hardlink_to(source)
    with pytest.raises(SpanToTestContractError, match="source trace"):
        preflight_output_paths(source, [source_alias], overwrite=True)

    first = tmp_path / "first.json"
    first.write_text("owned", encoding="utf-8")
    second = tmp_path / "second.json"
    second.hardlink_to(first)
    with pytest.raises(SpanToTestContractError, match="must be distinct"):
        preflight_output_paths(source, [first, second], overwrite=True)


def test_cli_writes_only_explicit_outputs_and_preserves_source(tmp_path, capsys) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(SOURCE_PATH.read_bytes())
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    fixture_path = tmp_path / "fixture.json"
    receipt_path = tmp_path / "receipt.json"
    test_path = tmp_path / "test_fixture.py"
    rc = main(
        [
            "span-to-test",
            str(source),
            "--span",
            "failing-tool",
            "-o",
            str(fixture_path),
            "--receipt-output",
            str(receipt_path),
            "--pytest-output",
            str(test_path),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == ""
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fixture.json",
        "receipt.json",
        "source.json",
        "test_fixture.py",
    ]
    assert evaluate_expected_assertion(json.loads(fixture_path.read_text(encoding="utf-8")))
    skeleton = test_path.read_text(encoding="utf-8")
    assert "evaluate_expected_assertion" in skeleton
    assert not any(sentinel in skeleton for sentinel in SENTINELS)


def test_cli_refuses_accidental_overwrite(tmp_path, capsys) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("owned", encoding="utf-8")
    rc = main(
        [
            "span-to-test",
            str(SOURCE_PATH),
            "--span",
            "failing-tool",
            "-o",
            str(fixture_path),
        ]
    )
    assert rc == 2
    assert "already exists" in capsys.readouterr().err
    assert fixture_path.read_text(encoding="utf-8") == "owned"


def test_cli_receipt_defaults_to_stdout(tmp_path, capsys) -> None:
    fixture_path = tmp_path / "fixture.json"
    assert (
        main(
            [
                "span-to-test",
                str(SOURCE_PATH),
                "--span",
                "failing-tool",
                "-o",
                str(fixture_path),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema_version"] == RECEIPT_VERSION


def test_pytest_skeleton_refuses_non_sibling_path() -> None:
    with pytest.raises(SpanToTestContractError, match="sibling JSON"):
        render_pytest_skeleton("../fixture.json")
