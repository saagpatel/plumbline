# Stable failure messages and explicit synthetic list construction improve test readability.
# ruff: noqa: EM101, PERF401, TRY003

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from plumbline.cli import main
from plumbline.span_gaps import SpanGapContractError, analyze_span_gaps, load_json_file

FIXTURES = Path(__file__).parent / "fixtures" / "span-gaps"
SCHEMA = Path(__file__).parents[1] / "schema" / "plan-tool-span-gap-report.schema.json"


def fixture(name: str) -> dict:
    return load_json_file(FIXTURES / name)


def codes(report: dict) -> set[str]:
    return {item["code"] for item in report["findings"]}


@pytest.mark.parametrize(
    "name",
    ["healthy-single.json", "healthy-multi-agent.json", "retry-compaction.plumbline.json"],
)
def test_healthy_single_multi_parallel_retry_delegation_and_compaction_pass(name: str) -> None:
    report = analyze_span_gaps(fixture(name))
    assert report["disposition"] == "PASS"
    assert report["findings"] == []


def test_partial_capture_is_unknown_not_a_structural_false_positive() -> None:
    report = analyze_span_gaps(fixture("partial-capture.json"))
    assert report["disposition"] == "UNKNOWN"
    assert {item["classification"] for item in report["findings"]} == {"UNKNOWN"}
    assert {"MISSING_PARENT", "ORPHAN_TOOL_EXECUTION"} <= codes(report)


def test_malformed_graph_reports_direct_structural_evidence() -> None:
    report = analyze_span_gaps(fixture("malformed-graph.json"))
    assert report["disposition"] == "FAIL"
    assert {
        "BROKEN_SPAN_REFERENCE",
        "BROKEN_TRACE_REFERENCE",
        "CREATE_WITHOUT_INVOKE",
        "DUPLICATE_ID",
        "IMPOSSIBLE_ORDERING",
        "INVOKE_WITHOUT_CREATED_AGENT",
        "MULTIPLE_PARENTS",
        "TOOL_RESULT_WITHOUT_INVOCATION",
    } <= codes(report)
    evidence_ids = {item["evidence_id"] for item in report["evidence_index"]}
    assert all(set(item["evidence_refs"]) <= evidence_ids for item in report["findings"])


def test_cycle_is_detected_without_recursion_loop() -> None:
    report = analyze_span_gaps(fixture("cycle.json"))
    assert "CYCLE" in codes(report)
    assert report["disposition"] == "FAIL"


def test_plan_without_execution_and_disconnected_outcome_are_detected() -> None:
    trace = {
        "capture_scope": {"plans": "complete", "outcomes": "complete"},
        "spans": [
            {
                "trace_id": "1" * 32,
                "span_id": "1" * 16,
                "name": "plan",
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
                "attributes": {"gen_ai.operation.name": "plan"},
            },
            {
                "trace_id": "1" * 32,
                "span_id": "2" * 16,
                "name": "outcome",
                "start_time_unix_nano": 3,
                "end_time_unix_nano": 4,
                "attributes": {},
            },
        ],
    }
    assert {"PLAN_STEP_WITHOUT_EXECUTION", "DISCONNECTED_OUTCOME"} <= codes(
        analyze_span_gaps(trace)
    )


def test_workflow_subtree_escape_is_detected() -> None:
    trace = {
        "capture_scope": {"spans": "complete"},
        "spans": [
            {
                "trace_id": "2" * 32,
                "span_id": "1" * 16,
                "name": "invoke_workflow alpha",
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 100,
                "attributes": {
                    "gen_ai.operation.name": "invoke_workflow",
                    "gen_ai.workflow.id": "alpha",
                },
            },
            {
                "trace_id": "2" * 32,
                "span_id": "2" * 16,
                "name": "invoke_agent outside",
                "start_time_unix_nano": 2,
                "end_time_unix_nano": 90,
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
            },
            {
                "trace_id": "2" * 32,
                "span_id": "3" * 16,
                "parent_span_id": "2" * 16,
                "name": "execute_tool escaped",
                "start_time_unix_nano": 3,
                "end_time_unix_nano": 4,
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.workflow.id": "alpha",
                },
            },
        ],
    }
    assert "WORKFLOW_SUBTREE_ESCAPE" in codes(analyze_span_gaps(trace))


def test_ambiguous_configurable_mapping_is_explainable() -> None:
    mapping = {
        "schema_version": "PlanToolSpanRoleMapV1",
        "mode": "replace",
        "mappings": [
            {"id": "a", "role": "plan", "match": {"span_name_prefixes": ["vendor.do"]}},
            {
                "id": "b",
                "role": "execute_tool",
                "match": {"span_name_prefixes": ["vendor.do"]},
            },
        ],
    }
    trace = {
        "capture_scope": {"spans": "complete"},
        "spans": [
            {
                "trace_id": "3" * 32,
                "span_id": "3" * 16,
                "name": "vendor.do",
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
                "attributes": {},
            }
        ],
    }
    report = analyze_span_gaps(trace, mapping)
    assert "AMBIGUOUS_MAPPING" in codes(report)
    assert report["evidence_index"][0]["mapping_ids"] == ["a", "b"]


def test_vendor_alias_fixture_passes_and_preserves_raw_provenance() -> None:
    report = analyze_span_gaps(fixture("vendor-aliases.json"), fixture("vendor-alias-map.json"))
    assert report["disposition"] == "PASS"
    assert report["mapping"]["rule_count"] == 7
    assert report["mapping"]["unmapped_node_count"] == 0


def test_nested_otlp_shape_and_attribute_arrays_are_supported() -> None:
    trace = {
        "capture_scope": {"spans": "complete", "events": "complete"},
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "6" * 32,
                                "spanId": "1" * 16,
                                "name": "invoke_agent",
                                "startTimeUnixNano": 1,
                                "endTimeUnixNano": 20,
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "invoke_agent"},
                                    }
                                ],
                            },
                            {
                                "traceId": "6" * 32,
                                "spanId": "2" * 16,
                                "parentSpanId": "1" * 16,
                                "name": "execute_tool read",
                                "startTimeUnixNano": 2,
                                "endTimeUnixNano": 10,
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "execute_tool"},
                                    },
                                    {
                                        "key": "gen_ai.tool.call.id",
                                        "value": {"stringValue": "otlp-call"},
                                    },
                                ],
                                "events": [
                                    {
                                        "name": "tool_result",
                                        "timeUnixNano": 9,
                                        "attributes": [
                                            {
                                                "key": "gen_ai.tool.call.id",
                                                "value": {"stringValue": "otlp-call"},
                                            }
                                        ],
                                    }
                                ],
                            },
                        ]
                    }
                ]
            }
        ],
    }
    report = analyze_span_gaps(trace)
    assert report["disposition"] == "PASS"
    assert report["input"]["format"] == "otel"


def test_report_is_deterministic_schema_valid_and_scrubbed() -> None:
    trace = fixture("malformed-graph.json")
    trace["spans"][0]["name"] = "create_agent /Users/alice secret@example.com"
    first = analyze_span_gaps(trace)
    second = analyze_span_gaps(trace)
    assert first == second
    assert not list(Draft202012Validator(json.loads(SCHEMA.read_text())).iter_errors(first))
    rendered = json.dumps(first)
    assert "/Users/alice" not in rendered
    assert "secret@example.com" not in rendered


def test_analyzer_makes_no_network_or_model_calls(monkeypatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    assert analyze_span_gaps(fixture("healthy-single.json"))["disposition"] == "PASS"


def test_bounded_large_trace_performance() -> None:
    trace_id = "4" * 32
    spans = [
        {
            "trace_id": trace_id,
            "span_id": "1" * 16,
            "name": "invoke_agent root",
            "start_time_unix_nano": 0,
            "end_time_unix_nano": 100_000,
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
        }
    ]
    for index in range(1, 3001):
        spans.append(
            {
                "trace_id": trace_id,
                "span_id": f"{index + 1:016x}",
                "parent_span_id": "1" * 16,
                "name": "execute_tool synthetic",
                "start_time_unix_nano": index,
                "end_time_unix_nano": index + 1,
                "attributes": {"gen_ai.operation.name": "execute_tool"},
            }
        )
    started = time.monotonic()
    report = analyze_span_gaps({"capture_scope": {"spans": "complete"}, "spans": spans})
    assert time.monotonic() - started < 3.0
    assert report["input"]["node_count"] == 3001
    assert report["disposition"] == "PASS"


def test_cli_human_json_exit_codes_and_invalid_input(tmp_path, capsys) -> None:
    healthy = FIXTURES / "healthy-single.json"
    assert main(["span-gaps", str(healthy)]) == 0
    assert "Plan/tool span gaps: PASS" in capsys.readouterr().out
    assert main(["span-gaps", str(healthy), "--format", "json", "--gate"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "PlanToolSpanGapReportV1"
    assert main(["span-gaps", str(FIXTURES / "malformed-graph.json"), "--gate"]) == 1
    capsys.readouterr()
    assert main(["span-gaps", str(FIXTURES / "partial-capture.json"), "--gate"]) == 3
    capsys.readouterr()
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    assert main(["span-gaps", str(invalid), "--gate"]) == 2
    assert "INVALID" in capsys.readouterr().err


def test_invalid_mapping_fails_closed() -> None:
    with pytest.raises(SpanGapContractError, match="schema_version"):
        analyze_span_gaps(fixture("healthy-single.json"), {"mappings": []})
