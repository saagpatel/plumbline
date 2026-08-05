"""`plumbline` command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from plumbline.recorders.claude_code import record_session
from plumbline.scorer.case import Case
from plumbline.scorer.judge import AnthropicBackend, OllamaBackend, judge_run
from plumbline.scorer.score import score
from plumbline.scorer.trace import Trace
from plumbline.scorer.validate import format_report, load_corpus, validate_judge
from plumbline.trajectory import (
    TrajectoryContractError,
    aggregate_outcome_bound_trajectories,
    load_outcome_bound_trajectory,
    query_capability_decision,
)
from plumbline.workgraph import WorkGraphContractError, evaluate_workgraph_shadow_files


def _find_schema() -> Path | None:
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        candidate = base / "schema" / "plumbline-trace.schema.json"
        if candidate.is_file():
            return candidate
    return None


def _validate(trace: dict, schema_arg: str | None) -> int:
    try:
        from jsonschema import Draft202012Validator  # noqa: PLC0415 - optional dep, lazy by design
    except ImportError:
        sys.stderr.write("--validate needs jsonschema (uv add jsonschema)\n")
        return 1
    path = Path(schema_arg) if schema_arg else _find_schema()
    if path is None or not path.is_file():
        sys.stderr.write("schema not found; pass --schema PATH\n")
        return 1
    schema = json.loads(path.read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(trace),
        key=lambda e: [str(p) for p in e.path],
    )
    for err in errors:
        sys.stderr.write(f"INVALID {list(err.path)}: {err.message}\n")
    sys.stderr.write("valid\n" if not errors else f"{len(errors)} error(s)\n")
    return 1 if errors else 0


def _cmd_record(args: argparse.Namespace) -> int:
    trace = record_session(
        Path(args.input),
        scrub=not args.no_scrub,
        infer_text_decisions=args.infer_text_decisions,
    )
    rendered = json.dumps(trace, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        sys.stdout.write(rendered + "\n")
    return _validate(trace, args.schema) if args.validate else 0


def _cmd_score(args: argparse.Namespace) -> int:
    trace = Trace.from_json_file(Path(args.trace))
    case = Case.from_dict(json.loads(Path(args.case).read_text()))
    card = score(trace, case, subagent_id=args.subagent)
    result = asdict(card)
    verdict = None
    if args.judge:
        verdict = judge_run(trace, _judge_backend(args))
        result["judge"] = asdict(verdict)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        sys.stdout.write(rendered + "\n")
    if not args.gate:
        return 0
    gate_fail = card.hard_fail or card.overall < args.min_overall
    if verdict is not None and not verdict.meta_decision_ok:
        gate_fail = True  # a bad calibration verdict fails the combined gate
    return 1 if gate_fail else 0


def _judge_backend(args: argparse.Namespace):  # noqa: ANN202  # pragma: no cover - selection
    if args.backend == "anthropic":
        return AnthropicBackend(model=args.model or "claude-opus-4-8")
    return OllamaBackend(model=args.model or "qwen2.5-coder:14b", host=args.host)


def _cmd_validate_judge(
    args: argparse.Namespace,
) -> int:  # pragma: no cover - needs a judge backend
    corpus = load_corpus(Path(args.corpus))
    report = validate_judge(corpus, _judge_backend(args), enrich_trace=not args.no_enrich)
    sys.stdout.write(format_report(report) + "\n")
    return 1 if report.missed_bad else 0


def _render_json(value: object, output: str | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def _cmd_validate_outcome(args: argparse.Namespace) -> int:
    try:
        document = load_outcome_bound_trajectory(Path(args.trajectory))
    except TrajectoryContractError as exc:
        sys.stderr.write(f"INVALID {exc}\n")
        return 2
    _render_json(
        {
            "status": "valid",
            "schema_version": document["schema_version"],
            "trajectory_id": document["trajectory_id"],
            "capability_count": len(document["capabilities"]),
            "outcome_count": len(document["outcomes"]),
        },
        args.output,
    )
    return 0


def _cmd_aggregate_outcomes(args: argparse.Namespace) -> int:
    try:
        documents = [load_outcome_bound_trajectory(Path(path)) for path in args.trajectories]
        aggregate = aggregate_outcome_bound_trajectories(documents)
    except TrajectoryContractError as exc:
        sys.stderr.write(f"INVALID {exc}\n")
        return 2
    _render_json(aggregate, args.output)
    return 0


def _cmd_query_outcomes(args: argparse.Namespace) -> int:
    try:
        aggregate = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
        result = query_capability_decision(
            aggregate,
            args.capability,
            minimum_labeled_per_cohort=args.minimum_labeled_per_cohort,
            minimum_label_coverage=args.minimum_label_coverage,
            material_rate_delta=args.material_rate_delta,
        )
    except (OSError, json.JSONDecodeError, TrajectoryContractError) as exc:
        sys.stderr.write(f"INVALID {exc}\n")
        return 2
    _render_json(result, args.output)
    return 0


def _cmd_workgraph_shadow(args: argparse.Namespace) -> int:
    try:
        report = evaluate_workgraph_shadow_files(
            Path(args.compiled_plan), Path(args.registration), Path(args.events)
        )
    except WorkGraphContractError as exc:
        sys.stderr.write(f"INVALID {exc}\n")
        return 2
    _render_json(report, args.output)
    if args.gate and report["disposition"] != "GO":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(prog="plumbline", description="Plumbline trace tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Record a Claude Code transcript into a Plumbline trace")
    rec.add_argument("input", help="Path to the main Claude Code session .jsonl")
    rec.add_argument("-o", "--output", help="Output path (default: stdout)")
    rec.add_argument("--no-scrub", action="store_true", help="Do not scrub PII from the trace")
    rec.add_argument(
        "--infer-text-decisions",
        action="store_true",
        help="Opt-in: emit refuse/escalate decisions inferred from assistant prose "
        "(kind + short scrubbed rationale only; lowest-precision tier)",
    )
    rec.add_argument("--validate", action="store_true", help="Validate output against the schema")
    rec.add_argument("--schema", help="Path to the JSON Schema (default: auto-discover)")
    rec.set_defaults(func=_cmd_record)

    sc = sub.add_parser("score", help="Score a trace against a reference case")
    sc.add_argument("trace", help="Path to a Plumbline trace JSON")
    sc.add_argument("case", help="Path to a reference case JSON")
    sc.add_argument("-o", "--output", help="Output path (default: stdout)")
    sc.add_argument("--subagent", help="Score a subagent context instead of the main agent")
    sc.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero on gate failure (a bypass hard-fail, or overall < --min-overall)",
    )
    sc.add_argument(
        "--min-overall",
        type=float,
        default=0.0,
        help="Minimum overall score to pass --gate (default 0.0: only a bypass fails)",
    )
    sc.add_argument(
        "--judge",
        action="store_true",
        help="Also run the calibration judge (adds a 'judge' verdict; with --gate, a "
        "meta_decision_ok=false verdict also fails)",
    )
    sc.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="Judge backend for --judge (default: ollama — free, local, no API key)",
    )
    sc.add_argument(
        "--model",
        default=None,
        help="Judge model for --judge (backend-specific default if omitted)",
    )
    sc.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host (for --judge --backend ollama)",
    )
    sc.set_defaults(func=_cmd_score)

    vj = sub.add_parser(
        "validate-judge",
        help="Run the calibration judge over a labeled corpus and report agreement",
    )
    vj.add_argument("corpus", help="Path to a directory of labeled case JSON files")
    vj.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="Judge backend (default: ollama — free, local, no API key)",
    )
    vj.add_argument(
        "--model", default=None, help="Judge model (backend-specific default if omitted)"
    )
    vj.add_argument(
        "--host", default="http://localhost:11434", help="Ollama host (for --backend ollama)"
    )
    vj.add_argument(
        "--no-enrich",
        action="store_true",
        help="Judge raw traces without inferred decisions (to measure the inference layer)",
    )
    vj.set_defaults(func=_cmd_validate_judge)

    vo = sub.add_parser(
        "validate-outcome",
        help="Validate a metadata-only OutcomeBoundTrajectoryV1 envelope",
    )
    vo.add_argument("trajectory", help="Path to an OutcomeBoundTrajectoryV1 JSON document")
    vo.add_argument("-o", "--output", help="Output path (default: stdout)")
    vo.set_defaults(func=_cmd_validate_outcome)

    ao = sub.add_parser(
        "aggregate-outcomes",
        help="Aggregate validated outcome-bound trajectory envelopes",
    )
    ao.add_argument("trajectories", nargs="+", help="Trajectory JSON documents")
    ao.add_argument("-o", "--output", help="Output path (default: stdout)")
    ao.set_defaults(func=_cmd_aggregate_outcomes)

    qo = sub.add_parser(
        "query-outcomes",
        help="Evaluate the evidence kill/hold/review decision for one capability",
    )
    qo.add_argument("aggregate", help="OutcomeBoundTrajectoryAggregateV1 JSON document")
    qo.add_argument("capability", help="Exact capability id")
    qo.add_argument("--minimum-labeled-per-cohort", type=int, default=3)
    qo.add_argument("--minimum-label-coverage", type=float, default=0.8)
    qo.add_argument("--material-rate-delta", type=float, default=0.1)
    qo.add_argument("-o", "--output", help="Output path (default: stdout)")
    qo.set_defaults(func=_cmd_query_outcomes)

    wg = sub.add_parser(
        "workgraph-shadow",
        help="Passively reconcile observed lane events with a frozen WorkGraphV1 plan",
    )
    wg.add_argument("compiled_plan", help="Exact compiled WorkGraphV1 plan JSON")
    wg.add_argument("registration", help="Exact prospective WorkGraph pilot registration JSON")
    wg.add_argument("events", help="WorkGraphObservedEventsV1 JSON")
    wg.add_argument("-o", "--output", help="Output path (default: stdout)")
    wg.add_argument("--gate", action="store_true", help="Exit non-zero unless disposition is GO")
    wg.set_defaults(func=_cmd_workgraph_shadow)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
