"""`plumbline` command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from plumbline.recorders.claude_code import record_session
from plumbline.scorer.case import Case
from plumbline.scorer.score import score
from plumbline.scorer.trace import Trace


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
    trace = record_session(Path(args.input), scrub=not args.no_scrub)
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
    rendered = json.dumps(asdict(card), indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        sys.stdout.write(rendered + "\n")
    if args.gate and (card.hard_fail or card.overall < args.min_overall):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plumbline", description="Plumbline trace tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Record a Claude Code transcript into a Plumbline trace")
    rec.add_argument("input", help="Path to the main Claude Code session .jsonl")
    rec.add_argument("-o", "--output", help="Output path (default: stdout)")
    rec.add_argument("--no-scrub", action="store_true", help="Do not scrub PII from the trace")
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
    sc.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
