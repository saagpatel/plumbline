# Plumbline

**An open, OpenTelemetry-compatible agent-trace schema with a first-class decision-path overlay — plus
(coming) an offline scorer that grades a trace your harness already produced.**

Plumbline answers a question nothing else does cleanly today: *given the trace of an agent run my own
harness produced, did the agent take the right path — the right tools, in the right order, with the right
arguments, and did it stay on-plan?* — and lets you gate that in CI.

## Why

As of mid-2026, generic trajectory scoring is solved (LangChain `agentevals`, DeepEval, Promptfoo,
Google ADK). The gap is the **intersection** none of them fills:

- **Offline, trace-in → score-out.** Existing tools make you instrument your agent with their SDK
  (DeepEval), drive the run themselves (Promptfoo), route through their span store (Phoenix), or run the
  agent inside their framework (Inspect). None scores *a trace you already have*.
- **The harness-control layer.** Subagent hierarchies, hook/guardrail verdicts, memory read/writes,
  context-compaction boundaries, permission-mode changes — OpenTelemetry GenAI leaves these to unmerged
  proposals; the rich models that exist (`agentevals`) are LangGraph-locked. Plumbline makes them
  first-class.
- **Decision calibration.** Everyone scores "task done / right tool." Plumbline also captures the
  *meta-decision* — refuse, escalate, take the sanctioned path — so the scorer can grade operating
  judgment, not just task completion.

## Status — phased

| Phase | Deliverable | State |
|---|---|---|
| **0** | **Trace schema + JSON Schema + a worked example** | **this repo** |
| 1 | Thin Claude Code recorder: `*.jsonl` → Plumbline trace (with a PII scrubber) | next |
| 2 | Offline decision-path scorer (composes existing trajectory metrics + a calibration judge) | planned |
| 3 | CI gate (bare exit code) + an OTel `semantic-conventions-genai` extension proposal | planned |

## The schema

- Human spec: [`SCHEMA.md`](SCHEMA.md)
- Machine schema (JSON Schema 2020-12): [`schema/plumbline-trace.schema.json`](schema/plumbline-trace.schema.json)
- Worked, synthetic, PII-free example: [`examples/example-run.plumbline.json`](examples/example-run.plumbline.json)

Validate the example against the schema:

```sh
uvx check-jsonschema --schemafile schema/plumbline-trace.schema.json examples/example-run.plumbline.json
```

## Design stance

Reuse the standard layer (`gen_ai.*` + OTLP transport) for backend portability; own the overlay
(`agent.decision.*`, `harness.*`) that differentiates a coding-agent harness. Harness-agnostic schema,
harness-specific recorders. No PII in any published trace.

## License

MIT — see [`LICENSE`](LICENSE).
