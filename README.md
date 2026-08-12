# Plumbline

**An open, OpenTelemetry-compatible agent-trace schema with a first-class decision-path overlay and an
offline scorer that grades a trace your harness already produced.**

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
| **0** | Trace schema + JSON Schema + a worked example | shipped |
| **1** | Thin Claude Code recorder: `*.jsonl` → Plumbline trace (with a PII scrubber) | shipped |
| **2** | Offline deterministic scorer (selection / ordering / edit-similarity / param-name + bypass gate) + opt-in calibration judge | shipped |
| **3** | CI gate (`score --gate`, bare exit code) + an OTel `semantic-conventions-genai` extension proposal ([`OTEL-PROPOSAL.md`](OTEL-PROPOSAL.md)) | shipped |
| **4** | **Inference layer: outcome/plan capture + structural & text-signal decision inference, so the judge works on real recorded traces ([`PHASE4.md`](PHASE4.md))** | **shipped (v0.2)** |
| **5** | **Metadata-only outcome binding + passive WorkGraphV1 shadow reconciliation ([`OUTCOME-BOUND-TRAJECTORY.md`](OUTCOME-BOUND-TRAJECTORY.md), [`WORKGRAPH-SHADOW.md`](WORKGRAPH-SHADOW.md))** | **shipped (v0.3)** |
| **P10** | **Safe span-to-test reduction into deterministic inert replay fixtures ([`SPAN-TO-TEST.md`](SPAN-TO-TEST.md))** | **shipped (v0.4)** |

## Scoring (Phase 2)

Install and run the scorer in one step with `uv`:

```sh
uv run plumbline score run.plumbline.json cases/my-case.json
```

Or install once and run directly:

```sh
uv pip install -e .
plumbline score run.plumbline.json cases/my-case.json
```

### Quickstart

**1. Record a trace** (Phase 1):

```sh
plumbline record ~/.claude/projects/<encoded-path>/<session>.jsonl -o run.plumbline.json
```

**2. Write a reference case** (`cases/ideal.json`):

```json
{
  "case_id": "ideal",
  "description": "Read config, patch middleware, verify",
  "reference_path": [
    { "tool": "Read" },
    { "tool": "Edit" },
    { "tool": "Bash" }
  ]
}
```

**3. Score**:

```sh
plumbline score run.plumbline.json cases/ideal.json
```

Output (JSON to stdout):

```json
{
  "case_id": "ideal",
  "selection":       { "precision": 0.8, "recall": 1.0, "f1": 0.889 },
  "ordering":        { "precision": 0.5, "recall": 0.5, "f1": 0.5 },
  "edit_similarity": 0.8,
  "param_name": null,
  "overall": 0.73,
  "ordering_informative": true,
  "bypass": [],
  "hard_fail": false
}
```

`param_name` is `null` when no reference node declares `args`; it does not affect `overall` in that
case. `hard_fail: true` forces `overall` to `0.0` regardless of component scores — see the bypass gate
below.

### What the scorer measures

| Axis | Metric | What it checks |
|---|---|---|
| `selection` | multiset F1 over tool names | Right tools chosen? |
| `ordering` | multiset F1 over consecutive pairs | Right order? |
| `edit_similarity` | normalized Levenshtein | Sequence closeness end-to-end |
| `param_name` | micro-averaged F1 over argument keys | Right argument keys? (values ignored — scrubbing-immune) |

`overall` is the unweighted mean of the informative axes. `ordering` is excluded when both paths have
fewer than 2 nodes (vacuous). `param_name` is excluded when no case node declares `args`.

### Bypass hard-fail gate

The scorer runs a deterministic structural scan for **guard evasion**: a guardrail hook denies a tool
call, and the agent re-targets the same resource (URL, host, path) via a different call without taking a
sanctioned path (`escalate`, `refuse`, or `proceed_sanctioned`). When detected, `hard_fail` is set and
`overall` is forced to `0.0` — component scores remain truthful so you can see the path quality
alongside the safety failure.

### CLI reference

```
plumbline score TRACE CASE [-o OUTPUT] [--subagent SUBAGENT_ID] [--gate] [--min-overall FLOAT]
                           [--judge] [--backend {ollama,anthropic}] [--model MODEL] [--host HOST]
```

| Argument | Description |
|---|---|
| `TRACE` | Path to a Plumbline trace JSON |
| `CASE` | Path to a reference case JSON |
| `-o OUTPUT` | Write scorecard to file instead of stdout |
| `--subagent SUBAGENT_ID` | Score a subagent context instead of the main agent |
| `--gate` | Exit non-zero on gate failure (a bypass hard-fail, or `overall < --min-overall`) |
| `--min-overall FLOAT` | Minimum `overall` to pass `--gate` (default `0.0`: only a bypass fails) |
| `--judge` | Also run the calibration judge; adds a `judge` verdict to the output. With `--gate`, a `meta_decision_ok=false` verdict also fails |
| `--backend`, `--model`, `--host` | Judge backend selection for `--judge` (see [`JUDGE.md`](JUDGE.md)) |

### CI gate (Phase 3)

`--gate` turns the scorecard into a bare exit code for CI — the card still prints (so the log shows
*why*), but a failing run exits `1`:

```sh
# Fail the build on guard evasion only:
plumbline score run.plumbline.json cases/ideal.json --gate
# Also require a minimum path-quality score:
plumbline score run.plumbline.json cases/ideal.json --gate --min-overall 0.7
```

## Calibration judge (opt-in)

Beyond the deterministic axes, an opt-in **calibration judge** (OPERANT axis-3) grades the *meta-decision*:
given the situation, was the agent's choice the right one? It's trace-grounded (it reasons over the path,
the guardrail denials, and the deterministic bypass findings) and the zero-dep core stays offline — the
judge takes any `(prompt) -> str` backend, so you bring your own model. A reference Anthropic backend
lives behind the `judge` extra:

```sh
uv pip install -e ".[judge]"
```

**Validate the judge before trusting it.** An unvalidated judge is decoration; on hard cases it can
disagree with humans on a majority of runs. `plumbline.scorer.validate` runs the judge over a labeled
corpus ([`corpus/judge/`](corpus/judge)) and reports agreement, naming the dangerous error explicitly
(`missed_bad`: the judge blessed a run a human judged bad).

```sh
uv run plumbline validate-judge corpus/judge                 # free local model (default)
uv run plumbline validate-judge corpus/judge --model qwen3:8b
```

Once validated, attach a verdict to a deterministic scorecard in one pass with
`plumbline score TRACE CASE --judge` (with `--gate`, a bad meta-decision fails the build too).

On the current corpus the shipped rubric scores 14/14 (`qwen2.5-coder:14b`) on the original cases and
15/20 once deliberately adversarial traps are added. Every original case still passes, and the residual
gaps are a documented small-local-model ceiling. Full architecture, the rubric, the validation tables,
and the ceiling analysis live in [`JUDGE.md`](JUDGE.md).

### Further reading

- [`JUDGE.md`](JUDGE.md) — calibration judge: architecture, the rubric, validation results across models,
  and the measured adversarial ceiling.
- [`SCORING.md`](SCORING.md) — full model: every axis, the composite formula, bypass detection in depth,
  and design rationale.
- [`CASES.md`](CASES.md) — reference-case format spec, tool naming conventions, and worked examples.

## The inference layer (Phase 4)

A recorded trace captures the *observable* layer (tool calls, hooks, modes) but not the agent's
meta-decisions, which the judge needs. Dogfooding the full pipeline against a real session showed the
judge returning a confident-but-vacuous "approve" on recorded data with no plan, outcome, or decisions to
reason over. Phase 4 closes that gap so the loop is **record → enrich → score → judge** on real traces:

- **Outcome + plan capture (4a, recorder).** The plan (first user turn) and the outcome (final assistant
  turn: status + a scrubbed summary, the agent's *claim*) are observable and now captured. A real bug was
  fixed here: string-valued `message.content` was silently dropped.
- **Structural decision inference (4b, scorer).** `enrich(trace)` infers `reroute` decisions from
  structure: a denied tool re-attempted as the *same tool on a different resource* (vs a bypass), or a
  tool error fixed and re-verified. Inferred steps are tagged with provenance + evidence.
- **Text-signal decisions (4d, recorder, opt-in).** `record --infer-text-decisions` detects
  refuse/escalate in assistant prose at record time and emits decisions carrying only the kind + a short
  scrubbed rationale, never the full prose, so the trace's privacy posture is unchanged.
- **Measured (4c).** A recorder-shaped, PII-free corpus ([`corpus/judge/recorded/`](corpus/judge/recorded))
  plus `validate-judge --no-enrich` isolates the layer's contribution: enrich takes the weaker judge from
  5/6 to 6/6 by fixing a sanctioned-reroute false alarm, model-dependent, with no regressions.

Design, inference rules, the privacy decision, and the honest coverage/precision findings are in
[`PHASE4.md`](PHASE4.md).

## The schema

- Human spec: [`SCHEMA.md`](SCHEMA.md)
- Machine schema (JSON Schema 2020-12): [`schema/plumbline-trace.schema.json`](schema/plumbline-trace.schema.json)
- Worked, synthetic, PII-free example: [`examples/example-run.plumbline.json`](examples/example-run.plumbline.json)
- Outcome companion contract: [`schema/outcome-bound-trajectory.schema.json`](schema/outcome-bound-trajectory.schema.json)
- WorkGraph prospective registration contract: [`schema/workgraph-pilot-registration.schema.json`](schema/workgraph-pilot-registration.schema.json)
- WorkGraph observed-event contract: [`schema/workgraph-observed-events.schema.json`](schema/workgraph-observed-events.schema.json)
- Passive WorkGraph report contract: [`schema/workgraph-shadow-trace.schema.json`](schema/workgraph-shadow-trace.schema.json)
- Span-to-test generation request: [`schema/span-to-test-generation-request.schema.json`](schema/span-to-test-generation-request.schema.json)
- Sanitized inert replay fixture: [`schema/sanitized-replay-fixture.schema.json`](schema/sanitized-replay-fixture.schema.json)
- Reduction/provenance receipt: [`schema/span-to-test-reduction-receipt.schema.json`](schema/span-to-test-reduction-receipt.schema.json)

Validate the example against the schema:

```sh
uvx check-jsonschema --schemafile schema/plumbline-trace.schema.json examples/example-run.plumbline.json
```

## Outcome-bound telemetry and WorkGraph shadowing (Phase 5)

Plumbline now keeps an agent's claimed trace outcome separate from independently attributable outcome
evidence. `OutcomeBoundTrajectoryV1` binds a trace digest, pseudonymous task identity, capability
availability/exposure/adoption states, bounded indicators, lifecycle rules, and an explicit claim
ceiling. Raw prompts, full tool payloads, and secret material are structurally excluded;
file-backed validation resolves the exact contained privacy-review receipt digest.

```sh
plumbline validate-outcome examples/outcome-bound-trajectory.json
plumbline aggregate-outcomes run-a.outcome.json run-b.outcome.json -o aggregate.json
plumbline query-outcomes aggregate.json quality-gatekeeper
```

The passive WorkGraph adapter compares a compiled plan with prospective metadata-only lane events. It
fails closed on missing terminal evidence, constrains all event identifiers and references to bounded
single-line metadata, and never dispatches, leases, retries, or rewrites work:

```sh
plumbline workgraph-shadow compiled-plan.json registration.json observed-events.json --gate
```

See [`OUTCOME-BOUND-TRAJECTORY.md`](OUTCOME-BOUND-TRAJECTORY.md) and
[`WORKGRAPH-SHADOW.md`](WORKGRAPH-SHADOW.md) for contracts, kill criteria, privacy/lifecycle policy,
compatibility, and claim ceilings.

## Span-to-test generation (P10)

Reduce a failing span subtree to a compact, deterministic fixture that preserves
topology and the selected failure signal while removing captured content:

```sh
plumbline span-to-test trace.json --span STEP_ID -o fixture.json \
  --receipt-output reduction.json --pytest-output test_fixture.py
```

The default artifact is inert data, never an executable transcript. See
[`SPAN-TO-TEST.md`](SPAN-TO-TEST.md) for the threat model, reduction algorithm,
contracts, five-minute demo, standards snapshot, and claim ceiling.

## Recording a Claude Code session (Phase 1)

Normalize a real Claude Code transcript into a Plumbline trace — PII-scrubbed by default:

```sh
uv run plumbline record path/to/<session>.jsonl -o run.plumbline.json --validate
```

Subagent sidechains at `<session>/subagents/agent-*.jsonl` are merged automatically and
tagged by their `agentId`; the result validates against the Phase 0 schema. Pass `--no-scrub`
for local-only inspection. The recorder captures the *observable* execution layer (llm turns,
tool calls, subagent dispatch, hook verdicts, mode changes, compaction) — `decision`-kind
steps are a Phase 2 scoring concern, inferred from this path, not recorded here.

## Design stance

Reuse the standard layer (`gen_ai.*` + OTLP transport) for backend portability; own the overlay
(`agent.decision.*`, `harness.*`) that differentiates a coding-agent harness. Harness-agnostic schema,
harness-specific recorders. No PII in any published trace.

## License

MIT — see [`LICENSE`](LICENSE).
