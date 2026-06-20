# Plumbline Trace Schema — v0.1 (draft)

> Status: **Phase 0 / draft.** This is the contract everything else in Plumbline hangs off
> (the recorder produces it; the scorer consumes it). Breaking changes are expected until v1.0.
> Machine-checkable schema: [`schema/plumbline-trace.schema.json`](schema/plumbline-trace.schema.json).
> Worked instance: [`examples/example-run.plumbline.json`](examples/example-run.plumbline.json).

## What this is

Plumbline is an open, OpenTelemetry-compatible trace format for **a single agent run**, designed so the
**decision path** can be scored offline — *which tools fired, in what order, with what arguments, and
whether the agent stayed on-plan* — not just the final output.

It exists because of a specific, verified gap (mid-2026): generic trajectory scoring is solved
(`agentevals`, DeepEval, Promptfoo, Google ADK), but every tool either (a) makes you instrument your
agent with its SDK, (b) drives the run itself, or (c) routes the trace through its own span store —
**none ingests "the trace my harness already produced" as a first-class, portable, offline input.** And
none models the **harness-control layer** (subagent hierarchies, hook/guardrail verdicts, memory
read/writes, context-compaction boundaries, permission-mode changes), which OpenTelemetry GenAI leaves
to unmerged proposals.

Plumbline fills exactly that intersection.

## Design principles

1. **Reuse the standard layer.** Model/token/tool attributes use OTel `gen_ai.*` names, and a trace maps
   cleanly onto OTLP spans — so any OTLP backend can ingest it. We do not reinvent what OTel standardized.
2. **Own the overlay.** The differentiating layer — `agent.decision.*` and `harness.*` — is first-class
   here precisely because OTel will not model it for years.
3. **Offline, trace-in.** A Plumbline trace is a self-contained artifact. No running server, no SDK
   wrapped around your agent, no in-framework execution required to read or score it.
4. **Harness-agnostic schema, harness-specific recorders.** The schema names no vendor. Recorders
   (Claude Code first) normalize a harness's native logs into it.
5. **No PII.** `workspace`, `tool.arguments`, and memory keys can carry sensitive data. Recorders MUST
   offer a scrubbing pass, and every published sample MUST be synthetic or scrubbed.

## Top-level shape

```
PlumblineTrace
├── plumbline_version : "0.1.x"
├── run               : Run          # run-level context + the on-plan anchor
└── steps             : Step[]       # the decision DAG
```

## Run

The run-level context and — critically — the **plan anchor** the decision path is scored against.

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | Stable id (e.g. harness session id). |
| `harness` | object | `{ name, version?, entrypoint? }` — e.g. `claude-code`, `cursor`, `aider`. |
| `started_at` / `ended_at` | date-time | ISO 8601. |
| `model` | string? | Primary / launch model. |
| `workspace` | object? | `{ cwd?, git_branch? }`. **Scrub before publishing.** |
| `plan` | object? | The intent the run is graded against (see below). |
| `outcome` | object? | `{ status, summary? }` — terminal result. |

### The plan anchor

```
plan : { source, statement, items? }
```

`source ∈ {user_prompt, todo_list, approved_plan, system}`. The `statement` is the goal in prose;
`items[]` is an optional structured checklist (`{id, text, status}`). This is what makes *on-plan*
scoring possible later: a `decision` step can reference a `plan_ref`, and the scorer can ask "did the
realized path advance the stated plan, or drift off it?"

## Step — the decision DAG

Every node in the run is a `Step`: an OTel-shaped span plus the overlay. The DAG is reconstructible from
four join keys (all derived from real harness transcripts — see the OTel mapping appendix):

| Key | Meaning |
|---|---|
| `step_id` | Unique id for this step. |
| `parent_step_id` | DAG parent (the nesting / conversation spine). |
| `caused_by` | The step that *triggered* this one (a hook's gated call; a result's originating call). |
| `subagent_id` | Which agent context the step lives in. `null` = root/main; any value = a subagent sidechain. |

Common fields: `kind`, `started_at`, `ended_at?`, `status ∈ {ok, error, interrupted}?`,
`attribution?` (`{skill?, mcp_server?, mcp_tool?, agent?}`), and a kind-specific `attributes` bag.

### Step kinds

The **standard layer** (maps to OTel `gen_ai` spans):

| kind | Required attributes | Purpose |
|---|---|---|
| `llm` | — | A model generation turn. `gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`, `agent.reasoning?`. |
| `tool_call` | `gen_ai.tool.name` | A tool invocation. `gen_ai.tool.call.id`, `tool.arguments` (scrubbed), `tool.result.kind`, typed result fields. |
| `agent` | `agent.type` | A subagent dispatch. `agent.name?`, `agent.model?`, `agent.spawns_subagent_id` → child steps carry that `subagent_id`. |

The **overlay layer** (the harness-control layer nothing else models):

| kind | Required attributes | Purpose |
|---|---|---|
| `decision` | `agent.decision.kind` ∈ {proceed, proceed_sanctioned, refuse, escalate, reroute} | A meta-decision. `agent.decision.rationale?`, `agent.decision.on_plan?`, `agent.decision.plan_ref?`. The calibration lens. |
| `hook` | `harness.hook.name`, `harness.hook.verdict` ∈ {allow, deny, modify, warn} | A guardrail verdict. `harness.hook.event`, `harness.hook.prevented_continuation?`, `harness.hook.target_step_id?`. |
| `memory` | `harness.memory.op` ∈ {read, write, update, delete} | A persistent-memory operation. `harness.memory.scope?`, `harness.memory.key?` (scrubbed). |
| `compaction` | — | A context-compaction boundary. `harness.compaction.reason`, `harness.compaction.tokens_before/after?`. |
| `mode_change` | `harness.mode.to` | A permission-mode / mode transition. `harness.mode.kind`, `harness.mode.from`. |

`attributes` is intentionally an **open bag** — recorders may add keys — but the JSON Schema enforces the
required keys above per kind, so the overlay can never be silently dropped.

## OTel overlay mapping (appendix)

Plumbline is designed to round-trip with OTLP. The intent is that a Plumbline trace can be emitted as a
tree of OTel spans and, later, contributed back as an OTel `semantic-conventions-genai` *extension*
proposal — handing adoption to the SIG rather than forking the standard.

| Plumbline | OpenTelemetry GenAI (status mid-2026) |
|---|---|
| `step.kind = llm` | `gen_ai` **chat** span — *merged, Experimental*. |
| `step.kind = tool_call` | `gen_ai` **execute_tool** span — *merged, Experimental*. |
| `step.kind = agent` + `subagent_id` | `invoke_agent` span — *merged*; but agent-to-agent **handoff ordering** is **unmerged** (issue #2664, "experts needed"). Plumbline pins it down via `subagent_id` + `parent_step_id`. |
| `step.kind = decision` (`agent.decision.*`) | **No OTel convention.** Plumbline-owned. |
| `step.kind = hook` (`harness.hook.*`) | **Not standardized** (early SIG discussion only). Plumbline-owned. |
| `step.kind = memory` (`harness.memory.*`) | **Proposal-only** (Traceloop RFC #3460, vendor draft). Plumbline-owned. |
| `step.kind = compaction` (`harness.compaction.*`) | **No OTel attribute.** Plumbline-owned. |
| `step.kind = mode_change` (`harness.mode.*`) | **Nonexistent** in OTel GenAI. Plumbline-owned. |
| `gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.tool.name` | Reused verbatim from the OTel attribute registry. |
| `parent_step_id` / `caused_by` | OTel span parent + span links. |

Net: Plumbline reuses the merged, output-layer `gen_ai.*` primitives for portability, and adds the
control/decision layer that differentiates a coding-agent harness.

## Versioning

`plumbline_version` is `0.1.x` for this draft. Pre-1.0, the overlay namespaces (`agent.decision.*`,
`harness.*`) may change as the schema is validated against more harnesses and against the OTel proposal
process. The standard `gen_ai.*` layer tracks the OTel v1.36 baseline.
