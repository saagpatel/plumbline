# Proposal: Harness-Control Overlay for OTel GenAI Semantic Conventions

**Repository:** [Plumbline](https://github.com/saagpatel/plumbline) (MIT)
**Status:** Draft for SIG discussion
**Schema version this document tracks:** Plumbline `0.1.x`
**OTel baseline:** v1.36, `semantic-conventions/docs/gen-ai/`

---

## 1. Motivation — the gap

OpenTelemetry GenAI (mid-2026) has made meaningful progress on the *output layer* of agent systems:

- `gen_ai` **chat** spans — merged, Experimental
- `gen_ai` **execute_tool** spans — merged, Experimental
- `gen_ai` **invoke_agent** span — merged

But a structural gap remains: none of the merged conventions model the **harness-control layer** that
governs how a coding-agent harness actually behaves during a run. Specifically:

| Layer | OTel GenAI status (mid-2026) |
|---|---|
| Agent-to-agent handoff ordering | `invoke_agent` merged; **handoff ordering unresolved** (issue #2664, "experts needed") |
| Guardrail / hook verdicts | **Early SIG discussion only** — no merged attribute |
| Memory read/write ops | **Proposal-only** (Traceloop RFC #3460, vendor draft) — not merged |
| Context-compaction boundaries | **No attribute** — not discussed in any merged convention |
| Permission-mode changes | **Nonexistent** in OTel GenAI — no registered attribute |

The consequence is that anyone who builds an agent harness with hooks, subagent hierarchies, memory
operations, or permission gating must either invent a private schema or leave that signal invisible to
their observability backend. Two outcomes follow: (a) safety and control telemetry is dark, and (b) the
same invented schema proliferates across vendors with no interoperability.

This document proposes a concrete set of span kinds and attributes — the Plumbline decision-path overlay
— as a candidate extension to the OTel GenAI semantic conventions. Plumbline has implemented these
attributes as a shipped, open-source schema (JSON Schema 2020-12) and a working Claude Code recorder. We
are offering it as a reference producer and asking the SIG to consider adopting the overlay as an official
extension.

---

## 2. What we reuse vs. what we add

### Reused verbatim from the OTel attribute registry

These attributes appear on standard-layer spans and are not redefined:

| Attribute | Span kind | Source |
|---|---|---|
| `gen_ai.request.model` | `llm` | OTel `gen_ai` attribute registry |
| `gen_ai.usage.input_tokens` | `llm` | OTel `gen_ai` attribute registry |
| `gen_ai.usage.output_tokens` | `llm` | OTel `gen_ai` attribute registry |
| `gen_ai.response.finish_reasons` | `llm` | OTel `gen_ai` attribute registry |
| `gen_ai.tool.name` | `tool_call` | OTel `gen_ai` attribute registry |
| `gen_ai.tool.call.id` | `tool_call` | OTel `gen_ai` attribute registry |

Plumbline traces that contain `llm` and `tool_call` steps are already valid OTLP consumers of the merged
conventions — no translation needed.

### Added by this proposal

Two new namespaces:

- **`agent.decision.*`** — the meta-decision layer: did the agent proceed, refuse, escalate, or take a
  sanctioned alternate path? This is the calibration signal that separates operating judgment from task
  completion.
- **`harness.*`** — the harness-control layer: hook verdicts, memory operations, context-compaction
  events, and permission-mode transitions.

These are not vendor attributes. They describe harness behaviors that are structurally present in any
sufficiently capable agent harness (Claude Code, Cursor, Aider, OpenHands), regardless of the underlying
model or provider.

---

## 3. Proposed attributes

The tables below are derived directly from `schema/plumbline-trace.schema.json` `allOf` constraint
blocks. Attribute names, types, and enum values are taken verbatim from the schema.

### 3.1 `decision` span

A `decision` span captures a meta-decision the agent made: which path to take, whether to proceed,
and whether the chosen path stayed on the stated plan. It is the primary calibration signal.

| Attribute | Type | Required | Enum values | Semantics |
|---|---|---|---|---|
| `agent.decision.kind` | string | **Required** | `proceed`, `proceed_sanctioned`, `refuse`, `escalate`, `reroute` | The type of decision taken. `proceed_sanctioned` = the agent chose a guarded/indirect path rather than the direct one. `reroute` = the agent redirected to a different resource or approach mid-task. |
| `agent.decision.rationale` | string | Optional | — | Human-readable rationale the agent produced for the decision. |
| `agent.decision.on_plan` | boolean | Optional | — | Whether the agent assessed this decision as on-plan with the run's stated intent. |
| `agent.decision.plan_ref` | string | Optional | — | The `plan.items[].id` this decision advances or departs from. |

**`agent.decision.kind` enum semantics:**

| Value | Meaning |
|---|---|
| `proceed` | Normal continuation; no special gating condition. |
| `proceed_sanctioned` | A guarded path exists (a hook might deny the direct route); the agent proactively chose the sanctioned alternate. |
| `refuse` | The agent declined to take an action, typically citing a safety or policy reason. |
| `escalate` | The agent surfaced a condition to the operator rather than attempting resolution autonomously. |
| `reroute` | The agent redirected mid-task to a different resource, tool, or subpath. |

### 3.2 `hook` span

A `hook` span captures the verdict of a guardrail or policy hook that ran before, during, or after a
tool call. This is the primary safety telemetry signal; without it, hook denials are invisible to
observability backends.

| Attribute | Type | Required | Enum values | Semantics |
|---|---|---|---|---|
| `harness.hook.name` | string | **Required** | — | Identifier for the hook that ran (e.g. `bash-egress-guard`, `PreToolUse`). |
| `harness.hook.verdict` | string | **Required** | `allow`, `deny`, `modify`, `warn` | The hook's decision. |
| `harness.hook.event` | string | Optional | — | The harness event that triggered the hook (e.g. `PreToolUse`, `PostToolUse`). |
| `harness.hook.prevented_continuation` | boolean | Optional | — | Whether the hook's verdict halted the originating tool call. |
| `harness.hook.target_step_id` | string | Optional | — | The `step_id` of the tool call this hook verdict applies to. Allows structural join without relying on `caused_by`. |
| `harness.hook.reason` | string | Optional | — | Human-readable explanation for a `deny` or `modify` verdict. |

**`harness.hook.verdict` enum semantics:**

| Value | Meaning |
|---|---|
| `allow` | Hook ran and permitted the call to proceed unchanged. |
| `deny` | Hook blocked the call. `harness.hook.prevented_continuation` should be `true`. |
| `modify` | Hook allowed the call but mutated its arguments or output. |
| `warn` | Hook allowed the call but emitted a diagnostic. |

### 3.3 `memory` span

A `memory` span captures a persistent-memory operation: reading, writing, updating, or deleting an entry
in the agent's cross-session or project-scoped memory store.

| Attribute | Type | Required | Enum values | Semantics |
|---|---|---|---|---|
| `harness.memory.op` | string | **Required** | `read`, `write`, `update`, `delete` | The type of memory operation performed. |
| `harness.memory.scope` | string | Optional | — | The scope of the memory store accessed (e.g. `project`, `global`, `session`). |
| `harness.memory.key` | string | Optional | — | The key or path within the memory store. **MUST be scrubbed** if it contains filenames, paths, or user-identifying content before publication. |

### 3.4 `compaction` span

A `compaction` span marks a context-compaction boundary: the point at which the harness truncated,
summarized, or otherwise reduced the active context window. This boundary is structurally important for
scoring because tool calls and decisions made before vs. after compaction have different information
access; without this span, a scorer cannot distinguish "the agent forgot" from "the agent never knew."

| Attribute | Type | Required | Enum values | Semantics |
|---|---|---|---|---|
| `harness.compaction.reason` | string | Optional | — | Why compaction was triggered. Common values: `auto` (threshold hit), `manual` (operator-requested). |
| `harness.compaction.tokens_before` | integer | Optional | — | Approximate token count in the context window before compaction. |
| `harness.compaction.tokens_after` | integer | Optional | — | Approximate token count in the context window after compaction. |

Note: the JSON Schema enforces no required attributes on `compaction` beyond the common step fields. The
`harness.compaction.*` attributes are recommended when the harness has access to them.

### 3.5 `mode_change` span

A `mode_change` span records a transition in the harness's permission or operating mode. In harnesses
with layered permission systems, mode transitions change what tool calls are allowed, which hooks are
active, and what the agent can do autonomously.

| Attribute | Type | Required | Enum values | Semantics |
|---|---|---|---|---|
| `harness.mode.to` | string | **Required** | — | The mode the harness transitioned into. |
| `harness.mode.from` | string | Optional | — | The mode the harness transitioned from. |
| `harness.mode.kind` | string | Optional | — | The type of mode being changed (e.g. `permission_mode`, `agent_mode`). |

---

## 4. Span and link mapping

### How Plumbline steps map onto OTLP

Each Plumbline step is an OTLP span. The four join keys map as follows:

| Plumbline field | OTLP equivalent | Notes |
|---|---|---|
| `step_id` | Span ID | Stable within a trace. |
| `parent_step_id` | Span parent | Forms the conversation/nesting spine — the structural hierarchy. |
| `caused_by` | Span link | A causal link, not a parent-child relationship. Used when a hook is triggered by a tool call (`hook.caused_by = tool_call.step_id`) or when a decision follows from a result. |
| `subagent_id` | Resource attribute or span attribute on the `invoke_agent` span | Pins the sidechain context. All steps from a subagent carry the same `subagent_id`; the root agent's steps carry `null`. |

### Resolving handoff ordering (OTel issue #2664)

The `invoke_agent` span is now merged, but the ordering guarantee for agent-to-agent handoffs remains
unresolved. Plumbline addresses this via two structural constraints:

1. An `agent`-kind step has `agent.spawns_subagent_id` pointing to the subagent context it spawns.
2. All steps from the spawned subagent carry `subagent_id` equal to that value, and their
   `parent_step_id` chains back to the `agent` step.

This makes it unambiguous which LLM turns and tool calls belong to which agent sidechain and in what
order they occurred relative to the parent. The OTLP rendering is: the `agent` step becomes the
`invoke_agent` span; subagent steps are child spans of it in their own trace segment. When the SIG
resolves issue #2664, Plumbline's `subagent_id` field provides the reference implementation for the
binding.

### `caused_by` as a span link

In OTLP, a span link (as opposed to parent-child) is the correct primitive for causal but non-nested
relationships. Specifically:

- A `hook` step triggered by a `tool_call` uses `caused_by` pointing to the tool call's `step_id`. The
  hook is not a child of the tool call in execution terms; it is a side-effect triggered by it.
- A `decision` step following a hook verdict uses `caused_by` pointing to the hook step.

This preserves the distinction between the **conversation spine** (parent-child via `parent_step_id`)
and **causal triggers** (links via `caused_by`), which matters for replay and scoring.

---

## 5. Worked example

The following span tree is derived from `examples/example-run.plumbline.json` (synthetic, PII-free).
The run's plan: add a rate-limit guard to the public API and verify it with the test suite.

```
run_synthetic_0001
│
├── s1  [mode_change]  harness.mode.kind=permission_mode  from=default  to=acceptEdits
│
├── s2  [llm]  gen_ai.request.model=claude-opus-4-8  in=1840  out=220
│   │
│   ├── s3  [tool_call]  gen_ai.tool.name=Read  → /repo/src/api.py
│   │
│   ├── s4  [decision]  caused_by=s3
│   │         agent.decision.kind=proceed_sanctioned
│   │         agent.decision.rationale="Route through sanctioned config path"
│   │         agent.decision.on_plan=true  plan_ref=p2
│   │
│   ├── s5  [agent]  agent.type=code-reviewer  spawns_subagent_id=agent_rev1
│   │   ├── s6  [llm]  subagent_id=agent_rev1  in=900  out=140
│   │   ├── s7  [tool_call]  subagent_id=agent_rev1  gen_ai.tool.name=Read
│   │   └── s8  [llm]  subagent_id=agent_rev1  in=1100  out=310
│   │
│   ├── s9  [tool_call]  gen_ai.tool.name=Bash  status=interrupted
│   │         (curl to non-allowlisted external host)
│   │
│   ├── s10 [hook]  caused_by=s9
│   │         harness.hook.name=bash-egress-guard
│   │         harness.hook.event=PreToolUse
│   │         harness.hook.verdict=deny
│   │         harness.hook.prevented_continuation=true
│   │
│   ├── s11 [decision]  caused_by=s10
│   │         agent.decision.kind=escalate
│   │         agent.decision.rationale="Egress guard-denied; surface to operator"
│   │         agent.decision.on_plan=true
│   │
│   ├── s12 [memory]  attribution.skill=bank
│   │         harness.memory.op=write  scope=project
│   │         key=lessons/egress-guard-blocks-curl
│   │
│   ├── s13 [tool_call]  gen_ai.tool.name=Edit  → /repo/config/middleware.py
│   ├── s14 [tool_call]  gen_ai.tool.name=Bash  (pytest -q)  exit_code=0
│   │
│   └── s15 [compaction]
│           harness.compaction.reason=auto
│           harness.compaction.tokens_before=142000
│           harness.compaction.tokens_after=38000
│
└── s16 [llm]  parent=s15  gen_ai.request.model=claude-opus-4-8  in=38500  out=180
```

Key observations for the SIG:

- **s4 (`proceed_sanctioned`) as a span link target:** a scorer can ask "before s4, was the agent on a
  path that a hook would have blocked?" and use the `agent.decision.kind` value to confirm the agent
  self-corrected proactively — this signal is invisible without the decision overlay.
- **s9 → s10 (`caused_by`):** the hook deny is a span link from s10 to s9, not a child span. The Bash
  call is interrupted (`status=interrupted`); the hook is a side-effect that reads the same timestamp.
  Without `caused_by`, the causal relationship must be inferred from timestamp proximity — fragile.
- **s10 → s11 (`caused_by`):** the escalation decision is caused by the hook deny, not by the LLM turn
  directly. This chain (`tool_call → hook → decision`) is the safety telemetry loop: it shows the
  harness blocked the call and the agent escalated rather than retrying.
- **s5 → s6/s7/s8 (subagent_id):** the `code-reviewer` subagent's three steps are tagged with
  `subagent_id=agent_rev1`. Their ordering relative to the parent (s5) is unambiguous via
  `parent_step_id` chaining, resolving the ordering concern from issue #2664.
- **s15 (compaction):** without this span, the token count at s16 (38,500) looks anomalous versus the
  count at s2 (1,840 in, trending up). With it, the scorer knows this is a fresh context window post-
  compaction and can adjust its information-access assumptions accordingly.

---

## 6. What we're asking the SIG

### Primary ask

Adopt the five overlay span kinds (`decision`, `hook`, `memory`, `compaction`, `mode_change`) and their
attribute namespaces (`agent.decision.*`, `harness.*`) as an **official OTel GenAI extension**, with
Plumbline as the reference producer.

"Extension" here means: the attributes are registered in the OTel attribute registry, the span kinds
appear in the GenAI semantic conventions document, and the status is initially Experimental — matching
the status of the merged `chat` and `execute_tool` spans.

### Secondary asks

1. **Resolve issue #2664 (handoff ordering) by adopting `subagent_id` as a first-class span attribute.**
   Plumbline's `subagent_id` field provides a working binding: every step in a subagent sidechain carries
   the same value, and that value matches `agent.spawns_subagent_id` on the parent `invoke_agent` span.

2. **Register `harness.hook.verdict` and `harness.hook.name` as the canonical guardrail telemetry
   attributes.** The SIG has discussed hook telemetry without merging anything; this proposal gives that
   discussion a concrete schema to react to.

3. **Register `harness.compaction.*` as the context-window boundary signal.** There is currently no
   attribute for this in any GenAI convention. Every harness with a finite context window compacts;
   making the boundary observable is a prerequisite for accurate trajectory scoring.

### What we are not asking

- We are not asking the SIG to adopt Plumbline's scoring model, CI gate, or recorder.
- We are not asking to replace the merged `gen_ai.*` attributes — Plumbline reuses them verbatim.
- We are not proposing new transport semantics — OTLP parent/link semantics handle everything described
  above without modification.

### Reference materials

| Resource | Location |
|---|---|
| JSON Schema (authoritative attribute names) | `schema/plumbline-trace.schema.json` |
| Human schema spec | `SCHEMA.md` |
| Synthetic worked example | `examples/example-run.plumbline.json` |
| Recorder (Claude Code → Plumbline) | `src/plumbline/recorders/claude_code.py` |
| License | MIT (`LICENSE`) |

---

## Appendix: Full attribute index

All attributes proposed in this document, in alphabetical order.

| Attribute | Span kind | Type | Required | Values |
|---|---|---|---|---|
| `agent.decision.kind` | `decision` | string | Required | `proceed`, `proceed_sanctioned`, `refuse`, `escalate`, `reroute` |
| `agent.decision.on_plan` | `decision` | boolean | Optional | — |
| `agent.decision.plan_ref` | `decision` | string | Optional | — |
| `agent.decision.rationale` | `decision` | string | Optional | — |
| `harness.compaction.reason` | `compaction` | string | Optional | — |
| `harness.compaction.tokens_after` | `compaction` | integer | Optional | — |
| `harness.compaction.tokens_before` | `compaction` | integer | Optional | — |
| `harness.hook.event` | `hook` | string | Optional | — |
| `harness.hook.name` | `hook` | string | Required | — |
| `harness.hook.prevented_continuation` | `hook` | boolean | Optional | — |
| `harness.hook.reason` | `hook` | string | Optional | — |
| `harness.hook.target_step_id` | `hook` | string | Optional | — |
| `harness.hook.verdict` | `hook` | string | Required | `allow`, `deny`, `modify`, `warn` |
| `harness.memory.key` | `memory` | string | Optional | Scrub before publishing |
| `harness.memory.op` | `memory` | string | Required | `read`, `write`, `update`, `delete` |
| `harness.memory.scope` | `memory` | string | Optional | — |
| `harness.mode.from` | `mode_change` | string | Optional | — |
| `harness.mode.kind` | `mode_change` | string | Optional | — |
| `harness.mode.to` | `mode_change` | string | Required | — |

Reused OTel attributes (not proposed here, already registered):

| Attribute | Span kind | Source |
|---|---|---|
| `gen_ai.request.model` | `llm` | OTel attribute registry |
| `gen_ai.response.finish_reasons` | `llm` | OTel attribute registry |
| `gen_ai.tool.call.id` | `tool_call` | OTel attribute registry |
| `gen_ai.tool.name` | `tool_call` | OTel attribute registry |
| `gen_ai.usage.input_tokens` | `llm` | OTel attribute registry |
| `gen_ai.usage.output_tokens` | `llm` | OTel attribute registry |
