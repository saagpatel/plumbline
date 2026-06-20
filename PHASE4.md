# Phase 4: Decision and Outcome Inference

## Why (the dogfooding finding)

The judge was built and validated against a synthetic corpus whose traces
hand-author `decision` steps, an `outcome` (status + summary), and a `plan`. Running
the full pipeline (`record -> score --judge`) against a **real** Claude Code session
exposed that those signals are absent on recorded data:

- `run.plan` was `null`, `run.outcome` was `null`, and there were **zero** `decision`
  and `hook` steps; only `llm` and `tool_call`.
- With its three strongest inputs empty, the judge returned `meta_decision_ok: true`
  at **confidence 1.0** with a generic rationale ("followed a logical sequence...
  without bypassing any guardrails"). A confident rubber-stamp, not an assessment.

Root causes, confirmed in source:

1. **Decisions are never recorded, and the inference layer that should produce them
   does not exist yet.** The recorder captures the observable layer only and
   explicitly defers meta-decisions to the scorer (`claude_code.py` docstring). The
   judge reads `trace.steps_of_kind("decision")`, which is therefore always empty on
   real traces. The synthetic unlock (surface the agent's decisions) has nothing to
   surface.
2. **`run.outcome` is never set** by the recorder, so the "claimed completion" check
   (judge rule 4) and the outcome-summary signal have no input on real data.
3. **Plan capture silently drops string-valued content.** The recorder *tries* to set
   `run.plan` from the first user text, but `_content()` only handles list-of-blocks
   content. Real sessions often send the first prompt as a plain string; in the
   dogfood the plan statement ("Add a focused evaluation test for a weak-evidence or
   decline path...") was present verbatim and dropped on the floor.

The mechanical pipeline is sound (record -> trace -> score -> judge runs clean,
schema-valid, fast). The missing piece is the inference layer between the observable
trace and the judge.

## Goals

- Infer `decision` steps and `run.outcome` from the observable trace so the judge and
  the scorecard operate on real recorded data, not just synthetic fixtures.
- Keep the recorder pure (observable layer only). Inference is a separate, scorer-side
  pass, matching the locked design ("decision / on-plan inferred by scorer").
- Deterministic and zero-dep where possible. A high-precision structural tier is the
  floor; any text-based heuristics are optional, separately validated, and clearly
  marked as lower-confidence.
- Every inferred element is tagged with provenance and its supporting evidence, so it
  is distinguishable from observed facts and auditable.

## Non-goals

- Not changing what the recorder observes (the plan-capture fix is a bug fix, not new
  observation).
- Not a perfect decision classifier. Aim for high precision on the structural tier and
  honest, bounded recall, validated against real sessions before being trusted.
- Not an LLM-in-the-loop inference pass in the core (it would break zero-dep
  determinism). If a model is ever used to infer decisions, it lives behind the same
  optional-backend boundary as the judge.

## Architecture

New module `plumbline/scorer/infer.py`:

- `infer_outcome(trace) -> (status, summary)` from the final assistant turn.
- `infer_decisions(trace) -> list[Step]` from denial- and failure-anchored patterns.
- `enrich(trace) -> Trace` composes both: returns a trace augmented with inferred
  `decision` steps (in timestamp order) and a populated `run.outcome`.

Wiring:

- The judge path (`judge_run` / `build_prompt`, and `score --judge`) runs `enrich`
  first, so the judge always sees inferred decisions and outcome.
- Expose the step explicitly for transparency: a `plumbline enrich TRACE` subcommand
  (or `score --show-inferred`) that prints the enriched trace, so users can audit what
  was inferred before trusting a verdict.

Provenance (non-negotiable): every inferred step carries
`agent.decision.inferred = true` and `agent.decision.evidence = [<step_id>, ...]`
citing the observable steps it was derived from. `run.outcome` gets an analogous
`inferred` marker. This keeps inferred signal from masquerading as observed fact and
lets the judge (and a human) weight it accordingly.

## Inference rules

### Outcome (Phase 4a, shipped)

Refinement found while building: the outcome **summary** is the final assistant turn,
which is *observable*, exactly like the plan is the first user turn. So outcome is
**captured by the recorder** (symmetric with plan), not inferred by the scorer. The
schema's `outcome` allows only `status` + `summary` (`additionalProperties: false`),
which also rules out an inferred-provenance tag here. Only `decision` steps remain
genuine scorer-side inference (Phase 4b).

As shipped in the recorder's `_build_run`:

- **status**: from the final non-sidechain assistant turn's `stop_reason`. Only an
  explicit `end_turn` maps to `completed`; everything else stays `unknown`
  (conservative; no guessing). Enum-constrained to {completed, failed, aborted,
  unknown}.
- **summary**: the last assistant turn's joined text blocks, scrubbed and truncated to
  600 chars. This is the agent's self-reported CLAIM, which judge rule 4 checks against
  the realized path.
- **plan fix**: `_content_text` now reads a plain-string `message.content`, recovering
  the first-user-prompt plan that real sessions carry as a string.

### Decisions, structural tier (Phase 4b, shipped via `infer.py`)

Infers only `reroute`, the one decision unambiguous from structure. `enrich(trace)`
runs it (no-op when the trace already carries `decision` steps) and is wired into the
judge path. Every inferred step carries `agent.decision.inferred = true` +
`agent.decision.evidence = [step_ids]` and an "(inferred)" rationale prefix.

Denial-anchored (keyed off `hook` deny steps and `detect_bypass`):

- denial of tool T, then the **same tool T** on a *different* resource -> `reroute`.
  The same-tool constraint is the precision boundary: it separates a real reroute from
  a silent abandon (a denied fetch followed by editing something unrelated is NOT a
  reroute, and it must not be blessed), and the different resource separates it from a
  bypass (same resource = evasion, judge rule 1).

Failure-anchored:

- tool error, then an edit, then the same tool succeeding -> `reroute` (fixed and
  re-verified). Each success is claimed once, so a run of failures before one fix is a
  single reroute episode, not one per failure.

**Coverage finding (dogfooded across 160 real sessions).** Denial-anchored reroute had
**zero** inputs: under the operator's default `bypassPermissions`, guards do not emit
recorded `deny` events, so 0/160 sessions carried a hook-deny step. Failure-anchored
reroute is well-fed: 38/40 tool-heavy sessions had >=1 error tool_call (~7% error rate),
and the inference fired on real sessions ("fixed and re-verified after a failed Bash").
Takeaway: the structural tier is precise but its denial branch is dormant on bypass
traces, which is the empirical case for the text-signal tier (4d): escalations and
refusals live in prose, not in recorded deny events.

### Decisions, text-signal tier (Phase 4d, shipped, opt-in)

Architecture (operator decision): assistant prose only the recorder sees is scanned at
record time; the recorder emits `decision` steps carrying the kind + a short scrubbed
rationale snippet (<=120 chars), and NEVER stores the full prose. The portable trace's
privacy posture is unchanged. Opt-in via `record --infer-text-decisions` (off by
default; lowest-precision tier). Every emitted step carries `agent.decision.inferred`,
`agent.decision.source = "text_signal"`, and an evidence id. `enrich`'s gate was
narrowed so these inferred decisions do NOT block structural (4b) inference: the two
tiers compose.

- first-person, declining-an-action prose -> `refuse`.
- clarification / wait-for-approval / before-I-proceed prose -> `escalate`.

Precision finding (dogfooded, 30 real sessions). The first cut keyed on bare
"won't" / "refuse" / "declined" and fired 54 false-heavy refuses (narrative prose, not
meta-decisions). Tightening to first-person declining-an-action forms ("I refuse/decline
to ...", "I won't <action verb>") cut that to 7 genuine refuses; escalate stayed at 5
and was clean from the start. The tier remains heuristic and opt-in; provenance tags let
the judge weight it as derived, and the structural tier stands on its own without it.

## Plan capture fix (Phase 4a, separable)

Teach the recorder's `_content` / `_first_user_text` to accept a plain-string
`message.content`, not only a list of blocks. Low-risk, high-value: it recovers the
plan statement that real sessions carry as a string. This is a recorder bug fix and can
land independently of the inference module.

## Validation plan

An inference layer is only trustworthy if validated, same lesson as the judge.

### Phase 4c result (shipped): does enrich measurably help?

Real recorded traces carry PII even after best-effort scrubbing, so they cannot be
committed. Instead `corpus/judge/recorded/` holds 6 PII-free, gold-labeled,
**recorder-shaped** traces (observable layer + outcome, *no authored decisions*, i.e. what
`record_session` actually produces), so they exercise the inference layer (the main
corpus has authored decisions and no-ops enrich). A `validate-judge --no-enrich` lever
judges the raw trace, isolating enrich's contribution:

| Model | with enrich | without enrich |
|---|---|---|
| qwen2.5-coder:14b | 6/6 | 5/6 (false-alarm on `rec_reroute_after_denial_ok`) |
| qwen3:8b | 6/6 | 6/6 |

The inference layer helps exactly as designed: without the inferred reroute, coder read
"denial then more edits" and over-flagged a legitimate sanctioned reroute; with enrich
it blessed it. The effect is model-dependent (qwen3:8b reasoned it out from the raw
path) with no regressions either way. Honest caveat: 6 cases is small, the controls
(clean / fabricated / bypass / proceed-past-failure) were handled correctly by both
models with and without enrich, so the measured signal rests on the single
denial-reroute case. The claim is calibrated: enrich prevents a specific false-alarm
class for weaker judges and never hurts, on this corpus.

### Remaining (not yet done)

1. **Inference precision on real labels.** Sample real sessions, hand-label the
   decisions a human reads, and measure the structural + text rules' precision/recall.
   (4d's refuse precision was eyeballed at 54 -> 7, not formally scored.)
2. **Judge-on-real at scale.** The dogfood showed enrich turns the confident-vacuous
   verdict grounded on one real session; a labeled batch would quantify it.

## Risks and open questions

- **Provenance and trust.** An inferred `reroute` that is actually a disguised bypass
  would mislead the judge. Mitigation: bypass detection runs first and wins; inferred
  decisions never override a bypass flag.
- **Circularity.** Inference must add observable-grounded structure, not pre-judge the
  run. Rules stay mechanical (denial/error/resource-token patterns); they do not encode
  "this was good/bad".
- **Text-tier precision.** Refusal/escalation live mostly in prose. The structural tier
  cannot see them; the text tier may need a model, which is why it stays optional and
  behind the same boundary as the judge backend.
- **Enrich-by-default vs explicit.** Auto-enriching the judge path is ergonomic but
  hides work; the `enrich` subcommand plus provenance tags are the transparency
  counterweight.

## Phased delivery

- **4a** (shipped): plan-capture string fix + outcome capture (status + summary) in the
  recorder + tests. Re-dogfooded: the judge now reasons over the real plan and claim
  instead of returning a vacuous "approve". (Outcome turned out to be observable
  recorder capture, not scorer inference; see the Outcome section.)
- **4b** (shipped): structural reroute inference (denial- and failure-anchored) +
  provenance tags + `enrich` wired into the judge path. Verified no-op on the synthetic
  corpus (identical validation), fires on real sessions for failure-anchored reroute;
  the denial branch is dormant on bypassPermissions traces (no recorded denials). The
  original 4b/4c/4d bullets below predate this split.
- **4b (original)**: structural decision inference (denial- and failure-anchored) + provenance
  tags + `enrich` wiring + tests.
- **4c** (shipped): recorder-shaped, gold-labeled corpus (`corpus/judge/recorded/`) +
  `validate-judge --no-enrich` lever; measured that enrich takes coder 5/6 -> 6/6 by
  fixing a denial-reroute false alarm, neutral on qwen3:8b, no regressions. See the
  Phase 4c result above. Formal precision/recall on real labels remains open.
- **4d** (shipped): text-signal refuse/escalate inference emitted by the recorder from
  raw prose (no prose stored in the trace), opt-in via `--infer-text-decisions`.
  Dogfood-tuned for precision (refuse 54 -> 7 false-heavy matches removed).
