# Outcome-bound trajectory telemetry

`OutcomeBoundTrajectoryV1` is a metadata-only companion to a Plumbline trace. It joins an exact trace
digest, a pseudonymous task identity, capability observations, independently attributable outcomes,
and bounded cost/rework indicators. It does not replace the trace schema or introduce another
collector.

Machine contract: [`schema/outcome-bound-trajectory.schema.json`](schema/outcome-bound-trajectory.schema.json)
and the stricter cross-field validator in `plumbline.trajectory`.

## Why a companion envelope

`run.outcome` in a Plumbline trace is the agent's terminal claim. It is useful execution evidence, but
it is not an independent outcome label. Putting external test, operator, or product observations into
the trace would blur those authorities and encourage self-reported completion to be scored as success.
The companion envelope therefore digest-references the trace and keeps each outcome's source,
authority, confidence, timestamp, and independence explicit.

The envelope also references `CodexCapabilityUsageSnapshotV1` by digest. That snapshot measures
observed skill reads and attributable tool activity within its stated window. It is an availability and
exposure baseline—not proof that a capability changed behavior or outcomes.

## State model

Capability observations use distinct states:

- `available`: the capability was present in a digest-bound snapshot or catalog.
- `exposed`: the run has a trace span showing the capability reached the agent workflow.
- `adopted`: exposure is joined to a named outcome observation. This is still an observation, not a
  causal claim.

Claims use a separate progression:

- `observation`: one or more bound measurements.
- `correlation`: a descriptive association with independent outcome labels.
- `experiment`: a declared design reference and independently observed outcomes.
- `causality`: reserved for a design that can support that claim; the local validator requires the
  same minimum bindings as an experiment, while external methodological review remains necessary.

No state is inferred merely because a skill was installed, configured, or read.

## Privacy, authorization, and lifecycle

The public profile is fixed at `metadata-only-v1`:

- `raw_prompts_included`, `full_tool_payloads_included`, and `secret_material_included` must all be
  `false`.
- Task identity must be pseudonymous.
- Evidence and span references carry identities/digests, not payload content.
- Identifier, authority, provenance, and reference fields accept only bounded opaque
  metadata syntax; free-form or multiline payload text fails validation.
- `authorization_audience` names who may read the envelope.
- `privacy.verification` binds a named ruleset/verifier to an exact contained,
  non-symlink privacy-review receipt and timestamp. File-backed validation resolves
  and hashes those receipt bytes. It does not prove independent authorship or that a
  downstream store enforced the review.
- `retention_days`, `expires_at`, and `deletion_mode` are mandatory and cross-checked.
- Private source deletion remains the source owner's job. `delete-envelope-and-private-source` is a
  requested lifecycle mode, not proof that a downstream store complied.

OpenTelemetry's GenAI conventions keep model/tool content attributes opt-in; Plumbline follows that
boundary and uses the standard `gen_ai.*` layer where it applies. OpenAI Agents SDK tracing similarly
supports custom processors but can include sensitive data unless configured otherwise, so a producer
must explicitly select this metadata-only profile rather than assuming a tracing default is safe.

Primary references:

- [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/)
- [OpenTelemetry GenAI span conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/model/gen-ai/spans.yaml)
- [OpenTelemetry security guidance](https://opentelemetry.io/docs/security/)
- [OpenAI Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/)

## Validation and decision query

```sh
plumbline validate-outcome examples/outcome-bound-trajectory.json
plumbline aggregate-outcomes run-a.outcome.json run-b.outcome.json -o aggregate.json
plumbline query-outcomes aggregate.json quality-gatekeeper
```

Aggregation is deterministic, streaming, and descriptive. It preserves availability,
exposure, and adoption counts separately. Only known independent outcome IDs explicitly
bound to the exact capability state enter that state's cohort; unknown labels are excluded and
conflicting terminal labels fail closed. An `available` observation with outcome references cannot
coexist with `exposed` or `adopted` observations for the same capability because that would make
control-versus-exposure attribution ambiguous. Such an envelope fails validation instead of silently
promoting the available-only label into the exposed cohort. The query has four terminal outcomes:

- `STOP_LOW_LABEL_COVERAGE`: the kill criterion fires because labels are too incomplete.
- `HOLD_INSUFFICIENT_COHORTS`: exposed and available-only cohorts are too small.
- `DO_NOT_SCALE_NO_DECISION_CHANGE`: observed pass-rate movement is below the declared materiality
  threshold.
- `REVIEW_CORRELATION`: a material association exists; experiment or stronger identification is
  required before any causal scale decision.

The query never returns an automatic “scale” verdict. That is intentional: one run or a selected
observational cohort cannot establish causal value.

## Producer and consumer boundary

| Surface | Responsibility |
|---|---|
| Plumbline | Portable trace, outcome-bound envelope, validation, aggregation, bounded query |
| Private recorder | Collection, protected raw storage, task pseudonymization, source deletion |
| Capability-usage snapshot | Cross-session availability/exposure baseline with its own claim limits |
| OPERANT | Controlled evaluation/cohort evidence and explicit admission |
| Execution evidence | Consume the digest and claim ceiling; never copy spans or payloads |

## Compatibility and rollback

The existing `PlumblineTrace` contract remains unchanged. Consumers that do not understand the new
envelope continue to score traces normally. Rollback is removal of the envelope producer/consumer; no
trace migration is required. A producer version and `compatible_with` list are mandatory so future
schema revisions cannot silently masquerade as V1.

## Falsification and kill criteria

Do not scale collection when independent label coverage remains below the declared threshold, both
cohorts cannot be populated without selection bias, the decision query repeatedly returns no decision
change, retention/deletion cannot be enforced, or the metadata profile cannot exclude sensitive
content. An envelope that passes schema validation does not prove any of those operating conditions.
