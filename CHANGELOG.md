# Changelog

All notable changes to Plumbline are recorded here. The trace **schema** version
(`plumbline_version`, currently `0.1.0`) is independent of this package version; Phase 4
populates existing schema fields and did not change the schema.

## 0.2.0

The inference layer: make the calibration judge work on real recorded traces, not just
hand-authored synthetic ones. Dogfooding the full `record -> score --judge` pipeline
against a real session exposed that the judge was reasoning over empty plan / outcome /
decisions on recorded data; Phase 4 closes that gap.

### Added

- **Calibration judge** (`scorer/judge.py`): a rule-based, trace-grounded rubric over a
  pluggable backend (free local `OllamaBackend`, opt-in `AnthropicBackend`). Validated
  on a labeled corpus; see [`JUDGE.md`](JUDGE.md).
- **`plumbline score --judge`**: deterministic scorecard + calibration verdict in one
  command; with `--gate`, a `meta_decision_ok=false` verdict also fails the build.
- **Outcome + plan capture (4a)**: the recorder captures `run.outcome` (status + a
  scrubbed summary) from the final assistant turn and fixes plan capture for
  string-valued `message.content` (was silently dropped).
- **Structural decision inference (4b)**: `scorer/infer.py` `enrich(trace)` infers
  `reroute` decisions (same-tool-different-resource after a denial; fix-and-re-verify
  after an error), provenance- and evidence-tagged; wired into `judge_run`.
- **Text-signal decision inference (4d)**: opt-in `record --infer-text-decisions`
  detects refuse/escalate in assistant prose at record time, emitting decisions with
  only the kind + a short scrubbed rationale, no prose stored in the trace.
- **Validation tooling (4c)**: `corpus/judge/recorded/` (recorder-shaped, PII-free,
  gold-labeled) + `validate-judge --no-enrich` to measure the inference layer. Measured:
  enrich takes the weaker judge 5/6 -> 6/6 on that tier, no regressions.

### Notes

- The judge has a documented adversarial ceiling and a small-local-model reasoning
  limit (see [`JUDGE.md`](JUDGE.md)); treat `missed_bad` as the metric that matters.
- The structural inference denial branch is dormant under `bypassPermissions` (no
  recorded deny events); the text-signal tier exists for that reason. See
  [`PHASE4.md`](PHASE4.md).

## 0.1.0

- Trace schema + JSON Schema + worked example (Phase 0).
- Claude Code recorder: `*.jsonl` -> Plumbline trace, with a PII scrubber (Phase 1).
- Offline deterministic scorer: selection / ordering / edit-similarity / param-name +
  bypass hard-fail gate (Phase 2).
- CI gate (`score --gate`) + OTel extension proposal (Phase 3).
