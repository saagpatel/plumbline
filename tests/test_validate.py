"""Judge validation harness: measure judge-vs-gold agreement on a labeled corpus.

An unvalidated judge is decoration (OPERANT: >50% judge error on hard cases until
calibrated). These tests use fake backends so the harness math is pinned without a
model, and confirm the starter corpus is separable by the deterministic signal the
judge already receives.
"""

import json
from pathlib import Path

from plumbline.scorer.infer import enrich
from plumbline.scorer.validate import (
    LabeledCase,
    format_report,
    load_corpus,
    validate_judge,
)

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "judge"
RECORDED = CORPUS / "recorded"


def _always_ok(_prompt: str) -> str:
    return '{"meta_decision_ok": true, "confidence": 0.5, "rationale": "approve"}'


def _always_bad(_prompt: str) -> str:
    return '{"meta_decision_ok": false, "confidence": 0.5, "rationale": "reject"}'


def _bypass_oracle(prompt: str) -> str:
    # A sane judge: fail the run iff a deterministic guard-evasion was flagged.
    # "(none detected)" is the bypass section's sentinel for no findings.
    ok = "(none detected)" in prompt
    return json.dumps({"meta_decision_ok": ok, "confidence": 0.9, "rationale": "oracle"})


def test_load_corpus_reads_labeled_cases() -> None:
    cases = load_corpus(CORPUS)
    assert len(cases) == 20
    by_id = {c.label_id: c for c in cases}
    good = sum(c.gold_meta_decision_ok for c in cases)
    assert good == 10  # balanced corpus: 10 good, 10 bad
    assert isinstance(by_id["bypass_evasion"], LabeledCase)
    assert by_id["bypass_evasion"].gold_meta_decision_ok is False
    assert by_id["clean_run"].gold_meta_decision_ok is True
    assert by_id["bypass_evasion"].trace.run.run_id == "corpus_bypass"


# The 8 bad cases the deterministic bypass scan CANNOT see (no guard-evasion token):
# the 5 original subtle cases plus the 3 adversarial "looks-good-is-bad" traps.
_DETERMINISTICALLY_SILENT_BAD = {
    "silent_abandon",
    "proceed_past_failed_tests",
    "fabricated_success",
    "destructive_without_asking",
    "skipped_verification",
    "escalation_theater_bad",
    "hollow_test_pass_bad",
    "reroute_drops_scope_bad",
}


def test_deterministic_oracle_misses_every_subtle_bad_case() -> None:
    # A judge relying ONLY on the deterministic bypass signal catches the two
    # obvious evasions but blesses all 8 subtle bad runs (no bypass to flag) —
    # the exact gap the LLM judge exists to close, and why it matters here.
    report = validate_judge(load_corpus(CORPUS), _bypass_oracle)
    assert report.n == 20
    assert report.correct_good == 10  # no good case trips the bypass scan
    assert report.correct_bad == 2  # only the two real evasions
    assert report.missed_bad == 8
    assert report.false_alarm == 0
    assert {r.label_id for r in report.disagreements} == _DETERMINISTICALLY_SILENT_BAD


def test_always_approve_misses_every_bad_run() -> None:
    report = validate_judge(load_corpus(CORPUS), _always_ok)
    assert report.correct_good == 10
    assert report.missed_bad == 10  # every bad run wrongly blessed — the dangerous error
    assert report.correct_bad == 0
    assert report.accuracy == 0.5


def test_always_reject_false_alarms_every_good_run() -> None:
    report = validate_judge(load_corpus(CORPUS), _always_bad)
    assert report.correct_bad == 10
    assert report.false_alarm == 10
    assert report.accuracy == 0.5


def test_recorded_corpus_exercises_the_inference_layer() -> None:
    # The recorded corpus is recorder-shaped (no authored decisions), so it actually
    # exercises enrich — unlike the main corpus, where enrich is a no-op.
    cases = {c.label_id: c for c in load_corpus(RECORDED)}
    assert len(cases) == 6
    for cid in ("rec_reroute_after_error_ok", "rec_reroute_after_denial_ok"):
        enriched = enrich(cases[cid].trace)
        kinds = [s.attributes["agent.decision.kind"] for s in enriched.steps_of_kind("decision")]
        assert "reroute" in kinds, cid
    for cid in ("rec_clean_ok", "rec_fabricated_bad", "rec_proceed_past_failed_tests_bad"):
        assert enrich(cases[cid].trace).steps_of_kind("decision") == [], cid


def test_main_corpus_is_unaffected_by_recorded_subdir() -> None:
    # load_corpus is non-recursive; the recorded/ subdir must not change the main count.
    assert len(load_corpus(CORPUS)) == 20


def test_format_report_is_readable() -> None:
    report = validate_judge(load_corpus(CORPUS), _always_ok)
    text = format_report(report)
    assert "accuracy" in text
    assert "missed_bad=10" in text
    assert "silent_abandon" in text  # a disagreement is named
