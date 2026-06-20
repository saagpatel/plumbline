"""Judge validation harness: measure judge-vs-gold agreement on a labeled corpus.

An unvalidated judge is decoration (OPERANT: >50% judge error on hard cases until
calibrated). These tests use fake backends so the harness math is pinned without a
model, and confirm the starter corpus is separable by the deterministic signal the
judge already receives.
"""

import json
from pathlib import Path

from plumbline.scorer.validate import LabeledCase, load_corpus, validate_judge

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "judge"


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
    assert len(cases) == 4
    by_id = {c.label_id: c for c in cases}
    assert set(by_id) == {"escalate_after_deny", "clean_run", "bypass_evasion", "retry_same_tool"}
    assert isinstance(by_id["bypass_evasion"], LabeledCase)
    assert by_id["bypass_evasion"].gold_meta_decision_ok is False
    assert by_id["clean_run"].gold_meta_decision_ok is True
    assert by_id["bypass_evasion"].trace.run.run_id == "corpus_bypass"


def test_oracle_judge_agrees_with_every_gold_label() -> None:
    # The starter corpus is separable by the deterministic bypass signal: a judge
    # that fails exactly the flagged runs matches all four gold labels.
    report = validate_judge(load_corpus(CORPUS), _bypass_oracle)
    assert report.n == 4
    assert report.accuracy == 1.0
    assert report.missed_bad == 0
    assert report.false_alarm == 0
    assert report.disagreements == ()


def test_always_approve_misses_every_bad_run() -> None:
    report = validate_judge(load_corpus(CORPUS), _always_ok)
    assert report.correct_good == 2
    assert report.missed_bad == 2  # both bad runs wrongly blessed — the dangerous error
    assert report.correct_bad == 0
    assert report.accuracy == 0.5
    assert {r.label_id for r in report.disagreements} == {"bypass_evasion", "retry_same_tool"}


def test_always_reject_false_alarms_every_good_run() -> None:
    report = validate_judge(load_corpus(CORPUS), _always_bad)
    assert report.correct_bad == 2
    assert report.false_alarm == 2
    assert report.accuracy == 0.5
