"""Validate the calibration judge against a human-labeled corpus.

A judge you haven't measured against human labels is decoration — on hard cases
an unvalidated judge can disagree with humans on a majority of runs. This harness
runs the judge over a corpus of labeled traces and reports agreement, with the
confusion cells named for what they mean operationally:

* ``correct_good`` / ``correct_bad`` — judge matched the human label.
* ``false_alarm`` — judge failed a run a human judged fine (annoying, not unsafe).
* ``missed_bad`` — judge blessed a run a human judged bad (e.g. a bypass). This is
  the dangerous error: a judge that misses bad meta-decisions is worse than none.

A labeled case is a JSON file: ``{label_id, gold_meta_decision_ok, note, trace}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plumbline.scorer.judge import judge_run
from plumbline.scorer.trace import Trace

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from plumbline.scorer.judge import JudgeBackend


@dataclass(frozen=True)
class LabeledCase:
    """A trace plus the human's gold meta-decision verdict."""

    label_id: str
    gold_meta_decision_ok: bool
    note: str
    trace: Trace


@dataclass(frozen=True)
class CaseResult:
    """The judge's verdict on one labeled case, compared to gold."""

    label_id: str
    gold: bool
    predicted: bool
    confidence: float

    @property
    def agree(self) -> bool:
        return self.gold == self.predicted


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate judge-vs-gold agreement over a labeled corpus."""

    results: tuple[CaseResult, ...]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def agreements(self) -> int:
        return sum(1 for r in self.results if r.agree)

    @property
    def accuracy(self) -> float:
        return self.agreements / self.n if self.n else 0.0

    @property
    def correct_good(self) -> int:
        return sum(1 for r in self.results if r.gold and r.predicted)

    @property
    def correct_bad(self) -> int:
        return sum(1 for r in self.results if not r.gold and not r.predicted)

    @property
    def false_alarm(self) -> int:
        """Gold good, judge failed it — over-strict but not unsafe."""
        return sum(1 for r in self.results if r.gold and not r.predicted)

    @property
    def missed_bad(self) -> int:
        """Gold bad, judge blessed it — the dangerous error."""
        return sum(1 for r in self.results if not r.gold and r.predicted)

    @property
    def disagreements(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.agree)


def load_corpus(directory: Path) -> list[LabeledCase]:
    """Load every ``*.json`` labeled case in a directory, sorted by label_id."""
    cases: list[LabeledCase] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.append(
            LabeledCase(
                label_id=data["label_id"],
                gold_meta_decision_ok=bool(data["gold_meta_decision_ok"]),
                note=data.get("note", ""),
                trace=Trace.from_dict(data["trace"]),
            )
        )
    return cases


def validate_judge(corpus: Sequence[LabeledCase], backend: JudgeBackend) -> ValidationReport:
    """Run the judge over each labeled case and tally agreement with gold."""
    results = []
    for case in corpus:
        verdict = judge_run(case.trace, backend)
        results.append(
            CaseResult(
                label_id=case.label_id,
                gold=case.gold_meta_decision_ok,
                predicted=verdict.meta_decision_ok,
                confidence=verdict.confidence,
            )
        )
    return ValidationReport(results=tuple(results))
