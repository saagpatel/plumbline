"""Fail-closed parsing for human gold labels."""

import json
from pathlib import Path

import pytest

from plumbline.scorer.validate import load_corpus


def test_load_corpus_rejects_non_boolean_gold_label(tmp_path) -> None:
    trace_path = Path(__file__).parent.parent / "examples" / "example-run.plumbline.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "label_id": "synthetic",
                "gold_meta_decision_ok": "false",
                "trace": trace,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"gold_meta_decision_ok.*boolean"):
        load_corpus(tmp_path)
