"""PII scrubber behavior."""

from plumbline.scrub import scrub_obj, scrub_text


def test_scrub_home_path() -> None:
    assert "/Users/alice" not in scrub_text("see /Users/alice/secret.txt")


def test_scrub_email() -> None:
    assert "a@b.com" not in scrub_text("mail a@b.com now")


def test_scrub_long_hex_token() -> None:
    secret = "deadbeef" * 5  # 40 hex chars
    assert secret not in scrub_text(f"token {secret}")


def test_scrub_prefixed_token() -> None:
    assert "sk-ABCD1234EFGH5678IJKL" not in scrub_text("token sk-ABCD1234EFGH5678IJKL")


def test_scrub_keeps_ordinary_identifiers() -> None:
    # Must NOT over-redact: these look prefix-ish but are not secrets.
    text = "skip_empty_lines patient-monitoring pattern_match_score ghost_mode"
    assert scrub_text(text) == text


def test_scrub_obj_is_deep() -> None:
    obj = {"a": ["/Users/bob/x", {"b": "c@d.com"}], "n": 3, "ok": True}
    out = scrub_obj(obj)
    blob = str(out)
    assert "/Users/bob" not in blob
    assert "c@d.com" not in blob
    # non-strings pass through untouched
    assert out["n"] == 3
    assert out["ok"] is True
