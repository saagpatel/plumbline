"""Best-effort PII scrubbing for traces destined for publication.

The recorder runs this by default. It is deliberately conservative and tuned to
avoid *corrupting* a trace: it redacts high-signal secret shapes (emails,
prefixed API tokens, long hex blobs) and collapses the home-directory prefix,
while leaving ordinary identifiers and sub-paths intact so the decision path
stays legible.

Scope and limits:
- It collapses the home prefix (`/Users/<name>`, `/home/<name>`) to `~`. Deeper
  path segments (project/file names) are preserved because they are usually
  trace-meaningful; review them by hand if a directory name is itself sensitive.
- It does NOT touch UUID/opaque identifiers: `run_id`, `step_id`, `caused_by`
  etc. are structural join keys, and redacting them would break the trace DAG.

This is not a guarantee of anonymity — review any trace before sharing it.
"""

from __future__ import annotations

import re
from typing import Any

_CREDENTIAL = r"(?:password|passwd|token|secret|api[_-]?key|aws_secret_access_key)"
_ENV_LABEL = rf"(?:[\w]+_)?{_CREDENTIAL}="
_OPTION = rf"--(?:[\w-]+-)?{_CREDENTIAL}"
_ASSIGNMENT_LABEL = rf"(?:{_ENV_LABEL}|{_OPTION}=)"


def _credential_patterns(*, plan: bool) -> tuple[tuple[re.Pattern[str], str], ...]:
    single = r"[^']*"
    double = r'(?:\\[\s\S]|[^"\\])*'
    end = r"\Z"
    if plan:
        # A closing quote can be on a later line. Only an unfinished quote
        # stops at the newline, preserving the next plan instruction.
        single = rf"(?:{single}(?=')|[^'\n]*)"
        double_line = r'(?:\\[^\n]|[^"\\\n])*(?:\\(?=\n|\Z))?'
        double = rf'(?:{double}(?=")|{double_line})'
        end = r"(?=\n|\Z)"

    # Shell quoting can split a single argument into adjacent segments.
    word_part = rf"""(?:\\[\s\S]|'{single}(?:'|{end})|"{double}(?:"|{end})|[^\s\\'";|&()<>])"""
    label = rf"\b{_ENV_LABEL}|{_OPTION}(?:=|[ \t]+)"
    return (
        (
            re.compile(rf"(?i)('{_ASSIGNMENT_LABEL}){single}('|{end}){word_part}*"),
            r"\1<redacted>\2",
        ),
        (
            re.compile(rf'(?i)("{_ASSIGNMENT_LABEL}){double}(?:\\(?=\Z))?("|{end}){word_part}*'),
            r"\1<redacted>\2",
        ),
        (re.compile(rf"(?i)({label}){word_part}+"), r"\1<redacted>"),
    )


_CREDENTIAL_PATTERNS = _credential_patterns(plan=False)
_PLAN_CREDENTIAL_PATTERNS = _credential_patterns(plan=True)
# Token rules require a real separator after the provider prefix so ordinary
# words ("skip_...", "patient-...", "pattern_...") are not mistaken for secrets.
_PATTERNS = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b(?:sk|ghp|gho|ghs|ghu|ghr|xox[baprs])[-_][A-Za-z0-9._-]{16,}"), "<token>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}"), "<token>"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<hex>"),
    (re.compile(r"/(?:Users|home)/[^/\s\"']+"), "~"),
)


def scrub_text(value: str, *, plan: bool = False) -> str:
    """Redact high-signal PII shapes from a single string."""
    credentials = _PLAN_CREDENTIAL_PATTERNS if plan else _CREDENTIAL_PATTERNS
    for pattern, replacement in (*credentials, *_PATTERNS):
        value = pattern.sub(replacement, value)
    return value


def scrub_argument(key: str, value: str) -> str:
    """A credential-named tool argument carries its label outside its value."""
    if re.fullmatch(_ASSIGNMENT_LABEL, key + "=", re.IGNORECASE):
        return "<redacted>"
    return scrub_text(value)


def scrub_obj(value: Any) -> Any:  # noqa: ANN401 - intentionally walks arbitrary JSON
    """Recursively scrub every string in a JSON-shaped structure."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [scrub_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_obj(item) for key, item in value.items()}
    return value
