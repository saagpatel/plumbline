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

# Ordered: specific shapes first. Token rules require a real separator after the
# provider prefix so ordinary words ("skip_...", "patient-...", "pattern_...")
# are NOT mistaken for secrets.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b(?:sk|ghp|gho|ghs|ghu|ghr|xox[baprs])[-_][A-Za-z0-9._-]{16,}"), "<token>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}"), "<token>"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<hex>"),
    (re.compile(r"/(?:Users|home)/[^/\s\"']+"), "~"),
)


def scrub_text(value: str) -> str:
    """Redact high-signal PII shapes from a single string."""
    out = value
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def scrub_obj(value: Any) -> Any:  # noqa: ANN401 - intentionally walks arbitrary JSON
    """Recursively scrub every string in a JSON-shaped structure."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [scrub_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_obj(item) for key, item in value.items()}
    return value
