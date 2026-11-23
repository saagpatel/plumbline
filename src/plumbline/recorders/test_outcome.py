"""Recognize bounded test-runner summaries without retaining result text."""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNTS = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|errors?|todo|pending)")


def _summary_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in _COUNTS.finditer(text):
        kind = match.group("kind")
        if kind.startswith("error"):
            key = "errors"
        elif kind in {"todo", "pending"}:
            key = "skipped"
        else:
            key = kind
        counts[key] += int(match.group("count"))
    return counts


def _pytest(output: str) -> dict[str, int | str] | None:
    pattern = re.compile(
        r"^\s*=*\s*(?P<counts>\d+\s+(?:passed|failed|skipped|errors?)(?:\s*,\s*"
        r"\d+\s+(?:passed|failed|skipped|errors?))*)(?:,\s*\d+\s+warnings?)?"
        r"\s+in\s+\d+(?:\.\d+)?s\s*=*\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(output))
    if not matches:
        return None
    return {"runner": "pytest", **_summary_counts(matches[-1].group("counts"))}


def _test_line(output: str, runner: str) -> dict[str, int | str] | None:
    if runner == "vitest":
        pattern = r"^[ \t]*Tests[ \t]+(.+?)[ \t]+\((\d+)\)[ \t]*$"
    else:
        pattern = r"^[ \t]*Tests:[ \t]*(.+?),[ \t]*(\d+) total[ \t]*$"
    matches = list(re.finditer(pattern, output, re.MULTILINE))
    if not matches:
        return None
    fields, total = matches[-1].groups()
    if not all(_COUNTS.fullmatch(field.strip()) for field in re.split(r"[|,]", fields)):
        return None
    counts = _summary_counts(fields)
    return {"runner": runner, **counts} if sum(counts.values()) == int(total) else None


def _unittest(output: str) -> dict[str, int | str] | None:
    result = re.search(
        r"^Ran (\d+) tests? in [\d.]+s\s*\n(OK|FAILED)(?: \(([^)\n]+)\))?[ \t]*$",
        output,
        re.MULTILINE,
    )
    if not result:
        return None
    total, status, details = result.groups()
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    categories = {
        "failures": "failed",
        "errors": "errors",
        "skipped": "skipped",
        "expected failures": "skipped",
        "unexpected successes": "failed",
    }
    for field in details.split(", ") if details else ():
        label, _, value = field.partition("=")
        if label not in categories or not value.isdecimal():
            return None
        counts[categories[label]] += int(value)
    if (status == "FAILED") != bool(counts["failed"] or counts["errors"]):
        return None
    counts["passed"] = max(0, int(total) - sum(counts.values()))
    return {"runner": "unittest", **counts}


def _cargo(output: str) -> dict[str, int | str] | None:
    pattern = re.compile(
        r"^test result:\s*(?:ok|FAILED)\.\s*(\d+) passed;\s*(\d+) failed;\s*"
        r"(\d+) ignored;[^\n]*$",
        re.MULTILINE,
    )
    summaries = [tuple(map(int, match.groups())) for match in pattern.finditer(output)]
    total_tests = sum(passed + failed + ignored for passed, failed, ignored in summaries)
    if not summaries or total_tests == 0:
        return None
    return {
        "runner": "cargo test",
        "passed": sum(item[0] for item in summaries),
        "failed": sum(item[1] for item in summaries),
        "skipped": sum(item[2] for item in summaries),
        "errors": 0,
    }


def _go(output: str) -> dict[str, int | str] | None:
    passed = re.findall(r"^ok[ \t]+\S+[ \t]+[\d.]+s[ \t]*$", output, re.MULTILINE)
    failed = re.findall(
        r"^FAIL[ \t]+\S+(?:[ \t]+(?:[\d.]+s|\[build failed\]))?[ \t]*$", output, re.MULTILINE
    )
    if not passed and not failed:
        return None
    outcome: dict[str, int | str] = {
        "runner": "go test",
        "passed_packages": len(passed),
        "failed_packages": len(failed),
    }
    tests = re.findall(r"^--- (PASS|FAIL|SKIP): \S+ \([\d.]+s\)$", output, re.MULTILINE)
    if tests:
        outcome.update(
            passed=tests.count("PASS"), failed=tests.count("FAIL"), skipped=tests.count("SKIP")
        )
    return outcome


def parse_test_outcome(output: str) -> dict[str, int | str] | None:
    """Return runner name and summary counts, without preserving output content."""
    clean = _ANSI.sub("", output)
    return (
        _pytest(clean)
        or _unittest(clean)
        or _test_line(clean, "vitest")
        or _test_line(clean, "jest")
        or _cargo(clean)
        or _go(clean)
    )
