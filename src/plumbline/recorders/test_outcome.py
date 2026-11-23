"""Recognize bounded test-runner summaries without retaining result text."""

from __future__ import annotations

import re
from collections import Counter

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNTS = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|errors?|todo|pending)")
_PYTEST_KIND = r"passed|failed|skipped|errors?|deselected|xfailed|xpassed|warnings?"


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
    field = rf"\d+[ \t]+(?:{_PYTEST_KIND})"
    pattern = (
        rf"^[ \t=]*(?P<counts>{field}(?:[ \t]*,[ \t]*{field})*)"
        r"[ \t]+in[ \t]+\d+(?:\.\d+)?s(?:[ \t]+\(\d+:\d{2}:\d{2}\))?[ \t=]*$"
    )
    matches = list(re.finditer(pattern, output, re.MULTILINE))
    if not matches:
        return None
    counts: Counter[str] = Counter(passed=0, failed=0, skipped=0, errors=0)
    for match in matches:
        for field_text in match["counts"].split(","):
            value, kind = field_text.split()
            if kind not in {"warning", "warnings"}:
                counts["errors" if kind == "error" else kind] += int(value)
    return {"runner": "pytest", **counts}


def _test_line(output: str, runner: str) -> dict[str, int | str] | None:
    if runner == "vitest":
        pattern = r"^[ \t]*Tests[ \t]+(.+?)[ \t]+\((\d+)\)[ \t]*$"
    else:
        pattern = r"^[ \t]*Tests:[ \t]*(.+?),[ \t]*(\d+) total[ \t]*$"
    matches = list(re.finditer(pattern, output, re.MULTILINE))
    if not matches:
        return None
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in matches:
        fields, total = match.groups()
        if not all(_COUNTS.fullmatch(field.strip()) for field in re.split(r"[|,]", fields)):
            return None
        result_counts = _summary_counts(fields)
        if sum(result_counts.values()) != int(total):
            return None
        for key, value in result_counts.items():
            counts[key] += value
    return {"runner": runner, **counts}


def _unittest(output: str) -> dict[str, int | str] | None:
    results = list(
        re.finditer(
            r"^Ran (\d+) tests? in [\d.]+s\s*\n(OK|FAILED)(?: \(([^)\n]+)\))?[ \t]*$",
            output,
            re.MULTILINE,
        )
    )
    if not results:
        return None
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    categories = {
        "failures": "failed",
        "errors": "errors",
        "skipped": "skipped",
        "expected failures": "skipped",
        "unexpected successes": "failed",
    }
    for result in results:
        total, status, details = result.groups()
        result_counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        for field in details.split(", ") if details else ():
            label, _, value = field.partition("=")
            if label not in categories or not value.isdecimal():
                return None
            result_counts[categories[label]] += int(value)
        if (status == "FAILED") != bool(result_counts["failed"] or result_counts["errors"]):
            return None
        result_counts["passed"] = max(0, int(total) - sum(result_counts.values()))
        for key, value in result_counts.items():
            counts[key] += value
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


def _go(output: str, command: str | None) -> dict[str, int | str] | None:
    passed = re.findall(r"^ok[ \t]+\S+[ \t]+(?:[\d.]+s|\(cached\))[ \t]*$", output, re.MULTILINE)
    failed = re.findall(
        r"^FAIL[ \t]+\S+(?:[ \t]+(?:[\d.]+s|\[(?:build|setup) failed\]))?[ \t]*$",
        output,
        re.MULTILINE,
    )
    final_failure = output.rstrip().rpartition("\n")[2].strip() == "FAIL"
    if not passed and not failed:
        go_command = isinstance(command, str) and re.match(
            r"^\s*(?:\S*/)?go\s+test(?:\s|$)", command
        )
        if not (final_failure and go_command):
            return None
    outcome: dict[str, int | str] = {
        "runner": "go test",
        "passed_packages": len(passed),
        "failed_packages": len(failed),
    }
    if final_failure and not failed:
        outcome["errors"] = 1  # Aggregate failure with no observed failed package.
    tests = re.findall(r"^--- (PASS|FAIL|SKIP): \S+ \([\d.]+s\)$", output, re.MULTILINE)
    if tests:
        outcome.update(
            passed=tests.count("PASS"), failed=tests.count("FAIL"), skipped=tests.count("SKIP")
        )
    return outcome


def parse_test_outcome(output: str, *, command: str | None = None) -> dict[str, int | str] | None:
    """Return runner name and summary counts, without preserving output content."""
    clean = _ANSI.sub("", output)
    parsed = (
        _pytest(clean),
        _unittest(clean),
        _test_line(clean, "vitest"),
        _test_line(clean, "jest"),
        _cargo(clean),
        _go(clean, command),
    )
    outcomes = [outcome for outcome in parsed if outcome is not None]
    if not outcomes:
        return None
    if len(outcomes) == 1:
        return outcomes[0]
    counts: Counter[str] = Counter()
    for outcome in outcomes:
        counts.update({key: value for key, value in outcome.items() if isinstance(value, int)})
    return {"runner": "multiple", **counts}
