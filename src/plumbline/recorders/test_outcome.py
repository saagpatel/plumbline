"""Recognize bounded test-runner summaries without retaining result text."""

from __future__ import annotations

import re
from collections import Counter

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNTS = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|errors?|todo|pending)")
_PYTEST_KIND = r"passed|failed|skipped|errors?|deselected|xfailed|xpassed|warnings?"


_JS_RUNNERS = ("jest", "vitest")
_RUNNER_PREFIXES = {
    ("pytest",): ("pytest",),
    ("uv", "run", "pytest"): ("pytest",),
    ("uv", "run", "--frozen", "pytest"): ("pytest",),
    ("poetry", "run", "pytest"): ("pytest",),
    ("python", "-m", "pytest"): ("pytest",),
    ("python", "-m", "unittest"): ("unittest",),
    ("jest",): ("jest",),
    ("vitest",): ("vitest",),
    ("npx", "jest"): ("jest",),
    ("npx", "vitest"): ("vitest",),
    ("npm", "test"): _JS_RUNNERS,
    ("pnpm", "test"): _JS_RUNNERS,
    ("yarn", "test"): _JS_RUNNERS,
    ("cargo", "test"): ("cargo test",),
    ("go", "test"): ("go test",),
}
_LAUNCHER = r"(?:pytest|python[\d.]*|jest|vitest|npx|npm|pnpm|yarn|cargo|go)"
_START_FAILURE = re.compile(
    rf"(?:\b{_LAUNCHER}: (?:command not found|not found|no such file or directory)\b"
    rf"|\bcommand not found: {_LAUNCHER}\b)",
    re.IGNORECASE,
)


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


def _suite_failures(output: str, runner: str) -> int:
    heading = "Test Files" if runner == "vitest" else "Test Suites:"
    summaries = re.findall(rf"^[ \t]*{heading}[ \t]+(.+)$", output, re.MULTILINE)
    return sum(int(n) for line in summaries for n in re.findall(r"\b(\d+) failed\b", line))


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
    suite_failures = _suite_failures(output, runner)
    if not matches and not suite_failures:
        return None
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": suite_failures}
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


def _go(output: str) -> dict[str, int | str] | None:
    passed = re.findall(r"^ok[ \t]+\S+[ \t]+(?:[\d.]+s|\(cached\))[ \t]*$", output, re.MULTILINE)
    detail = r"(?:[ \t]+(?:[\d.]+s|\[(?:build|setup) failed\]))"
    failed = re.findall(rf"^FAIL[ \t]+\S+{detail}?[ \t]*$", output, re.MULTILINE)
    final_failure = output.rstrip().rpartition("\n")[2].strip() == "FAIL"
    if not passed and not failed and not final_failure:
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


def parse_test_outcome(output: str, *, command: str | None = None) -> dict[str, object] | None:
    """Return runner name and summary counts, without preserving output content."""
    runners = command_runners(command)
    if not runners:
        return None
    clean = _ANSI.sub("", output)
    if _START_FAILURE.search(clean):
        return None
    # Package scripts may invoke Jest or Vitest. Require one matching summary format.
    parsed = (
        _pytest(clean) if "pytest" in runners else None,
        _unittest(clean) if "unittest" in runners else None,
        _test_line(clean, "vitest") if "vitest" in runners else None,
        _test_line(clean, "jest") if "jest" in runners else None,
        _cargo(clean) if "cargo test" in runners else None,
        _go(clean) if "go test" in runners else None,
    )
    outcomes = [outcome for outcome in parsed if outcome is not None]
    return dict(outcomes[0]) if len(outcomes) == 1 else None


def _literal_command(command: str | None) -> str:
    """Mask quoted arguments. Leave dynamic or malformed shell syntax unclassified."""
    if not isinstance(command, str) or re.search(r"[\\$`]", command):
        return ""
    literal = re.sub(r"'[^']*'|\"[^\"]*\"", ":", command)
    return "" if re.search(r"""['"(){}#]""", literal) else literal


def _runner_prefix(stage: str) -> tuple[str, ...]:
    words = re.split(r"[ \t]+", stage.strip(" \t"))
    while words and re.match(r"[A-Za-z_]\w*=", words[0], re.ASCII):
        words.pop(0)
    if not words:
        return ()
    words[0] = words[0].rsplit("/", 1)[-1]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", words[0]):
        words[0] = "python"
    for prefix, names in _RUNNER_PREFIXES.items():
        if tuple(words[: len(prefix)]) == prefix:
            return names
    return ()


def mentions_test_invocation(command: str) -> bool:
    """Identify runner stages for judge previews, even when counts are unattributable."""
    if _runner_prefix(command):
        return True  # A literal executable can have dynamic arguments.
    return any(_runner_prefix(stage) for stage in re.split(r"[;&|\n]", _literal_command(command)))


def command_runners(command: str | None) -> tuple[str, ...]:
    """Candidate runner labels for a single invocation, optionally filtered on stdout."""
    literal = _literal_command(command).strip(" \t\n")
    # Allow a trailing stderr merge on either stage, without swallowing other redirects.
    literal = re.sub(r"[ \t]+2>&1(?=[ \t]*(?:\||$))", "", literal)
    if re.search(r"[;&<>\n]", literal):
        return ()
    runner, pipe, output_filter = literal.partition("|")
    if pipe and not re.fullmatch(
        r"(?:tail|head)(?:[ \t]+(?:-\d+|-[nc][ \t]*[+-]?\d+|--(?:lines|bytes)=[+-]?\d+))?",
        output_filter.strip(" \t"),
    ):
        return ()
    return _runner_prefix(runner)
