# Purpose

Plumbline is an OpenTelemetry-compatible agent-trace schema with an offline scorer and a decision-path
overlay. The Python package supports Python 3.11 and newer and has zero runtime dependencies.

# Layout

- `src/plumbline/`: package code, including recorder, scorer, contracts, and the packaged trace schema.
- `schema/`: source-of-truth JSON Schemas.
- `tests/` and `tests/fixtures/`: synthetic regression coverage and fixtures.
- `docs` at the repository root: design and contract references such as `SCHEMA.md` and `JUDGE.md`.

# Setup

Use the Python version in `.python-version` and the repository's pinned uv requirement. From the
repository root, install the committed development lock:

```sh
uv sync --frozen --group dev
```

# Checks

Run `make check` for lock, lint, formatting, typing, tests, and build checks. Focused test runs should
use `uv run --frozen pytest ... --no-cov`; the full test command enforces the configured 80% coverage
floor.

# Boundaries

- Keep the runtime dependency set empty; validation and model integrations remain optional.
- The deterministic scorer is offline.
- Use synthetic data only for development and verification. Do not read or record real transcripts.
- Do not use `--judge` or contact Ollama or Anthropic to verify code or documentation.
- Do not add provider calls or secrets to CI.
- Do not claim that the judge verifies test success: it receives outcome claims and statuses, not
  captured tool output.

# Delivery

Keep changes scoped to the requested behavior, preserve the Python 3.11 floor unless the owner decides
otherwise, and update the relevant tests and documentation with code changes. Report the commands run
and distinguish local verification from provider or human evidence. Do not publish or release from
this repository without explicit task authority.
