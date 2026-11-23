.PHONY: check

check:
	@command -v uv >/dev/null 2>&1 || { printf 'install uv 0.13.9 or a later 0.13.x: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 127; }
	uv lock --check
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .
	uv run --frozen ty check
	uv run --frozen pytest
	uv build
