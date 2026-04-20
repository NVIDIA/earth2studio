# Earth2Studio – Claude Code Guide

## Project Rules

The authoritative coding rules live in `.cursor/rules/`. Read the relevant rule
file(s) before writing or reviewing code:

| Rule file | Topic |
|---|---|
| `e2s-000-python-style-guide.mdc` | Style, formatting, type hints, license header |
| `e2s-001-dependency-management.mdc` | Adding / updating dependencies |
| `e2s-002-api-documentation.mdc` | Docstrings and public API docs |
| `e2s-003-unit-testing.mdc` | Writing and structuring tests |
| `e2s-004-data-sources.mdc` | Implementing `DataSource` classes |
| `e2s-005-forecast-sources.mdc` | Implementing `ForecastSource` classes |
| `e2s-006-dataframe-sources.mdc` | Implementing `DataFrameSource` classes |
| `e2s-007-forecast-frame-sources.mdc` | Implementing `ForecastFrameSource` classes |
| `e2s-008-lexicon-usage.mdc` | Variable lexicons and coordinate conventions |
| `e2s-009-prognostic-models.mdc` | Implementing prognostic models |
| `e2s-010-diagnostic-models.mdc` | Implementing diagnostic models |
| `e2s-011-examples.mdc` | Writing gallery examples |
| `e2s-012-time-tolerance.mdc` | Time tolerance patterns |
| `e2s-013-assimilation-models.mdc` | Implementing data assimilation models |

## Key Conventions (quick reference)

- **License header** — every `.py` file in `earth2studio/` must start with the SPDX Apache-2.0
  header.
- **Type hints** — all public functions must be fully typed; the codebase is mypy-clean.
- **Logging** — use `loguru.logger`, never `print()`, inside `earth2studio/`.
- **Formatting** — black; run `/format` or `make format`.
- **Linting** — ruff + mypy; run `/lint` or `make lint` before opening a PR.

## Skills

Custom skills live in `.claude/skills/`. Invoke the relevant skill before starting
a matching task:

| Skill | Purpose |
|---|---|
| `create-data-source` | Create a new data source wrapper |
| `create-prognostic-wrapper` | Create a new prognostic model (px) wrapper |
| `validate-data-source` | Validate a newly implemented data source |

## Custom Commands

Use these slash commands (defined in `.claude/commands/`) to run common tasks:

| Command | Action |
|---|---|
| `/format` | Auto-format code with black |
| `/lint` | Run all linters (ruff, mypy, pre-commit checks) |
| `/license` | Check SPDX license headers on all Python files |
| `/test` | Run tests for a specific tox environment |
| `/test-full` | Run the full test suite with coverage |
| `/docs` | Build docs — fast, full (with examples), or targeted single example |
| `/docs-serve` | Build docs then serve locally at <http://localhost:8000> |
| `/release` | Bump version, update changelog, strip example tags, open release PR |
