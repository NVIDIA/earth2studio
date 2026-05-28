# AGENTS.md — Earth2Studio AI Agent Entry Point

AI agent skills for NVIDIA Earth2Studio weather and climate inference
library. Skills live in **`skills/`** (repo root) and coding rules live
in **`.agents/rules/`**. Skills are task-specific guides for building
inference pipelines, integrating data sources, wrapping models, and
managing releases.

## Skills directory

### Discovery and installation

- `skills/earth2studio-discover/` — Find which models, data sources,
  and examples fit a weather/climate use case
- `skills/earth2studio-install/` — Install Earth2Studio, select model
  extras, configure environment variables

### Inference

- `skills/earth2studio-deterministic-forecast/` — Build deterministic
  (single-member) forecast scripts with model selection, data source,
  IO backend, and nsteps calculation

### Data

- `skills/earth2studio-data-fetch/` — Download weather/climate data for
  specific variables, times, and lead times using Earth2Studio data
  sources

### Implementation (new components)

- `skills/earth2studio-create-datasource/` — Create a new data source
  wrapper (DataSource, ForecastSource, DataFrameSource, or
  ForecastFrameSource) end-to-end
- `skills/earth2studio-create-prognostic/` — Create a new prognostic
  model (px) wrapper from a reference inference script end-to-end

### Developer

- `skills/developer-release-rebase/` — Prepare a new minor alpha
  release (rebase, bump version, changelog, PR)

## Python environment

Always use `uv run` to execute Python commands (e.g. `uv run pytest`,
`uv run black`). Alternatively, use the project virtualenv at `.venv/`
(e.g. `.venv/bin/python`). **Never use the system Python directly** —
it does not have project dependencies installed.

## Key conventions (quick reference)

- **License header** — every `.py` file in `earth2studio/` must start
  with the SPDX Apache-2.0 header.
- **Type hints** — all public functions must be fully typed; the
  codebase is mypy-clean.
- **Logging** — use `loguru.logger`, never `print()`, inside
  `earth2studio/`.
- **Formatting** — black; run `make format`.
- **Linting** — ruff + mypy; run `make lint` before opening a PR.

## Developer rules

The authoritative coding rules live in `.agents/rules/`. These are for
**developers contributing to Earth2Studio** (not end users). Read the
relevant rule file(s) before writing or reviewing code:

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

## Resources

### Documentation

- [Earth2Studio Docs](https://nvidia.github.io/earth2studio/)
- [API Reference](https://nvidia.github.io/earth2studio/modules/index.html)

### Examples

- [Gallery Examples](https://nvidia.github.io/earth2studio/examples/index.html)

### Support

- [File a Bug](https://github.com/NVIDIA/earth2studio/issues/new)
- [All Issues](https://github.com/NVIDIA/earth2studio/issues)
