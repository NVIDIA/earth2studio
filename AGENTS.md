# Earth2Studio Agent Guide

Earth2Studio is an AI weather and climate Python package with a large catalog of data sources
and prognostic, diagnostic, and data assimilation models. It builds on PyTorch, xarray, and CuPy.

Task guides live in [`skills/`](./skills/). Use the routes below to find relevant documentation.

## Route the task

- **User:** uses or extends Earth2Studio in their own Python project, even if they
  are a software developer by profession.
- **Developer:** actively develops the Earth2Studio package itself, including its
  code, tests, documentation, and agent skills.

If unclear, ask:
**"Are you using or extending Earth2Studio in your own Python project, or developing
the Earth2Studio package itself?"**

Prefer files in the local checkout; otherwise use the matching release on GitHub
or the [documentation site](https://nvidia.github.io/earth2studio/main/).
Read and cite only the sources relevant to the task.

## For users

| Task | Start here |
| --- | --- |
| Install and configure | [Install guide][install], [command options][install-options] |
| Select a forecast model | [Prognostics][prognostics], [model implementations][px] |
| Find diagnostics or assimilation | [Diagnostics][diagnostics], [assimilation][assimilation] |
| Choose a data source | [Analysis][analysis], [forecast][forecast], [observations][observations] |
| Build a workflow | [Examples][examples], [recipes][recipes], [component guide][components] |
| Compare forecast skill | [Published scorecards][scorecards], [evaluation recipe][eval] |
| Compute or reproduce scorecards | [Scorecard recipe][scorecard-recipe], [metrics][statistics] |
| Resolve setup problems | [Troubleshooting][troubleshooting], [FAQ][faq] |

The [install-selector script][install-script] builds commands from `install_options.yml`;
[the generator][install-generator] prepares it for the docs.

Choose models by region, variables, forecast horizon, and GPU memory. Compare scorecards
by metric and lead time. See [evaluation configuration][scorecard-config],
[score data][scorecard-data], and the [page generator][scorecard-generator].

## User skills

Activate the matching skill with your agent's skill tool, or read its `SKILL.md` directly.

### Install skills if needed

Install skills directly from the Earth2Studio GitHub repository for current guidance.
Run from your project (requires Node.js/npm) and select your agent when prompted:

```bash
npx skills add https://github.com/NVIDIA/earth2studio --skill earth2studio-install
npx skills add https://github.com/NVIDIA/earth2studio --skill earth2studio-discover
```

Set the `--skill` value to any listed skill name. These commands use the default branch.
Reading local skills needs no installation. To register the version in your GitHub
checkout, run from its root:

```bash
npx skills add . --skill earth2studio-install
```

For the Python package, follow the install skill: use the install guide's uv setup,
select extras from `install_options.yml`, and give the user install and verification commands.

### Available user skills

- [earth2studio-discover](./skills/earth2studio-discover/SKILL.md): choose models, sources, and examples.
- [earth2studio-install](./skills/earth2studio-install/SKILL.md): install and configure the package.
- [earth2studio-deterministic-forecast](./skills/earth2studio-deterministic-forecast/SKILL.md)
  for single-member forecast scripts.
- [earth2studio-couple-models](./skills/earth2studio-couple-models/SKILL.md): run connected models.
- [earth2studio-data-fetch](./skills/earth2studio-data-fetch/SKILL.md): download weather/climate data.

## For developers

Start with [CONTRIBUTING.md](./CONTRIBUTING.md) and the
[developer guide](./docs/userguide/developer/overview.md).

### Implementation skills (developers only)

These skills are for adding or migrating components in Earth2Studio itself.
Activate the matching skill; use `npx skills add` above if installation is needed.

- [earth2studio-create-datasource](./skills/earth2studio-create-datasource/SKILL.md)
- [earth2studio-create-prognostic](./skills/earth2studio-create-prognostic/SKILL.md)
- [earth2studio-create-diagnostic](./skills/earth2studio-create-diagnostic/SKILL.md)
- [earth2studio-create-assimilation](./skills/earth2studio-create-assimilation/SKILL.md)

Creating or editing repository skills is developer-only work. See the
[skill development guide](./docs/userguide/developer/skills.md).

### Key conventions

- **Python:** use `uv run` or `.venv/bin/python`, never system Python.
- **License:** start package `.py` files with the SPDX Apache-2.0 header.
- **Types:** fully type public functions; check with mypy.
- **Logging:** use `loguru.logger` in `earth2studio/`.
- **Checks:** run `make format`, `make lint`, and relevant tests before opening a PR.

### Developer rules

Read relevant files in [`.agents/rules/`](./.agents/rules/) before changing package code:

| Rule file | Topic |
| --- | --- |
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

- [API reference](https://nvidia.github.io/earth2studio/main/modules/)
- [Example gallery](https://nvidia.github.io/earth2studio/main/examples/)
- [Issues and support](https://github.com/NVIDIA/earth2studio/issues)

[install]: ./docs/userguide/about/install.md
[install-options]: ./docs/userguide/about/install_options.yml
[install-script]: ./docs/assets/javascripts/e2s-install-selector.js
[install-generator]: ./docs/generate_install_options.py
[prognostics]: ./docs/modules/models_px.md
[px]: ./earth2studio/models/px/
[diagnostics]: ./docs/modules/models_dx.md
[assimilation]: ./docs/modules/models_da.md
[analysis]: ./docs/modules/datasources_analysis.md
[forecast]: ./docs/modules/datasources_forecast.md
[observations]: ./docs/modules/datasources_dataframe.md
[examples]: ./examples/
[recipes]: ./recipes/
[components]: ./docs/userguide/components/
[scorecards]: https://nvidia.github.io/earth2studio/main/scorecard/generated/
[eval]: ./recipes/eval/README.md
[scorecard-recipe]: ./recipes/eval/scorecard/README.md
[statistics]: ./docs/modules/statistics.md
[troubleshooting]: ./docs/userguide/support/troubleshooting.md
[faq]: ./docs/userguide/support/faq.md
[scorecard-config]: ./docs/scorecard/config/
[scorecard-data]: https://huggingface.co/datasets/nvidia/earth2studio-assets/tree/main/scorecard
[scorecard-generator]: ./docs/generate_scorecard.py
