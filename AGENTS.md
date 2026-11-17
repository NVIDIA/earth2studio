# Earth2Studio Agent Guide

First ask, unless already specified: **"Which Earth2Studio version would you like:
the latest release, main, or a specific version?"**

- **Latest:** resolve the latest stable tag from GitHub releases, then read
  `https://github.com/NVIDIA/earth2studio/blob/<version>-rc/AGENTS.md`.
- **Main:** read `https://github.com/NVIDIA/earth2studio/blob/main/AGENTS.md`.
- **Specific version:** read the corresponding `<version>-rc/AGENTS.md` at that repository.
  Preserve an explicitly supplied `-rc` suffix rather than appending it twice.

Verify the branch and file exist; if unavailable, explain and ask before substituting.
Carry the choice forward without asking again. Use that branch's skills and documentation;
prefer local files only when they match. Released package installs use the matching release tag.
For a first forecast, start with [Earth2Studio - Get Started][get-started].

Earth2Studio is an AI weather and climate Python package with a large catalog of data sources
and prognostic, diagnostic, and data assimilation models. It builds on PyTorch, xarray, and CuPy.
Its goal is to make these components easy to discover, extend, and combine into reproducible,
GPU-accelerated workflows for forecasting, downscaling, assimilation, and evaluation.

Task guides live in [`skills/`](./skills/). Use the routes below to find relevant documentation.

Use the selected version's sources or matching documentation on the
[documentation site](https://nvidia.github.io/earth2studio/main/).
Read and cite only the sources relevant to the task.

## Documentation routes

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

## Skills

Activate the matching skill with your agent's skill tool, or read its `SKILL.md` directly.

### Install skills if needed

For main, install skills directly from the Earth2Studio GitHub repository.
Run from your project (requires Node.js/npm) and select your agent when prompted:

```bash
npx skills add https://github.com/NVIDIA/earth2studio --skill earth2studio-install
npx skills add https://github.com/NVIDIA/earth2studio --skill earth2studio-discover
```

Set the `--skill` value to any listed skill name. These commands use the default branch.
For another version, read skills from its selected RC branch or register them from
a matching checkout. Reading local skills needs no installation. From the checkout root:

```bash
npx skills add . --skill earth2studio-install
```

For the Python package, follow the install skill: use the install guide's uv setup,
select extras from `install_options.yml`, and give the user install and verification commands.

### Available skills

- [earth2studio-get-started][get-started]: set up, run, and plot your first UCAST forecast.
- [earth2studio-discover](./skills/earth2studio-discover/SKILL.md): choose models, sources, and examples.
- [earth2studio-install](./skills/earth2studio-install/SKILL.md): install and configure the package.
- [earth2studio-deterministic-forecast](./skills/earth2studio-deterministic-forecast/SKILL.md)
  for single-member forecasts with optional diagnostic postprocessing.
- [earth2studio-couple-models](./skills/earth2studio-couple-models/SKILL.md): run connected models.
- [earth2studio-data-fetch](./skills/earth2studio-data-fetch/SKILL.md): download weather/climate data.
- [earth2studio-create-datasource](./skills/earth2studio-create-datasource/SKILL.md): wrap a data source.
- [earth2studio-create-prognostic](./skills/earth2studio-create-prognostic/SKILL.md):
  wrap a forecast model.
- [earth2studio-create-diagnostic](./skills/earth2studio-create-diagnostic/SKILL.md): wrap a diagnostic.
- [earth2studio-create-assimilation](./skills/earth2studio-create-assimilation/SKILL.md): wrap assimilation.

## Common tools

Use [uv](https://docs.astral.sh/uv/) or the project's virtualenv for Python commands.
Follow the install skill for version selection, extras, and compatible GPU dependency wheels.

| Task | Command or guide |
| --- | --- |
| Initialize a new project | `uv init --python=3.13` in a new project directory |
| Add dependencies | `uv add <package>`; see the [install guide][install] |
| Run Python | `uv run python <script.py>` or `.venv/bin/python <script.py>` |
| Format this repository | `make format` |
| Lint this repository | `make lint` |
| Run selected tests | `uv run pytest <test-path>` |
| Check changed files | `uv run pre-commit run --files <paths>` |

Keep an existing project's Python version and tooling unless a change is needed.
Repository make targets are defined in [Makefile](./Makefile). See the
[style guide](./docs/userguide/developer/style.md),
[environment setup](./docs/userguide/developer/overview.md), and
[skill development guide](./docs/userguide/developer/skills.md) for details.

## Resources

- [API reference](https://nvidia.github.io/earth2studio/main/modules/)
- [Example gallery](https://nvidia.github.io/earth2studio/main/examples/)
- [Issues and support](https://github.com/NVIDIA/earth2studio/issues)

[install]: ./docs/userguide/about/install.md
[get-started]: ./skills/earth2studio-get-started/SKILL.md
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
