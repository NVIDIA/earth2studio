<!-- markdownlint-disable MD033 MD041 -->

> [!IMPORTANT]
> **1.0.0**: This is a major release with breaking changes from pre-1.0 releases.
> See the [migration guide](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/MIGRATION.md)
> if you need help transitioning, or point your agent to it.

<div align="center">

# NVIDIA Earth2Studio

**Next-generation AI weather and climate modeling.**

[![python version][e2studio_python_img]][e2studio_python_url]
[![license][e2studio_license_img]][e2studio_license_url]
[![coverage][e2studio_cov_img]][e2studio_cov_url]
[![uv][e2studio_uv_img]][e2studio_uv_url]

[Install][e2studio_install_url] | [User-Guide][e2studio_userguide_url] |
[Examples][e2studio_examples_url] | [API][e2studio_api_url] |
[Skills](./skills)

</div>

Access a leading collection of weather and climate AI models, production-ready data sources,
composable inference APIs, and GPU-accelerated workflows in one Python package.

![Earth2Studio hero][e2studio_hero_img]

## For agents

Paste this into your coding agent (Claude, Codex, OpenCode, Cursor, etc.):

```text
Help me get started with NVIDIA Earth2Studio. Read
https://github.com/NVIDIA/earth2studio/blob/main/AGENTS.md and follow its guidance
for my task. If I have a local checkout, use its AGENTS.md instead.
```

[`AGENTS.md`](./AGENTS.md) routes usage and development tasks to the relevant
documentation and skills. Then ask for what you want:

```text
Summarize new models and features in the latest Earth2Studio release.
Include the version, release date, and links to the release notes.

Create me a 10-day AI forecasting pipeline starting from the latest GFS analysis.
Set up my environment for FourCastNet3 inference.
Recommend the best model for a 2-week global forecast on one GPU.
Fetch ERA5 surface winds for March 2024 and save them to Zarr.
```

To keep the skills installed, add them from this repository:

```bash
npx skills add NVIDIA/earth2studio --skill earth2studio-install
npx skills add NVIDIA/earth2studio --skill earth2studio-discover
```

User workflow skills and developer-only package implementation skills are listed
in [`AGENTS.md`](./AGENTS.md).

## For humans

| Start here | |
| --- | --- |
| [**Install**][e2studio_install_url] | Environments, model-specific extras, hardware |
| [**Examples**][e2studio_examples_url] | Deterministic, ensemble, diagnostic and coupled workflows |
| [**User Guide**][e2studio_userguide_url] | Components, design philosophy and extending the package |
| [**API**][e2studio_api_url] | Every [prognostic][e2studio_px_api], [diagnostic][e2studio_dx_api] and [data source][e2studio_data_api] |

## Main Features

- [**Forecast**][e2studio_px_api] with global and regional prognostic AI models.
- [**Diagnose**][e2studio_dx_api] derived fields such as precipitation, wind speed and downscaled output.
- [**Assimilate**][e2studio_da_api] observations into global analyses.
- [**Fetch**][e2studio_data_api] analysis, forecast and observation data from public sources.
- [**Compose**][e2studio_examples_url] ensembles and coupled multi-model workflows.
- [**Store**][e2studio_io_api] results to Zarr, NetCDF and more.

## Documentation

Everything else (model zoo, data sources, news, tutorials) lives on the
[documentation site][e2studio_docs_url]. See the [changelog](./CHANGELOG.md) for what
is new, [CONTRIBUTING](./CONTRIBUTING.md) to get involved, and
[Earth-2 Open Models](https://huggingface.co/collections/nvidia/earth-2) for NVIDIA's
open weights and [PhysicsNeMo][physicsnemo_repo_url] for training recipes.
For deploying workflows as an inference service, see
[PhysicsNeMo Serve](https://github.com/NVIDIA/physicsnemo-serve).

> [!NOTE]
> Earth2Studio is an interface to third‑party models, checkpoints, and datasets.
> Licenses for these assets are owned by their providers. Ensure you have the rights
> to download, use, and (if applicable) redistribute each model and dataset.

## License

Apache License 2.0, see the [LICENSE file][e2studio_license_url].

<!-- Badge links -->

[e2studio_python_img]: https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13%20|%203.14-blue?style=flat-square&logo=python
[e2studio_license_img]: https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square
[e2studio_cov_img]: https://img.shields.io/codecov/c/github/nvidia/earth2studio?style=flat-square&logo=codecov
[e2studio_uv_img]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fuv%2Fmain%2Fassets%2Fbadge%2Fv0.json&style=flat-square

[e2studio_python_url]: https://www.python.org/downloads/
[e2studio_license_url]: ./LICENSE
[e2studio_cov_url]: ./test/
[e2studio_uv_url]: https://github.com/astral-sh/uv

<!-- Doc links -->
[e2studio_docs_url]: https://nvidia.github.io/earth2studio/
[e2studio_install_url]: https://nvidia.github.io/earth2studio/main/userguide/about/install/
[e2studio_userguide_url]: https://nvidia.github.io/earth2studio/main/userguide/
[e2studio_examples_url]: https://nvidia.github.io/earth2studio/main/examples/
[e2studio_api_url]: https://nvidia.github.io/earth2studio/main/modules/
[e2studio_px_api]: https://nvidia.github.io/earth2studio/main/modules/models_px/
[e2studio_dx_api]: https://nvidia.github.io/earth2studio/main/modules/models_dx/
[e2studio_da_api]: https://nvidia.github.io/earth2studio/main/modules/models_da/
[e2studio_io_api]: https://nvidia.github.io/earth2studio/main/modules/io/
[e2studio_data_api]: https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/

<!-- Misc links -->
[physicsnemo_repo_url]: https://github.com/NVIDIA/physicsnemo
[e2studio_hero_img]: https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/1e19e907c909d87888dfb0acf7c17fa809ed2cc8/readme/earth2studio_hero_image.png
