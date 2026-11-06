<!-- markdownlint-disable MD033 MD041 -->

> [!IMPORTANT]
> **1.0.0**: This tier has been released. For a migration guide please point your agent
> to this [migration guide](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/MIGRATION.md)
> to transition.

<div align="center">

# NVIDIA Earth2Studio

**Weather and climate AI inference, for humans and agents.**

[![python version][e2studio_python_img]][e2studio_python_url]
[![license][e2studio_license_img]][e2studio_license_url]
[![coverage][e2studio_cov_img]][e2studio_cov_url]
[![uv][e2studio_uv_img]][e2studio_uv_url]

[Install][e2studio_install_url] | [User-Guide][e2studio_userguide_url] |
[Examples][e2studio_examples_url] | [API][e2studio_api_url] |
[Skills](./skills)

</div>

Earth2Studio is a Python toolkit for running AI Earth system models *fast*: one
unified API over models, data sources, IO backends and SciML tooling. Swap a model
or a data source and the rest of the pipeline keeps working.

## For agents

Install the Earth2Studio skills, restart your agent (Claude, Codex, OpenCode, etc.)
and ask:

```bash
npx skills add NVIDIA/earth2studio --skill earth2studio-install
npx skills add NVIDIA/earth2studio --skill earth2studio-discover
npx skills add NVIDIA/earth2studio --skill earth2studio-data-fetch
npx skills add NVIDIA/earth2studio --skill earth2studio-deterministic-forecast
```

```text
Create me a 10-day AI forecasting pipeline starting from the latest GFS analysis.
Set up my environment for FourCastNet3 inference.
Recommend the best model for a 2-week global forecast on one GPU.
Fetch ERA5 surface winds for March 2024 and save them to Zarr.
```

More skills for extending Earth2Studio (new prognostic, diagnostic, data source,
assimilation and coupled models) live in [`skills/`](./skills).

## For humans

| Start here | |
| --- | --- |
| [**Install**][e2studio_install_url] | Environments, model-specific extras, hardware |
| [**Examples**][e2studio_examples_url] | Deterministic, ensemble, diagnostic and coupled workflows |
| [**User Guide**][e2studio_userguide_url] | Components, design philosophy and extending the package |
| [**API**][e2studio_api_url] | Every [prognostic][e2studio_px_api], [diagnostic][e2studio_dx_api] and [data source][e2studio_data_api] |

## Documentation

Everything else (model zoo, data sources, news, tutorials) lives on the
[documentation site][e2studio_docs_url]. See the [changelog](./CHANGELOG.md) for what
is new, [CONTRIBUTING](./CONTRIBUTING.md) to get involved, and
[Earth-2 Open Models](https://huggingface.co/collections/nvidia/earth-2) for NVIDIA's
open weights and [PhysicsNeMo][physicsnemo_repo_url] for training recipes.

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
[e2studio_data_api]: https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/

<!-- Misc links -->
[physicsnemo_repo_url]: https://github.com/NVIDIA/physicsnemo
