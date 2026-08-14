<!-- markdownlint-disable MD002 MD033 MD041 MD053 -->
<div align="center">

# NVIDIA Earth2Studio

[![python version][e2studio_python_img]][e2studio_python_url]
[![license][e2studio_license_img]][e2studio_license_url]
[![coverage][e2studio_cov_img]][e2studio_cov_url]
[![mypy][e2studio_mypy_img]][e2studio_mypy_url]
[![format][e2studio_format_img]][e2studio_format_url]
[![ruff][e2studio_ruff_img]][e2studio_ruff_url]
[![uv][e2studio_uv_img]][e2studio_uv_url]

Earth2Studio is a Python-based package designed to get users up and running
with AI Earth system models *fast*.
Our mission is to enable everyone to build, research and explore AI driven weather and
climate science.

<!-- markdownlint-disable MD036 -->
**- Earth2Studio Documentation -**
<!-- markdownlint-enable MD036 -->

[Install][e2studio_install_url] | [User-Guide][e2studio_userguide_url] |
[Examples][e2studio_examples_url] | [API][e2studio_api_url]

![Earth2Studio README hero](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-hero.png?v2)

</div>

## Quick start

Running AI weather prediction can be done with just a few lines of code.

- For detailed installation steps, including model-specific installations, see the
    [install guide][e2studio_install_url].
- See the [examples][e2studio_examples_url] gallery providing different inference
    workflow samples.
- Swap out [data sources][e2studio_data_api] or [models][e2studio_px_api] depending on
    your use case!

### Tutorial

[![Earth2Studio Tutorial](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-quickstart-video.png?v1)](https://www.youtube.com/watch?v=Sog6aCapZeA)

### Agent-assisted setup

Automate setup with your preferred coding agent using NVIDIA Earth2Studio skills.
Install the Earth2Studio skill set, then ask your favorite agent (Claude, Codex, OpenCode, etc) to
recommend a model, configure an environment, or run a first deterministic forecast.
Find more Earth2Studio skills in the [NVIDIA Skills catalog](https://build.nvidia.com/skills?q=earth2studio).

![Earth2Studio agentic setup](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-agent-setup.png?v1)

```bash
npx skills add NVIDIA/skills --skill earth2studio-install
npx skills add NVIDIA/skills --skill earth2studio-discover
npx skills add NVIDIA/skills --skill earth2studio-data-fetch
npx skills add NVIDIA/skills --skill earth2studio-deterministic-forecast
```

Example agent prompts:

```text
Use the Earth2Studio discover skill to recommend a starter forecast workflow.
Use the Earth2Studio install skill to set up my environment for FourCastNet3 inference.
Create a script to fetch ERA5 surface winds data for March 2024.
Create a deterministic forecast workflow with GFS, FourCastNet3, and a Zarr output store.
```

### NVIDIA FourCastNet3

```python
from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run

model = FCN3.load_model(FCN3.load_default_package())
data = GFS()
io = ZarrBackend("outputs/fcn3_forecast.zarr")
run(["2025-01-01T00:00:00"], 10, model, data, io)
```

### ECMWF AIFS

```python
from earth2studio.models.px import AIFS
from earth2studio.data import IFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run

model = AIFS.load_model(AIFS.load_default_package())
data = IFS()
io = ZarrBackend("outputs/aifs_forecast.zarr")
run(["2025-01-01T00:00:00"], 10, model, data, io)
```

### Google Graphcast

```python
from earth2studio.models.px import GraphCastOperational
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run

package = GraphCastOperational.load_default_package()
model = GraphCastOperational.load_model(package)
data = GFS()
io = ZarrBackend("outputs/graphcast_operational_forecast.zarr")
run(["2025-01-01T00:00:00"], 4, model, data, io)
```

> [!IMPORTANT]
> Earth2Studio is an interface to third‑party models, checkpoints, and datasets.
> Licenses for these assets are owned by their providers.
> Ensure you have the rights to download, use, and (if applicable) redistribute each
> model and dataset.
> Links to the original license and source are often provided in the API docs for each
> model/data source.

## Latest News

> [!NOTE]
> As of version `0.14.0`, Earth2Studio TOML default installs now target CUDA 13.

- [**Aurora v1.5**](https://nvidia.github.io/earth2studio/modules/generated/models/px/earth2studio.models.px.Aurora1p5.html),
    Microsoft Aurora v1.5 deterministic and ensemble model wrapper for global
    weather forecasting.
- [**StormCast CONUS**](https://nvidia.github.io/earth2studio/modules/generated/models/px/earth2studio.models.px.StormCastCONUS.html),
    StormCast CONUS prognostic model for convective-scale forecasting over the
    contiguous United States.
- [**Dynamical.org Sources**](https://nvidia.github.io/earth2studio/modules/generated/data/earth2studio.data.DynamicalGFS.html),
    a comprehensive suite of analysis and forecast data sources reading from
    anonymous Icechunk repositories (AIFS, GFS, GEFS, HRRR, MRMS, ICON-EU, IFS-ENS).
- [**EarthMover Data Sources**](https://nvidia.github.io/earth2studio/modules/generated/data/earth2studio.data.EarthMoverERA5.html),
    EarthMover ERA5 0.25-degree reanalysis and IFS 0.1-degree forecast sources
    hosted by BrightBand.
- [**StormScope NSRDB**](https://nvidia.github.io/earth2studio/modules/generated/models/dx/earth2studio.models.dx.StormScopeDxNSRDB.html),
    solar irradiance (GHI) estimation diagnostic model.

For a complete list of latest features and improvements see the [changelog](./CHANGELOG.md).

## Overview

Earth2Studio is an *AI inference pipeline toolkit* focused on weather and climate
applications that is designed to ride on top of different AI frameworks, model
architectures, data sources and SciML tooling while providing a unified API.

<div align="center">

![Earth2Studio model zoo](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-model-zoo.png?v3)
![Earth2Studio data sources](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-data-sources.png?v3)

</div>

The composability of the different core components in Earth2Studio easily allows the
development and deployment of increasingly complex pipelines that may chain multiple
data sources, AI models and other modules together.

<div align="center">

![Earth2Studio composable pipelines](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/readme/v2/earth2studio-readme-composability.png?v2)

</div>

The unified ecosystem of Earth2Studio provides users the opportunity to rapidly
swap out components for alternatives.
In addition to the largest model zoo of weather/climate AI models, Earth2Studio is
packed with useful functionality such as optimized data access to cloud data stores,
statistical operations and more to accelerate your pipelines.

### Earth-2 Open Models

Access state of the art Nvidia open models for climate and weather: [Earth-2 Open Models](https://huggingface.co/collections/nvidia/earth-2).
For training recipes for these models, see the [PhysicsNeMo repository][physicsnemo_repo_url].

## Features

Earth2Studio package focuses on supplying you the tools to build your own
workflows, pipelines, APIs, or packages using modular components including:

For a more complete list of features, be sure to view the [documentation][e2studio_docs_url].
Don't see what you need?
Great news, extension and customization are at the heart of our [design][e2studio_customization_url].

## Contributors

Check out the [contributing](CONTRIBUTING.md) document for details about the technical
requirements and the user guide for higher level philosophy, structure, and design.

## License

Earth2Studio is provided under the Apache License 2.0, refer to the
[LICENSE file][e2studio_license_url] for full license text.

<!-- Badge links -->

[e2studio_python_img]: https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13%20|%203.14-blue?style=flat-square&logo=python
[e2studio_license_img]: https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square
[e2studio_format_img]: https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square
[e2studio_mypy_img]: https://img.shields.io/badge/mypy-Checked-blue?style=flat-square&labelColor=grey
[e2studio_cov_img]: https://img.shields.io/codecov/c/github/nvidia/earth2studio?style=flat-square&logo=codecov
[e2studio_ruff_img]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square
[e2studio_uv_img]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fuv%2Fmain%2Fassets%2Fbadge%2Fv0.json&style=flat-square

[e2studio_python_url]: https://www.python.org/downloads/
[e2studio_license_url]: ./LICENSE
[e2studio_format_url]: https://github.com/psf/black
[e2studio_cov_url]: ./test/
[e2studio_mypy_url]: https://mypy-lang.org/
[e2studio_ruff_url]: https://github.com/astral-sh/ruff
[e2studio_uv_url]: https://github.com/astral-sh/uv

<!-- Doc links -->
[e2studio_docs_url]: https://nvidia.github.io/earth2studio/
[e2studio_install_url]: https://nvidia.github.io/earth2studio/userguide/about/install.html
[e2studio_userguide_url]: https://nvidia.github.io/earth2studio/userguide/
[e2studio_examples_url]: https://nvidia.github.io/earth2studio/examples/
[e2studio_api_url]: https://nvidia.github.io/earth2studio/modules/
[e2studio_customization_url]: https://nvidia.github.io/earth2studio/examples/extend/index.html
[e2studio_px_url]: https://nvidia.github.io/earth2studio/userguide/components/prognostic.html
[e2studio_px_api]: https://nvidia.github.io/earth2studio/modules/models_px.html
[e2studio_dx_url]: https://nvidia.github.io/earth2studio/userguide/components/diagnostic.html
[e2studio_dx_api]: https://nvidia.github.io/earth2studio/modules/models_dx.html
[e2studio_data_url]: https://nvidia.github.io/earth2studio/userguide/components/datasources.html
[e2studio_data_api]: https://nvidia.github.io/earth2studio/modules/datasources_analysis.html
[e2studio_io_url]: https://nvidia.github.io/earth2studio/userguide/components/io.html
[e2studio_io_api]: https://nvidia.github.io/earth2studio/modules/io.html
[e2studio_pb_url]: https://nvidia.github.io/earth2studio/userguide/components/perturbation.html
[e2studio_pb_api]: https://nvidia.github.io/earth2studio/modules/perturbation.html
[e2studio_stat_url]: https://nvidia.github.io/earth2studio/userguide/components/statistics.html
[e2studio_stat_api]: https://nvidia.github.io/earth2studio/modules/statistics.html
[e2studio_lex_url]: https://nvidia.github.io/earth2studio/userguide/advanced/lexicon.html

<!-- Misc links -->
[physicsnemo_repo_url]: https://github.com/NVIDIA/physicsnemo
