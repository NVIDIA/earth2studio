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

![Earth2Studio Banner](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/raw/main/0.2.0/earth2studio_feature_banner.png?id=1)

</div>

## Quick start

Running AI weather prediction can be done with just a few lines of code.

- For detailed installation steps, including model-specific installations, see the
    [install guide][e2studio_install_url].
- See the [examples][e2studio_examples_url] gallery providing different inference
    workflow samples.
- Swap out [data sources][e2studio_data_api] or [models][e2studio_px_api] depending on
    your use case!

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

- **CMIP6 datasource** has been added to improve support for usecases that are focused
    on climate modeling. See [data source APIs](e2studio_api_url) for more information.
- [**Ai2 Climate Emulator (ACE) 2 ERA5 model**](https://arxiv.org/pdf/2411.11268v1) has
    been added which is a 1 degree, 6 hour time-step, forecast model that supports long
    roll outs with user specified SST forcing.
- [**Climate in a Bottle**](https://blogs.nvidia.com/blog/earth2-generative-ai-foundation-model-global-climate-kilometer-scale-resolution/)
    model APIs have been extended with the addition of tropical Cyclone guidance
    diagnostic and cBottle video prognostic model.

For a complete list of latest features and improvements see the [changelog](./CHANGELOG.md).

## Overview

Earth2Studio is an *AI inference pipeline toolkit* focused on weather and climate
applications that is designed to ride on top of different AI frameworks, model
architectures, data sources and SciML tooling while providing a unified API.

<div align="center">

![Earth2Studio Overview 1](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/resolve/main/0.9.0/earth2studio-readme-overview-1.png?id=1)

</div>

The composability of the different core components in Earth2Studio easily allows the
development and deployment of increasingly complex pipelines that may chain multiple
data sources, AI models and other modules together.

<div align="center">

![Earth2Studio Overview 1](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/resolve/main/0.9.0/earth2studio-readme-overview-2.png?id=1)

</div>

The unified ecosystem of Earth2Studio provides users the opportunity to rapidly
swap out components for alternatives.
In addition to the largest model zoo of weather/climate AI models, Earth2Studio is
packed with useful functionality such as optimized data access to cloud data stores,
statistical operations and more to accelerate your pipelines.

<div align="center">

![Earth2Studio Overview 1](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/resolve/main/0.9.0/earth2studio-readme-overview-3.webp?id=1)

</div>

Earth2Studio can be used for seamless deployment of Earth-2 models trained in
[PhysicsNeMo][physicsnemo_repo_url].

## Features

Earth2Studio package focuses on supplying users the tools to build their own
workflows, pipelines, APIs, packages, etc. via modular components including:

<details>
<summary>Prognostic Models</summary>

[Prognostic models][e2studio_px_url]
    in Earth2Studio perform time integration, taking atmospheric fields at a specific
    time and auto-regressively predicting the same fields into the future (typically 6
    hours per step), enabling both single time-step predictions and extended time-series
    forecasting.

Earth2Studio maintains the largest collection of pre-trained state-of-the-art AI
    weather/climate models ranging from global forecast models to regional specialized
    models, covering various resolutions, architectures, and forecasting capabilities to
    suit different computational and accuracy requirements.

Available models include but are not limited to:

| Model | Resolution | Architecture | Time Step | Coverage |
|-------|------------|--------------|-----------|----------|
| GraphCast Small | 1.0° | Graph Neural Network | 6h | Global |
| GraphCast Operational | 0.25° | Graph Neural Network | 6h | Global |
| Pangu 3hr | 0.25° | Transformer | 3h | Global |
| Pangu 6hr | 0.25° | Transformer | 6h | Global |
| Pangu 24hr | 0.25° | Transformer | 24h | Global |
| Aurora | 0.25° | Transformer | 6h | Global |
| FuXi | 0.25° | Transformer | 6h | Global |
| AIFS | 0.25° | Transformer | 6h | Global |
| AIFS Ensemble | 0.25° | Transformer Ensemble | 6h | Global |
| StormCast | 3km | Diffusion + Regression | 1h | Regional (US) |
| SFNO | 0.25° | Neural Operator | 6h | Global |
| DLESyM | 0.25° | Convolutional | 6h | Global |

For a complete list, see the [prognostic model API docs][e2studio_px_api].

</details>

<details>
<summary>Diagnostic Models</summary>

[Diagnostic models][e2studio_dx_url] in Earth2Studio perform time-independent
    transformations, typically taking geospatial fields at a specific time and
    predicting new derived quantities without performing time integration enabling users
    to build pipelines to predict specific quantities of interest that may not be
    provided by forecasting models.

Earth2Studio contains a growing collection of specialized diagnostic models for
    various phenomena including precipitation prediction, tropical cyclone tracking,
    solar radiation estimation, wind gust forecasting, and more.

Available diagnostics include but are not limited to:

| Model | Resolution | Architecture | Coverage | Output |
|-------|------------|--------------|----------|--------|
| PrecipitationAFNO | 0.25° | Neural Operator  | Global | Total precipitation |
| SolarRadiationAFNO1H | 0.25° | Neural Operator  | Global | Surface solar radiation |
| WindgustAFNO | 0.25° | AFNO | Global | Maximum wind gust |
| TCTrackerVitart | 0.25° | Algorithmic | Global | TC tracks & properties |
| CBottleInfill | 100km | Diffusion | Global | Global climate sample |
| CBottleSR | 5km | Diffusion | Regional / Global | High-res climate |
| CorrDiff | Variable | Diffusion | Regional | Fine-scale weather |
| CorrDiffTaiwan | 2km | Diffusion | Regional (Taiwan) | Taiwan fine-scale weather |

For a complete list, see the [diagnostic model API docs][e2studio_dx_api].

</details>

<details>
<summary>Datasources</summary>

[Data sources][e2studio_data_url]
    in Earth2Studio provide a standardized API for accessing weather and climate
    datasets from various providers (numerical models, data assimilation results, and
    AI-generated data), enabling seamless integration of initial conditions for model
    inference and validation data for scoring across different data formats and storage
    systems.

Earth2Studio includes data sources ranging from operational weather models (GFS, HRRR,
    IFS) and reanalysis datasets (ERA5 via ARCO, CDS) to AI-generated climate data
    (cBottle) and local file systems. Fetching data is just plain easy, Earth2Studio
    handles the complicated parts giving the users an easy to use Xarray data array of
    requested data under a shared package wide [vocabulary][e2studio_lex_url] and
    coordinate system.

Available data sources include but are not limited to:

| Data Source | Type | Resolution | Coverage | Data Format |
|-------------|------|------------|----------|-------------|
| GFS | Operational | 0.25° | Global | GRIB2 |
| GFS_FX | Forecast | 0.25° | Global | GRIB2 |
| HRRR | Operational | 3km | Regional (US) | GRIB2 |
| HRRR_FX | Forecast | 3km | Regional (US) | GRIB2 |
| ARCO ERA5 | Reanalysis | 0.25° | Global | Zarr |
| CDS | Reanalysis | 0.25° | Global | NetCDF |
| IFS | Operational | 0.25° | Global | GRIB2 |
| NCAR_ERA5 | Reanalysis | 0.25° | Global | NetCDF |
| WeatherBench2 | Reanalysis | 0.25° | Global | Zarr |
| GEFS_FX | Ensemble Forecast | 0.25° | Global | GRIB2 |
| CBottle3D | AI Generated | 100km | Global | HEALPix |

For a complete list, see the [data source API docs][e2studio_data_api].

</details>

<details>
<summary>IO Backends</summary>

[IO backends][e2studio_io_url] in
    Earth2Studio provides a standardized interface for writing and storing
    pipeline outputs across different file formats and storage systems enabling users
    to store inference outputs for later processing.

Earth2Studio includes IO backends ranging from traditional scientific formats (NetCDF)
    and modern cloud-optimized formats (Zarr) to in-memory storage backends.

Available IO backends include:

| IO Backend | Format | Features | Location |
|------------|--------|----------|----------|
| ZarrBackend | Zarr | Compression, Chunking | In-Memory/Local |
| AsyncZarrBackend | Zarr | Async writes, Parallel I/O | In-Memory/Local/Remote |
| NetCDF4Backend | NetCDF4 | CF-compliant, Metadata | In-Memory/Local |
| XarrayBackend | Xarray Dataset | Rich metadata, Analysis-ready | In-Memory |
| KVBackend | Key-Value| Fast Temporary Access | In-Memory |

For a complete list, see the [IO API docs][e2studio_io_api].

</details>

<details>
<summary>Perturbation Methods</summary>

[Perturbation methods][e2studio_pb_url]
    in Earth2Studio provide a standardized interface for adding noise
    to data arrays, typically enabling the creation of ensembling forecast pipelines
    that capture uncertainty in weather and climate predictions.

Available perturbations include but are not limited to:

| Perturbation Method | Type | Spatial Correlation | Temporal Correlation |
|---------------------|------|-------------------|---------------------|
| Gaussian | Noise | None | None |
| Correlated SphericalGaussian | Noise | Spherical | AR(1) process |
| Spherical Gaussian | Noise | Spherical (Matern) | None |
| Brown | Noise | 2D Fourier | None |
| Bred Vector | Dynamical | Model-dependent | Model-dependent |
| Hemispheric Centred Bred Vector | Dynamical | Hemispheric | Model-dependent |

For a complete list, see the [perturbations API docs][e2studio_pb_url].

</details>

<details>
<summary>Statistics / Metrics</summary>

[Statistics and metrics][e2studio_stat_url]
    in Earth2Studio provide operations typically useful for in-pipeline evaluation of
    forecast performance across different dimensions (spatial, temporal, ensemble)
    through various statistical measures including error metrics, correlation
    coefficients, and ensemble verification statistics.

Available operations include but are not limited to:

| Statistic | Type | Application |
|-----------|------|-------------|
| RMSE | Error Metric | Forecast accuracy |
| ACC | Correlation | Pattern correlation |
| CRPS | Ensemble Metric | Probabilistic skill |
| Rank Histogram | Ensemble Metric | Ensemble reliability |
| Standard Deviation | Moment | Spread measure |
| Spread-Skill Ratio | Ensemble Metric | Ensemble calibration |

For a complete list, see the [statistics API docs][e2studio_stat_api].

</details>

For a more complete list of features, be sure to view the [documentation][e2studio_docs_url].
Don't see what you need?
Great news, extension and customization are at the heart of our [design][e2studio_customization_url].

## Contributors

Check out the [contributing](CONTRIBUTING.md) document for details about the technical
requirements and the userguide for higher level philosophy, structure, and design.

## License

Earth2Studio is provided under the Apache License 2.0, please see the
[LICENSE file][e2studio_license_url] for full license text.

<!-- Badge links -->

[e2studio_python_img]: https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-blue?style=flat-square&logo=python
[e2studio_license_img]: https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square
[e2studio_format_img]: https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square
[e2studio_mypy_img]: https://img.shields.io/badge/mypy-Checked-blue?style=flat-square&labelColor=grey
[e2studio_cov_img]: https://img.shields.io/codecov/c/github/nickgeneva/earth2studio?style=flat-square&logo=codecov
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
[e2studio_px_api]: https://nvidia.github.io/earth2studio/modules/models.html#earth2studio-models-px-prognostic
[e2studio_dx_url]: https://nvidia.github.io/earth2studio/userguide/components/diagnostic.html
[e2studio_dx_api]: https://nvidia.github.io/earth2studio/modules/models.html#earth2studio-models-dx-diagnostic
[e2studio_data_url]: https://nvidia.github.io/earth2studio/userguide/components/datasources.html
[e2studio_data_api]: https://nvidia.github.io/earth2studio/modules/datasources.html
[e2studio_io_url]: https://nvidia.github.io/earth2studio/userguide/components/io.html
[e2studio_io_api]: https://nvidia.github.io/earth2studio/modules/io.html
[e2studio_pb_url]: https://nvidia.github.io/earth2studio/userguide/components/perturbation.html
[e2studio_pb_api]: https://nvidia.github.io/earth2studio/modules/perturbation.html
[e2studio_stat_url]: https://nvidia.github.io/earth2studio/userguide/components/statistics.html
[e2studio_stat_api]: https://nvidia.github.io/earth2studio/modules/statistics.html
[e2studio_lex_url]: https://nvidia.github.io/earth2studio/userguide/advanced/lexicon.html

<!-- Misc links -->
[physicsnemo_repo_url]: https://github.com/NVIDIA/physicsnemo
