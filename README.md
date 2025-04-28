<!-- markdownlint-disable MD002 MD033 MD041 MD053 -->
<div align="center">

![Earth2Studio Banner](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/raw/main/0.2.0/earth2studio_banner.png)

[![python version][e2studio_python_img]][e2studio_python_url]
[![license][e2studio_license_img]][e2studio_license_url]
[![coverage][e2studio_cov_img]][e2studio_cov_url]
[![mypy][e2studio_mypy_img]][e2studio_mypy_url]
[![format][e2studio_format_img]][e2studio_format_url]
[![ruff][e2studio_ruff_img]][e2studio_ruff_url]
[![uv][e2studio_uv_img]][e2studio_uv_url]

Earth2Studio is a Python-based package designed to get users up and running
with AI weather and climate models *fast*.
Our mission is to enable everyone to build, research and explore AI driven meteorology.

<!-- markdownlint-disable MD036 -->
**- Earth2Studio Documentation -**
<!-- markdownlint-enable MD036 -->

[Install][e2studio_install_url] | [User-Guide][e2studio_userguide_url] |
[Examples][e2studio_examples_url] | [API][e2studio_api_url]

![Earth2Studio Banner](https://huggingface.co/datasets/NickGeneva/Earth2StudioAssets/raw/main/0.2.0/earth2studio_feature_banner.png)

</div>

## Quick start

Install Earth2Studio:

```bash
pip install earth2studio[dlwp]
```

Run a deterministic weather prediction in just a few lines of code:

```python
from earth2studio.models.px import DLWP
from earth2studio.data import GFS
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic as run

model = DLWP.load_model(DLWP.load_default_package())
ds = GFS()
io = NetCDF4Backend("output.nc")

run(["2024-01-01"], 10, model, ds, io)
```

## Features

Earth2Studio provides access to pre-trained AI weather models and inference
features through an easy to use and extendable Python interface.
This package focuses on supplying users the tools to build their own
workflows, pipelines, APIs, packages, etc. via modular components including:

- Collection of pre-trained weather/climate prediction models
- Collection of pre-trained diagnostic weather models
- Variety of online and on-prem data sources for initialization, scoring, analysis, etc.
- IO utilities for exporting predicted data to user friendly formats
- Suite of perturbation methods for building ensemble predictions
- Sample workflows and examples for common tasks / use cases
- Seamless integration into other Nvidia packages including [PhysicsNeMo][physicsnemo_repo_url]

For a more complete list of feature set, be sure to view the [documentation][e2studio_docs_url].
Don't see what you need?
Great news, extension and customization are at the heart of our [design][e2studio_customization_url].

## Contributors

Check out the [Contributing](CONTRIBUTING.md) document for details about the technical
requirements and the userguide for higher level philosophy, structure, and design.

## License

Earth2Studio is provided under the Apache License 2.0, please see
[LICENSE file][e2studio_license_url] for full license text.

<!-- Badge links -->

[e2studio_python_img]: https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?style=flat-square&logo=python
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

<!-- Misc links -->
[physicsnemo_repo_url]: https://github.com/NVIDIA/physicsnemo
