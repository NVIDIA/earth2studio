<!-- markdownlint-disable MD033 MD041 MD053 -->
<div align="center">

# Earth-2 Inference Studio

[![python version][e2studio_python_img]][e2studio_python_url]
[![license][e2studio_license_img]][e2studio_license_url]
[![format][e2studio_format_img]][e2studio_format_url]

Earth-2 Inference Studio is a Python-based package designed to get users up and running
with AI weather and climate models *fast*.
Our mission is to enable everyone to build, research and explore AI driven meteorology.

<!-- markdownlint-disable MD036 -->
**- Earth-2 Inference Studio Documentation -**
<!-- markdownlint-enable MD036 -->

[Home](e2studio_docs_url) | [User-Guide](e2studio_userguide_url) |
[Examples](e2studio_examples_url) | [API](e2studio_api_url)

</div>

## Quick start

Install Earth-2 Inference Studio from source:

```bash
git clone https://gitlab-master.nvidia.com/modulus/earth-2/earth2-inference-studio.git

cd earth2-inference-studio

pip install .
```

Run a deterministic weather prediction in just a few lines of code:

```python
from earth2studio.models.px import DLWP
from earth2studio.data import GFS
from earth2studio.io import NetCDF
from earth2studio import wf_simple

model = DLWP.load_model(DLWP.load_default_package())
ds = GFS()
io = NetCDF("output.nc")

wf_simple(time=["2024-01-01"], nsteps=10, model, ds, io)
```

## Features

Earth-2 Inference Studio focuses on providing users the tools to build their own
workflows, pipelines, APIs, pacakges, etc. via modular components including:

- Collection of pre-trained weather/climate prediction models
- Collection of pre-trained diagnostic weather models
- Variety of online and on-prem data sources for initialization, scoring, analysis, etc.
- IO utilities for exporting predicted data to user friendly formats
- Suite of perturbation methods for building ensemble predictions
- Sample workflows and examples for common tasks / use cases
- Seamless integration into other Nvidia packages including [Modulus](modulus_repo_url).

For a more complete list of feature set, be sure to view the [documentation](e2studio_features_url).
Don't see what you need?
Great news, extension and customization are at the heart of our [design](e2studio_customization_url).

## Contributors

Check out the [Design](DESIGN.md) document for details about the repo's philosphy,
structure, and design.

## License

Earth-2 Inference Studio is provided under the Apache License 2.0, please see
[LICENSE file](e2studio_license_url) for full license text.

<!-- Badge links -->

[e2studio_python_img]: https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python
[e2studio_license_img]: https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square
[e2studio_format_img]: https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square

[e2studio_python_url]: https://www.python.org/downloads/
[e2studio_license_url]: ./LICENSE
[e2studio_format_url]: https://github.com/psf/black

<!-- Doc links -->
[e2studio_docs_url]: https://www.python.org/downloads/
[e2studio_userguide_url]: https://www.python.org/downloads/
[e2studio_examples_url]: https://www.python.org/downloads/
[e2studio_api_url]: https://www.python.org/downloads/
[e2studio_customization_url]: https://www.python.org/downloads/

<!-- Misc links -->
[modulus_repo_url]: https://github.com/NVIDIA/modulus
