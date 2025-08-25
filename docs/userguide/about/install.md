<!-- markdownlint-disable MD025 -->

(install_guide)=

# Install

:::{admonition} Base Install Limitations
:class: warning

The base pip install does not guarantee all functionality and/or examples are
operational due to optional dependencies.
We encourage users that face package issues to familiarize themselves with the optional
model installs and suggested environment set up for the most complete experience.
:::

## Install using Pip

To get the latest release of Earth2Studio, install from the Python index:

```bash
pip install earth2studio
```

## Install using uv (recommended)

This package is developed using [uv](https://docs.astral.sh/uv/getting-started/installation/)
and it's recommended that users use an uv project for the best install experience:

```bash
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

:::{admonition} uv install
:class: note

The use of the latest git release tag for the package install with uv is intentional.
This will allow uv to handle any complicated dependency conditions and automatically
handle url based dependencies.
This is not achievable using the [pypi registry](https://docs.astral.sh/uv/pip/compatibility/#transitive-url-dependencies)
but makes installing optional packages much easier down the line.
:::

## Install Main Branch

To install the latest main branch version of Earth2Studio main branch:

```bash
pip install "earth2studio @ git+https://github.com/NVIDIA/earth2studio"
```

or if you are using uv:

```bash
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git"
```

## Verify Installation

```bash
python
# or when using uv
uv run python

>>> import earth2studio
>>> earth2studio.__version__
```

(optional_dependencies)=

## Optional Dependencies

:::{admonition} uv Package Manager
:class: tip
For developers [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
should be used.
uv is **not required** for just using Earth2Studio thus both pip and uv commands are
included.
uv commands assume Earth2Studio has already been added to the project using *git source*
used in the above sections.
:::

:::{admonition} uv Package Manager
:class: warning
uv commands assume Earth2Studio has already been added to the project with the *git*
link used in the above sections.
:::

(model_dependencies)=

### Model Dependencies

Models typically require additional dependencies which are not installed by default.
Use the optional install commands to add these dependencies.

#### Prognostics

::::::{tab-set}
:::::{tab-item} AIFS
Notes: The AIFS model requires additional dependencies for data processing and
visualization. This includes the use of [flash-attention](https://github.com/Dao-AILab/flash-attention)
which can take a long time to build on some systems.
See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
for known suggestions/fixes related to this install process.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[aifs] --no-build-isolation
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra aifs
```

:::
::::
:::::
:::::{tab-item} Aurora
Note: The shipped Aurora package has a restricted dependency which is incompatible with
other Earth2Studio dependencies, thus it is suggested to use the forked variant.

::::{tab-set}
:::{tab-item} pip

```bash
pip install "microsoft-aurora @ git+https://github.com/NickGeneva/aurora.git@ab41cf1de67d5dcc723b96fc9a6219e4b548d181"
pip install earth2studio[aurora]
```

:::
:::{tab-item} uv

```bash
# Patched fork
uv add earth2studio --extra aurora-fork
# Original package from msc
uv add earth2studio --extra aurora
```

:::
::::
:::::
:::::{tab-item} DLWP
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[dlwp]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra dlwp
```

:::
::::
:::::
:::::{tab-item} DLESyM
Notes: For all DLESyM models, [Earth2Grid](https://github.com/NVlabs/earth2grid) needs to
be installed manually.

::::{tab-set}
:::{tab-item} pip

```bash
pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@aefb10793aa372bae7d0951d627a89e2983fd0ca"
pip install earth2studio[dlesym]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra dlesym
```

:::
::::
:::::
:::::{tab-item} FourCastNet
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[fcn]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra fcn
```

:::
::::
:::::
:::::{tab-item} FourCastNet 3
Notes: Recommended to install [torch-harmonics](https://github.com/NVIDIA/torch-harmonics)
with CUDA extensions for best performance which can take a long time to build on some
systems.
See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
for known suggestions/fixes related to this install process.

::::{tab-set}
:::{tab-item} pip

```bash
export FORCE_CUDA_EXTENSION=1
pip install --no-build-isolation torch-harmonics==0.8.0
pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@49280812513d8f1daf872a2e9343855a6adb3acf"
pip install earth2studio[fcn3]
```

:::
:::{tab-item} uv

```bash
export FORCE_CUDA_EXTENSION=1
uv add torch-harmonics==0.8.0 --no-build-isolation
uv add earth2studio --extra fcn3
```

:::
::::
:::::
:::::{tab-item} FengWu
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA and Python version.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[fengwu]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra fengwu
```

:::
::::
:::::
:::::{tab-item} FuXi
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[fuxi]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra fuxi
```

:::
::::
:::::
:::::{tab-item} GraphCast
Notes: The GraphCast models (operational and small) require additional dependencies for JAX and Haiku.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[graphcast]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra graphcast
```

:::
::::
:::::
:::::{tab-item} Pangu
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[pangu]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra pangu
```

:::
::::
:::::
:::::{tab-item} SFNO
Notes: Requires [Modulus-Makani](https://github.com/NVIDIA/modulus-makani) to be
installed manually.

::::{tab-set}
:::{tab-item} pip

```bash
pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@49280812513d8f1daf872a2e9343855a6adb3acf"
pip install earth2studio[sfno]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra sfno
```

:::
::::
:::::
:::::{tab-item} StormCast
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[stormcast]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra stormcast
```

:::
::::
:::::
:::::{tab-item} InterpModAFNO
Notes: Requires a base prognostic model to be installed.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[interp-modafno]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra interp-modafno
```

:::
::::
:::::
::::::

#### Diagnostics

::::::{tab-set}
:::::{tab-item} CBottle
Notes: Additional dependencies needed for CBottle3D data source, CBottle video
prognostic, CBottleInfill diagnostic and CBottleSR diagnostic.

::::{tab-set}
:::{tab-item} pip

```bash
pip install hatchling
pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@aefb10793aa372bae7d0951d627a89e2983fd0ca"
pip install --no-build-isolation "cbottle @ git+https://github.com/NickGeneva/cBottle.git@9b24c2f81096a340444e7e0966fe93223a4d9532"
pip install earth2studio[cbottle]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra cbottle
```

:::
::::
:::::
:::::{tab-item} ClimateNet
Notes: No additional dependencies are needed for ClimateNet but included for
completeness.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[climatenet]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra climatenet
```

:::
::::
:::::
:::::{tab-item} CorrDiff
Notes: Additional dependencies for all CorrDiff models.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[corrdiff]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra corrdiff
```

:::
::::
:::::
:::::{tab-item} Cyclone Trackers
Notes: Additional dependencies for all cyclone tracking models. Only Python 3.12 and
below support.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra cyclone
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[cyclone]
```

:::
::::
:::::
:::::{tab-item} Derived
Notes: Additional dependencies for all derived diagnostic models.
No additional dependencies are needed for the derived models at the moment but included
for completeness.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[derived]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra derived
```

:::
::::
:::::
:::::{tab-item} Precipitation AFNO
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[precip-afno]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra precip-afno
```

:::
::::
:::::
:::::{tab-item} Precipitation AFNO V2
Notes: Improved version of the Precipitation AFNO model with enhanced accuracy.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[precip-afno-v2]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra precip-afno-v2
```

:::
::::
:::::
:::::{tab-item} Solar Radiation AFNO
Notes: Requires physicsnemo package for zenith angle calculations.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[solarradiation-afno]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra solarradiation-afno
```

:::
::::
:::::
:::::{tab-item} Windgust AFNO
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[windgust-afno]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra windgust-afno
```

:::
::::
:::::
::::::

### Submodule Dependencies

A few features in various submodules require some specific dependencies that have been
deemed too specific to warrant an addition to the core dependencies.
These can be installed with a submodule wide install group:

::::::{tab-set}
:::::{tab-item} Data

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[data]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra data
```

:::
::::
:::::
:::::{tab-item} Perturbation
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[perturbation]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra perturbation
```

:::
::::
:::::
:::::{tab-item} Statistics
::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[statistics]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra statistics
```

:::
::::
:::::
::::::

### Install All Optional Dependencies

In Earth2Studio, it's recommended that users pick and choose the optional dependencies that
are needed for their use case.
Installing everything at once and for all models is only expected to work in a few
specific environments and may not include support for every model depending on
conflicts.
This is only supported using uv and when using github as the source, [not pypi registry](https://docs.astral.sh/uv/pip/compatibility/#transitive-url-dependencies).
To install a best effort all optional dependencies group, use the following:

::::{tab-set}
:::{tab-item} uv

```bash
uv sync
uv add earth2studio --extra all
```

:::
::::

(install_environments)=

# Environments

For the best experience, we recommend creating a fresh environment whether that be using
uv, a Docker container or even a Conda environment.
Below are some recipes for creating a handful of environments for setting up
Earth2Studio in an isolated environment.
For developer environments, please refer to the {ref}`developer_overview`.

## uv Project

Using uv is the recommend way to set up a local Python environment for Earth2Studio.
Assuming [uv is installed](https://docs.astral.sh/uv/getting-started/installation/), use
the following commands:

```bash
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

or if you are already inside an existing uv project:

```bash
uv venv --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

(pytorch_container_environment)=

## PyTorch Docker Container

For a docker environment the [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
provides a good base with many dependencies already installed and optimized for NVIDIA
hardware.
In container instances, using a virtual environment is often [not necessary](https://docs.astral.sh/uv/pip/environments/#using-arbitrary-python-environments).
It is recommend using the following commands to install using the container's Python
interpreter:

```bash
docker run -it -t nvcr.io/nvidia/pytorch:25.05-py3

>>> apt-get update && apt-get install -y git make curl && rm -rf /var/lib/apt/lists/*
>>> unset PIP_CONSTRAINT
>>> curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
>>> uv pip install --system --break-system-packages "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

<!-- markdownlint-disable MD013 -->
:::{admonition} Extra Dependencies
:class: note

To add extra dependencies adjust the `uv pip install` command like you would normally
do with pip e.g.

```bash
uv pip install --system \
    --break-system-packages \
    "earth2studio[all]@git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

:::

<!-- markdownlint-enable MD013 -->

## Custom Container

For a dedicated docker container the following can be used to get started.
There is some complexity to undo the pip constraints from the PyTorch container, but
otherwise the install process is the same.

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Disable constraint files in the container
ENV PIP_CONSTRAINT=
# Install Earth2Studio and dependencies
RUN uv pip install --system --break-system-packages "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

## Conda Environment

It is no longer recommend to use any conda environment manager for Earth2Studio in favor
of uv if possible.
This is because the virtual environments set up by uv makes the system-wide conda
environments not needed unless some system dependencies are required.
However this demonstrates that in principle Earth2Studio can be installed using standard
package tooling.

```bash
conda create -n earth2studio python=3.12
conda activate earth2studio

uv pip install --system --break-system-packages "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
```

# System Recommendations

## Software

Earth2Studio does not have any specific software version requirements.
The following are recommended to closely match development and automation environments,
minimizing the chance for unforeseen incompatibilities:

- OS: Ubuntu 24.04 LTS
- Python Version: 3.12
- CUDA Version: 12.8

## Hardware

Earth2Studio does not have any specific hardware requirements, if PyTorch can run then
many features of Earth2Studio should run as well.
However, most models do require a GPU with sufficient memory and compute score to run
without complications.
The recommended hardware for the majority of models supported in Earth2Studio is:

| GPU | GPU Memory (GB) | Precision | # of GPUs | Disk Space (GB) |
|-----|-----------------|-----------|-----------|-----------------|
| [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) with compute capability ≥ 8.9 | ≥40 | FP32 | 1 | 128 |

This includes cards such as:

- L40S
- RTX A6000
- H100
- B200

We encourage users to experiment on different hardware for their specific needs and
use case.

(configuration_userguide)=

# Configuration

Earth2Studio uses a few environment variables to configure various parts of the package.
The important ones are:

- `EARTH2STUDIO_CACHE`: The location of the cache used for Earth2Studio. This is a file
path where things like models and cached data from data sources will be stored.
- `EARTH2STUDIO_PACKAGE_TIMEOUT`: The max number of seconds for a download operation of
a model package file from a remote store such as NGC, Huggingface or S3.
