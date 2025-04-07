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

## Install using UV (Recommended)

This package is developed using [uv](https://docs.astral.sh/uv/getting-started/installation/)
and its recommended that users use UV for the best install experience:

```bash
uv venv --python=3.12
uv pip install earth2studio
```

## Install from Source

To install the latest main branch version of Earth2Studio:

```bash
git clone https://github.com/NVIDIA/earth2studio.git
cd earth2-inference-studio
pip install .
```

or if you are using uv:

```bash
uv venv --python=3.12
uv pip install "earth2studio @ git+https://github.com/NVIDIA/earth2studio"
```

## Verify Installation

```bash
python
# or when using uv
uv run python

>>> import earth2studio
>>> earth2studio.__version__
```

## Optional Dependencies

:::{admonition} uv Package Manager
:class: warning
From this point forward, using [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
will be the default package manager over pip in the documentation.
Due to the complexities of interfacing with multiple models with different dependency
requirements, Earth2Studio relies on uv to create a reproducible runtime environment.
uv is **not required**, and all installs can be replaced with pip command variants that
are included but have limited support.
:::

(data_dependencies)=

### Datasource Dependencies

Some data sources require additional dependencies, libraries or specific Python versions
to install.
To install all dependencies

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra data
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[data]
```

:::
::::

(model_dependencies)=

### Model Dependencies

Models typically require additional dependencies which are not installed by default.
Use the optional install commands to add these dependencies.

#### Prognostics

::::::{tab-set}
:::::{tab-item} Aurora
Note: The shipped Aurora package has a restricted dependency which is incompatible with
other Earth2Studio dependiences, thus it is suggested to use the forked variant.

::::{tab-set}
:::{tab-item} uv

```bash
# Patched fork
uv pip install earth2studio --extra aurora-fork
# Original package from msc
uv pip install earth2studio --extra aurora
```

:::
:::{tab-item} pip

```bash
pip install "microsoft-aurora @ git+https://github.com/NickGeneva/aurora.git@ab41cf1de67d5dcc723b96fc9a6219e4b548d181"
```

:::
::::
:::::
:::::{tab-item} FourCastNet
::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra fcn
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[fcn]
```

:::
::::
:::::
:::::{tab-item} FengWu
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra fengwu
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[fengwu]
```

:::
::::
:::::
:::::{tab-item} FuXi
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra fuxi
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[fuxi]
```

:::
::::
:::::
:::::{tab-item} Pangu
Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra pangu
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[pangu]
```

:::
::::
:::::
:::::{tab-item} SFNO
Notes: Requires [Modulus-Makani](https://github.com/NVIDIA/modulus-makani) to be
installed manually.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra sfno
```

:::
:::{tab-item} pip

```bash
pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"
pip install earth2studio[sfno]
```

:::
::::
:::::
:::::{tab-item} StormCast
::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra stormcast
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[stormcast]
```

:::
::::
:::::
::::::

#### Diagnostics

::::::{tab-set}
:::::{tab-item} ClimateNet
Notes: No additional dependencies are needed for ClimateNet but included for
completeness.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra climatenet
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[climatenet]
```

:::
::::
:::::
:::::{tab-item} CorrDiff
Notes: Additional dependencies for all CorrDiff models.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra corrdiff
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[corrdiff]
```

:::
::::
:::::
:::::{tab-item} Precipitation AFNO
::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra precip-afno
```

:::
:::{tab-item} pip

```bash
pip install earth2studio[precip-afno]
```

:::
::::
:::::
::::::

## Install All Optional Dependencies

In Earth2Studio, its suggested that users pick and choose the optional dependencies that
are needed for their use case.
Installing everything at once and for all models is only expected to work in a few
golden environments and may not include support for every model depending on conflicts.
To install a best effort all optional dependencies group, use the following:

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install earth2studio --extra all
```

:::
::::

(install_environments)=

# Environments

For the best experience we recommend creating a fresh environment whether that be using
a Docker container or a Conda environment.
Below are some recipes for creating a handful of environments that we recommend for
setting up Earth2Studio to run all build in models.

## PhysicsNeMo Docker Container

The recommended environment to run Earth2Studio in is the [PhysicsNeMo docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo).
This is the environment the team develops with and is the primary test bed.
You can install Earth2Studio in a running container directly:

```bash
docker run -i -t nvcr.io/nvidia/physicsnemo/physicsnemo:25.03

>>> pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"

>>> pip install earth2studio[all]
```

or build your own Earth2Studio container using a Dockerfile:

```dockerfile
FROM nvcr.io/nvidia/physicsnemo/physicsnemo:25.03

RUN pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"

RUN pip install earth2studio[all]
```

## PyTorch Docker Container

PhysicsNeMo docker container is shipped with some packages that are not directly needed by
Earth2Studio.
Thus, some may prefer to install from the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
Note that for ONNX models to work we will need a [specific install](https://onnxruntime.ai/docs/install/#install-onnx-runtime-ort-1):

```bash
docker run -i -t nvcr.io/nvidia/pytorch:25.02-py3

>>> pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"

>>> pip install earth2studio[all]
```

## Conda Environment

For instances where Docker is not an option, we recommend creating a Conda environment.
Ensuring the PyTorch is running on your GPU is essential, make sure you are [installing](https://pytorch.org/get-started/locally/)
the correct PyTorch for your hardware and CUDA is accessible.

```bash
conda create -n earth2studio python=3.12
conda activate earth2studio

pip install torch
conda install eccodes python-eccodes -c conda-forge
pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"

pip install earth2studio[all]
```

(configuration_userguide)=

# Configuration

Earth2Studio uses a few environment variables to configure various parts of the package.
The import ones are:

- `EARTH2STUDIO_CACHE`: The location of the cache used for Earth2Studio. This is a file
path where things like models and cached data from data sources will be stored.
- `EARTH2STUDIO_PACKAGE_TIMEOUT`: The max number of seconds for a download operation of
a model package file from a remote store such as NGC, Huggingface or S3.
