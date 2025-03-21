<!-- markdownlint-disable MD025 -->

(install_guide)=

# Install

## Install from PyPi

To get the latest release of Earth2Studio, install from the Python index:

```bash
pip install earth2studio
```

## Install from Source

To install the latest main branch version of Earth2Studio:

```bash
git clone https://github.com/NVIDIA/earth2studio.git

cd earth2-inference-studio

pip install .
```

Verify installation using:

```bash
python

>>> import earth2studio
>>> earth2studio.__version__
```

:::{admonition} Base Install Limitations
:class: warning

The base pip install does not guarantee all functionality and/or examples are
operational due to optional dependencies.
We encourage users that face package issues to familiarize themselves with the optional
model installs and suggested environment set up for the most complete experience.
:::

(data_dependencies)=

## Datasource Dependencies

Some data sources require additional dependencies, libraries or specific Python versions
to install.
To install all dependencies

```bash
pip install earth2studio[data]
```

(model_dependencies)=

## Model Dependencies

Some models require additional dependencies which are not installed by default.
Use the optional install commands to add these dependencies.

::::{tab-set}
:::{tab-item} CorrDiff

Notes: Additional dependencies for all CorrDiff models.

```bash
pip install earth2studio[corrdiff]
```

:::

:::{tab-item} FengWu

Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

```bash
pip install earth2studio[fengwu]
```

:::

:::{tab-item} FuXi

Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

```bash
pip install earth2studio[fuxi]
```

:::

:::{tab-item} Pangu

Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/). May need
manual install depending on CUDA version.

```bash
pip install earth2studio[pangu]
```

:::

:::{tab-item} SFNO
Notes: Requires [Modulus-Makani](https://github.com/NVIDIA/modulus-makani) to be
installed manually.

```bash
pip install "makani[all] @ git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0"
pip install earth2studio[sfno]
```

:::
::::

Using `pip install earth2studio[all]` will install all optional functionality dependencies.

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
