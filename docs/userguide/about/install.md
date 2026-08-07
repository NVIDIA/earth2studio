<!-- markdownlint-disable MD025 MD046 -->

# Install { #install_guide }

!!! warning "Base Install Limitations"
    The base pip install does not guarantee all functionality and/or examples are
    operational due to optional dependencies.
    We encourage users that face package issues to familiarize themselves with the optional
    model installs and suggested environment set up for the most complete experience.

## Install using Pip

Earth2Studio runs on [PyTorch](https://pytorch.org/get-started/locally/); **make sure it
is installed correctly for your system first**.
To get the latest release of Earth2Studio, install from the Python index.

```bash
pip install earth2studio
```

## Install using uv (recommended)

This package is developed using [uv](https://docs.astral.sh/uv/getting-started/installation/)
and it's recommended that users use an uv project for the best install experience:

```bash
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.13
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

??? info "uv Install"
    The use of the latest git release tag for the package install with uv is intentional.
    This will allow uv to handle any complicated dependency conditions and automatically
    handle url based dependencies.
    This is not achievable using the [pypi registry](https://docs.astral.sh/uv/pip/compatibility/#transitive-url-dependencies)
    but makes installing optional packages much easier down the line.

## Install Main Branch

To install the latest main branch version of Earth2Studio:

```bash
pip install hatchling # Optional if install process builds the wheel
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

## Optional Dependencies { #optional_dependencies }

??? info "uv Package Manager"
    For developers [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
    should be used.
    uv is **not required** for just using Earth2Studio thus both pip and uv commands are
    included.
    uv commands assume Earth2Studio has already been added to the project using *git source*
    used in the above sections.

!!! warning "Suggested prerequisites"
    Installing the base package before attempting any optional dependency groups is
    recommended if using pip.

    uv commands assume Earth2Studio has already been added to the project with the *git*
    link used in the above sections.

### Model Dependencies { #model_dependencies }

Models typically require additional dependencies which are not installed by default.
Use the optional install commands to add these dependencies.

#### Prognostics

=== "AIFS"
    Notes: The AIFS model requires additional dependencies for data processing and
    visualization. This includes the use of [flash-attention](https://github.com/Dao-AILab/flash-attention)
    which can take a long time to build on some systems.
    See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
    for known suggestions/fixes related to this install process.

    === "pip"

        ```bash
        pip install earth2studio[aifs] --no-build-isolation
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aifs
        ```

=== "AIFS Ensemble"
    Notes: The AIFS Ensemble model relies on updated ECMWF checkpoints with ensemble
    sampling support. Similar to the deterministic AIFS variant this extra depends on
    [flash-attention](https://github.com/Dao-AILab/flash-attention), which can take a long
    time to compile. See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
    for compilation tips.

    === "pip"

        ```bash
        pip install earth2studio[aifsens] --no-build-isolation
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aifsens
        ```

=== "AIFS2"
    Notes: This model depends on
    [flash-attention](https://github.com/Dao-AILab/flash-attention), which can take a long
    time to compile. See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
    for compilation tips.

    === "pip"

        ```bash
        pip install earth2studio[aifs2] --no-build-isolation
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aifs2
        ```

=== "AIFS2 Ensemble"
    Notes: This model depends on
    [flash-attention](https://github.com/Dao-AILab/flash-attention), which can take a long
    time to compile. See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
    for compilation tips.

    === "pip"

        ```bash
        pip install earth2studio[aifs2ens] --no-build-isolation
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aifs2ens
        ```

=== "Atlas"
    Notes: The Atlas model depends on [natten](https://github.com/SHI-Labs/NATTEN), which
    can take a long time to compile.

    === "pip"

        ```bash
        pip install --no-build-isolation "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"
        pip install earth2studio[atlas]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra atlas
        ```

=== "Aurora"
    Notes: The Aurora model relies on the [Microsoft Aurora](https://github.com/microsoft/aurora)
    package for inference.

    === "pip"

        ```bash
        pip install earth2studio[aurora]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aurora
        ```

=== "Aurora v1.5"
    Notes: The Aurora v1.5 model relies on the [Microsoft Aurora](https://github.com/microsoft/aurora)
    package for inference.

    === "pip"

        ```bash
        pip install earth2studio[aurora]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra aurora
        ```

=== "DLWP"
    === "pip"

        ```bash
        pip install earth2studio[dlwp]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra dlwp
        ```

=== "DLESyM"
    Notes: For all DLESyM models, [Earth2Grid](https://github.com/NVlabs/earth2grid) needs to
    be installed manually for pip users.

    === "pip"

        ```bash
        pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"
        pip install earth2studio[dlesym]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra dlesym
        ```

=== "FourCastNet"
    === "pip"

        ```bash
        pip install earth2studio[fcn]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra fcn
        ```

=== "FourCastNet 3"
    Notes: Recommended to install [torch-harmonics](https://github.com/NVIDIA/torch-harmonics)
    with CUDA extensions for best performance which can take a long time to build on some
    systems.
    See the [troubleshooting docs](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html)
    for known suggestions/fixes related to this install process.

    === "pip"

        ```bash
        export FORCE_CUDA_EXTENSION=1
        pip install --no-build-isolation "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"
        pip install "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"
        pip install earth2studio[fcn3]
        ```

    === "uv"

        ```bash
        export FORCE_CUDA_EXTENSION=1
        uv add earth2studio --extra fcn3
        ```

=== "FengWu"
    Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/#python-installs).
    This might have specific pip installation steps depending on your CUDA version.

    === "pip"

        ```bash
        pip install earth2studio[fengwu]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra fengwu
        ```

=== "FuXi"
    Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/#python-installs).
    This might have specific pip installation steps depending on your CUDA version.

    === "pip"

        ```bash
        pip install earth2studio[fuxi]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra fuxi
        ```

=== "GraphCast"
    Notes: The GraphCast models (operational and small) require additional dependencies
    for JAX and Haiku. The GraphCast package must be installed from the Google DeepMind
    repository.

    === "pip"

        ```bash
        pip install "graphcast @ git+https://github.com/google-deepmind/graphcast.git@7077d40a36db6541e3ed72ccaed1c0d202fa6014"
        pip install "earth2studio[graphcast]"
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra graphcast
        ```

=== "Pangu"
    Notes: Requires [ONNX GPU Runtime](https://onnxruntime.ai/docs/install/#python-installs).
    This might have specific pip installation steps depending on your CUDA version.

    === "pip"

        ```bash
        pip install earth2studio[pangu]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra pangu
        ```

=== "SFNO"
    Notes: Requires [Makani](https://github.com/NVIDIA/makani) to be
    installed manually.

    === "pip"

        ```bash
        pip install --no-build-isolation "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"
        pip install "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"
        pip install earth2studio[sfno]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra sfno
        ```

=== "StormCast"
    === "pip"

        ```bash
        pip install earth2studio[stormcast]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra stormcast
        ```

=== "StormCast-CONUS"
    === "pip"
        Notes: The StormCast-CONUS model depends on [natten](https://github.com/SHI-Labs/NATTEN),
        which can take a long time to compile.

        ```bash
        pip install earth2studio[stormcast-conus]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra stormcast-conus
        ```

=== "StormScope"
    Notes: The StormScope model depends on [natten](https://github.com/SHI-Labs/NATTEN),
    which can take a long time to compile. [Earth2Grid](https://github.com/NVlabs/earth2grid)
    needs to be installed manually for pip users.

    === "pip"

        ```bash
        pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"
        pip install earth2studio[stormscope]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra stormscope
        ```

=== "UCast"
    Notes: The UCast model does not require additional Python packages beyond the
    base Earth2Studio install. Install the model extra anyway so environments can
    select the UCast dependency group consistently.

    === "pip"

        ```bash
        pip install earth2studio[ucast]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra ucast
        ```

=== "InterpModAFNO"
    Notes: Requires a base prognostic model to be installed.

    === "pip"

        ```bash
        pip install earth2studio[interp-modafno]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra interp-modafno
        ```

#### Diagnostics

=== "CBottle"
    Notes: Additional dependencies needed for CBottle3D data source, CBottle video
    prognostic, CBottleInfill diagnostic and CBottleSR diagnostic.

    === "pip"

        ```bash
        pip install hatchling
        pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"
        pip install --no-build-isolation "cbottle @ git+https://github.com/NickGeneva/cBottle.git@e48c7eb518d49d4a92b2a1397d683e765c02c354"
        pip install earth2studio[cbottle]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra cbottle
        ```

=== "ClimateNet"
    Notes: No additional dependencies are needed for ClimateNet but included for
    completeness.

    === "pip"

        ```bash
        pip install earth2studio[climatenet]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra climatenet
        ```

=== "CorrDiff"
    Notes: Additional dependencies for all CorrDiff models.

    === "pip"

        ```bash
        pip install earth2studio[corrdiff]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra corrdiff
        ```

=== "CorrDiff COSMO-ERA5"
    Notes: Additional dependencies for the `CorrDiffCosmoEra5` model. This model needs
    the RoPE / NATTEN attention backend from `nvidia-physicsnemo`, which is not on PyPI
    yet, so physicsnemo must be installed from a pinned git commit. The `uv` path picks
    this up automatically from `[tool.uv.sources]`; the `pip` path installs it explicitly.

    === "pip"

        ```bash
        pip install "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@ced75d93d014f70bb691372788eee2d201171c12"
        pip install earth2studio[cosmo]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra cosmo
        ```

=== "Cyclone Trackers"
    Notes: Additional dependencies for cyclone tracking models `TCTrackerVitart` and `TCTrackerWuDuan`.

    === "uv"

        ```bash
        uv pip install earth2studio --extra cyclone
        ```

    === "pip"

        ```bash
        pip install earth2studio[cyclone]
        ```

        `TempestExtremes` is not provided as a Python library and must be installed
        separately by the user. Installation instructions can be found on the
        [TempestExtremes GitHub page](https://github.com/ClimateGlobalChange/tempestextremes?tab=readme-ov-file#installation-via-cmake-recommended).

        When compiling `TempestExtremes` via CMake, executables are placed in a `bin`
        directory inside the `TempestExtremes` source tree by default (i.e.
        `/path/to/tempestextremes/bin`). Because these binaries are not
        automatically added to the system `PATH`, the `detect_cmd` and `stitch_cmd`
        entries in the pipeline configuration must reference the full path to the
        `DetectNodes` and `StitchNodes` executables, e.g.
        `/path/to/tempestextremes/bin/DetectNodes ...`. When using the provided
        Docker container, the binaries are copied to `/usr/local/bin` and are therefore
        available on the `PATH`; in that case only the executable names are needed
        (e.g. `DetectNodes ...`). Examples for both commands are provided in the
        docstring of the `TempestExtremes` class and in the TC tracking recipe.

=== "Derived"
    Notes: Additional dependencies for all derived diagnostic models.
    No additional dependencies are needed for the derived models at the moment but included
    for completeness.

    === "pip"

        ```bash
        pip install earth2studio[derived]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra derived
        ```

=== "ORBIT-2"
    Notes: The ORBIT-2 diagnostic model requires the climate-learn package. This needs to be
    installed manually for pip users.

    === "pip"

        ```bash
        pip install "climate-learn @ git+https://github.com/NickGeneva/ORBIT-2@5b2d80a8ba4dc95029211ef2b8530d3663f65d39"
        pip install earth2studio[orbit]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra orbit
        ```

=== "Precipitation AFNO"
    === "pip"

        ```bash
        pip install earth2studio[precip-afno]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra precip-afno
        ```

=== "Precipitation AFNO V2"
    Notes: Improved version of the Precipitation AFNO model with enhanced accuracy.

    === "pip"

        ```bash
        pip install earth2studio[precip-afno-v2]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra precip-afno-v2
        ```

=== "Solar Radiation AFNO"
    Notes: Requires physicsnemo package for zenith angle calculations.

    === "pip"

        ```bash
        pip install earth2studio[solarradiation-afno]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra solarradiation-afno
        ```

=== "Windgust AFNO"
    === "pip"

        ```bash
        pip install earth2studio[windgust-afno]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra windgust-afno
        ```

#### Data Assimilation

!!! warning "Warning"
    Data assimilation model APIs are currently **in Beta** and may change in future
    releases. Expect possible breaking changes as these APIs mature.

!!! warning "Warning"
    All data assimilation models require [CuPy](https://docs.cupy.dev/en/stable/) and [cuDF](https://docs.rapids.ai/api/cudf/stable/),
    which are CUDA-dependent libraries.
    The default installation uses CUDA 12 (i.e., `cupy-cuda12x`, `cudf-cu12`).
    If your system uses a different CUDA version, you may need to adjust the dependencies.

=== "HealDA"
    === "pip"

        ```bash
        pip install hatchling
        pip install --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"
        pip install earth2studio[da-healda]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra da-healda
        ```

=== "InterpEquirectangular"
    === "pip"

        ```bash
        pip install earth2studio[da-interp]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra da-interp
        ```

=== "StormCast SDA"
    === "pip"

        ```bash
        pip install earth2studio[da-stormcast]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra da-stormcast
        ```

### Submodule Dependencies

A few features in various submodules require some specific dependencies that have been
deemed too specific to warrant an addition to the core dependencies.
These can be installed with a submodule wide install group:

=== "Data"

    === "pip"

        ```bash
        pip install earth2studio[data]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra data
        ```

=== "Perturbation"
    === "pip"

        ```bash
        pip install earth2studio[perturbation]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra perturbation
        ```

=== "Statistics"
    === "pip"

        ```bash
        pip install earth2studio[statistics]
        ```

    === "uv"

        ```bash
        uv add earth2studio --extra statistics
        ```

### Install All Optional Dependencies

In Earth2Studio, it's recommended that users pick and choose the optional dependencies that
are needed for their use case.
Installing everything at once and for all models is only expected to work in a few
specific environments and may not include support for every model depending on
conflicts.
This is only supported using uv and when using github as the source, [not pypi registry](https://docs.astral.sh/uv/pip/compatibility/#transitive-url-dependencies).
To install a best effort all optional dependencies group, use the following:

=== "uv"

    ```bash
    uv sync
    uv add earth2studio --extra all
    ```

# Environments { #install_environments }

For the best experience, we recommend creating a fresh environment whether that be using
uv, a Docker container or even a Conda environment.
Below are some recipes for creating a handful of environments for setting up
Earth2Studio in an isolated environment.
For developer environments, refer to the [Developer Overview](../developer/overview.md#developer_overview).

## uv Project

Using uv is the recommended way to set up a local Python environment for Earth2Studio.
Assuming [uv is installed](https://docs.astral.sh/uv/getting-started/installation/), use
the following commands:

```bash
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.13
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

or if you are already inside an existing uv project:

```bash
uv venv --python=3.13
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

## Docker Container { #pytorch_container_environment }

For a docker environment, the recommended process is to still use `uv` help install
packages for you.
The [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
typically provides a good base with many dependencies already installed and optimized
for NVIDIA hardware.
In container instances, using a virtual environment is often [not necessary](https://docs.astral.sh/uv/pip/environments/#using-arbitrary-python-environments).
It is recommended to use the following commands to install using the container's Python
interpreter:

```bash
docker run -it -t nvcr.io/nvidia/pytorch:26.04-py3

>>> apt-get update && apt-get install -y git make curl cmake python3-dev \
    libeccodes-tools libeccodes-dev
>>> unset PIP_CONSTRAINT
>>> curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
>>> uv pip install --system --break-system-packages "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

<!-- markdownlint-disable MD013 -->
!!! note "Extra Dependencies"
    To add extra dependencies adjust the `uv pip install` command like you would normally
    do with pip, for example:

    ```bash
    uv pip install --system \
        --break-system-packages \
        "earth2studio[aifs,data]@git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
    ```

??? warning "Earth2Studio in Docker"
    Some models and dependencies have specific system requirements (for example, CUDA
    versions) that may require a different container than the one listed here. If you are
    comfortable with Docker, refer to the [testing Dockerfile](https://github.com/NVIDIA/earth2studio/blob/main/test/Dockerfile)
    as a reference for building a general-purpose Earth2Studio image.

## Conda Environment

It is no longer recommended to use any conda environment manager for Earth2Studio in
favor of uv if possible.
This is because the virtual environments set up by uv makes the system-wide conda
environments not needed unless some system dependencies are required.
However this demonstrates that in principle Earth2Studio can be installed using standard
package tooling.

```bash
conda create -n earth2studio python=3.13
conda activate earth2studio

uv pip install --system --break-system-packages "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

# System Recommendations

## Software

Earth2Studio does not have any specific software version requirements.
The following are recommended to closely match development and automation environments,
minimizing the chance for unexpected incompatibilities:

- OS: Ubuntu 24.04 LTS
- Python Version: 3.13
- CUDA Version: 13.0

## Hardware

Earth2Studio does not have any specific hardware requirements, if PyTorch can run then
many features of Earth2Studio should run as well.
However, most models do require a GPU with sufficient memory and compute score to run
without complications.
The recommended hardware for the majority of models supported in Earth2Studio is:

| GPU | GPU Memory (GB) | Precision | # of GPUs | Disk Space (GB) |
| --- | --------------- | --------- | --------- | --------------- |
| [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) with compute capability ≥ 8.9 | ≥40 | FP32 | 1 | 128 |

This includes cards such as:

- L40S
- RTX A6000
- H100
- B200

We encourage users to experiment on different hardware for their specific needs and
use case.

# Configuration { #configuration_userguide }

Earth2Studio uses a few environment variables to configure various parts of the package.
The important ones are:

- `EARTH2STUDIO_CACHE`: The general cache location used for Earth2Studio. This is a file
path where things like models and cached data from data sources will be stored. Defaults to
`~/.cache/earth2studio`.
- `EARTH2STUDIO_DATA_CACHE`: The cache location specifically for data sources. If set,
this overrides `EARTH2STUDIO_CACHE` for data source caching operations.
- `EARTH2STUDIO_MODEL_CACHE`: The cache location specifically for model packages. If
    set, this overrides `EARTH2STUDIO_CACHE` for model checkpoint caching operations.
- `EARTH2STUDIO_PACKAGE_TIMEOUT`: The max number of seconds for a download operation of
a model package file from a remote store such as NGC, Huggingface or S3.
