<!-- markdownlint-disable MD033 MD046 -->

# Install { #install_guide }

!!! warning "Base Install Limitations"
    **Base install scope:** The base install only includes the core Earth2Studio
    package. Many models, data sources, examples, and workflows require optional
    dependency groups. Use the selector below to choose the install that matches the
    component you want to use.

    **GPU dependency compatibility:** GPU-dependent model dependencies can be sensitive
    to the PyTorch and CUDA versions in your environment.

!!! important "Prerequisites"
    - **PyTorch:** See the [PyTorch install guide](https://pytorch.org/get-started/locally/)
      and make sure PyTorch is installed correctly on your system first.
    - **Python environment:** Initialize an appropriate Python environment. Python 3.13
      is the recommended version.
    - **Package manager:** uv is recommended, but pip is also supported.

## Install Selector { #install-command }

<section class="e2s-install-selector" data-e2s-install-selector></section>

## Verify Installation

```bash
python -c "import earth2studio; print(earth2studio.__version__)"
```

If you installed with uv and want to run inside the uv project:

```bash
uv run python -c "import earth2studio; print(earth2studio.__version__)"
```

## Environments { #install_environments }

For the best experience, create a fresh environment with uv, Docker, or another
environment manager. For developer environments, refer to the
[Developer Overview](../developer/overview.md#developer_overview).

### uv Project

Using uv is the recommended way to set up a local Python environment for Earth2Studio.
Assuming [uv is installed](https://docs.astral.sh/uv/getting-started/installation/), use
the following commands:

```bash
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.13
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

### Docker Container { #pytorch_container_environment }

For a Docker environment, the recommended process is to use `uv` inside a container.
The [NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
typically provides a good base with many dependencies already installed and optimized
for NVIDIA hardware.

```bash
docker run -it -t nvcr.io/nvidia/pytorch:26.04-py3

apt-get update && apt-get install -y git make curl cmake python3-dev \
    libeccodes-tools libeccodes-dev
unset PIP_CONSTRAINT
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv pip install --system --break-system-packages \
    "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
```

!!! note "Extra Dependencies"
    Add extras to the `uv pip install` command in the same way you would for pip:

    ```bash
    uv pip install --system --break-system-packages \
        "earth2studio[aifs,data] @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0"
    ```

??? warning "Earth2Studio in Docker"
    Some models and dependencies have specific system requirements, such as CUDA
    versions, that may require a different container. If you are comfortable with
    Docker, refer to the
    [testing Dockerfile](https://github.com/NVIDIA/earth2studio/blob/main/test/Dockerfile)
    as a reference for a general-purpose Earth2Studio image.

### Conda Environment

It is no longer recommended to use conda environment managers for Earth2Studio when uv
is available. If conda is required for your system, use it only to create the Python
environment and install Earth2Studio with standard Python tooling.

```bash
conda create -n earth2studio python=3.13
conda activate earth2studio
pip install earth2studio
```

## System Recommendations

### Software

Earth2Studio does not have specific software version requirements. The following
versions are recommended to closely match development and automation environments:

- OS: Ubuntu 24.04 LTS
- Python Version: 3.13
- CUDA Version: 13.0

### Hardware

Earth2Studio does not have specific hardware requirements. If PyTorch can run, many
features of Earth2Studio should run as well. Most models do require a GPU with
sufficient memory and compute capability.

| GPU | GPU Memory (GB) | Precision | # of GPUs | Disk Space (GB) |
| --- | --------------- | --------- | --------- | --------------- |
| [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) with compute capability >= 8.9 | >=40 | FP32 | 1 | 128 |

## Configuration { #configuration_userguide }

Earth2Studio uses a few environment variables to configure package behavior:

- `EARTH2STUDIO_CACHE`: General cache location for models and cached data. Defaults to
  `~/.cache/earth2studio`.
- `EARTH2STUDIO_DATA_CACHE`: Cache location for data sources. If set, this overrides
  `EARTH2STUDIO_CACHE` for data source caching.
- `EARTH2STUDIO_MODEL_CACHE`: Cache location for model packages. If set, this overrides
  `EARTH2STUDIO_CACHE` for model checkpoint caching.
- `EARTH2STUDIO_PACKAGE_TIMEOUT`: Maximum number of seconds for a model package download
  from a remote store such as NGC, Hugging Face, or S3.
