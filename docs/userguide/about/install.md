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
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.6.0"
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
Notes: The AIFS model requires additional dependencies for data processing and visualization.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[aifs]
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
pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"
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
:::::{tab-item} ClimateNet
Notes: No additional dependencies are needed for ClimateNet but included for
completeness.

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[climatenet]
```