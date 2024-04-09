# Install

🚧 Under construction 🚧

## Install from PyPi

To get the latest release of Earth2Studio install from the Python index:

```bash
pip install earth2studio
```

## Install from Source

To install the latest main branch version of Earth2Studio:

```bash
git clone https://gitlab-master.nvidia.com/modulus/earth-2/earth2-inference-studio.git

cd earth2-inference-studio

pip install .
```

:::{admonition} Base Install Limitations
:class: warning

The base pip install does not guarentee all functionality and/or examples are
operational due to optional dependencies.
We encourage users that face package issues to familize themselves with the optional
model installs and suggested enviroment set up for the most complete experience.
:::

## Model Dependencies

Some models require additional dependencies which are not installed by default.
Use the optional install commands to add these dependencies.

```{list-table}
    :widths: 25 40 15
    :header-rows: 1


   * - Model
     - Install Command
     - Install Notes
   * - Pangu
     - `pip install earth2studio[pangu]`
     - ONNX Runtime
   * - FengWu
     - `pip install earth2studio[fengwu]`
     - ONNX Runtime
   * - SFNO
     - `pip install earth2studio[sfno]`
     - [Modulus-Makani](https://github.com/NVIDIA/modulus-makani)
```

## For Developers

For developers, use an edittable install of Earth-2 Studio with the `dev` option:

```bash
git clone https://gitlab-master.nvidia.com/modulus/earth-2/earth2-inference-studio.git

cd earth2-inference-studio

pip install -e .[dev]
```

Earth-2 Studio uses pre-commit which also should be installed immediately by developers:

```bash
pre-commit install

>>> pre-commit installed at .git/hooks/pre-commit
```

To install documentation dependencies use:

```bash
pip install .[docs]
```

## Configuration

Earth2Studio uses a few enviroment variables to configure various parts of the package.
The import ones are:

- `EARTH2STUDIO_CACHE`: The location of the cache used for Earth2Studio. This is a file
path where things like models and cached data from data sources will be stored.
