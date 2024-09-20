<!-- markdownlint-disable MD025 -->

(developer_overview)=

# Overview

This guide assumes you have looked at the repository and are familiar with the majority
information found in the {ref}`userguide`.

## Environment Setup

When developing Earth2Studio, it's suggested a docker environment is used detailed in
the [install environments](#install_environments) section.
Clone the repo and use an editable install of Earth2Studio with the `dev` option:

```bash
git clone https://github.com/NVIDIA/earth2studio.git

cd earth2-inference-studio

pip install -e .[dev]
```

To install documentation dependencies, use:

```bash
pip install .[docs]
```

:::{note}
When working with models, additional optional dependencies may be required. See the
[model dependencies](#model_dependencies) section for details.
:::

## Pre-commit

For Earth2Studio development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline, but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

To install `pre-commit` run the following inside the Earth2Studio repository folder:

```bash
pip install pre-commit
pre-commit install

>>> pre-commit installed at .git/hooks/pre-commit
```

Once the above commands are executed, the pre-commit hooks will be activated and all
the commits will be checked for appropriate formatting, linting, etc.
