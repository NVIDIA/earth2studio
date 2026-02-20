<!-- markdownlint-disable MD025 -->

(developer_overview)=

# Overview

This guide assumes you have reviewed the repository and are familiar with the majority
information found in the {ref}`userguide`.

## Environment Setup

When developing Earth2Studio, using uv virtual environment is suggested and will be used
in the documentation.
To create a local development environment with a git repo and uv, follow the steps
below.
Clone the repo and use uv to create a Python 3.12 virtual environment and sync it with
the `dev` dependency group:

```bash
# Replace with your fork
git clone https://github.com/NVIDIA/earth2studio.git --single-branch

cd earth2studio

uv venv --python=3.12
uv sync # This will install the base and developer dependencies
```

This should create a Python virtual environment inside of the local Earth2Studio git
repository.
To install the base, developer, documentation dependencies, use:

```bash
uv sync --group={docs}
```

:::{note}
When working with models, additional optional dependencies may be required.
See the [model dependencies](#model_dependencies) section for details on the optional
dependencies inside the package.
Use the `uv sync --extra <optional dep>` command instead of the pip install.
Take note of the difference between the optional dependency groups used here and the
extra dependencies when using specific models.
:::

## Pre-commit

For Earth2Studio development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline, but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

To install `pre-commit` run the following inside the Earth2Studio repository folder:

```bash
uv pip install pre-commit
uv run pre-commit install

>>> pre-commit installed at .git/hooks/pre-commit
```

After the above commands are executed, the pre-commit hooks are activated and all
the commits are checked for appropriate formatting, linting.
