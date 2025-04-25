# Recipe Guide

Earth2Studio recipes are reference solutions that focus on solving specific use cases.
Recipes provide more complex implementations that may require deeper domain knowledge
and familiarity with the codebase.

:::{admonition} Warning
:class: warning

Earth2Studio recipes are in beta, thus may dramatically change/get moved.
:::

## What is a Recipe?

While examples demonstrate how to use features, recipes focus on how to solve specific
problems.
Recipes are not intended to be turnkey solutions with every potential option; rather
reference or boilerplate projects for different common use cases.
Here's how they differ:

### Examples

- Self-contained [JupyText](https://jupytext.readthedocs.io/en/latest/) format
- Focus on demonstrating specific features
- Include graphical output for sphinx gallery
- Requirements must be part of earth2studio + docs dep group
- Single GPU, A100 (40G) max resource requirement
- Runs in less than 10 minutes

### Recipes

- Require multiple support files
- Focus on providing complete solutions
- Configuration YAMLs/JSONs
- Pythonic interaction through main.py
- Requirements provided in pyproject.toml
- May require more specific hardware or system dependencies

See {ref}`examples_userguide` for more information on creating an example.

## Recipe Structure

The [recipe template](https://github.com/NVIDIA/earth2studio/recipes/template/),
defines what a recipe should contain and look like including folders, README sections,
support files, etc.
Recipes have the following structure:

```text
recipes/
 |- recipe-name/
    cfg/
    |- config.yaml # Configuration files
    src/
    |- *.py # Support files
    test/
    |- test.py # Test files
    main.py # Main interaction script
    README.md # Recipe documentation
    requirements.txt # Exported requirements
    pyproject.toml # Project configuration
```

## Recipe Guidelines

1. **Code Standards**

    - Must follow all pre-commit standards and code practices
    - Must include end-to-end tests validating functionality
    - **No** images/data files/other file formats allowed in the repo, can link to
        externally hosted
    - Implementation or re-implementation of components that exist in Earth2Studio
        package is not encouraged. Upstreaming is a priority

2. **Testing**

    - Should include end-to-end tests with expected results to verify functionality
    - Not included in the main pytest suite
    - Tested following major release cycles

3. **Ownership**

    - Must have designated owners (listed as authors in `pyproject.toml`)
    - Owners responsible for fixing bugs and updates
    - Recipes can and may be removed based on their use / relevance

4. **Dependencies**

    - Must use shipped versions of Earth2Studio (not main branch)
    - Dependencies managed through `pyproject.toml` using [uv](https://docs.astral.sh/uv/)
    - A `requirements.txt` should also be provided for pip install

5. **Documentation**

    - Clear README.md explaining purpose and usage
    - Can include JuPyText notebooks for interactive documentation as the main.py
    - Should reference related examples when applicable
    - Links to the recipes should be added to the recipe index
    - Recipes can provide related plotting functionality

## Creating a New Recipe

To create a new recipe, you can use the [template](https://github.com/NVIDIA/earth2studio/recipes/template/)
provided in the recipes directory:

```bash
cp -r recipes/template recipes/my_new_recipe
cd recipes/my_new_recipe
```

Update the `pyproject.toml` to reflect the correct meta data:

```toml
[project]
name = "newrecipe"
version = "0.1.0"
description = "My New Recipe"
readme = "README.md"
requires-python = ">=3.10"
authors = [
  { name="My Name", email = "optional@example.com" },
]
```

Inside the recipe dependencies can be added / removed using [uv](https://docs.astral.sh/uv/).
For example:

```bash
uv sync
uv add cartopy

# Update the requirements.txt
uv export --format requirements-txt --no-hashes >> requirements.txt
```

Because recipes are more extensive, expect a longer review timeline.
Making iterative updates expanding functionality is preferred.
