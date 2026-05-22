# Install.md Templates

This file contains templates for adding model installation instructions to
`docs/userguide/about/install.md`.

## Location

Add entries to the **Prognostics** section (starting around line 107) in
**alphabetical order** among existing `:::::{tab-item}` blocks.

## Basic Template

```markdown
:::::{tab-item} ModelName
Notes: <Any special installation notes, dependencies, or warnings>

::::{tab-set}
:::{tab-item} pip

```bash
pip install earth2studio[model-name]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra model-name
```

:::
::::
:::::
```

## Template with Complex pip Install

For models requiring manual git packages (uv handles these automatically):

```markdown
:::::{tab-item} ModelName
Notes: <Model description>. For pip users, [Package](https://github.com/...)
needs to be installed manually.

::::{tab-set}
:::{tab-item} pip

```bash
pip install --no-build-isolation "package @ git+https://github.com/org/repo@commit"
pip install earth2studio[model-name]
```

:::
:::{tab-item} uv

```bash
uv add earth2studio --extra model-name
```

:::
::::
:::::
```

## Example: AIFS

```markdown
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
```

## Insertion Rules

1. Find the correct alphabetical position within the Prognostics section
2. The extra name must match the dependency group added to `pyproject.toml`
3. Document any pip prerequisites that uv handles automatically
