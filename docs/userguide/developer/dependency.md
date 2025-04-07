# Dependency Management

Earth2Studio uses [uv](https://github.com/astral-sh/uv) as its primary package manager
to handle complex dependency relationships between different models, data sources, and
utilities.
This choice helps maintain a clean, reproducible environment while avoiding dependency
conflicts and bloat.

## Base Dependencies

The base installation provides core functionality that is focused on the relative
ecosystem surrounding the following four packages:

- `torch`: Deep learning framework
- `xarray`: N-D labeled arrays and datasets
- `zarr`: Array storage format
- `fsspec`: Filesystem interfaces

With these base dependencies, most IO methods, data sources, perturbation methods, and
utilities should be functional. However, specific features may require additional
optional dependencies.

:::{admonition} Adding Packages
:class: note
uv commands should always be used to update package lists inside the pyproject.toml.
Refer to the [uv documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/)
for details about this if unfamiliar.
:::

## Optional Dependencies

Earth2Studio uses uv's dependency groups to organize optional features:

```toml
# Example pyproject.toml structure
[project.optional-dependencies]
# Data
data = [
    "cdsapi>=0.7.5",
    "eccodes>=2.38.0",
    "ecmwf-opendata>=0.3.3",
    "herbie-data",
]
# PX Models
aurora = [
    "microsoft-aurora>=1.5.0",
]
aurora-fork = [
    "microsoft-aurora", # optional fork install without timm package version conflict
]
dlwp = [
    "nvidia-physicsnemo>=1.0.1",
]
```

Each model and major feature set has its own optional dependency group.
This allows users to:

1. Install only what they need
2. Avoid potential conflicts between different models
3. Keep the base installation lightweight

Earth2Studio also attempts to maintain a best effort "all" optional group that can be
updated accordingly.

### Handling Optional Imports

When developing features that require optional dependencies, always use graceful error
handling:

```python
try:
    import optional_package
except ImportError:
    raise ImportError(
        "Optional dependency 'optional_package' is required for this feature. "
        "Install with: uv pip install earth2studio --extra feature_name"
    )
```

### Managing Conflicts

Earth2Studio uses uv's conflict resolution system to explicitly handle package
incompatibilities:

```toml
# Example conflict resolution in pyproject.toml
[tool.uv]
conflicts = [
    [
      { extra = "aurora" },
      { extra = "dlwp" },
    ],
]
```

All potential conflicts should be documented in the pyproject.toml file to ensure
reproducible environments.

## Developer Dependencies

Development dependencies are managed separately using uv dependency groups:

```toml
[dependency-groups]
dev = [
    "black==24.1.0",
    "coverage>=6.5.0",
    "interrogate>=1.5.0",
    "hatch>=1.14.0",
    "mypy",
    ...
]
```

These dependencies are not included in the distributed package wheel.
For more information on dependency management with uv, see:

- [uv Dependency Documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv Conflict Resolution](https://docs.astral.sh/uv/reference/settings/#conflicts)
