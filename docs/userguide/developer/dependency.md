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
handling for the user to communicate that additional dependency group needs to be
installed.
A good sanity check if things have properly been caught is by building the API docs
with `make docs`.
The pattern to properly handle the use of optional dependencies is:

1. Use a try / except in the import for ImportErrors. Set respective package objects to
    None if not installed.
2. Place the `@check_extra_imports` decorator on the class and/or function that requires
    the optional import.
    This will check to make sure that the required imports are installed when the user
    attempts to use the respective feature.

For example in the FourCastNet model:

```python
try:
    from physicsnemo.models.afno import AFNO
except ImportError:
    AFNO = None

from earth2studio.utils import check_extra_imports


# Use the decorator on classes
@check_extra_imports("fcn", [AFNO])
class FCN(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    ...
    # Use the decorator on functions
    @classmethod
    @check_extra_imports("fcn", [AFNO])
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:

    ...
```

When the user attempts to use FCN without the required extra packages, the following
error will occur:

```bash
uv run python
Python 3.12.9 (main, Mar 17 2025, 21:01:58) [Clang 20.1.0 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from earth2studio.models.px import FCN
>>> FCN.load_model(FCN.load_default_package())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../earth2studio/utils/imports.py", line 78, in _wrapper
    _check_dependencies(f"{obj.__module__}.{obj.__name__}")
  File ".../earth2studio/utils/imports.py", line 61, in _check_dependencies
    raise ExtraDependencyError(extra_name, obj_name)
earth2studio.utils.imports.ExtraDependencyError: Extra dependency group 'fcn' is
    required for earth2studio.models.px.fcn.load_model.
Install with: uv pip install earth2studio --extra fcn
or refer to the install documentation.
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
# Example dependency group in pyproject.toml
[dependency-groups]
dev = [
    "black==24.1.0",
    "coverage>=6.5.0",
    "interrogate>=1.5.0",
    "hatch>=1.14.0",
    "mypy"
]
```

These dependencies are not included in the distributed package wheel.
For more information on dependency management with uv, see:

- [uv Dependency Documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv Conflict Resolution](https://docs.astral.sh/uv/reference/settings/#conflicts)
