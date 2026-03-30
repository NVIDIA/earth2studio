(developer_dependency)=

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
- `xarray`: Multi-dimensional (N-D) labeled arrays and datasets
- `zarr`: Array storage format
- `fsspec`: Filesystem interfaces

With these base dependencies, most IO methods, data sources, perturbation methods, and
utilities should be functional. However, specific features can require additional
optional dependencies.

:::{admonition} Adding Packages
:class: note
uv commands should always be used to update package lists inside the `pyproject.toml`.
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
handling for the user to communicate if an additional dependency group needs to be
installed.
A good sanity check is to run `make docs` from the repository root to build the API
documentation and verify that optional imports are handled correctly. For targets and
setup, refer to {ref}`building_documentation`.

It is important to handle the absence of optional dependencies in a standardized way for
the user.
Earth2Studio has some utilities to provide informative errors to users that should be
used when some optional dependency group is needed to be installed:

1. Use a `try` or `except` around the optional import for `ImportError`. Set respective
    package objects to `None` if not installed. The file should **not** error on import;
    optional imports should be evaluated lazily when needed.
2. In the catch of the import try catch, instantiate a `OptionalDependencyFailure`. The
    only parameter needed is the name of the dependency group that needs to be
    installed.
3. Place the `@check_optional_dependencies` decorator on the class and/or function that
    requires the optional import.
    This will check to make sure that the required imports are installed when the user
    attempts to use the respective feature.

For example in the FourCastNet model:

```python
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
try:
    from physicsnemo.models.afno import AFNO
except ImportError:
    OptionalDependencyFailure("fcn")
    AFNO = None


# Use the decorator on classes
@check_optional_dependencies()
class FCN(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    ...
    # Use the decorator on functions
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:

    ...
```

In this example, you can import the FCN module without errors. Only when you use FCN
without the required extra packages installed will the following error appear:

```text
uv run python
Python 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from earth2studio.models.px import FCN
>>> FCN.load_model(FCN.load_default_package())
┌────────────────────────────────────────────────────────────────────┐
│ Earth2Studio Extra Dependency Error                                │
│ This error typically indicates an extra dependency group is        │
│ needed.                                                            │
│ Don't panic, this is usually an easy fix.                          │
├────────────────────────────────────────────────────────────────────┤
│ This feature ('earth2studio.models.px.fcn.load_model') is marked   │
│ needing optional dependency group 'fcn'.                           │
│                                                                    │
│ uv install with: `uv add earth2studio --extra fcn`                 │
│                                                                    │
│ For more information (such as pip install instructions), visit the │
│ install documentation:                                             │
│ https://nvidia.github.io/earth2studio/userguide/about/install.html │
├────────────────────────────────────────────────────────────────────┤
│ ╭────────────── Traceback (most recent call last) ───────────────╮ │
│ │ .../earth2studio/earth2studio/models/px/… │ │
│ │ in <module>                                                    │ │
│ │                                                                │ │
│ │    34 from earth2studio.utils.type import CoordSystem          │ │
│ │    35                                                          │ │
│ │    36 try:                                                     │ │
│ │ ❱  37 │   from physicsnemo.models.afno import AFNO             │ │
│ │    38 except ImportError:                                      │ │
│ │    39 │   OptionalDependencyFailure("fcn")                     │ │
│ │    40 │   AFNO = None                                          │ │
│ ╰────────────────────────────────────────────────────────────────╯ │
│ ImportError: No module named 'physicsnemo'                         │
└────────────────────────────────────────────────────────────────────┘
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../earth2studio/earth2studio/utils/imports.py", line 172, in _wrapper
    _check_deps(f"{obj.__module__}.{obj.__name__}")
  File ".../earth2studio/earth2studio/utils/imports.py", line 149, in _check_deps
    raise OptionalDependencyError(
earth2studio.utils.imports.OptionalDependencyError: Optional dependency import error
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

All potential conflicts should be documented in the `pyproject.toml` file to ensure
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
For more information on dependency management with uv, refer to the following:

- [uv Dependency Documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv Conflict Resolution](https://docs.astral.sh/uv/reference/settings/#conflicts)
