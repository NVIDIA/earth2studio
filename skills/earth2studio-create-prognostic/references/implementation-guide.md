# Implementation Guide — Prognostic Model Wrapper

> **Table of Contents**
>
> 1. [Dependencies](#dependencies)
> 2. [Skeleton Template](#skeleton-template)
> 3. [Coordinate System](#coordinate-system)
> 4. [Forward Pass](#forward-pass)
> 5. [Model Loading](#model-loading)
> 6. [Registration & Documentation](#registration--documentation)

---

## Dependencies

### pyproject.toml group format

```toml
model-name = ["package1>=version", "package2"]
```

- Group name: lowercase-hyphenated model name
- Add alphabetically in `[project.optional-dependencies]`
- Add to `all` aggregate (px models line)

### Optional dependency imports pattern

```python
try:
    import optional_package
except ImportError:
    OptionalDependencyFailure("model-name")
    optional_package = None
```

---

## Skeleton Template

Every `.py` file must start with the license header from `test/_license/header.txt`.

### Canonical method ordering

1. `__init__` — constructor
2. `input_coords` — input coordinate system
3. `output_coords` — output coordinate system (decorated `@batch_coords()`)
4. `load_default_package` — classmethod returning default Package
5. `load_model` — classmethod loading model from package
6. `to` — device management (optional, only for non-PyTorch state)
7. Private/support methods (e.g., `_prepare_input`, `_normalize`)
8. `__call__` — single-step forward (decorated `@batch_func()`)
9. `_default_generator` — batch-decorated generator (`@batch_func()`)
10. `create_iterator` — public time-integration entry point

### Triple inheritance

```python
class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
```

### Complete skeleton

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

# Optional dependency imports
try:
    import optional_package
except ImportError:
    OptionalDependencyFailure("model-name")
    optional_package = None

VARIABLES = [...]  # List of variable names from E2STUDIO_VOCAB


class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-line description.

    Extended description of the model, its source, and relevant details.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core model instance
    ...additional params...

    Note
    ----
    For more information see: <link to paper/repo>
    """

    # 1. Constructor
    def __init__(self, core_model, ...):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))
        # TODO: Initialize model

    # 2. Input coordinates
    def input_coords(self) -> CoordSystem:
        # TODO: Define input coordinates
        pass

    # 3. Output coordinates
    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        # TODO: Define output coordinates
        pass

    # 4. Default package location
    @classmethod
    def load_default_package(cls) -> Package:
        # TODO: Default checkpoint location
        pass

    # 5. Load model from package
    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> PrognosticModel:
        # TODO: Load model from package
        pass

    # 6. Device management (optional)
    def to(self, device: torch.device | str) -> PrognosticModel:
        # TODO: Device management (only if non-PyTorch state)
        pass

    # 7. Private/support methods go here

    # 8. Single step forward
    @batch_func()
    def __call__(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
        # TODO: Single step forward
        pass

    # 9. Batch-decorated generator
    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Single step forward
        pass

    # 10. Public iterator entry point
    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Setup, then yield from self._default_generator(x, coords)
        pass
```

---

## Coordinate System

### input_coords rules

- `batch` first with `np.empty(0)` — MUST be first
- `time` is `np.empty(0)` (dynamic)
- `lead_time` starts at `np.timedelta64(0, "h")` (or negative for history)
- `variable`: array of E2STUDIO_VOCAB names
- `lat`: 90 to -90 (north to south)
- `lon`: 0 to 360

Verify all model variables exist in `earth2studio/lexicon/base.py` (282 entries).
Standard pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.

### input_coords template

```python
def input_coords(self) -> CoordSystem:
    """Input coordinate system of the prognostic model.

    Returns
    -------
    CoordSystem
        Coordinate system dictionary
    """
    return OrderedDict(
        {
            "batch": np.empty(0),
            "time": np.empty(0),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, num_lat, endpoint=...),
            "lon": np.linspace(0, 360, num_lon, endpoint=False),
        }
    )
```

### output_coords template

```python
@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    """Output coordinate system.

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinates to validate and transform

    Returns
    -------
    CoordSystem
        Output coordinates with updated lead_time

    Raises
    ------
    ValueError
        If input coordinates are invalid
    """
    target_input_coords = self.input_coords()

    # Validate dimensions exist at correct indices
    handshake_dim(input_coords, "lead_time", 2)
    handshake_dim(input_coords, "variable", 3)
    handshake_dim(input_coords, "lat", 4)
    handshake_dim(input_coords, "lon", 5)

    # Validate coordinate values match
    handshake_coords(input_coords, target_input_coords, "variable")
    handshake_coords(input_coords, target_input_coords, "lat")
    handshake_coords(input_coords, target_input_coords, "lon")

    output_coords = input_coords.copy()
    output_coords["lead_time"] = input_coords["lead_time"] + np.array([self._time_step])
    return output_coords
```

---

## Forward Pass

### `__call__` template

```python
@batch_func()
def __call__(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
    """Run prognostic model 1 step.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    coords : CoordSystem
        Input coordinate system

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Output tensor and coordinates one time step ahead
    """
    target_input_coords = self.input_coords()
    handshake_coords(coords, target_input_coords, "variable")
    handshake_dim(coords, "variable", 3)

    # Move to device
    device = self.device_buffer.device
    x = x.to(device)

    # TODO: Reshape input tensor for the core model
    # TODO: Call core model
    # TODO: Reshape output tensor back to earth2studio format

    out_coords = self.output_coords(coords)
    return output, out_coords
```

Key notes:

- `@batch_func()` handles batch dimension
- Input shape: `(batch, time, lead_time, variable, lat, lon)`
- Reshape to model format → call model → reshape back

### `_default_generator` template

```python
@batch_func()
def _default_generator(
    self, x: torch.Tensor, coords: CoordSystem
) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
    """Batch-decorated generator for time integration.

    Parameters
    ----------
    x : torch.Tensor
        Initial condition tensor
    coords : CoordSystem
        Initial coordinate system

    Yields
    ------
    tuple[torch.Tensor, CoordSystem]
        Predicted state and coordinates at each time step
    """
    # MUST yield initial condition first (step 0)
    yield x, coords

    # Time integration loop (runs indefinitely)
    current_x = x
    current_coords = coords
    while True:
        # Apply front hook (for perturbation injection)
        current_x, current_coords = self.front_hook(current_x, current_coords)

        # Forward step
        current_x, current_coords = self.__call__(current_x, current_coords)

        # Apply rear hook (for post-processing)
        current_x, current_coords = self.rear_hook(current_x, current_coords)

        yield current_x, current_coords
```

### create_iterator template

```python
def create_iterator(
    self, x: torch.Tensor, coords: CoordSystem
) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
    """Create time-integration iterator.

    Parameters
    ----------
    x : torch.Tensor
        Initial condition tensor
    coords : CoordSystem
        Initial coordinate system

    Yields
    ------
    tuple[torch.Tensor, CoordSystem]
        Predicted state and coordinates at each time step
    """
    yield from self._default_generator(x, coords)
```

---

## Model Loading

### load_default_package template

```python
@classmethod
def load_default_package(cls) -> Package:
    """Default pre-trained model package on <source>.

    Returns
    -------
    Package
        Model package
    """
    return Package(
        "hf://org/repo@commit",  # Lock to specific commit
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )
```

Lock HuggingFace URLs to commit: `hf://org/repo@commit`.

### load_model template

```python
@classmethod
@check_optional_dependencies()
def load_model(cls, package: Package) -> PrognosticModel:
    """Load prognostic model from package.

    Parameters
    ----------
    package : Package
        Model package with checkpoint files

    Returns
    -------
    PrognosticModel
        Loaded model instance
    """
    checkpoint_path = package.resolve("model.pt")
    # NOTE: weights_only=False is required when loading full model objects (not just
    # state dicts). Only load checkpoints from trusted, verified sources (e.g.,
    # HuggingFace repos pinned to a commit hash via load_default_package).
    core_model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    core_model.eval()
    return cls(core_model)
```

Key patterns:

- Use `package.resolve("filename")`
- Load with `map_location="cpu"`
- Set `eval()` mode
- Use `@check_optional_dependencies()`

### .to() template (optional)

Only override if non-PyTorch state exists (ONNX, JAX). Otherwise omit.

```python
def to(self, device: torch.device | str) -> PrognosticModel:
    """Move model to device.

    Parameters
    ----------
    device : torch.device | str
        Target device

    Returns
    -------
    PrognosticModel
        Model on target device
    """
    super().to(device)
    # If using ONNX Runtime, destroy and recreate session on new device
    return self
```

---

## Registration & Documentation

### Register in `__init__.py`

Edit `earth2studio/models/px/__init__.py` — add import alphabetically.

### Add to docs

Edit `docs/modules/models_px.rst` — add class alphabetically to autosummary.

### Update install.md

Add to `docs/userguide/about/install.md` Prognostics section (alphabetically).

#### Basic install template

````markdown
:::::{tab-item} ModelName
Notes: <Any special installation notes>

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
````

#### Template with manual pip packages

````markdown
:::::{tab-item} ModelName
Notes: For pip users, [Package](https://github.com/...) needs manual install.

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
````

### Update CHANGELOG.md

Under `### Added`:

```markdown
- Added <ModelName> prognostic model (`<ClassName>`) with <brief description>
```

If new dependencies, under `### Dependencies`:

```markdown
- Added `<model-name>` optional dependency group for <ModelName> model
```

### Docstring requirements

- NumPy-style docstrings on all public methods
- Complete `Parameters`/`Returns`/`Raises` sections
- Type hints on all public methods
- Reference URL in class `Note` section
