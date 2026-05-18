---
name: create-prognostic-wrapper
description: Create a new Earth2Studio prognostic model (px) wrapper from a reference inference script or repository
argument-hint: URL or local path to reference inference script/repo (optional — will be asked if not provided)
---

# Create Prognostic Model Wrapper

Create a new Earth2Studio prognostic model wrapper by following every step below in order.
Each confirmation gate marked by starting with:

```markdown
### **[CONFIRM — <Title>]**
```

requires explicit user approval before proceeding.

---

## Step 0 — Obtain Reference Script / Repository

If `$ARGUMENTS` is provided, use it as the reference inference script or repository.

- If it is a URL, use WebFetch to retrieve the content.
- If it is a local file path, read it directly.

If `$ARGUMENTS` is empty or not provided, ask the user:

> Please provide a reference inference script or
> repository URL/path that demonstrates how this model
> runs inference. This will be used to understand the
> model architecture, dependencies, input/output shapes,
> and variable mapping.

Store the reference code content for use in subsequent steps.

---

## Step 1 — Examine Reference & Propose Dependencies

### 1a. Analyze the reference code

Examine the reference inference script/repo to identify:

- **Python packages** required (e.g., `torch`, `onnxruntime`, `einops`, `timm`, custom packages)
- **Model architecture** (PyTorch, ONNX, JAX, etc.)
- **Input/output tensor shapes** and variable names
- **Time step** of the model (e.g., 6h, 24h)
- **Spatial resolution** (lat/lon grid dimensions and spacing)
- **Checkpoint format** (`.pt`, `.onnx`, `.safetensors`, etc.)

### 1b. Propose pyproject.toml dependency group

Propose a new optional dependency group for `pyproject.toml`. Follow the existing pattern:

```toml
# In [project.optional-dependencies] section of pyproject.toml
model-name = [
    "package1>=version",
    "package2",
]
```

The group name should be lowercase-hyphenated (e.g., `pangu`, `aurora`, `stormcast`).

Look at the existing groups in `pyproject.toml`
(lines ~59-257) for reference on naming and version
pinning conventions.

**Also propose adding the new group to the `all` aggregate** in the appropriate line (px models line).

### **[CONFIRM — Dependencies]**

Present to the user:

1. The proposed dependency group name
2. The list of packages with versions
3. Ask if the packages and group name look correct

---

## Step 2 — Add Dependencies to pyproject.toml

After confirmation, edit `pyproject.toml`:

1. Add the new optional dependency group in alphabetical order
   among the per-model extras
2. Add the group to the `all` aggregate (in the px models line)

---

## Step 3 — Create Skeleton Class File

### 3a. Determine class name and file name

Based on the model name from the reference, propose:

- **Class name**: PascalCase (e.g., `Pangu24`, `Aurora`, `StormCast`)
- **File name**: lowercase with underscores
  (e.g., `pangu.py`, `aurora.py`, `stormcast.py`)
- **File path**: `earth2studio/models/px/<filename>.py`

### 3b. Write skeleton with pseudocode

Create the file with the full structure but pseudocode
implementations. Every `.py` file in `earth2studio/`
**must** start with the license header found in:

```text
test/_license/header.txt
```

Read this file and copy the header verbatim to the top of the new file.

The skeleton must use **triple inheritance** and follow this exact structure.

**Method ordering** — methods in the class must appear in
this canonical order:

1. `__init__` — constructor
2. `input_coords` — input coordinate system
3. `output_coords` — output coordinate system (decorated
   `@batch_coords()`)
4. `load_default_package` — classmethod returning default
   `Package`
5. `load_model` — classmethod loading model from package
6. `to` — device management (optional for pure
   `torch.nn.Module` wrappers where `super().to()` is
   sufficient; only needed for ONNX, JAX, or other
   non-PyTorch state)
7. Any private/support methods (e.g., helper functions for
   data conversion, JAX setup, normalization, etc.)
8. `__call__` — single-step forward (decorated
   `@batch_func()`)
9. `_default_generator` — batch-decorated generator used
   by `create_iterator` (decorated `@batch_func()`)
10. `create_iterator` — public time-integration entry point
    that delegates to `_default_generator`

```python
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

# Optional dependency imports (try/except pattern)
try:
    import optional_package
except ImportError:
    OptionalDependencyFailure("model-name")
    optional_package = None

VARIABLES = [...]  # List of variable names from E2STUDIO_VOCAB

class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-line description.

    Extended description of the model, its source,
    and any relevant details.

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
        # TODO: Initialize model
        self.register_buffer(
            "device_buffer", torch.empty(0)
        )
        pass

    # 2. Input coordinates
    def input_coords(self) -> CoordSystem:
        # TODO: Define input coordinates
        pass

    # 3. Output coordinates
    @batch_coords()
    def output_coords(
        self, input_coords: CoordSystem,
    ) -> CoordSystem:
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
    def load_model(
        cls, package: Package,
    ) -> PrognosticModel:
        # TODO: Load model from package
        pass

    # 6. Device management (optional — see note below)
    def to(
        self, device: torch.device | str,
    ) -> PrognosticModel:
        # TODO: Device management
        pass

    # 7. Private/support methods go here
    # e.g., _load_checkpoint(), _prepare_input(), etc.

    # 8. Single step forward
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # TODO: Single step forward
        pass

    # 9. Batch-decorated generator
    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Yield initial condition, then loop
        pass

    # 10. Public iterator entry point
    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Setup, then yield from
        #       self._default_generator(x, coords)
        pass
```

### **[CONFIRM — Skeleton]**

Present to the user:

1. The proposed class name
2. The proposed file name and path
3. Ask if these are acceptable

---

## Step 4 — Implement Coordinate System

### 4a. Map variables to E2STUDIO_VOCAB

Read `earth2studio/lexicon/base.py` and verify every
variable the model uses exists in `E2STUDIO_VOCAB`.
The vocab contains 282 entries including:

| Category | Examples |
|---|---|
| Surface wind | `u10m`, `v10m`, `ws10m`, `u100m`, `v100m` |
| Surface temp | `t2m`, `d2m`, `sst`, `skt` |
| Humidity | `r2m`, `q2m`, `tcwv` |
| Pressure | `sp`, `msl` |
| Precipitation | `tp`, `lsp`, `cp`, `tp06` |
| Pressure-level | `u50`-`u1000`, `v50`-`v1000`, `z50`-`z1000` |
| Cloud/radiation | `tcc`, `rlut`, `rsut` |

Pressure levels available: 50, 100, 150, 200, 250,
300, 400, 500, 600, 700, 850, 925, 1000.

If a variable in the reference model does NOT exist
in `E2STUDIO_VOCAB`, flag it to the user and discuss
whether to:

- Map it to an existing vocab entry
- Propose adding a new vocab entry (separate step)

### 4b. Implement input_coords

```python
def input_coords(self) -> CoordSystem:
    """Input coordinate system of the prognostic model.

    Returns
    -------
    CoordSystem
        Coordinate system dictionary
    """
    return OrderedDict({
        "batch": np.empty(0),          # MUST be first, MUST be np.empty(0)
        "time": np.empty(0),           # Dynamic time dimension
        "lead_time": np.array([np.timedelta64(0, "h")]),  # Initial lead time (0h)
        "variable": np.array(VARIABLES),
        "lat": np.linspace(90, -90, num_lat, endpoint=...),  # From reference
        "lon": np.linspace(0, 360, num_lon, endpoint=False),  # From reference
    })
```

**Rules:**

- `batch` is always first with `np.empty(0)`
- `time` is `np.empty(0)` (dynamic)
- `lead_time` starts at `np.timedelta64(0, "h")`
  unless the model requires history (multiple past
  time steps)
- If the model needs history, `lead_time` should
  contain negative timedeltas (e.g., `[-6h, 0h]`)
- `lat` typically goes from 90 to -90 (north to south)
- `lon` typically goes from 0 to 360

### 4c. Implement output_coords

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

### **[CONFIRM — Coordinates]**

Present to the user:

1. The variable list and any mapping issues with
   `E2STUDIO_VOCAB`
2. The spatial dimensions (lat/lon grid size and spacing)
3. Whether the model needs history (multiple lead times
   in input)
4. The model time step (e.g., 6h, 24h)

---

## Step 5 — Implement Forward Pass

### 5a. Implement `__call__`

```python
@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
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

    # Run forward pass
    # TODO: Reshape input tensor for the core model
    # TODO: Call core model
    # TODO: Reshape output tensor back to earth2studio format

    out_coords = self.output_coords(coords)
    return output, out_coords
```

Key implementation notes:

- The `@batch_func()` decorator handles the batch
  dimension — inside `__call__`, `x` has shape
  `(batch, time, lead_time, variable, lat, lon)`
- Reshape from earth2studio's
  `(batch, time, lead_time, variable, lat, lon)` to
  whatever the core model expects
- Reshape back to
  `(batch, time, lead_time, variable, lat, lon)` after
  forward pass
- All tensor operations should happen on GPU when
  possible

### 5b. Implement create_iterator

```python
@batch_func()
def create_iterator(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
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
    # MUST yield initial condition first (step 0)
    yield x, coords

    # Time integration loop (runs indefinitely)
    current_x = x
    current_coords = coords
    while True:
        # Apply front hook (for perturbation injection, etc.)
        current_x, current_coords = self.front_hook(current_x, current_coords)

        # Forward step
        current_x, current_coords = self.__call__(current_x, current_coords)

        # Apply rear hook (for post-processing, etc.)
        current_x, current_coords = self.rear_hook(current_x, current_coords)

        yield current_x, current_coords
```

### **[CONFIRM — Forward Pass]**

Show the user the pseudocode for `__call__`
(especially the reshape logic) and `create_iterator`.
Ask:

1. Does the tensor reshaping logic look correct for
   this model?
2. Are there any special considerations (e.g., multiple
   ONNX models, interleaved time steps)?

---

## Step 6 — Implement Model Loading

### 6a. Implement load_default_package

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
        "hf://org/repo",  # or ngc://, s3://, local path
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )
```

### 6b. Implement load_model

```python
@classmethod
@check_optional_dependencies()
def load_model(
    cls,
    package: Package,
) -> PrognosticModel:
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
    # Resolve checkpoint files
    checkpoint_path = package.resolve("model.pt")

    # Load model
    core_model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    core_model.eval()

    return cls(core_model)
```

**Key patterns:**

- Use `package.resolve("filename")` to get cached
  file paths
- Load with `map_location="cpu"` then let user call
  `.to(device)`
- Set model to `eval()` mode
- Do NOT over-populate `load_model()` API — only
  expose essential parameters
- Use `@check_optional_dependencies()` if the model
  has optional deps

### 6c. Implement .to()

> **Note:** When the wrapper inherits from `torch.nn.Module`,
> `super().to(device)` already handles moving all registered
> parameters, buffers, and sub-modules. A custom `to()`
> override is only needed when there is non-PyTorch state to
> manage (e.g., ONNX Runtime sessions that must be destroyed
> and recreated on a new device, or JAX device placement).
> If `super().to(device)` is sufficient, you can omit the
> override entirely.

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
    # If using PyTorch, super().to(device) handles it
    return self
```

### **[CONFIRM — Model Loading]**

Present to the user:

1. The checkpoint URL/path for `load_default_package`
2. The checkpoint file names and loading logic
3. Whether there are multiple checkpoint files
4. The `.to()` implementation (especially if ONNX or
   non-PyTorch backend)

---

## Step 7 — Register the Model

### 7a. Add to `__init__.py`

Edit `earth2studio/models/px/__init__.py`:

- Add import in alphabetical order:
  `from earth2studio.models.px.<filename> import <ClassName>`

### 7b. Verify pyproject.toml

Confirm the dependency group was added in Step 2 and is included in the `all` aggregate.

---

## Step 8 — Verify Style, Documentation, Format & Lint

Before testing, verify the wrapper passes all code quality checks.

### 8a. Run formatting

```bash
make format
```

This runs `black` on the codebase. Fix any formatting issues in the new wrapper file and test file.

### 8b. Run linting

```bash
make lint
```

This runs `ruff` and `mypy`. Common issues to watch
for:

- Missing type annotations on public functions
- Unused imports
- Import ordering issues
- Type errors from incorrect return types or missing
  `CoordSystem` annotations

Fix all errors before proceeding.

### 8c. Check license headers

```bash
make license
```

Verify that both the wrapper file
(`earth2studio/models/px/<filename>.py`) and the test
file (`test/models/px/test_<filename>.py`) have the
correct SPDX Apache-2.0 license header.

### 8d. Verify documentation

Check that:

- The class docstring follows NumPy-style formatting
  with `Parameters`, `Note`, etc.
- All public methods (`__call__`, `create_iterator`,
  `input_coords`, `output_coords`, `to`,
  `load_default_package`, `load_model`) have complete
  docstrings with `Parameters`, `Returns`, `Raises`
  sections as applicable
- Type hints are present on all public method
  signatures

If any checks fail, fix the issues and re-run until all pass cleanly.

---

## Step 9 — Test the Wrapper

Use the **`create-prognostic-tests`** skill to test the wrapper:

1. Smoke test with random data
2. Data fetch validation
3. Pytest unit tests (mock and package tests)
4. Side-by-side comparison scripts

Invoke the skill:

```text
/skill create-prognostic-tests <ModelName>
```

---

## Reminders

- **DO NOT** make a general base class with intent to reuse the wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose essential parameters
- **DO** add the model to `docs/modules/models.rst`
  in the `earth2studio.models.px` section
  (alphabetical order)
- **DO** use `loguru.logger` for logging, never `print()`, inside `earth2studio/`
- **DO** ensure all public functions have full type hints
- **DO** run formatting (`make format`) and linting (`make lint`) before finalizing
