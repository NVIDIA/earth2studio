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
**must** start with this license header:

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
```

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

## Step 9 — Test Forward Pass with Random Data

Write and run a quick smoke test script:

```python
import torch
import numpy as np
from earth2studio.models.px import ModelName

# Load model (or construct with dummy weights for testing)
model = ModelName(...)  # Use dummy/test weights if real ones aren't available
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Get input coords
input_coords = model.input_coords()

# Create random input tensor
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape)

# Test __call__
output, out_coords = model(x, input_coords)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output lead_time: {out_coords['lead_time']}")

# Test create_iterator (3 steps)
iterator = model.create_iterator(x, input_coords)
for i, (step_x, step_coords) in enumerate(iterator):
    print(f"Step {i}: shape={step_x.shape}, lead_time={step_coords['lead_time']}")
    if i >= 3:
        break
```

Report results to the user.

---

## Step 10 — Test Data Fetch with Random Source

Test that the model's coordinate system works with Earth2Studio's data pipeline:

```python
import numpy as np
from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

model = ModelName(...)

# Create time array
time = np.array([np.datetime64("2024-01-01T00:00")])

# Fetch data using model's input coords
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"])

print(f"Fetched data shape: {x.shape}")
print(f"Variables: {input_coords['variable']}")
```

Report results to the user. This validates the coordinate system is compatible with the data pipeline.

---

## Step 11 — Write Pytest Unit Tests

Create `test/models/px/test_<filename>.py` following the existing test patterns.

### Test file structure

```python
# License header (same SPDX Apache-2.0 header as above)

import gc
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import ModelName
from earth2studio.utils import handshake_dim


class PhooModelName(torch.nn.Module):
    """Dummy model that performs a simple deterministic operation."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1  # Simple deterministic operation for testing


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create a dummy model package for testing."""
    tmp_path = tmp_path_factory.mktemp("data")
    # Export dummy model to checkpoint format
    model = PhooModelName()
    torch.save(model, tmp_path / "model.pt")
    return Package(str(tmp_path))


class TestModelNameMock:
    @pytest.mark.parametrize(
        "time",
        [
            np.array([
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_call(self, test_package, time, device):
        """Test single forward pass."""
        model = ModelName.load_model(test_package)
        model = model.to(device)

        # Fetch input data
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Run forward
        out, out_coords = model(x, coords)

        # Validate output
        assert out.shape == x.shape
        assert isinstance(out_coords, OrderedDict)
        handshake_dim(out_coords, "variable", 3)
        # Additional model-specific assertions

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_iter(self, test_package, ensemble, device):
        """Test iterator produces correct sequence."""
        model = ModelName.load_model(test_package)
        model = model.to(device)

        # Create input
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Add ensemble dim
        x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        # Test iterator
        iterator = model.create_iterator(x, coords)
        assert isinstance(iterator, Iterable)

        # Skip initial condition
        next(iterator)

        for i, (step_x, step_coords) in enumerate(iterator):
            assert step_x.shape[0] == ensemble
            if i >= 4:
                break

        del model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "coords",
        [
            OrderedDict({
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["wrong_var"]),
                "lat": np.linspace(90, -90, 10),
                "lon": np.linspace(0, 360, 20),
            }),
        ],
    )
    def test_model_exceptions(self, test_package, coords):
        """Test model raises on invalid coordinates."""
        model = ModelName.load_model(test_package)
        x = torch.randn(
            1, 1, 1,
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        with pytest.raises((KeyError, ValueError)):
            model(x, coords)


@pytest.mark.package
def test_model_package():
    """Integration test with real model weights."""
    model = ModelName.from_pretrained()
    input_coords = model.input_coords()
    time = np.array([np.datetime64("2024-01-01T00:00")])
    input_coords["time"] = time
    ds = Random(input_coords)
    x, coords = fetch_data(ds, time, input_coords["variable"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(model.device_buffer.device)
    out, out_coords = model(x, coords)
    assert out.shape == x.shape
```

Adapt the dummy model (`PhooModelName`) to match the
actual model's input/output interface so the wrapper's
reshaping logic is exercised.

---

## Step 11b — Run Tests

### 11b-i. Run mock tests (no package flag)

First, run the unit tests that use mocked / dummy models.
These do **not** require downloading real checkpoints and
should run quickly on any machine:

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "not package" -v
```

All mock tests must pass before proceeding. Fix any
failures and re-run until green.

### 11b-ii. Run the package integration test

Once the mock tests pass, run the `@pytest.mark.package`
test which exercises `from_pretrained()` with real model
weights:

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "package" -v
```

### **[CONFIRM — Package Test]**

Before executing the package test, warn the user:

> The package / integration test will:
>
> - **Download the model checkpoint** (may be several GB)
>   to the local cache
> - **Require GPU compute** for models that need CUDA
>   (the test will skip on CPU-only machines if
>   `torch.cuda.is_available()` is `False`)
> - Take significantly longer than the mock tests
>
> Do you want to proceed with the package test?

Only run the package test after the user confirms. Report
the results back to the user. If the package test fails,
debug and fix the wrapper or test, then re-run.

---

## Step 12 — Provide E2E Validation Scripts and Comparison Plots

Create three runnable scripts that validate the Earth2Studio
wrapper produces output equivalent to the model's native
inference pipeline. These scripts are intended to be
**attached to the PR** for reviewer verification and serve
as end-to-end integration tests.

### 12a. Earth2Studio reference script (`e2s_reference.py`)

A self-contained script that runs inference through
`run.deterministic` and writes output to a single file.

```python
"""<ModelName> inference using the Earth2Studio run.deterministic API.

Loads the model, fetches initial conditions from the
appropriate data source, and runs a multi-step rollout.
Output is written to a single NetCDF file for comparison
with the native-framework reference output.
"""

import earth2studio.run as run
from earth2studio.data import ModelDataSource  # e.g., SamudrACEData
from earth2studio.io import NetCDF4Backend
from earth2studio.models.px import ModelName

# Configuration — match exactly what the native reference uses
SCENARIO = "..."          # If applicable
N_STEPS = 40              # Total forward steps
TIME = ["2000-01-01T00:00:00"]
OUTPUT_FILE = "outputs/e2s_reference_output.nc"

print("Loading model via Earth2Studio...")
package = ModelName.load_default_package()
model = ModelName.load_model(package, ...)
data = ModelDataSource(...)

print(f"Running deterministic forecast ({N_STEPS} steps)...")
io = run.deterministic(
    time=TIME,
    nsteps=N_STEPS,
    prognostic=model,
    data=data,
    io=NetCDF4Backend(
        file_name=OUTPUT_FILE,
        backend_kwargs={"mode": "w"},
    ),
)

print(f"Results saved to {OUTPUT_FILE}")
```

**Key requirements:**

- Use the **same initial conditions, forcing data, and
  scenario** as the native reference script
- Use `run.deterministic` (not manual iteration) so the
  full E2S pipeline (data fetch, coord alignment, IO
  write) is exercised
- Output to a **NetCDF file** with all model variables

### 12b. Native-framework reference script (`native_reference.py`)

A self-contained script that runs inference using the
model's **original framework** (not Earth2Studio).

```python
"""<ModelName> inference using the native framework.

Downloads model artifacts, then runs inference using the
original API / config-driven pipeline.
"""

import os
# ... native framework imports ...

# Download / resolve model artifacts
# (e.g., snapshot_download from HuggingFace)
repo_dir = download_model(...)

# Run inference using the native API
native_inference(
    config="inference_config.yaml",
    n_steps=...,
    output_dir="outputs/native_reference_output",
)

print("Done.")
```

**Key requirements:**

- Use the **same checkpoint, IC files, and config** as
  the E2S script
- Write output to a **known directory structure** with
  predictable file names
- Pin random seeds if the model has any stochastic
  components (`torch.manual_seed(0)`,
  `np.random.seed(0)`)

### 12c. Comparison script (`compare.py`)

A script that loads both outputs, aligns them by time /
lead-time, and produces:

1. **Global-mean timeseries** — per variable, 3-panel
   (reference, E2S, difference)
2. **Spatial maps** — at selected lead times, 3-panel
   (reference, E2S, difference with RMSE/MAE stats)
3. **Summary statistics** — per-variable RMSE, MAE,
   max absolute difference, and `np.allclose` result

```python
"""Compare <ModelName> forecasts from native and E2S pipelines.

Plots selected variables as timeseries (global mean) and
spatial maps at selected lead times, plus the difference.

Usage:
    python compare.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from earth2studio.lexicon.<model_lexicon> import ModelLexicon

# ---- Variable definitions ----
# Map E2S variable names to native-framework names, grouped
# by component if the model has multiple components.
VARIABLES = {
    "var_key": {
        "e2s_name": "...",
        "native_name": "...",    # Native framework name
        "component": "...",      # If applicable (e.g., "atmosphere", "ocean")
        "label": "Description [units]",
        "cmap": "RdYlBu_r",
        "vmin": ..., "vmax": ...,
    },
    # ... more variables ...
}

# ---- Load datasets ----
ds_ref = xr.open_dataset("outputs/native_reference_output/predictions.nc")
ds_e2s = xr.open_dataset("outputs/e2s_reference_output.nc")

# ---- Helper functions ----

def get_ref_series(da):
    """Extract (n_steps, lat, lon) from native output."""
    return da.isel(sample=0).values

def get_e2s_series(da, component="atmosphere"):
    """Extract (n_steps, lat, lon) from E2S output.

    Skips the IC at lead_time=0 since the native output
    typically contains only predictions.
    """
    vals = da.isel(time=0).values[1:]  # skip IC
    # If component has different step cadence, sub-sample:
    # if component == "ocean":
    #     indices = np.arange(N_INNER - 1, len(vals), N_INNER)
    #     vals = vals[indices]
    return vals

# ---- Figure 1: Global-mean timeseries ----
# ... 3-panel plot per variable ...

# ---- Figure 2: Spatial maps at selected steps ----
# ... 3-panel (ref, E2S, diff) per step ...

# ---- Summary statistics ----
for var_key, var_info in VARIABLES.items():
    # Compute RMSE, MAE, max abs diff, allclose
    ...

ds_ref.close()
ds_e2s.close()
```

### Important considerations for the comparison script

**Time / lead-time alignment:**

The E2S output from `run.deterministic` includes the
initial condition at `lead_time=0` as step 0, followed
by predictions at `lead_time=dt, 2*dt, ...`. The native
framework may write predictions only (no IC) starting
from the first forecast step. The comparison script must
account for this offset — typically by skipping the E2S
IC entry (`values[1:]`).

**Component-specific step cadences:**

If the model has components that update at different
rates (e.g., atmosphere every 6h, ocean every 5 days),
the native framework may write each component's output
at its own cadence while E2S writes all variables at
every atmosphere step. The comparison script must
sub-sample the E2S output at the slower component's
update boundaries to align with the native output.

**Variable name mapping:**

Use the model's lexicon class to map between E2S and
native variable names. Verify the mapping is correct by
checking a few variables at the IC step (where both
outputs should be identical).

**Grid alignment:**

Check whether both outputs use the same latitude
ordering (north-to-south vs south-to-north). If they
differ, flip one before computing differences.

### Deliverables for the PR

Attach the following to the pull request:

| File | Purpose |
|------|---------|
| `e2s_reference.py` | E2S inference script |
| `native_reference.py` (or `<framework>_reference.py`) | Native-framework inference script |
| `compare.py` | Comparison and plotting script |
| `compare_timeseries.png` | Global-mean timeseries plot |
| `compare_<var>.png` | Per-variable spatial map plots |

These do **not** get committed to the repository — they
are PR attachments for reviewer validation only.

### **[CONFIRM — Comparison Scripts]**

Present the three scripts to the user and ask:

1. Do the E2S and native reference scripts use identical
   inputs (IC, forcing, scenario, number of steps)?
2. Is the time alignment logic in `compare.py` correct
   for this model's output structure?
3. Are the comparison plots showing near-zero differences
   at the IC step and acceptable divergence at later
   steps?

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
