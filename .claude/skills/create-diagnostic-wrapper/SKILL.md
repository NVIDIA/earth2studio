---
name: create-diagnostic-wrapper
description: Create a new Earth2Studio diagnostic model (dx) wrapper from a reference inference script or repository. Handles both NN-based models (with AutoModelMixin and Package loading) and physics-based calculators (pure torch computations).
argument-hint: URL or local path to reference inference script/repo (optional — will be asked if not provided)
---

# Create Diagnostic Model Wrapper

Create a new Earth2Studio diagnostic model wrapper by following every step below in order.
Each confirmation gate marked by starting with:

```markdown
### **[CONFIRM — <Title>]**
```

requires explicit user approval before proceeding.

> **Environment note**: Use `uv run python` for all Python
> commands. The project uses a `uv`-managed virtual environment
> — never install packages globally or use bare `python`.

---

## Step 0 — Obtain Reference & Detect Model Type

### 0a. Obtain reference script / repository

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

### 0b. Detect model type — NN-based vs physics-based

After obtaining the reference, classify the model into one
of two categories:

**NN-based indicators** (checkpoint-backed neural network):

- Checkpoint loading (`torch.load`, `.from_pretrained`, ONNX runtime)
- Trained weights, normalization parameters (mean/std tensors)
- Model architecture classes (e.g., `torch.nn.Module` subclass with learned layers)
- Imports like `onnxruntime`, `timm`, `einops`, `physicsnemo`
- `.pt`, `.onnx`, `.safetensors`, `.mdlus` checkpoint files

**Physics-based indicators** (analytical / derived calculator):

- Analytical formulas (e.g., `sqrt(u² + v²)`, Clausius–Clapeyron)
- No trained weights or checkpoint files
- Parameterized by physical constants, pressure levels, or thresholds
- Pure `torch` math operations
- No model architecture classes

Present the detection result to the user:

> **Model type detected: [NN-based / Physics-based]**
>
> Evidence:
>
> - [list key indicators found]
>
> This determines branching in Steps 2, 3, and 6.
> NN-based models require dependency management and
> checkpoint loading. Physics-based models skip those
> steps.

Store the model type flag (`nn` or `physics`) for
conditional branching in Steps 2, 3, and 6.

### **[CONFIRM — Model Type]**

Ask the user to confirm the detected model type before
proceeding.

---

## Step 1 — Examine Reference & Propose Dependencies

### 1a. Analyze the reference code

Examine the reference inference script/repo to identify:

- **Python packages** required (e.g., `torch`, `onnxruntime`, `einops`, `timm`, custom packages)
- **Model architecture** (PyTorch, ONNX, JAX, etc.) — NN-based only
- **Input/output tensor shapes** and variable names
- **Spatial resolution** (lat/lon grid dimensions and spacing, or flexible for physics-based)
- **For NN-based**: Checkpoint format (`.pt`, `.onnx`, `.safetensors`, etc.)
- **For physics-based**: Input variable patterns (e.g., `[u{level}, v{level}]`)
  and output variable formulas (e.g., `ws = sqrt(u² + v²)`)

### 1b. Propose pyproject.toml dependency group

**NN-based models only:**

Propose a new optional dependency group for `pyproject.toml`. Follow the existing pattern:

```toml
# In [project.optional-dependencies] section of pyproject.toml
model-name = [
    "package1>=version",
    "package2",
]
```

The group name should be lowercase-hyphenated (e.g., `precip-afno`, `windgust-afno`, `climatenet`).

Look at the existing groups in `pyproject.toml`
(lines ~59-257) for reference on naming and version
pinning conventions.

**Also propose adding the new group to the `all` aggregate** in the appropriate line (dx models line).

**Physics-based models:**

State: "No external dependencies needed — physics-based model uses
only `torch` and `numpy` (already core dependencies)."

### **[CONFIRM — Dependencies]**

Present to the user:

1. The proposed dependency group name (NN-based) or
   "No dependencies" (physics-based)
2. The list of packages with versions (NN-based only)
3. Ask if the packages and group name look correct

---

## Step 2 — Add Dependencies to pyproject.toml

**NN-based models:**

After confirmation, edit `pyproject.toml`:

1. Add the new optional dependency group in alphabetical order
   among the per-model extras
2. Add the group to the `all` aggregate (in the dx models line)

**Physics-based models:**

Skip — no external dependencies for physics-based model.
State explicitly: "Step 2 skipped — physics-based model
has no external dependencies."

---

## Step 3 — Create Skeleton Class File

### 3a. Determine class name and file name

Based on the model name from the reference, propose:

- **Class name**: PascalCase (e.g., `PrecipitationAFNO`, `DerivedWS`, `WindgustAFNO`)
- **File name**: lowercase with underscores
  (e.g., `precipitation_afno.py`, `derived.py`, `wind_gust.py`)
- **File path**: `earth2studio/models/dx/<filename>.py`

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

Use one of the two skeleton templates depending on the
model type detected in Step 0.

#### NN-based skeleton (dual inheritance: `torch.nn.Module, AutoModelMixin`)

```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
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

VARIABLES = [...]  # List of input variable names from E2STUDIO_VOCAB

@check_optional_dependencies()
class ModelName(torch.nn.Module, AutoModelMixin):
    """One-line description.

    Extended description of the model, its source,
    and any relevant details.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core neural network model
    ...additional params...

    Note
    ----
    For more information see: <link to paper/repo>
    """

    # 1. Constructor
    def __init__(self, core_model: torch.nn.Module, ...):
        super().__init__()
        self.core_model = core_model
        self.register_buffer(
            "device_buffer", torch.empty(0)
        )
        # TODO: Register normalization buffers (center, scale, etc.)

    # 2. Input coordinates
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # TODO: Define input coordinates
        pass

    # 3. Output coordinates
    @batch_coords()
    def output_coords(
        self, input_coords: CoordSystem,
    ) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform

        Returns
        -------
        CoordSystem
            Output coordinate system

        Raises
        ------
        ValueError
            If input coordinates are invalid
        """
        # TODO: Validate and transform coordinates
        pass

    # 4. Forward pass
    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates
        """
        # TODO: Validate, forward, return
        pass

    # 5. Private forward computation (optional)
    def _forward(
        self, x: torch.Tensor, coords: CoordSystem,
    ) -> torch.Tensor:
        """Internal forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on device
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # TODO: normalize -> core_model -> denormalize
        pass

    # 6. Device management
    def to(
        self, device: torch.device | str,
    ) -> DiagnosticModel:
        """Move model to device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        DiagnosticModel
            Model on target device
        """
        # TODO: Device management
        pass

    # 7. Default package location
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package
        """
        # TODO: Default checkpoint location
        pass

    # 8. Load model from package
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package,
    ) -> DiagnosticModel:
        """Load diagnostic model from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files

        Returns
        -------
        DiagnosticModel
            Loaded model instance
        """
        # TODO: Load model from package
        pass
```

#### Physics-based skeleton (`torch.nn.Module` only)

```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class ModelName(torch.nn.Module):
    """One-line description.

    Extended description of the calculation, formula,
    and any physical basis.

    Parameters
    ----------
    levels : list[int | str], optional
        Pressure / height levels to compute for
    ...additional params...

    Note
    ----
    For more information see: <link to reference>
    """

    # 1. Constructor
    def __init__(
        self, levels: list[int | str] = [100],
    ) -> None:
        super().__init__()
        self.levels = levels
        self.in_variables = [...]   # Built from levels
        self.out_variables = [...]  # Built from levels

    # 2. Input coordinates
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # TODO: Define input coordinates
        pass

    # 3. Output coordinates
    @batch_coords()
    def output_coords(
        self, input_coords: CoordSystem,
    ) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform

        Returns
        -------
        CoordSystem
            Output coordinate system

        Raises
        ------
        ValueError
            If input coordinates are invalid
        """
        # TODO: Validate and transform coordinates
        pass

    # 4. Forward pass
    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates
        """
        # TODO: Physics computation
        pass

    # 5. Device management
    def to(
        self, device: torch.device | str,
    ) -> DiagnosticModel:
        """Move model to device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        DiagnosticModel
            Model on target device
        """
        super().to(device)
        return self
```

### Canonical method ordering for diagnostic models

Methods in the class **must** appear in this order:

1. `__init__` — constructor
2. `input_coords` — input coordinate system
3. `output_coords` — output coordinate system (decorated `@batch_coords()`)
4. `__call__` — forward pass (decorated `@torch.inference_mode()`, `@batch_func()`)
5. `_forward` — private computation method (optional helper)
6. `to` — device management
7. `load_default_package` — classmethod returning default `Package` (NN-based only)
8. `load_model` — classmethod loading model from package (NN-based only)

### Key differences from prognostic models

Be aware of these critical differences from the
`create-prognostic-wrapper` skill:

- **No `PrognosticMixin`** — diagnostic models do NOT
  inherit from `PrognosticMixin` (no iterator hooks needed)
- **No `create_iterator`** or `_default_generator` —
  diagnostic models do NOT perform time integration
- **Typically no `lead_time`** — most diagnostic models do
  not have a `lead_time` dimension. However, some models
  that consume forecast output at specific lead times (e.g.,
  solar radiation, wind gust) may include `lead_time` in
  their coordinate system. Only add it if the model
  genuinely needs temporal context
- **Match `handshake_dim` indices to coordinate position** —
  use the actual position of each dimension in the input
  `CoordSystem` OrderedDict. For simple models with
  `(batch, variable, lat, lon)` the indices are `1, 2, 3`.
  For models with `(batch, time, variable, lat, lon)` use
  `2, 3, 4`. You may also use negative indices (`-3, -2, -1`)
  if the model must accept flexible leading dimensions.
  Check existing models in `earth2studio/models/dx/` for
  the predominant convention used in the codebase.

### **[CONFIRM — Skeleton]**

Present to the user:

1. The proposed class name
2. The proposed file name and path
3. The detected model type (NN-based or physics-based)
4. Ask if these are acceptable

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

**NN-based** (fixed grid):

```python
def input_coords(self) -> CoordSystem:
    """Input coordinate system of diagnostic model.

    Returns
    -------
    CoordSystem
        Coordinate system dictionary
    """
    return OrderedDict({
        "batch": np.empty(0),          # MUST be first, MUST be np.empty(0)
        "time": np.empty(0),           # Dynamic time dimension (optional)
        "variable": np.array(VARIABLES),
        "lat": np.linspace(90, -90, num_lat, endpoint=...),  # From reference
        "lon": np.linspace(0, 360, num_lon, endpoint=False),  # From reference
    })
```

**Physics-based** (flexible grid):

```python
def input_coords(self) -> CoordSystem:
    """Input coordinate system of diagnostic model.

    Returns
    -------
    CoordSystem
        Coordinate system dictionary
    """
    return OrderedDict({
        "batch": np.empty(0),          # MUST be first, MUST be np.empty(0)
        "variable": np.array(self.in_variables),
        "lat": np.empty(0),            # Flexible — accepts any grid
        "lon": np.empty(0),            # Flexible — accepts any grid
    })
```

**Rules:**

- `batch` is always first with `np.empty(0)`
- `time` is `np.empty(0)` (dynamic) — include only if
  the model needs time information (e.g., for solar
  zenith angle calculations)
- **Typically no `lead_time`** — most diagnostic models do
  not include `lead_time`. Add it only if the model needs
  temporal context (e.g., solar radiation models that need
  time-of-day, or models that consume forecast output at
  specific lead times)
- `lat` typically goes from 90 to -90 (north to south)
  for NN-based; use `np.empty(0)` for physics-based
  (flexible grid)
- `lon` typically goes from 0 to 360 for NN-based;
  use `np.empty(0)` for physics-based (flexible grid)

### 4c. Implement output_coords

```python
@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    """Output coordinate system of diagnostic model.

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinates to validate and transform

    Returns
    -------
    CoordSystem
        Output coordinates with updated variable list

    Raises
    ------
    ValueError
        If input coordinates are invalid
    """
    target_input_coords = self.input_coords()

    # Validate dimensions exist at correct indices
    # Use the positional index matching each dim's position in input_coords
    # For (batch, variable, lat, lon) → 1, 2, 3
    # For (batch, time, variable, lat, lon) → 2, 3, 4
    # Negative indices (-3, -2, -1) also work for flexible leading dims
    handshake_dim(input_coords, "variable", 1)
    handshake_dim(input_coords, "lat", 2)
    handshake_dim(input_coords, "lon", 3)

    # Validate coordinate values match
    handshake_coords(input_coords, target_input_coords, "variable")
    handshake_coords(input_coords, target_input_coords, "lat")
    handshake_coords(input_coords, target_input_coords, "lon")

    output_coords = input_coords.copy()
    output_coords["variable"] = np.array(OUTPUT_VARIABLES)
    return output_coords
```

**Key points:**

- Use `@batch_coords()` decorator
- Use `handshake_dim` with indices matching each
  dimension's position in the `CoordSystem` OrderedDict.
  Use positive indices (e.g., `1, 2, 3`) or negative
  indices (e.g., `-3, -2, -1`) — match the convention
  of the most similar existing model in the codebase.
- Validate coordinate values with `handshake_coords`
- Output typically changes the `variable` array (different
  output variables from input variables)
- Unlike prognostic models, there is typically no
  `lead_time` to increment

### **[CONFIRM — Coordinates]**

Present to the user:

1. The input and output variable lists and any mapping
   issues with `E2STUDIO_VOCAB`
2. The spatial dimensions (lat/lon grid size and spacing,
   or "flexible" for physics-based)
3. Whether `lead_time` is needed (and why — most dx
   models omit it)
4. Whether `time` is included (and why, if so)

---

## Step 5 — Implement Forward Pass

### 5a. Implement `__call__`

```python
@torch.inference_mode()
@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Forward pass of diagnostic model.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    coords : CoordSystem
        Input coordinate system

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Output tensor and coordinates
    """
    # Get output coordinates (this also validates input coords)
    output_coords = self.output_coords(coords)

    # Move to device
    device = self.device_buffer.device  # NN-based
    # or: device = next(self.parameters()).device
    x = x.to(device)

    # Run forward pass
    out = self._forward(x, coords)

    return out, output_coords
```

Key implementation notes:

- The `@batch_func()` decorator handles the batch
  dimension — inside `__call__`, `x` has shape
  `(batch, ..., variable, lat, lon)` where `...` may
  include `time` and/or `lead_time` depending on the
  upstream pipeline
- Coordinate validation is handled inside `output_coords()`
  — no need to repeat `handshake_dim`/`handshake_coords`
  in `__call__` (follow the pattern of existing dx models)
- All tensor operations should happen on GPU when
  possible
- **No `create_iterator`**, no hooks, no yielding —
  diagnostic models perform a single forward pass only

### 5b. Implement `_forward`

**NN-based** (normalize → model → denormalize):

```python
def _forward(
    self, x: torch.Tensor, coords: CoordSystem,
) -> torch.Tensor:
    """Internal forward pass."""
    # Normalize input
    x = (x - self.center) / self.scale

    # Reshape for core model if needed
    # TODO: Reshape from (..., variable, lat, lon) to model format

    # Core model forward
    out = self.core_model(x)

    # Reshape output back
    # TODO: Reshape from model format to (..., out_variable, lat, lon)

    # Denormalize output (if needed)
    out = out * self.output_scale + self.output_center

    return out
```

**Physics-based** (direct torch math):

```python
@torch.inference_mode()
@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Forward pass of diagnostic."""
    output_coords = self.output_coords(coords)

    # Example: wind speed from u, v components
    u = x[..., ::2, :, :]    # Every other variable starting from 0
    v = x[..., 1::2, :, :]   # Every other variable starting from 1
    out = torch.sqrt(u**2 + v**2)

    return out, output_coords
```

For physics-based models, the computation typically
goes directly in `__call__` rather than a separate
`_forward` method.

### **[CONFIRM — Forward Pass]**

Show the user the pseudocode for `__call__`
(especially the reshape logic for NN-based, or the
formula for physics-based). Ask:

1. Does the computation logic look correct for this
   model?
2. Are there any special considerations (e.g., multiple
   ONNX models, clamping, special tensor layouts)?

---

## Step 6 — Implement Model Loading (NN-based only)

**Physics-based models:** Skip this step entirely.
State explicitly: "Step 6 skipped — physics-based models
have no checkpoints to load."

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
        "ngc://models/org/model@version",  # or hf://, s3://, local path
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
) -> DiagnosticModel:
    """Load diagnostic model from package.

    Parameters
    ----------
    package : Package
        Model package with checkpoint files

    Returns
    -------
    DiagnosticModel
        Loaded model instance
    """
    # Resolve checkpoint files
    checkpoint_path = package.resolve("model.pt")

    # Load model
    core_model = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    core_model.eval()
    core_model.requires_grad_(False)

    # Load any additional data files (normalization, masks, etc.)
    # center = torch.Tensor(np.load(package.resolve("center.npy")))
    # scale = torch.Tensor(np.load(package.resolve("scale.npy")))

    return cls(core_model)
```

**Key patterns:**

- Use `package.resolve("filename")` to get cached
  file paths
- Load with `map_location="cpu"` then let user call
  `.to(device)`
- Set model to `eval()` mode and `requires_grad_(False)`
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
def to(self, device: torch.device | str) -> DiagnosticModel:
    """Move model to device.

    Parameters
    ----------
    device : torch.device | str
        Target device

    Returns
    -------
    DiagnosticModel
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

Edit `earth2studio/models/dx/__init__.py`:

- Add import in alphabetical order:
  `from earth2studio.models.dx.<filename> import <ClassName>`
- Add `<ClassName>` to the `__all__` list in alphabetical order

### 7b. Verify pyproject.toml (NN-based only)

Confirm the dependency group was added in Step 2 and is included in the `all` aggregate.

For physics-based models, no pyproject.toml verification is needed.

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
(`earth2studio/models/dx/<filename>.py`) and the test
file (`test/models/dx/test_<filename>.py`) have the
correct SPDX Apache-2.0 license header.

### 8d. Verify documentation

Check that:

- The class docstring follows NumPy-style formatting
  with `Parameters`, `Note`, etc.
- All public methods (`__call__`, `input_coords`,
  `output_coords`, `to`, and for NN-based:
  `load_default_package`, `load_model`) have complete
  docstrings with `Parameters`, `Returns`, `Raises`
  sections as applicable
- Type hints are present on all public method
  signatures

If any checks fail, fix the issues and re-run until all pass cleanly.

---

## Step 9 — Test Forward Pass with Random Data

Write and run a quick smoke test script.

**Note:** Diagnostic models do NOT have `create_iterator` — only test `__call__`.

```python
import torch
import numpy as np
from earth2studio.models.dx import ModelName

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
print(f"Input variables: {input_coords['variable']}")
print(f"Output variables: {out_coords['variable']}")
```

Report results to the user. There is no `create_iterator`
to test — diagnostic models perform a single forward pass.

---

## Step 10 — Test Data Fetch with Random Source

Test that the model's coordinate system works with Earth2Studio's data pipeline:

```python
import numpy as np
from earth2studio.data import Random, fetch_data
from earth2studio.models.dx import ModelName

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

Create `test/models/dx/test_<filename>.py` following the existing test patterns.

### 11a. Test file structure

```python
# License header (same SPDX Apache-2.0 header as above)

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.dx import ModelName
from earth2studio.utils import handshake_dim


class PhooModelName(torch.nn.Module):
    """Dummy model that performs a simple deterministic operation."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[..., :NUM_OUTPUT_VARS, :, :]  # Simple slice for testing


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create a dummy model package for testing."""
    tmp_path = tmp_path_factory.mktemp("data")
    # Export dummy model to checkpoint format
    model = PhooModelName()
    torch.save(model, tmp_path / "model.pt")
    # Save any additional files (normalization, etc.)
    # np.save(tmp_path / "center.npy", np.zeros(NUM_VARS))
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
        # NN-based: use load_model
        model = ModelName.load_model(test_package)
        # Physics-based: construct directly instead
        # model = ModelName(levels=[1000])
        model = model.to(device)

        # Fetch input data
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Run forward
        out, out_coords = model(x, coords)

        # Validate output
        assert isinstance(out_coords, OrderedDict)
        handshake_dim(out_coords, "variable", 1)  # Adjust index to match model's coord ordering
        # Additional model-specific assertions
        # e.g., assert output variable count is correct

    @pytest.mark.parametrize(
        "coords",
        [
            OrderedDict({
                "batch": np.empty(0),
                "variable": np.array(["wrong_var"]),
                "lat": np.linspace(90, -90, 10),
                "lon": np.linspace(0, 360, 20),
            }),
        ],
    )
    def test_model_exceptions(self, test_package, coords):
        """Test model raises on invalid coordinates."""
        # NN-based: use load_model
        model = ModelName.load_model(test_package)
        # Physics-based: construct directly instead
        # model = ModelName(levels=[1000])
        x = torch.randn(
            1,
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        with pytest.raises((KeyError, ValueError)):
            model(x, coords)
```

**Note:** There is NO `test_model_iter` — diagnostic
models do not have `create_iterator`.

**Physics-based models — add `test_physics_correctness`:**

```python
def test_physics_correctness(self):
    """Verify physics formula against known values."""
    model = ModelName(levels=[1000])

    # Create known input values
    # e.g., for wind speed: u=3, v=4 → ws=5
    input_coords = model.input_coords()
    x = torch.zeros(1, len(input_coords["variable"]), 5, 5)
    x[:, 0, :, :] = 3.0  # u component
    x[:, 1, :, :] = 4.0  # v component

    out, _ = model(x, input_coords)

    expected = torch.full_like(out, 5.0)  # sqrt(3² + 4²) = 5
    assert torch.allclose(out, expected, rtol=1e-5)
```

**NN-based models — add `@pytest.mark.package` integration test:**

```python
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
    # Validate output shape and variables
    assert out_coords["variable"].shape[0] > 0
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
uv run python -m pytest test/models/dx/test_<filename>.py \
    -m "not package" -v
```

All mock tests must pass before proceeding. Fix any
failures and re-run until green.

### 11b-ii. Run the package integration test (NN-based only)

Once the mock tests pass, run the `@pytest.mark.package`
test which exercises `from_pretrained()` with real model
weights:

```bash
uv run python -m pytest test/models/dx/test_<filename>.py \
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

For physics-based models, there is no package test to run.
State: "No package test — physics-based model has no
checkpoints."

---

## Step 12 — Provide Side-by-Side Comparison Scripts

Present two scripts to the user:

### Reference script (without Earth2Studio)

Reconstruct a minimal inference script based on the original reference code:

**NN-based:**

```python
# Reference inference (no Earth2Studio)
import torch
# ... original model imports ...

# Load model
model = OriginalModel.from_pretrained("path/to/checkpoint")
model.eval().cuda()

# Prepare input
input_data = ...  # Load/prepare input data per original repo instructions

# Run inference
with torch.no_grad():
    output = model(input_data)

print(f"Output shape: {output.shape}")
```

**Physics-based:**

```python
# Reference calculation (no Earth2Studio)
import torch

# Hand-calculated / reference values
u = torch.tensor([3.0, 0.0, 1.0])
v = torch.tensor([4.0, 5.0, 0.0])

# Direct formula
ws = torch.sqrt(u**2 + v**2)

print(f"Wind speed: {ws}")  # Expected: [5.0, 5.0, 1.0]
```

### Earth2Studio equivalent

**NN-based:**

```python
# Earth2Studio inference
import torch
import numpy as np
from earth2studio.models.dx import ModelName
from earth2studio.data import Random, fetch_data

# Load model
model = ModelName.from_pretrained()
model = model.to("cuda")

# Prepare input via Earth2Studio data pipeline
time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)  # Replace with real data source
x, coords = fetch_data(ds, time, input_coords["variable"], device="cuda")

# Single forward pass (no iterator — this is a diagnostic model)
with torch.no_grad():
    output, out_coords = model(x, coords)

print(f"Output shape: {output.shape}")
print(f"Output variables: {out_coords['variable']}")
```

**Physics-based:**

```python
# Earth2Studio inference
import torch
import numpy as np
from earth2studio.models.dx import ModelName

# Construct model (no checkpoint needed)
model = ModelName(levels=[1000])

# Prepare input
input_coords = model.input_coords()
x = torch.zeros(1, len(input_coords["variable"]), 5, 5)
x[:, 0, :, :] = 3.0  # u component
x[:, 1, :, :] = 4.0  # v component

# Single forward pass
output, out_coords = model(x, input_coords)

print(f"Output shape: {output.shape}")
print(f"Output variables: {out_coords['variable']}")
print(f"Output values (should be 5.0): {output.mean():.1f}")
```

### **[CONFIRM — Comparison Scripts]**

Ask the user to compare the two scripts and verify the
Earth2Studio version is functionally equivalent to the
reference.

For physics-based models, also ask the user to verify
the output matches hand-calculated expected values.

---

## Reminders

- **DO NOT** make a general base class with intent to reuse the wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose essential parameters
- **DO NOT** add `lead_time` dimension unless the model genuinely needs temporal context
  (e.g., solar radiation, wind gust models that depend on forecast lead time)
- **DO NOT** add `create_iterator` or `_default_generator` — diagnostic models are single-pass
- **DO NOT** inherit from `PrognosticMixin` — diagnostic models do not need iterator hooks
- **DO** use `handshake_dim` indices matching each dimension's position in the
  `CoordSystem` OrderedDict — check existing dx models for the predominant convention
- **DO** add the model to `docs/modules/models.rst`
  in the `earth2studio.models.dx` section
  (alphabetical order)
- **DO** use `loguru.logger` for logging, never `print()`, inside `earth2studio/`
- **DO** ensure all public functions have full type hints
- **DO** run formatting (`make format`) and linting (`make lint`) before finalizing
- **DO** use `@torch.inference_mode()` on `__call__` for inference-only models
- **DO** set `eval()` and `requires_grad_(False)` on loaded NN models
- **DO** use `@batch_func()` on `__call__` and `@batch_coords()` on `output_coords`
- **DO** validate coordinates with `handshake_coords()` and `handshake_dim()`
- **DO** move tensors to device before operations: `x.to(device)`
- **DO** use `uv run python` for all Python commands (never bare `python`)
