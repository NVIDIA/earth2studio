---
name: create-assimilation-wrapper
description: >-
  Create a new Earth2Studio data assimilation model (da) wrapper from a
  reference inference script or repository. DA models ingest sparse
  observations (DataFrames) and/or gridded state arrays (DataArrays) and
  produce analysis output — they do NOT use @batch_func, @batch_coords,
  or PrognosticMixin.
argument-hint: URL or local path to reference inference script/repo (optional — will be asked if not provided)
---

# Create Assimilation Model Wrapper

Create a new Earth2Studio data assimilation model wrapper by following every step below in order.
Each confirmation gate marked by starting with:

```markdown
### **[CONFIRM — <Title>]**
```

requires explicit user approval before proceeding.

> **Environment note**: Use `uv run python` for all Python
> commands. The project uses a `uv`-managed virtual environment
> — never install packages globally or use bare `python`.

---

## Critical Differences from px/dx Models

DA models are **not** tensor-in / tensor-out like px and dx models. The entire I/O
contract is different. Review this table **before** writing any code:

| Aspect | px / dx | da |
| ------------------- | ------------------------------ | ---------------------------------------- |
| **Primary input** | `Tensor` + `CoordSystem` | `pd.DataFrame` or `xr.DataArray` |
| **Primary output** | `Tensor` + `CoordSystem` | `xr.DataArray` or `pd.DataFrame` |
| **Coord schemas** | `CoordSystem` only | `FrameSchema` + `CoordSystem` |
| **Coord returns** | Single dict | **Tuple** (even for single) |
| **Batch handling** | `@batch_func` / `@batch_coords` | **Neither** — N/A for DataFrame I/O |
| **PrognosticMixin** | Yes (px) / No (dx) | **No** |
| **Time integration** | `create_iterator` yields | `create_generator` with **send** |
| **Generator prime** | Yields initial condition | Yields `None` / state, `.send()` |
| **Generator cleanup** | N/A | Must handle `GeneratorExit` |
| **Init data** | No | Optional via `init_coords()` |
| **Time metadata** | Coordinate dimension | `obs.attrs["request_time"]` |
| **GPU data** | `Tensor` on device | **cupy** arrays, **cudf** DFs |
| **Input validation** | `handshake_dim`/`handshake_coords` | `validate_observation_fields()` |
| **Time filtering** | N/A | `filter_time_range()` |
| **Tensor conversion** | Already tensors | `dfseries_to_torch()` |
| **Device tracking** | `device_buffer` / `parameters()` | `device_buffer` + `@property` |

---

## Step 0 — Obtain Reference & Analyze Model

### 0a. Obtain reference script / repository

If `$ARGUMENTS` is provided, use it as the reference inference script or repository.

- If it is a URL, use WebFetch to retrieve the content.
- If it is a local file path, read it directly.

If `$ARGUMENTS` is empty or not provided, ask the user:

> Please provide a reference inference script or
> repository URL/path that demonstrates how this model
> runs inference. This will be used to understand the
> model architecture, dependencies, input/output shapes,
> observation schema, and variable mapping.

Store the reference code content for use in subsequent steps.

### 0b. Analyze reference model

After obtaining the reference, analyze it for:

- **Input types**: Does the model ingest `pd.DataFrame`
  (sparse observations), `xr.DataArray` (gridded fields),
  or both?
- **Output types**: Does it produce `xr.DataArray`
  (gridded analysis), `pd.DataFrame` (corrected
  observations), or both?
- **Stateful vs stateless**: Does the model maintain
  internal state across time steps (e.g., background field),
  or is each call independent? Stateless models return
  `None` from `init_coords()`.
- **`@torch.inference_mode()` safety**: Does the forward
  pass require gradients (e.g., DPS guidance through a
  denoiser)? If so, `@torch.inference_mode()` must be
  omitted and the reason documented.
- **Dependencies**: External packages required (physicsnemo, scipy, healpy, cudf, cupy, etc.)

Present the analysis to the user:

> **Model Analysis Summary**
>
> - **Input type(s):** [DataFrame / DataArray / both]
> - **Output type(s):** [DataArray / DataFrame / both]
> - **Stateful/Stateless:** [stateful — needs init data / stateless — no init data]
> - **Inference mode safe:** [yes / no — reason]
> - **Key dependencies:** [list]
>
> Evidence:
>
> - [list key indicators from reference code]

### **[CONFIRM — Model Analysis]**

Ask the user to confirm the analysis before proceeding.

---

## Step 1 — Examine Reference & Propose Dependencies

### 1a. Analyze the reference code

Examine the reference inference script/repo to identify:

- **Python packages** required (e.g., `physicsnemo`, `scipy`, `healpy`, `cudf`, `cupy`, custom packages)
- **Model architecture** (PyTorch module, ONNX, etc.)
- **Observation schema** (DataFrame columns, variable names, coordinate dimensions)
- **Output grid specification** (lat/lon resolution, projection, etc.)
- **Checkpoint format** (`.pt`, `.onnx`, `.safetensors`, `.mdlus`, etc.)

### 1b. Propose pyproject.toml dependency group

Propose a new optional dependency group for `pyproject.toml`.
The group name must follow the pattern `da-<model-name>`:

```toml
# In [project.optional-dependencies] section of pyproject.toml
da-model-name = [
    "package1>=version",
    "package2",
]
```

Look at the existing groups in `pyproject.toml` for reference on naming and version pinning conventions.

Also propose adding the new group to the `all` aggregate in the appropriate line (da models).

Highlight `cudf` and `cupy` as optional GPU acceleration
packages — these are not required but improve performance.

### **[CONFIRM — Dependencies]**

Present to the user:

1. The proposed dependency group name (`da-<model-name>`)
2. The list of packages with versions
3. Ask if the packages and group name look correct

---

## Step 2 — Add Dependencies to pyproject.toml

After confirmation, edit `pyproject.toml`:

1. Add the new optional dependency group in alphabetical order among the per-model extras
2. Add the group to the `all` aggregate (in the da models line)

---

## Step 3 — Create Skeleton Class File

### 3a. Determine class name and file name

Based on the model name from the reference, propose:

- **Class name**: PascalCase (e.g., `StormCastSDA`, `InterpEquirectangular`, `HealDA`)
- **File name**: lowercase with underscores (e.g., `sda_stormcast.py`, `interp.py`, `healda.py`)
- **File path**: `earth2studio/models/da/<filename>.py`

### 3b. Write skeleton with pseudocode

Create the file with the full structure but pseudocode implementations.
Every `.py` file in `earth2studio/` **must** start with this license
header:

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

Dual inheritance: `torch.nn.Module + AutoModelMixin` (NO `PrognosticMixin`).
Class-level `@check_optional_dependencies()` decorator.

### Canonical method ordering for DA models

Methods in the class **must** appear in this order (11 method slots):

1. `__init__` — register `device_buffer`, store model params, normalize time tolerance
2. `device` property — `return self.device_buffer.device`
3. `init_coords` — `None` for stateless, tuple of `CoordSystem`/`FrameSchema` for stateful
4. `input_coords` — tuple of `FrameSchema` (for DataFrame) or `CoordSystem` (for DataArray)
5. `output_coords` — accept `input_coords` tuple + `request_time` kwarg, return tuple
6. `load_default_package` — classmethod returning default `Package`
7. `load_model` — classmethod with `@check_optional_dependencies()`
8. `to` — device management, return `AssimilationModel`
9. Private/support methods (e.g., `_interpolate`, `_forward`, spatial lookups)
10. `__call__` — stateless forward, accept `*args: pd.DataFrame | xr.DataArray | None`
11. `create_generator` — bidirectional generator with send protocol

### Complete skeleton template

```python
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.models.da.utils import (
    dfseries_to_torch,
    filter_time_range,
    validate_observation_fields,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeTolerance

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment, misc]

try:
    from some_package import CoreModel
except ImportError:
    OptionalDependencyFailure("da-mymodel")
    CoreModel = None


@check_optional_dependencies()
class MyDAModel(torch.nn.Module, AutoModelMixin):
    """One-line description of the DA model.

    Extended description of the model, its source, observation types it
    handles, and the analysis output it produces.

    Parameters
    ----------
    model : torch.nn.Module
        Core neural network or inference model
    time_tolerance : TimeTolerance, optional
        Observation time tolerance window, by default np.timedelta64(3, "h")

    Note
    ----
    For more information see: <link to paper/repo>

    Badges
    ------
    region:global class:da product:atmos product:insitu
    """

    OUTPUT_VARIABLES = ["u10m", "v10m", "t2m"]

    # 1. Constructor
    def __init__(
        self,
        model: torch.nn.Module,
        time_tolerance: TimeTolerance = np.timedelta64(3, "h"),
    ) -> None:
        super().__init__()
        self._model = model
        self._tolerance = normalize_time_tolerance(time_tolerance)
        self.register_buffer("device_buffer", torch.empty(0))
        # TODO: Register normalization buffers, static fields, etc.

    # 2. Device property
    @property
    def device(self) -> torch.device:
        """Model device."""
        return self.device_buffer.device

    # 3. Init coords
    def init_coords(self) -> None:
        """Initialization coordinate system.

        Returns None for stateless models. Override to return a tuple of
        CoordSystem / FrameSchema for stateful models that require initial
        state data.

        Returns
        -------
        None
            No initialization data required (stateless model)
        """
        return None

        # For stateful models, return a tuple instead:
        # return (
        #     CoordSystem(OrderedDict({
        #         "time": np.empty(0),
        #         "lead_time": np.array([np.timedelta64(0, "h")]),
        #         "variable": np.array(self.OUTPUT_VARIABLES),
        #         "lat": self._lat,
        #         "lon": self._lon,
        #     })),
        # )

    # 4. Input coords
    def input_coords(self) -> tuple[FrameSchema]:
        """Input coordinate system specifying required DataFrame fields.

        Returns
        -------
        tuple[FrameSchema]
            Tuple containing FrameSchema with field names as keys.
            Use np.empty(0, dtype=...) for dynamic dimensions and
            np.array([...]) for enumerated allowed values.
        """
        return (
            FrameSchema({
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(self.OUTPUT_VARIABLES, dtype=str),
            }),
        )

    # 5. Output coords
    def output_coords(
        self,
        input_coords: tuple[FrameSchema | CoordSystem, ...],
        request_time: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[CoordSystem]:
        """Output coordinate system.

        Parameters
        ----------
        input_coords : tuple[FrameSchema | CoordSystem, ...]
            Input coordinate system(s)
        request_time : np.ndarray | None, optional
            Analysis valid time(s)

        Returns
        -------
        tuple[CoordSystem]
            Output coordinate system(s)
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

        # Extract variables from first input coord system
        if len(input_coords) > 0 and "variable" in input_coords[0]:
            variables = input_coords[0]["variable"]
        else:
            variables = np.array(self.OUTPUT_VARIABLES, dtype=str)

        return (
            CoordSystem(OrderedDict({
                "time": request_time,
                "variable": variables,
                "lat": np.linspace(90, -90, 181),
                "lon": np.linspace(0, 360, 360, endpoint=False),
            })),
        )

    # 6. Default package
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package with default checkpoint location
        """
        return Package(
            "hf://nvidia/my-da-model@<commit-sha>",
            cache_options={"same_names": True},
        )

    # 7. Load model
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        time_tolerance: TimeTolerance = np.timedelta64(3, "h"),
    ) -> AssimilationModel:
        """Load assimilation model from package.

        Parameters
        ----------
        package : Package
            Package containing model checkpoint and statistics
        time_tolerance : TimeTolerance, optional
            Observation time tolerance window

        Returns
        -------
        AssimilationModel
            Loaded assimilation model
        """
        # TODO: Load model from package
        model = CoreModel.from_checkpoint(package.resolve("model.mdlus"))
        model.eval()

        # Load normalization stats, static fields, etc.
        stats = np.load(package.resolve("stats.npy"))

        return cls(model=model, time_tolerance=time_tolerance)

    # 8. Device management
    def to(self, device: torch.device | str) -> AssimilationModel:
        """Move model to device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        AssimilationModel
            Model on target device
        """
        super().to(device)
        return self

    # 9. Private/support methods
    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Internal forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Preprocessed input tensor on device

        Returns
        -------
        torch.Tensor
            Raw model output tensor
        """
        # TODO: normalize -> model -> denormalize
        return self._model(inputs)

    # 10. Stateless forward pass
    @torch.inference_mode()
    def __call__(self, obs: pd.DataFrame | None = None) -> xr.DataArray:
        """Run single-step assimilation.

        Parameters
        ----------
        obs : pd.DataFrame | None, optional
            Observation DataFrame with required columns and
            'request_time' in attrs

        Returns
        -------
        xr.DataArray
            Analysis output on the same device as the model
        """
        if obs is None:
            raise ValueError("obs must be provided")

        # Validate required columns
        input_coords = self.input_coords()
        validate_observation_fields(obs, required_fields=list(input_coords[0].keys()))

        # Extract request_time from DataFrame attrs
        request_time = obs.attrs.get("request_time")
        if request_time is None:
            raise ValueError(
                "Observation DataFrame must have 'request_time' in attrs. "
                "This is typically set by earth2studio data sources."
            )

        # Get output coordinate system
        (output_coords,) = self.output_coords(
            input_coords, request_time=request_time
        )

        # Filter observations within time tolerance window
        filtered_obs = filter_time_range(
            obs,
            request_time=request_time,
            tolerance=self._tolerance,
            time_column="time",
        )

        # Convert DataFrame columns to torch tensors (zero-copy for cudf)
        obs_values = dfseries_to_torch(
            filtered_obs["observation"],
            dtype=torch.float32,
            device=self.device,
        )
        obs_lat = dfseries_to_torch(
            filtered_obs["lat"],
            dtype=torch.float32,
            device=self.device,
        )
        obs_lon = dfseries_to_torch(
            filtered_obs["lon"],
            dtype=torch.float32,
            device=self.device,
        )

        # TODO: Run model forward pass
        prediction = self._forward(obs_values)

        # Build output xr.DataArray with cupy (GPU) or numpy (CPU)
        if self.device.type == "cuda" and cp is not None:
            data = cp.asarray(prediction)
        else:
            data = prediction.cpu().numpy()

        return xr.DataArray(
            data=data,
            dims=list(output_coords.keys()),
            coords=output_coords,
        )

    # 11. Stateful generator
    def create_generator(
        self,
    ) -> Generator[xr.DataArray, pd.DataFrame | None, None]:
        """Creates a generator for sequential data assimilation.

        Yields the current analysis state and receives the next observations
        via generator.send(). Prime the generator with next(gen) before
        sending observations.

        Yields
        ------
        xr.DataArray
            Analysis at each step

        Receives
        --------
        pd.DataFrame | None
            Observations for the next step. Pass None for steps with no data.

        Example
        -------
        >>> gen = model.create_generator()
        >>> next(gen)                  # prime generator (yields None)
        >>> result = gen.send(obs_df)  # step 1 with observations
        >>> result = gen.send(None)    # step 2 without observations
        >>> gen.close()                # clean up
        """
        # Prime generator — yield None for stateless models
        observations = yield None  # type: ignore[misc]
        try:
            while True:
                result = self.__call__(observations)
                observations = yield result
        except GeneratorExit:
            logger.debug("MyDAModel generator clean up complete.")
```

For **stateful models** that maintain internal state, the `create_generator` pattern is:

```python
def create_generator(
    self,
    x: xr.DataArray,  # initial state
) -> Generator[xr.DataArray, pd.DataFrame | None, None]:
    """Creates a stateful generator for sequential data assimilation.

    Parameters
    ----------
    x : xr.DataArray
        Initial state DataArray

    Yields
    ------
    xr.DataArray
        Analysis at each step

    Receives
    --------
    pd.DataFrame | None
        Observations for the next step

    Example
    -------
    >>> gen = model.create_generator(x0)
    >>> state = next(gen)          # yields initial state
    >>> state = gen.send(obs_df)   # step 1 with observations
    >>> state = gen.send(None)     # step 2 without observations
    >>> gen.close()                # clean up
    """
    # Yield initial state to prime the generator
    obs = yield x

    try:
        while True:
            result = self.__call__(x, obs)
            x = result  # update internal state
            obs = yield result
    except GeneratorExit:
        logger.debug("MyDAModel generator clean up complete.")
```

### **[CONFIRM — Skeleton]**

Present to the user:

1. The proposed class name
2. The proposed file name and path
3. The canonical method ordering (11 methods)
4. Whether stateful or stateless generator pattern applies
5. Ask if these are acceptable

---

## Step 4 — Implement Coordinate System

### 4a. Map variables to E2STUDIO_VOCAB

Read `earth2studio/lexicon/base.py` and verify every variable the
model uses exists in `E2STUDIO_VOCAB`. The vocab contains 282
entries including:

| Category | Examples |
| --------------- | ------------------------------------------- |
| Surface wind | `u10m`, `v10m`, `ws10m`, `u100m`, `v100m` |
| Surface temp | `t2m`, `d2m`, `sst`, `skt` |
| Humidity | `r2m`, `q2m`, `tcwv` |
| Pressure | `sp`, `msl` |
| Precipitation | `tp`, `lsp`, `cp`, `tp06` |
| Pressure-level | `u50`-`u1000`, `v50`-`v1000`, `z50`-`z1000` |
| Cloud/radiation | `tcc`, `rlut`, `rsut` |

If a variable in the reference model does NOT exist in
`E2STUDIO_VOCAB`, flag it to the user and discuss whether to
map it to an existing vocab entry or propose adding a new one.

### 4b. Implement init_coords

**Stateless models** — return `None`:

```python
def init_coords(self) -> None:
    """No initialization data required."""
    return None
```

**Stateful models** — return a tuple of `CoordSystem` and/or `FrameSchema`:

```python
def init_coords(self) -> tuple[CoordSystem]:
    """Initialization coordinate system for the background state.

    Returns
    -------
    tuple[CoordSystem]
        Tuple containing one CoordSystem for the initial gridded state
    """
    return (
        CoordSystem(OrderedDict({
            "time": np.empty(0),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(self.OUTPUT_VARIABLES),
            "lat": self._lat,
            "lon": self._lon,
        })),
    )
```

### 4c. Implement input_coords

Return a **tuple** of `FrameSchema | CoordSystem` — one entry per
positional argument that `__call__` accepts.

For DataFrame observations, use `FrameSchema`:

```python
def input_coords(self) -> tuple[FrameSchema]:
    """Input coordinate system specifying required DataFrame fields.

    Returns
    -------
    tuple[FrameSchema]
        Tuple containing FrameSchema for observation DataFrame
    """
    return (
        FrameSchema({
            "time": np.empty(0, dtype="datetime64[ns]"),
            "lat": np.empty(0, dtype=np.float32),
            "lon": np.empty(0, dtype=np.float32),
            "observation": np.empty(0, dtype=np.float32),
            "variable": np.array(self.OUTPUT_VARIABLES, dtype=str),
        }),
    )
```

For models accepting multiple inputs (e.g., conventional + satellite observations):

```python
def input_coords(self) -> tuple[FrameSchema, FrameSchema]:
    conv_schema = FrameSchema({...})
    sat_schema = FrameSchema({...})
    return (conv_schema, sat_schema)
```

**Rules:**

- Use `np.empty(0, dtype=...)` for unbounded/dynamic dimensions (individual observations)
- Use `np.array([...])` for enumerated allowed values (variable names)
- Always return a **tuple**, even for a single input

### 4d. Implement output_coords

Accept the input coordinate tuple plus `request_time` kwargs.
Return a tuple of output `CoordSystem | FrameSchema`:

```python
def output_coords(
    self,
    input_coords: tuple[FrameSchema | CoordSystem, ...],
    request_time: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[CoordSystem]:
    """Output coordinate system.

    Parameters
    ----------
    input_coords : tuple[FrameSchema | CoordSystem, ...]
        Input coordinate system(s)
    request_time : np.ndarray | None, optional
        Analysis valid time(s)

    Returns
    -------
    tuple[CoordSystem]
        Output coordinate system(s)
    """
    if request_time is None:
        request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

    # Extract variables from first input coord system
    if len(input_coords) > 0 and "variable" in input_coords[0]:
        variables = input_coords[0]["variable"]
    else:
        variables = np.array(self.OUTPUT_VARIABLES, dtype=str)

    return (
        CoordSystem(OrderedDict({
            "time": request_time,
            "variable": variables,
            "lat": self._lat,
            "lon": self._lon,
        })),
    )
```

**Key points:**

- Validate `CoordSystem` inputs using `handshake_dim` / `handshake_size` / `handshake_coords`
- Validate `FrameSchema` inputs using `validate_observation_fields()`
- Always return a **tuple**, even for a single output

### **[CONFIRM — Coordinates]**

Present to the user:

1. The input and output variable lists and any mapping issues with `E2STUDIO_VOCAB`
2. Whether `init_coords` returns `None` (stateless) or a tuple (stateful)
3. The `FrameSchema` column names and types for observation inputs
4. The output grid specification (lat/lon dimensions)

---

## Step 5 — Implement Forward Pass

### 5a. Implement `__call__`

The stateless forward pass accepts typed arguments matching
`input_coords`, runs inference, and returns a tuple of
`xr.DataArray | pd.DataFrame`.

```python
@torch.inference_mode()
def __call__(self, obs: pd.DataFrame | None = None) -> xr.DataArray:
    """Run single-step assimilation.

    Parameters
    ----------
    obs : pd.DataFrame | None, optional
        Observation DataFrame

    Returns
    -------
    xr.DataArray
        Analysis output on the same device as the model
    """
    if obs is None:
        raise ValueError("obs must be provided")

    # 1. Validate required columns
    input_coords = self.input_coords()
    validate_observation_fields(obs, required_fields=list(input_coords[0].keys()))

    # 2. Extract request_time from DataFrame attrs
    request_time = obs.attrs.get("request_time")
    if request_time is None:
        raise ValueError(
            "Observation DataFrame must have 'request_time' in attrs. "
            "This is typically set by earth2studio data sources."
        )

    # 3. Get output coordinate system
    (output_coords,) = self.output_coords(input_coords, request_time=request_time)

    # 4. Filter observations within time tolerance window
    filtered_obs = filter_time_range(
        obs,
        request_time=request_time,
        tolerance=self._tolerance,
        time_column="time",
    )

    # 5. Convert DataFrame columns to torch tensors (zero-copy for cudf)
    obs_values = dfseries_to_torch(
        filtered_obs["observation"], dtype=torch.float32, device=self.device,
    )
    obs_lat = dfseries_to_torch(
        filtered_obs["lat"], dtype=torch.float32, device=self.device,
    )
    obs_lon = dfseries_to_torch(
        filtered_obs["lon"], dtype=torch.float32, device=self.device,
    )

    # 6. Run model forward pass
    prediction = self._forward(obs_values)

    # 7. Build output xr.DataArray with cupy (GPU) or numpy (CPU)
    if self.device.type == "cuda" and cp is not None:
        data = cp.asarray(prediction)
    else:
        data = prediction.cpu().numpy()

    return xr.DataArray(
        data=data,
        dims=list(output_coords.keys()),
        coords=output_coords,
    )
```

**Key points:**

- Use `@torch.inference_mode()` on `__call__` or on the internal `_forward` method — place it
  on `_forward` if preprocessing outside `_forward` needs gradients. Omit entirely if the
  forward pass requires gradients (e.g., DPS guidance); document the reason if omitted
- Expect `request_time` in `obs.attrs` — validate early
- Use `validate_observation_fields()` to check required DataFrame columns
- Use `filter_time_range()` for time-window filtering
- Use `dfseries_to_torch()` for zero-copy cudf→torch conversion
- Output data must live on the same device as the model (cupy for GPU, numpy for CPU)
- Do **NOT** use `@batch_func()` — this is a px/dx convention only

### 5b. Implement `create_generator`

**Stateless pattern** (no initial state, delegates to `__call__`):

```python
def create_generator(
    self,
) -> Generator[xr.DataArray, pd.DataFrame | None, None]:
    """Creates a generator for sequential data assimilation.

    Yields
    ------
    xr.DataArray
        Analysis at each step

    Receives
    --------
    pd.DataFrame | None
        Observations for the next step
    """
    # Prime generator — yield None for stateless models
    observations = yield None  # type: ignore[misc]
    try:
        while True:
            result = self.__call__(observations)
            observations = yield result
    except GeneratorExit:
        logger.debug("MyDAModel generator clean up complete.")
```

**Stateful pattern** (maintains background state across steps):

```python
def create_generator(
    self,
    x: xr.DataArray,  # initial state
) -> Generator[xr.DataArray, pd.DataFrame | None, None]:
    """Creates a stateful generator for sequential data assimilation.

    Parameters
    ----------
    x : xr.DataArray
        Initial state DataArray

    Yields
    ------
    xr.DataArray
        Analysis at each step

    Receives
    --------
    pd.DataFrame | None
        Observations for the next step
    """
    # Yield initial state to prime the generator
    obs = yield x

    try:
        while True:
            result = self.__call__(x, obs)
            x = result  # update internal state
            obs = yield result
    except GeneratorExit:
        logger.debug("MyDAModel generator clean up complete.")
```

**Key points:**

- Always yield first to prime the generator (`yield None` for stateless, `yield x` for stateful)
- Receive observations via `.send()`: `observations = yield result`
- **Always** handle `GeneratorExit` for clean-up logic (e.g., releasing GPU resources)
- Do **NOT** use `create_iterator` — that is a px convention only

### **[CONFIRM — Forward Pass]**

Show the user the implementation for `__call__` and `create_generator`. Ask:

1. Does the computation logic look correct?
2. Is `@torch.inference_mode()` safe, or does the model need gradient flow?
3. Are there any special considerations (multiple observation types, custom preprocessing)?

---

## Step 6 — Implement Model Loading

### 6a. Implement load_default_package

```python
@classmethod
def load_default_package(cls) -> Package:
    """Default pre-trained model package.

    Returns
    -------
    Package
        Model package with default checkpoint location
    """
    return Package(
        "hf://nvidia/my-da-model@<commit-sha>",
        cache_options={"same_names": True},
    )
```

### 6b. Implement load_model

```python
@classmethod
@check_optional_dependencies()
def load_model(
    cls,
    package: Package,
    time_tolerance: TimeTolerance = np.timedelta64(3, "h"),
) -> AssimilationModel:
    """Load assimilation model from package.

    Parameters
    ----------
    package : Package
        Package containing model checkpoint and statistics
    time_tolerance : TimeTolerance, optional
        Observation time tolerance window

    Returns
    -------
    AssimilationModel
        Loaded assimilation model
    """
    model = CoreModel.from_checkpoint(package.resolve("model.mdlus"))
    model.eval()

    # Load normalization stats, static fields, etc.
    stats = np.load(package.resolve("stats.npy"))

    return cls(model=model, time_tolerance=time_tolerance)
```

**Key patterns:**

- Decorate `load_model` with `@check_optional_dependencies()`
- Use `package.resolve("filename")` to get cached file paths
- Call `.eval()` on loaded neural network modules
- Only expose essential parameters — do **not** over-populate the API

### 6c. Implement .to()

> **Note:** When the wrapper inherits from `torch.nn.Module`,
> `super().to(device)` already handles moving all registered
> parameters, buffers, and sub-modules. A custom `to()`
> override is only needed when there is non-PyTorch state to
> manage (e.g., ONNX Runtime sessions, JAX device placement).

```python
def to(self, device: torch.device | str) -> AssimilationModel:
    """Move model to device.

    Parameters
    ----------
    device : torch.device | str
        Target device

    Returns
    -------
    AssimilationModel
        Model on target device
    """
    super().to(device)
    return self
```

### **[CONFIRM — Model Loading]**

Present to the user:

1. The checkpoint URL/path for `load_default_package`
2. The checkpoint file names and loading logic
3. Whether there are multiple checkpoint files
4. The `.to()` implementation

---

## Step 7 — Register the Model

### 7a. Add to `__init__.py`

Edit `earth2studio/models/da/__init__.py`:

- Add import in alphabetical order:
  `from earth2studio.models.da.<filename> import <ClassName>`
- If an `__all__` list exists, add `<ClassName>` to it in alphabetical order

### 7b. Verify pyproject.toml

Confirm the dependency group was added in Step 2 and is included in the `all` aggregate.

---

## Step 8 — Verify Style, Documentation, Format & Lint

Before testing, verify the wrapper passes all code quality checks.

### 8a. Run formatting

```bash
make format
```

This runs `black` on the codebase. Fix any formatting issues in
the new wrapper file.

### 8b. Run linting

```bash
make lint
```

This runs `ruff` and `mypy`. Common issues to watch for:

- Missing type annotations on public functions
- Unused imports
- Import ordering issues
- Type errors from incorrect return types or missing annotations

Fix all errors before proceeding.

### 8c. Check license headers

```bash
make license
```

Verify that the wrapper file
(`earth2studio/models/da/<filename>.py`) has the correct SPDX
Apache-2.0 license header (2024-2026 copyright years).

### 8d. Verify documentation

Check that:

- The class docstring follows NumPy-style formatting with
  `Parameters`, `Note`, `Badges` sections
- All public methods have complete docstrings with
  `Parameters`, `Returns`, `Raises` sections as applicable
- Type hints are present on all public method signatures
- The model is added to `docs/modules/models.rst` in the
  `earth2studio.models.da` section (alphabetical order)

If any checks fail, fix the issues and re-run until all pass cleanly.

---

## Reminders

- **DO** return tuples from `input_coords` and `output_coords`,
  even for single inputs/outputs
- **DO** use `FrameSchema` for tabular DataFrame inputs and
  `CoordSystem` for gridded outputs
- **DO** validate `request_time` from `obs.attrs` — it is set
  by earth2studio data sources
- **DO** use `validate_observation_fields()` to check required
  DataFrame columns early
- **DO** use `filter_time_range()` for time-window filtering
  of observations
- **DO** use `dfseries_to_torch()` for zero-copy cudf to torch
  column conversion
- **DO** prime `create_generator` with `yield None` (stateless)
  or `yield initial_state` (stateful) before the loop
- **DO** handle `GeneratorExit` in `create_generator` for
  clean-up
- **DO** register `device_buffer` and expose a `device` property
- **DO** return cupy arrays on GPU, numpy arrays on CPU for
  `xr.DataArray` output
- **DO** use `loguru.logger` for logging, never `print()`,
  inside `earth2studio/`
- **DO** ensure all public functions have full type hints
- **DO** run formatting (`make format`) and linting
  (`make lint`) before finalizing
- **DO** use `@torch.inference_mode()` on `__call__` for
  inference-only models
- **DO** set `eval()` on loaded NN models in `load_model`
- **DO** add the model to `docs/modules/models.rst` in the
  `earth2studio.models.da` section (alphabetical order)
- **DO** use `uv run python` for all Python commands (never
  bare `python`)
- **DO NOT** use `@batch_func()` or `@batch_coords()` — these
  are px/dx conventions only
- **DO NOT** use `PrognosticMixin` — DA models do not need
  iterator hooks
- **DO NOT** use `create_iterator` — DA uses
  `create_generator` with send protocol
- **DO NOT** assume tensor inputs — inputs are
  DataFrames/DataArrays
- **DO NOT** forget cudf/cupy optional import pattern
- **DO NOT** use `@torch.inference_mode()` if the forward pass
  requires gradients (e.g., DPS guidance); document the reason
  if omitted
- **DO NOT** attempt to make a general base class with intent
  to reuse the wrapper across models
- **DO NOT** over-populate the `load_model()` API — only
  expose essential parameters
