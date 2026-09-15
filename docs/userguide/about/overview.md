<!-- markdownlint-disable MD033 -->

# Overview { #overview_userguide }

Welcome to the Earth2Studio user guide.
This guide explains how Earth2Studio is organized, how its main components fit
together, and where to go when you want to install dependencies, run examples, or
extend the package.

Earth2Studio is a Python package for working with AI weather and climate models.
It provides model interfaces, data connectors, IO backends, workflow utilities, and
supporting tools that can be assembled into research or production inference
pipelines.

## Package Philosophy

Earth2Studio is intentionally built from small, explicit pieces.
Rather than hiding a full workflow inside a single object, the package keeps data
access, model execution, output writing, perturbations, statistics, and checkpointing
as separate components with shared interfaces.
This makes it easier to inspect intermediate state, replace one part of a workflow,
or bring your own model or dataset.

!!! tip "Design principles"
    - **Modular components**: choose the data source, model, IO backend, and workflow
      utilities that fit the task.
    - **Simple data movement**: tensors and coordinates stay visible at component
      boundaries.
    - **Consistent APIs**: similar components expose similar call patterns.
    - **Composable workflows**: built-in workflows are starting points, not closed
      systems.
    - **Extension first**: custom models, data sources, and IO backends can plug into
      the same interfaces.

## Pipeline Model

Most Earth2Studio workflows follow the same high-level shape:

<div class="grid cards" markdown>

- **1. Fetch state or observations**

    Data sources fetch initial states, forecasts, reanalysis data, observations, or
    local arrays.

- **2. Prepare tensor data**

    Workflow utilities convert Xarray or tabular data into tensors and coordinate
    dictionaries for model execution.

- **3. Run models**

    Prognostic, diagnostic, data-assimilation, and downscaling models transform the
    state.

- **4. Store or analyze results**

    IO backends, statistics, and checkpoints write outputs, summarize ensembles, or
    make workflows restartable.

</div>

This separation is the main mental model for the package.
If a workflow does not quite fit your use case, you can usually keep most components
unchanged and replace the one piece that needs custom behavior.

## Data Movement { #data_userguide }

Earth2Studio keeps the data exchanged between components explicit and inspectable.
Inside model workflows, the common representation is:

1. A PyTorch tensor (`torch.Tensor`) that holds the array data on the inference device.
2. An ordered coordinate dictionary (`CoordSystem`) that describes the tensor axes.

The tensor carries the values; the coordinate system explains what each dimension
means.
For example, a forecast tensor might be indexed by batch, lead time, variable,
latitude, and longitude.

!!! note
    Data is moved between Earth2Studio components in physical units. Normalization,
    scaling, and model-specific preprocessing should be handled inside the relevant
    model or component.

Data sources generally return Xarray objects because those objects are useful outside
Earth2Studio and keep coordinate metadata attached on the CPU.
Workflow utilities such as `earth2studio.data.fetch_data` and
`earth2studio.data.prep_data_array` then prepare tensors and coordinate dictionaries
for model execution.

## Coordinate Systems { #coordinates_userguide }

Coordinate dictionaries are ordered because their keys correspond to tensor
dimensions.
Earth2Studio uses a small set of common coordinate names across built-in components:

| Key | Description |
| --- | --- |
| `batch` | Free dimension for batching or ensemble-like axes. |
| `time` | Initialization or valid times as NumPy datetime values. |
| `lead_time` | Forecast step offsets as NumPy timedelta values. |
| `variable` | Earth2Studio variable identifiers. |
| `lat` | Latitude coordinate values. |
| `lon` | Longitude coordinate values, commonly `[0, 360)`. |

The coordinate dictionary does not need to be complicated.
A simple latitude-longitude grid might look like:

```python
from collections import OrderedDict

import numpy as np
import torch

x = torch.randn(181, 360)
coords = OrderedDict(
    {
        "lat": np.linspace(-90, 90, 181),
        "lon": np.linspace(0, 360, 360, endpoint=False),
    }
)
```

Models advertise the coordinates they expect and produce through `input_coords()` and
`output_coords()`.
This lets workflows validate compatibility before or during execution and makes model
rollouts easier to reason about.

## Where To Go Next

<div class="grid cards" markdown>

- **Install Earth2Studio**

    Set up the package, optional dependencies, cache locations, and model-specific
    extras.

    [:octicons-arrow-right-24: Install guide](install.md)

- **Learn the components**

    Read about prognostic models, diagnostic models, data sources, perturbations, IO,
    and statistics.

    [:octicons-arrow-right-24: Core components](../components/index.md)

- **Run examples**

    Jump directly into executable examples for forecasts, diagnostics, downscaling,
    data assimilation, and extension patterns.

    [:octicons-arrow-right-24: Examples](../../examples/index.md)

- **Contribute or extend**

    Find development setup, testing, style, documentation, and package-extension
    guidance.

    [:octicons-arrow-right-24: Developer guide](../developer/index.md)

</div>
