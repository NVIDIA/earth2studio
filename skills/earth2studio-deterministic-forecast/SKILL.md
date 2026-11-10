---
name: earth2studio-deterministic-forecast
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - inference
    - forecast
    - deterministic
description: >
  Build deterministic forecast scripts with Earth2Studio (model, data source,
  IO, inference). Do NOT use for ensemble, diagnostics, data-only fetch, or
  install.
---

# Earth2Studio Deterministic Forecast Skill

Guide users through building deterministic (single-member) weather forecast
inference scripts with `earth2studio.run.deterministic`, or with `Pipeline` when
they need multiple initial times, resume, or several output streams.

## Prerequisites

- Earth2Studio installed with the extra for the chosen model (see
  `earth2studio-install`; it owns Python version, extras, and environment setup)
- A CUDA-capable GPU for nearly all models (`deterministic` falls back to CPU when
  CUDA is unavailable, which is rarely practical)
- Network access for model weights and data

## Where to look

Models, data sources, and APIs change between releases. Read these instead of
recalling them, and cite what you used.

| Need | Where to look (repo path first, live page as a fallback) |
|------|------------------------------------------------|
| Prognostic models and their badges (region, VRAM, class) | `docs/modules/models_px.md` |
| What a model needs and how it steps | the model class in `earth2studio/models/px/`: `input_coords()`, `output_coords()`, `forcing_coords()`, `default_sources()` |
| Data sources | `docs/modules/datasources_analysis.md`, `datasources_forecast.md`, and `earth2studio/lexicon/` |
| IO backends | `earth2studio/io/` (`ZarrBackend`, `NetCDF4Backend`, `XarrayBackend`, `AsyncZarrBackend`, `KVBackend`, ...) |
| The built-in workflow | `deterministic` in `earth2studio/run/__init__.py` |
| Multi-time, multi-stream, resumable runs | `Pipeline`, `work_items`, `ProgressStore` in `earth2studio/run/` |
| Worked example | `examples/01_getting_started/01_deterministic_workflow.py` |
| What changed from 0.x | `MIGRATION.md` |

Live pages (note the `/main/` segment; URLs without it return 404):

```text
https://nvidia.github.io/earth2studio/main/modules/models_px/
https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/
https://nvidia.github.io/earth2studio/main/modules/datasources_forecast/
https://nvidia.github.io/earth2studio/main/modules/io/
https://nvidia.github.io/earth2studio/main/examples/
```

## Workflow

### 1. Gather requirements (skip what's already provided)

- Time horizon (hours/days/weeks) and initialization time(s)
- Variables of interest (t2m, wind, geopotential, etc.)
- Region (global or specific like CONUS)
- GPU/VRAM available

### 2. Select a model

Read the prognostic models page and filter by horizon, region, and VRAM. From the
model class, note:

- Input variables: `model.input_coords().coords["variable"]`
- Step size: the model's output `lead_time` step. Read it from
  `output_coords(input_coords())` rather than assuming 6 h.
- Forcing slots: if `forcing_coords()` is non-empty, the model needs extra data
  (for example static fields or time-varying forcing) that `run.deterministic`
  does not supply. Use `Pipeline.from_model` or the manual loop for those.

### 3. Select a data source

Start from `model.default_sources()`: it lists the model's recommended source for
each input slot (and forcing slot), or `None` if there is no recommendation. Use
a different source only if the user asks or the recommendation is unsuitable. Any
source must provide every input variable, so check its lexicon in
`earth2studio/lexicon/` (`VOCAB` keys). Regional models need a data source that
covers their region.

### 4. Select an IO backend

Default: `ZarrBackend`. Use `NetCDF4Backend` for legacy tools and `XarrayBackend`
for in-memory or small runs. IO backends are created from an output template and
do not grow afterwards. `deterministic` and `Pipeline` build the template for you;
only a manual loop has to.

### 5. Calculate nsteps

`nsteps = forecast_hours // model_step_hours`. A 5-day forecast with a 6 h step
gives `nsteps = 120 // 6 = 20`. The stored output has `nsteps + 1` lead times,
because the initial condition is written as lead time zero.

### 6. Decide whether to filter output

- Pass `output_coords` when the user asks for specific variables, which reduces
  output size. It remains a dictionary of coordinate subsets in this workflow.
- Omit it to save everything the model outputs.

### 7. Generate the script

```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import <DataSourceClass>
from earth2studio.io import <IOBackendClass>
from earth2studio.models.px import <ModelClass>
from earth2studio.run import deterministic

model = <ModelClass>.load_model(<ModelClass>.load_default_package())
data = <DataSourceClass>()
io = <IOBackendClass>("<output_path>")

# Include output_coords ONLY if the user requested specific variables
output_coords = OrderedDict({"variable": np.array(["t2m", "u10m"])})

io = deterministic(
    time=["YYYY-MM-DDTHH:MM:SS"],
    nsteps=<N>,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,  # omit to save all variables
    device=torch.device("cuda"),
)
```

Check the model class docstring for `load_model` arguments; some models take
options, and constructor arguments differ between models. The output store is
readable afterwards (for Zarr, `xr.open_zarr(...)`); dims are
`time, lead_time, variable, ...`.

### 8. Pipeline for larger runs

Use `Pipeline` instead of `deterministic` when the user wants several initial
times, resume after failure, per-stream variable filtering or regridding,
diagnostics alongside the forecast, or multi-GPU runs. Read `Pipeline` and
`work_items` in `earth2studio/run/` for the current signature:

```python
from earth2studio.run import Pipeline, work_items

items = work_items(times, final_lead_time=np.timedelta64(240, "h"))
Pipeline.from_model(model, data, output={"forecast": io}).run(items)
```

`times` is an array of `np.datetime64`. `final_lead_time` is a time, not a step
count.

### 9. Manual loop alternative

Only when the user explicitly wants a hand-written loop, follow this order. Read
the model's base protocol in `earth2studio/models/px/base.py` for the exact
semantics.

1. **Fetch**: `x = fetch_data(source, time, variable, lead_time, device=...)`
   using the model's declared variable and lead-time labels from `input_coords()`.
   It returns one DataArray. Pass `target_grid=infer_grid(model.input_coords())`
   (and a `Regridder` if you want a specific method) when the source grid differs.
   Validate with `handshake_dataarray`.
2. **Build the IO template** with `earth2studio.io.utils.output_template(...)`
   from `model.output_coords(model.input_coords())`, covering every time and lead
   time to be written, then `io.add_array(template)` once before the loop.
3. **Iterate**: `for step, y in enumerate(model.create_iterator(x)):`. The first
   yield is the first forecast, not the initial condition, so write the initial
   condition separately if you want it stored. Alternatively call
   `y, state = model.initialize(x)` then `y, state = model.step(y, state=state)`.
4. **Write**: `io.write(y)` (select variables with `.sel(variable=[...])` first
   if filtering). Writes locate positions by coordinate label.
5. **Finish**: call `io.flush()` before reading back and `io.close()` when done.

Never mutate `x` or `state` in place; forcing is fetched outside the model and
supplied to `initialize`/`step`.

### 10. Explain next steps

- Change forecast time or run several initializations (`Pipeline`)
- Read the output (`xr.open_zarr(...)`)
- Add a diagnostic model or ensemble: these have their own workflows

## Ownership

**Owns:** model selection, data source compatibility, IO backend selection,
nsteps calculation, generating `deterministic` / `Pipeline` scripts.

**Does not own:** ensembles, standalone diagnostics, data-only fetch
(`earth2studio-data-fetch`), installation (`earth2studio-install`), model
training, writing new models.

## Troubleshooting

See `references/troubleshooting.md` for common errors and solutions.

## Reminders

- **Read, don't recall.** Models, sources, and signatures change between releases.
- **Verify lexicon compatibility.** Model input variables must exist in the data
  source's `VOCAB`.
- **Use `load_default_package()`** for standard weights loading.
- **Time format** for `deterministic(time=...)` is ISO 8601 strings, datetimes, or
  `np.datetime64`.
- **Wind speed needs both components.** Include `u10m` and `v10m`.
- **nsteps is integer division** of horizon by the model step.
- **ZarrBackend is the default.** Suggest alternatives only for a specific need.
