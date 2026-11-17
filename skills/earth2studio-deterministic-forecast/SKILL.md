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
  Use when building Earth2Studio single-member forecast scripts, including
  diagnostic postprocessing of forecast fields with Pipeline. Do NOT use for
  ensembles, standalone diagnostics, data-only fetch, or installation.
---

# Earth2Studio Deterministic Forecast Skill

Guide users through building deterministic (single-member) weather forecast
inference scripts with `earth2studio.run.deterministic`, or with `Pipeline` when
they need diagnostic postprocessing, multiple initial times, resume, or several
output streams.

## Prerequisites

- Earth2Studio installed with extras for the prognostic and any diagnostic (see
  `earth2studio-install`; it owns Python version, extras, and environment setup)
- A CUDA-capable GPU for nearly all models (`deterministic` falls back to CPU when
  CUDA is unavailable, which is rarely practical)
- Network access for model weights and data

## Where to look

Models, data sources, and APIs change between releases. Read these instead of
recalling them, and cite what you used.

Use the selected version's [model contract](../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md),
[execution contract](../../docs/specs/E2S-11_EXECUTION_CONTRACT_SPEC.md) and
[IO specification](../../docs/specs/E2S-10_IO_SPEC.md) as the source of truth.
Consult these directly for behavior requirements; examples below do not define a separate contract.

| Need | Where to look (repo path first, live page as a fallback) |
|------|------------------------------------------------|
| Prognostic models and their badges (region, VRAM, class) | `docs/modules/models_px.md` |
| What a model needs and how it steps | the model class in `earth2studio/models/px/`: `input_coords()`, `output_coords()`, `forcing_coords()`, `default_sources()` |
| Diagnostic inputs, outputs, grid, and loading | `docs/modules/models_dx.md`, the class in `earth2studio/models/dx/`, and `earth2studio/models/dx/base.py` |
| Data sources | `docs/modules/datasources_analysis.md`, `datasources_forecast.md`, and `earth2studio/lexicon/` |
| IO backends | `earth2studio/io/` (`ZarrBackend`, `NetCDF4Backend`, `XarrayBackend`, `AsyncZarrBackend`, `KVBackend`, ...) |
| The built-in workflow | `deterministic` in `earth2studio/run/__init__.py` |
| Forecast + diagnostics, multi-time, resumable runs | `Pipeline.from_model`, `PrognosticRunner`, `work_items`, `ProgressStore` in `earth2studio/run/` |
| Worked example | `examples/01_getting_started/01_deterministic_workflow.py` |
| Pipeline streams, filtering, regridding, resume | `dev/examples/09_pipeline.py`, `test/run/test_pipeline.py`, `test/run/test_runner.py` |
| What changed from 0.x | `MIGRATION.md` |

Live pages (note the `/main/` segment; URLs without it return 404):

```text
https://nvidia.github.io/earth2studio/main/modules/models_px/
https://nvidia.github.io/earth2studio/main/modules/models_dx/
https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/
https://nvidia.github.io/earth2studio/main/modules/datasources_forecast/
https://nvidia.github.io/earth2studio/main/modules/io/
https://nvidia.github.io/earth2studio/main/examples/
```

## Workflow

### 1. Gather requirements (skip what's already provided)

- Time horizon (hours/days/weeks) and initialization time(s)
- Variables of interest (t2m, wind, geopotential, etc.)
- Diagnostic model or derived product, if requested; save forecast, diagnostics,
  or both, and at what output resolution?
- Region (global or specific like CONUS)
- GPU/VRAM available for the prognostic and diagnostics together

### 2. Select a model

Read the prognostic models page and filter by horizon, region, and VRAM. From the
model class, note:

- Input variables: `model.input_coords().coords["variable"]`
- Step size: the model's output `lead_time` step. Read it from
  `output_coords(input_coords())` rather than assuming 6 h.
- Forcing slots: if `forcing_coords()` is non-empty, the model needs extra data
  (for example static fields or time-varying forcing) that `run.deterministic`
  does not supply. Use `Pipeline.from_model` or the manual loop for those.

For diagnostic postprocessing, compare the prognostic's output signature with
the diagnostic's `input_coords()`: required variables and order, units/statistics,
grid/domain, and temporal dimensions must be compatible. Inspect the diagnostic's
`output_coords(...)` and loading arguments too. The built-in `PrognosticRunner`
path handles a single prognostic input/output and single-input, single-output
diagnostics. Multi-slot models need an explicit runner or coupling workflow.
Each diagnostic receives the forecast, not the preceding diagnostic's output.

### 3. Select a data source

Start from `model.default_sources()`: it lists the model's recommended source for
each input slot (and forcing slot), or `None` if there is no recommendation. Use
a different source only if the user asks or the recommendation is unsuitable. Any
source must provide every input variable, so check its lexicon in
`earth2studio/lexicon/` (`VOCAB` keys). Regional models need a data source that
covers their region.

`Pipeline.from_model` uses recommended sources when `source`/`forcing` are
omitted; otherwise pass one source per prognostic forcing slot as a tuple.
Diagnostics in this pipeline consume forecast fields, not their own data sources.
For off-grid source data, use `RegriddedSource` with a suitable regridder before
model inference; output regridding does not fix model input compatibility.

### 4. Select an IO backend

Default: `ZarrBackend`. Use `NetCDF4Backend` for legacy tools and `XarrayBackend`
for in-memory or small runs. IO backends are created from an output template and
do not grow afterwards. `deterministic` and `Pipeline` build the template for you;
only a manual loop has to.

### 5. Calculate nsteps

`nsteps = forecast_hours // model_step_hours`. A 5-day forecast with a 6 h step
gives `nsteps = 120 // 6 = 20`. Require an exact multiple rather than silently
truncating the requested horizon. The simple `deterministic` workflow writes
the initial condition at lead zero plus the forecast steps. Pipeline lead-zero
availability depends on the model signatures (section 8).

### 6. Decide whether to filter output

- For `deterministic`, pass `output_coords` when the user asks for specific
  variables. It remains a dictionary of coordinate subsets in this workflow.
- For `Pipeline`, use per-stream `output_variables` and `regridder` (section 8).
  Keep all diagnostic inputs until diagnostics have run.
- Omit variable filtering to save all variables in each selected output stream.

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
readable afterwards (for Zarr, `xr.open_zarr(...)`). Zarr stores each weather
variable as a separate array with dimensions `time, lead_time, ...`.

### 8. Pipeline for diagnostics and larger runs

Use `Pipeline` instead of `deterministic` when the user wants several initial
times, resume after failure, per-stream variable filtering or regridding,
diagnostics alongside the forecast, or multi-GPU runs. The 1.0.0-rc API uses
`Pipeline.from_model(..., diagnostics={stream_name: diagnostic})`, which wraps
`PrognosticRunner`. Read these classes and `work_items` for the installed version.

Example: FCN plus PrecipitationAFNO for 48 hours, with separate forecast and
precipitation stores. Their grids match and FCN supplies the diagnostic's 20
input variables. Install both model extras and configure FCN's recommended
source (`NCAR_ERA5` in this checkout) before running:

```python
import numpy as np

from earth2studio.io import ZarrBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import FCN
from earth2studio.run import Pipeline, work_items

model = FCN.load_model(FCN.load_default_package())
precip = PrecipitationAFNO.load_model(PrecipitationAFNO.load_default_package())
items = work_items(
    [np.datetime64("2024-01-01T00:00:00")],
    final_lead_time=np.timedelta64(48, "h"),
)
Pipeline.from_model(
    model,
    diagnostics={"precipitation": precip},
    output={
        "forecast": ZarrBackend("forecast.zarr"),
        "precipitation": ZarrBackend("precipitation.zarr"),
    },
    output_variables={"forecast": ["t2m", "u10m", "v10m"]},
    device="cuda",
).run(items)
```

- `work_items` takes a sequence of `np.datetime64`; omit `members` for a
  single-member run. `final_lead_time` is a duration, not a step count, and must
  be a non-negative multiple of the model step. Here 48 h means eight 6 h steps.
- `"forecast"` is reserved for prognostic output; diagnostic keys name separate
  streams. Each `output` entry gets its own backend. Omitted streams are not
  written; omit `"forecast"` to save only diagnostics. `run(items)` returns
  `None` and handles store planning, flushing, and closing.
- Diagnostics run before Pipeline's output filtering and regridding. Thus the
  example saves only three forecast variables while precipitation still receives
  all 20 inputs. `output_variables` entries filter each stream independently;
  omitted entries keep all its variables. PrecipitationAFNO outputs `tp06` here;
  read its actual output signature rather than copying labels from older examples.
- `regridder={"precipitation": regridder}` regrids that output after variable
  filtering. Its `source_grid` must match the diagnostic output grid and storage
  uses its `target_grid`. It does not regrid forecast fields before the diagnostic.
  Runner input mapping selects variables/domain; it is not interpolation. Choose
  a compatible diagnostic or explicitly adapt inputs before calling it.
- For this pair, both streams have leads 0, 6, ..., 48 h. Lead zero is a diagnostic
  of the initial condition, not a forecast step. `tp06` is a six-hour accumulation
  at each valid time, not a cumulative total since initialization. In general,
  the runner publishes the initial condition only when input ends at zero and
  output contains all input variables on the same grid; otherwise streams start
  at the first forecast. See `references/troubleshooting.md` for output-only
  diagnostic inputs and release differences.

For resume, pass `progress=ProgressStore(path)` to `Pipeline.from_model` and reuse
the full item list and stores; completed items are skipped, interrupted items
rerun in full. For multi-GPU runs, initialize a process group before `run` and
follow `dev/examples/09_pipeline.py` for rank-safe backend creation.

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
- Add diagnostic postprocessing with Pipeline (section 8)
- For probabilistic runs, use the ensemble workflow

## Ownership

**Owns:** model selection, data source compatibility, IO backend selection,
nsteps calculation, diagnostic postprocessing of single-member forecasts,
generating `deterministic` / `Pipeline` scripts.

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
- **Horizon must be divisible** by the model step; Pipeline takes a duration.
- **ZarrBackend is the default.** Suggest alternatives only for a specific need.
