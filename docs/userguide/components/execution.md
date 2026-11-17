# Pipelines and Runners { #execution_userguide }

Execution tools in `earth2studio.run` make it easy to run workflows composing
multiple components (models, data sources, etc.) over many dates and ensemble
members. A `Runner` executes one forecast, called a work item, and yields its
outputs. A `Pipeline` drives a runner over a list of work items and handles the
rest: distribution across ranks, output filtering, regridding, storage, scoring,
and resuming.

The models themselves can be called directly; see
[Prognostic Models](prognostic.md). Use `Pipeline` when you want this bookkeeping
handled for you, or to run at scale.

## Interface

### Work Items

A `WorkItem` describes one forecast: a reference `time`, a `final_lead_time`, and
optional ensemble `member_ids`. Build items with `work_items`:

```python
import numpy as np

from earth2studio.run import work_items

times = [np.datetime64("2024-01-01T00"), np.datetime64("2024-01-02T00")]
items = work_items(times, final_lead_time=np.timedelta64(240, "h"))
ensemble = work_items(
    times, final_lead_time=np.timedelta64(240, "h"), members=8, member_batch=4
)
```

`member_batch` sets how many members each item runs together. Each item produces
lead times from zero through its final lead time, which must be a multiple of the
model's step.

### Runners

A `Runner` defines the core execution loop for a `WorkItem` and declares its inputs
(data fetches) and outputs (prediction fields). Runners group outputs into
distinct **streams**: each stream is a named sequence of outputs with one set of
variables, one grid, and one lead-time pattern.

| Member | Purpose |
| --- | --- |
| `supports_member_batching` | Whether one item may carry several members. |
| `to(device)` | Place models on a device, and fetch inputs onto it. |
| `output_coords(final_lead_time)` | Declare each stream's output signature without running. |
| `data_requests(item)` | Describe every fetch an item will make, or `None` if not known upfront. |
| `run_item(item)` | Yield one mapping per step, from stream name to output field. |

A runner is built once with its models and sources, then reused for every item;
data is fetched, from live or predownloaded sources, inside `run_item`.

The built-in runners are:

- `PrognosticRunner` rolls out one prognostic model, yielding the `forecast`
  stream. Each diagnostic in `diagnostics` adds a stream under its key. It accepts
  an optional `perturbation` for initial-condition ensembles.
- `DiagnosticRunner` applies diagnostics directly to source data. Step `k`
  reads the source at the item's time plus `k * step`.
- `CoupledRunner` (`earth2studio.coupling`) runs several models together;
  see [Coupled Models](#coupled-models).

Sources default to each model's `default_sources()`. The protocol is defined in
`earth2studio.run.runner.Runner`; inheritance is not required.

### Pipeline

A `Pipeline` provides the outer orchestration layer around a `Runner`, managing
distributed execution of work items, tracking progress, and writing each output
stream to its store.

| Argument | Purpose |
| --- | --- |
| `runner` | Runner executing each item. |
| `output` | IO backend per stream; streams left out are not written. |
| `output_variables` | Variables kept per stream. |
| `regridder` | Regridder per stream, applied after variable selection. |
| `scorer` | Online scoring hook, called with each output before it is written. |
| `progress` | `ProgressStore` recording finished items so a rerun resumes. |
| `device` | Runner device; `"auto"` selects `cuda:{LOCAL_RANK}` or CPU. |

At least one of `output` and `scorer` is required.

Construct a `Pipeline` explicitly from a runner and per-stream settings, or use
`Pipeline.from_model(model, source, ...)` to build one for a single prognostic
model, as shown below.

See [E2S-11: Execution Contracts](../../specs/E2S-11_EXECUTION_CONTRACT_SPEC.md)
for the full work item, runner, pipeline, seeding, and progress requirements.

## Usage

### Run a Forecast

Given a loaded prognostic model, `from_model` runs it from a data source and writes
the `forecast` stream:

```python
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import Pipeline

Pipeline.from_model(
    model,
    GFS(),
    output={"forecast": ZarrBackend("forecast.zarr")},
    output_variables={"forecast": ["t2m", "u10m", "v10m"]},
).run(items)
```

The store is planned from all items before any runs, so its `time` axis spans every
initialization. The initial condition is written at lead time zero when the model's
outputs include its inputs. Pass ensemble items to write a store with an `ensemble`
dimension.

### Add Diagnostics or Use a Runner Directly

Each diagnostic yields its own stream, written to its own backend:

```python
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.perturbation import Gaussian
from earth2studio.run import PrognosticRunner

precip = PrecipitationAFNO.load_model(PrecipitationAFNO.load_default_package())
runner = PrognosticRunner(
    model,
    GFS(),
    diagnostics={"precip": precip},
    perturbation=Gaussian(noise_amplitude=0.05),
)
Pipeline(
    runner,
    output={"forecast": forecast_io, "precip": precip_io},
).run(ensemble)
```

Each member's perturbation is seeded from the runner's `seed`, the item time, and
the member ID, so results are reproducible regardless of how members are batched.
Stochastic models are seeded once per item; use `member_batch=1` to reproduce each
member independently.

Runners can also be used on their own, without `Pipeline`:

```python
for step in runner.run_item(items[0]):
    print(step["forecast"].lead_time.values)
```

### Resume and Distribute

A `ProgressStore` marks an item done once its outputs are flushed. Running the
same items again skips finished ones:

```python
from earth2studio.run import ProgressStore

progress = ProgressStore("out/.progress", tags={"model": "SFNO"})
Pipeline.from_model(model, GFS(), output=output, progress=progress).run(items)
```

The store refuses to resume a run whose stream layout, seed, member grouping, or
tags differ. It cannot tell models apart, so record the model and checkpoint in
`tags`. An interrupted item reruns in full.

For multi-GPU runs, launch the script with `torchrun` and initialize a process
group before calling `run`. Each rank runs a contiguous slice of the remaining
items, and rank 0 creates the stores. Build backends on every rank, letting rank 0
create storage first:

```python
from earth2studio.run.work import rank0_first

output = rank0_first(lambda: {"forecast": ZarrBackend("out/forecast.zarr")})
```

Distributed stores must use chunks of 1 along `time` and `ensemble`, which is the
`ZarrBackend` default; `Pipeline` checks this before running.

## Coupled Models

`CoupledRunner` runs several models as named components. `Connector` declarations
describe which component feeds another's inputs, including variable mapping, time
selection, and optional regridding. Each component runs at its own cadence and
writes its own stream; inputs not fed by a connector are fetched from sources.

Because `CoupledRunner` is a runner, the same `Pipeline` drives it. For example,
DLESyM's atmosphere and ocean, stepping every 6 and 48 hours:

```python
from earth2studio.coupling import Connector, CoupledRunner

runner = CoupledRunner(
    {"atmos": atmos, "ocean": ocean},
    [
        Connector("ocean", "atmos", time="hold"),  # Latest SST before each cycle
        Connector("atmos", "ocean"),               # Atmospheric means per ocean step
    ],
)
print(runner.describe(np.timedelta64(192, "h")))  # Inspect the derived schedule
Pipeline(runner, output={"atmos": atmos_io, "ocean": ocean_io}).run(items)
```

See [E2S-12: Coupling](../../specs/E2S-12_COUPLING_SPEC.md) for connector and
scheduling requirements, and the DLESyM seasonal example for a complete coupled
run.

## Write Your Own Runner

Any object with the four runner members works with `Pipeline`. Wrap a built-in
runner to change its behavior, such as stopping a rollout early, or write one for
a workflow the built-ins do not cover. Keep stream names stable across items and
runs, since they key output stores and resume. Declare every stream in
`output_coords`, and give outputs with different grids or cadences separate
streams. Return `None` from `data_requests` when fetches depend on results.
