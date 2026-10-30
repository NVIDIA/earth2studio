# Connector Patterns

Recipes for common exchanges. `H = np.timedelta64(1, "h")`. Each consumer slot
takes at most one forcing connector; every unconnected slot is fetched from its
source.

## Timing

| Need | Consumer declares | Connector |
| --- | --- | --- |
| Producer's field at the same valid time (consumer runs after it) | plain label, e.g. `["z500"]` at the step's lead | `Connector("a", "b")` |
| Latest field published earlier (lag; breaks feedback loops) | plain label | `Connector("a", "b", time="hold")` |
| Reduction of a faster producer over the consumer's step | qualified label `"tp:sum:24h"`, `"t2m:mean:24h"` | `Connector("a", "b")` (`exact` only) |
| Earlier frames on purpose (history) | slot `lead_time` values before the step's lead | `Connector("a", "b")` |

- Qualified labels reduce frames in `[T - window, T)`. The window must be a
  multiple of the producer's frame spacing.
- `hold` applies to plain labels only. For a lagged reduction, have the consumer
  declare the slot at an earlier lead time instead.
- Two `exact` connectors in a loop are a cycle (C11). Make one of them `hold`.
  The model order in the components dict only breaks ties between components
  that are ready at the same tick.

Same cadence, run in sequence (StormScope GOES feeding MRMS conditioning):

```python
runner = CoupledRunner({"goes": goes, "mrms": mrms}, [Connector("goes", "mrms")])
```

Slow ocean and fast atmosphere, with feedback:

```python
connectors = [
    Connector("atmos", "ocean", regridder=to_ocean),             # "z1000:mean:48h"
    Connector("ocean", "atmos", regridder=to_atmos, time="hold"),  # "sst"
]
```

## Variables

```python
Connector("a", "b", variables={"t2m": "temperature_2m"})  # dst label -> src variable
Connector("a", "b", variables=["sst"])                    # identity: same as {"sst": "sst"}
```

A qualified label renames its base: `{"tp": "precip"}` makes `tp:sum:24h`
reduce the producer's `precip`. Each base must be published by exactly one of the
producer's output slots (C4).

## Grids

A regridder is required whenever the grids differ (C5). Its source grid must
match the producer's output grid and its target grid the consumer's slot grid.

```python
from earth2studio.grids import BilinearRegridder, NearestRegridder, infer_grid

src = infer_grid(producer.output_coords(producer.input_coords()))
dst = infer_grid(consumer.forcing_coords())   # [slot] if it returns a tuple
Connector("a", "b", regridder=BilinearRegridder(src, dst))

# Land-only or ocean-only fields: take the nearest *valid* source point.
Connector("ocean", "atmos", regridder=NearestRegridder(src, dst, source_mask=ocean_mask))
```

Models built from the same region view (`model.with_region(region)`) share a grid
and need no regridder. Time interpolation and conservative regridding are not
supported.

## Conversions: diagnostics between models

Connectors select, reduce and regrid; they never convert. Derived variables
(`ws10m` from `u10m`/`v10m`), unit changes and vertical levels are a
`DiagnosticModel` component placed between the producer and the consumer. It
runs at each valid time its inputs arrive, needs at least one `exact` connector
(C2), and publishes its own stream.

```python
from earth2studio.models.dx import DerivedWS

ws = DerivedWS(levels=["10m"], grid=grid)
runner = CoupledRunner(
    {"a": a, "ws": ws, "b": b},
    [Connector("a", "ws"), Connector("ws", "b")],
)
```

```python
from earth2studio.models.dx import DerivedPressureLevels

# Model levels 50-137 (t50k, ...) plus sp; a, b per level or per interface.
levels = DerivedPressureLevels(
    ["t", "u", "v"], [850, 500], model_levels=range(50, 138), a=a, b=b, grid=grid
)
runner = CoupledRunner(
    {"atmos": atmos, "levels": levels, "downstream": downstream},
    [Connector("atmos", "levels"), Connector("levels", "downstream")],
)
```

Fusing two cadences in one model is not supported: split it into components,
as with DLESyM's atmosphere and ocean.

## State overwrite

Use this when a model carries a coupled field in its own state, such as SST in
an atmosphere model's inputs and outputs. The connector overwrites the named
variables before each call, and the overwrite is not published.

```python
Connector("ocean", "atmos", state=True, variables=["sst"], regridder=to_atmos)
```

Requirements:

- `variables` is required.
- The target slot is an input slot whose output slot in the same position
  declares those variables on the same grid.
- Several state connectors may share a slot if their variables differ.
- Values the producer cannot supply yet are kept, not fetched.

## Sources

```python
sources = {
    "atmos": GFS(),                         # covers the first input slot only
    "land": (land_ic, None, retrievals),    # input, forcing[0], forcing[1]
}
```

- Slots are ordered inputs first, then forcing, as in `default_sources()`.
- `None`, or a slot left out, falls back to the model's recommendation.
- Connected slots use their source only for frames at or before the item time
  that the producer does not hold.
- DataFrame slots are always fetched, and an empty frame arrives as `None`.

## Devices and seeds

```python
runner = CoupledRunner(models, connectors, devices={"impact": "cpu"})
runner.to("cuda")   # moves unpinned components; Pipeline(device=...) does this
runner.device_of("impact")
```

Each component's fetches and connected fields are staged onto its own device,
and regridding runs there. Stochastic components are seeded per member from
`seed`, offset per component name, so they draw independent noise.

## Streams

- Each output slot is one stream: `name` for a component with one output slot,
  otherwise `name.0`, `name.1`, ….
- Initial conditions are published only when the output slot declares the input
  slot's variables on the same grid. Otherwise the stream starts at the first
  forecast.
- `Pipeline(runner, output={stream: backend, ...})` takes one store per stream.
