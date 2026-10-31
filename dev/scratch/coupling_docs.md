<!-- markdownlint-disable -->
# Coupling: superseded documentation drafts

Collected from `earth2studio/nvcoupler/docs/` and its README as they stood on
PR #1229 before the package was rebuilt on the 1.0 contracts (see
`dev/spec/COUPLING_SPEC.md`). Kept for reference only: the `Component`,
`Field`/`State`, field dictionary, mediator, pull-adapter, YAML and run-sequence
DSL APIs described below no longer exist. The concepts sections (coupling
modes, cadences, NUOPC analogies) remain useful background for the 1.0 docs
write-up.

---

# nvcoupler

`earth2studio.nvcoupler` is a Python-native framework for coupled
Earth-system **inference**. Components exchange labeled `xarray.DataArray`
fields by standard name, while a `Driver` executes an explicit run sequence
on a shared clock.

The public data boundary is DataArray-only. Array payloads may use NumPy or
CuPy; backend choice is inferred from `DataArray.data`.

```python
import earth2studio.nvcoupler as nvc
```

## Quickstart

```python
from earth2studio.nvcoupler.testing import (
    atmos_ic,
    fake_atmos,
    fake_ocean,
    ocean_ic,
)

driver = nvc.couple(
    fake_atmos(),
    fake_ocean(),
    start="2024-01-01",
    stop="2024-01-05",
)
driver.initialize({
    "atmos": atmos_ic(),
    "ocean": ocean_ic(),
})
datasets = driver.run()  # dict[str, xr.Dataset]
```

Initial conditions are `dict[str, xr.DataArray]`. `Driver.run()` executes the
remaining clock and returns in-memory `xarray.Dataset` objects when
`collect=True`; `Driver.to_xarray()` exposes the same collected records.
There is no streaming output API.

## Core model

- `Field(array=..., standard_name=..., units=...)` wraps one DataArray. A
  Field cannot contain a `variable` dimension.
- `State` is a mapping of standard name to `Field`.
  `State.from_dataarray(...)` splits a multi-variable DataArray and
  `State.stack(...)` rebuilds one.
- `FieldDictionary` resolves raw aliases to canonical names, checks units,
  and describes derived fields with `CellMethod`.
- `Component` implementations advertise imports and exports. All callable
  and model interfaces consume and return `xr.DataArray`.
- `Connector` transfers fields through time policy, vertical interpolation,
  mask filling, and spatial regridding.
- `Driver` schedules components and connectors using a run-sequence DSL.

Components use `earth2studio.grids.GridDefinition`. Pass a registered grid
name or a grid object through `grid=`; use
`earth2studio.grids.PointGrid` for scattered destinations. Point connectors
require `sample="nearest"` or `sample="bilinear"`.

## Components and adapters

- `CallableComponent`: wraps `fn(state: xr.DataArray) -> xr.DataArray`.
- `PrognosticComponent`: wraps a DataArray-protocol prognostic model.
- `DiagnosticComponent`: stacks imported fields and calls a DataArray
  diagnostic model.
- `DataComponent`: fetches prescribed data into DataArrays.
- `VariableOverwriteAdapter`: injects imports into matching state variables.
- `ConditioningKwargAdapter`: calls a model conditioning method with a
  stacked conditioning DataArray.
- `PullAdapter`: serves imports through a temporary in-memory data source.

The package is inference-only. It does not provide training or gradient
preservation.

## Connector behavior

`Connector(src, dst)` matches the intersection of advertised standard names,
or an explicit `fields=[...]`. Processing order is:

1. constant or linear time policy;
2. vertical interpolation;
3. optional mask fill;
4. spatial regridding or point sampling.

Automatic spatial interpolation supports regular lat/lon sources. For other
grids, pass `regridder=`, a callable with the contract
`xr.DataArray -> xr.DataArray`. Vertical interpolation is likewise
DataArray-native and supports `PressureLevels` destinations from pressure or
hybrid sources.

Set both `window=` and `reduce=` for a running mean, sum, maximum, or minimum
that delivers a dictionary-declared derived field at window boundaries.

## Run-sequence DSL

Ordering determines coupling semantics: a connection before its source runs
is lagged; a connection after its source runs is sequential.

```text
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
```

`Driver(sequence=None)` and `couple()` derive a canonical lagged sequence.
Use explicit DSL when ordering is part of the experiment.

## Scope and limitations

- NumPy and CuPy payloads only.
- In-memory xarray output only.
- Units are checked, not converted.
- Automatic regridding is limited to regular lat/lon and point sampling.
- No checkpoint/restart, concurrent component execution, training API, or
  streaming IO.
- YAML stores configuration, not initial conditions or custom callable
  regridders.

## Documentation

- [Concepts](docs/concepts.md)
- [User guide](docs/user_guide.md)
- [API reference](docs/api_reference.md)
- [Run-sequence DSL and YAML](docs/dsl_and_yaml_reference.md)
- [Errors and troubleshooting](docs/errors_and_troubleshooting.md)
- [Design and roadmap](docs/design_and_roadmap.md)
# Concepts

## DataArray is the public data model

nvcoupler public APIs exchange `xarray.DataArray` objects. A DataArray carries
dimension names, coordinates, attributes, and a NumPy or CuPy payload. The
coupler infers the payload backend; users do not pass a backend flag.

`Field` represents one physical quantity:

```python
import numpy as np
import xarray as xr
import earth2studio.nvcoupler as nvc

array = xr.DataArray(
    np.full((8, 16), 290.0),
    dims=("lat", "lon"),
    coords={
        "lat": np.linspace(90, -90, 8),
        "lon": np.linspace(0, 360, 16, endpoint=False),
    },
)
field = nvc.Field(
    array=array,
    standard_name="sea_surface_temperature",
    units="K",
    valid_time=np.datetime64("2024-01-01"),
)
```

A Field cannot have a `variable` dimension. `State` stores Fields by standard
name and converts at the multi-variable boundary:

```python
state = nvc.State.from_dataarray(
    "initial",
    multi_variable_array,
    nvc.DEFAULT_DICTIONARY,
)
array = state.stack(["sea_surface_temperature", "air_temperature_2m"])
```

`State.from_dataarray` resolves each variable coordinate through the field
dictionary. `strict=False` skips unknown variables. `State.stack` requires
identical dimensions and coordinates and creates the `variable` axis before
the first spatial dimension.

## Standard names and derived fields

Components advertise canonical standard names. `FieldDictionary` maps raw
model names to those names and checks canonical units. Values are not
converted.

`CellMethod(base, method, window)` describes a derived field such as a 48-hour
mean. Windowed connectors and accumulation mediators use this declaration;
they do not infer semantics from name suffixes.

## Components

Every component follows:

`advertise -> realize -> initialize -> run* -> finalize`

- `CallableComponent` owns a DataArray state and calls
  `fn(xr.DataArray) -> xr.DataArray`.
- `PrognosticComponent` requires DataArray `input_coords()` and
  `output_coords(...)` signatures and a model call returning a DataArray.
- `DiagnosticComponent` stacks imports in model-variable order and returns a
  DataArray diagnostic output.
- `DataComponent` fetches prescribed data and publishes DataArray fields.
- `Mediator` accumulates connector deliveries and emits derived fields.

Stateful components receive one `xr.DataArray` in
`Driver.initialize({"name": initial_array})`. Data and diagnostic components,
and mediators, do not require an initial condition.

### Import adapters

An adapter owns the model invocation:

- `VariableOverwriteAdapter` replaces matching variables in the state and
  calls `model(state)`.
- `ConditioningKwargAdapter` stacks imports and calls
  `model.<method>(state, conditioning)`.
- `PullAdapter` installs a `StateDataSource` and calls `model(state)`.

For more than one conditioning field, provide `field_order`; channel order is
not guessed.

## Grids

Components expose an `earth2studio.grids.GridDefinition` through `grid=` or
grid inference from their state. Registered names are resolved with
`earth2studio.grids.resolve_grid`.

`PointGrid(latitude, longitude, x=None)` is the scattered-location grid. Its
spatial dimension is `x`, with latitude and longitude auxiliary coordinates.
A connector targeting a PointGrid needs `sample="nearest"` or
`sample="bilinear"`.

## Connectors

A connector matches source exports to destination imports by standard name.
Each transfer applies:

1. time policy;
2. vertical conversion;
3. mask fill;
4. spatial transform.

`time_policy="constant"` holds the newest source field.
`time_policy="linear"` extrapolates from two distinct valid times and falls
back to constant when history is insufficient or a field contains
`lead_time`/`window`.

Masks are boolean DataArrays where `True` is valid. `fill="zero"` or
`fill="nearest"` consumes the mask before regridding.

Automatic regridding supports regular one-dimensional latitude/longitude.
Custom regridders must have this exact contract:

```python
def regrid(array: xr.DataArray) -> xr.DataArray:
    ...
```

The returned DataArray must carry the destination dimensions and coordinates.

### Vertical interpolation

Fields may declare `PressureLevels` or `HybridLevels`. When a destination
requests different pressure levels, the connector calls the DataArray-native
`interp_to_pressure`. Hybrid interpolation also needs the source's surface
pressure field. Values are interpolated linearly in log pressure and clamped
at column ends.

### Windowed exchange

`Connector(..., window="48h", reduce="mean")` accumulates source fields and
delivers a matching derived import on each boundary. Supported reductions are
`mean`, `sum`, `max`, and `min`. Use an `AccumulationMediator` when several
sources or custom reduction behavior are required.

## Time and ordering

The clock step is the finest cadence. Component timesteps and sequence slots
must be positive multiples of it. Initial conditions are published at the
clock start; actions begin at `start + dt`.

Coupling mode is ordering:

- connect before source run: lagged;
- source run before connect: sequential.

Derived sequences use lagged connections by default. Explicit DSL is the
mechanism for changing that behavior.

## Results

With `collect=True`, the Driver stores cloned Fields in memory. `run()` and
`to_xarray()` return `dict[str, xr.Dataset]`, one Dataset per component, with
component ring times on the `time` axis. `steps()` yields live export States
after each driver step.

This is an inference runtime. It has no training API and no streaming output
backend.
# User guide

## Couple two components

```python
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import (
    atmos_ic,
    fake_atmos,
    fake_ocean,
    ocean_ic,
)

driver = nvc.couple(
    fake_atmos(),
    fake_ocean(),
    start="2024-01-01",
    stop="2024-01-05",
)
driver.initialize({
    "atmos": atmos_ic(),
    "ocean": ocean_ic(),
})
datasets = driver.run()
```

Initial conditions are DataArrays keyed by component name. Results are
in-memory Datasets keyed by component name.

Use `driver.describe()` before initialization to inspect fields, cadences,
connector policies, and the derived sequence.

## Write a callable component

The callable receives and returns one `xr.DataArray`:

```python
import xarray as xr

def step(state: xr.DataArray) -> xr.DataArray:
    temperature = state.sel(variable="t2m", drop=True) + 0.5
    return xr.concat(
        [temperature],
        xr.IndexVariable("variable", ["t2m"]),
    ).transpose(*state.dims)

component = nvc.CallableComponent(
    "surface",
    step,
    timestep="6h",
    exports=["air_temperature_2m"],
)
```

The initial condition must contain a `variable` coordinate whose raw names
resolve through the component dictionary or `variable_aliases`.

## Wrap a prognostic or diagnostic model

A prognostic model must expose DataArray signatures and return a DataArray:

```python
class Model:
    def input_coords(self) -> xr.DataArray:
        ...

    def output_coords(self, input_signature: xr.DataArray) -> xr.DataArray:
        ...

    def __call__(self, state: xr.DataArray) -> xr.DataArray:
        ...

component = nvc.PrognosticComponent(
    "atmos",
    Model(),
    imports=["sea_surface_temperature"],
    variable_aliases={"model_sst": "sea_surface_temperature"},
)
```

If `timestep` is omitted, it is inferred from the final input and output
`lead_time` coordinates. `next_input(previous, output) -> xr.DataArray`
customizes sliding-window reconstruction.

`DiagnosticComponent` follows the same DataArray model contract, but is
stateless: it stacks its imports in the model's declared variable order,
conforms them to the input signature, and publishes the output.

## Choose import behavior

The default `VariableOverwriteAdapter` replaces matching variables in the
component state before calling the model.

Use conditioning input when imports are not state channels:

```python
adapter = nvc.ConditioningKwargAdapter(
    field_order=["eastward_wind_10m", "air_temperature_2m"],
)
component = nvc.PrognosticComponent(
    "regional",
    model,
    imports=["eastward_wind_10m", "air_temperature_2m"],
    import_adapter=adapter,
)
```

This calls `model.call_with_conditioning(state, conditioning)`. Configure a
different method name with `method=`.

`PullAdapter` supports models that fetch from a settable data-source
attribute. It installs a `StateDataSource` over current imports and then calls
`model(state)`. Place the connector before the pulling component's run when
fresh forcing is required; `strict_time=True` checks alignment.

## Build a Driver explicitly

```python
atmos, ocean = fake_atmos(), fake_ocean()
driver = nvc.Driver(
    {"atmos": atmos, "ocean": ocean},
    clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=[
        ("ocean", "atmos"),
        nvc.Connector(atmos, ocean, window="48h", reduce="mean"),
    ],
)
```

With `sequence=None`, the Driver derives a lagged sequence. Pass DSL text when
ordering must be explicit:

```python
sequence = """
@6h
  atmos -> ocean
  ocean -> atmos
  atmos
@48h
  ocean
@
"""
```

A connect before its source run is lagged. A connect after the source run is
sequential.

## Regrid and sample

Components accept `grid=` as a registered Earth2Studio grid name or a
`GridDefinition`. If no grid is supplied, the component attempts to infer one
from its state or model signature.

For scattered targets:

```python
from earth2studio.grids import PointGrid

sites = PointGrid(
    latitude=[40.0, 41.5],
    longitude=[-105.0, -104.2],
    x=["site-a", "site-b"],
)
target = nvc.CallableComponent(
    "sites",
    site_step,
    timestep="6h",
    imports=["air_temperature_2m"],
    grid=sites,
)
connector = nvc.Connector(source, target, sample="bilinear")
```

Automatic mesh interpolation requires a regular lat/lon source. For any
other layout, provide a DataArray-native regridder:

```python
def regrid(array: xr.DataArray) -> xr.DataArray:
    # Return destination dimensions and coordinates.
    ...

connector = nvc.Connector(source, target, regridder=regrid)
```

## Masks

Declare export masks as DataArrays or arrays, with `True` meaning valid:

```python
component = nvc.CallableComponent(
    "ocean",
    step,
    timestep="24h",
    exports=["sea_surface_temperature"],
    export_masks={"sea_surface_temperature": ocean_mask},
)
connector = nvc.Connector(component, atmosphere, fill="nearest")
```

`fill="nearest"` or `"zero"` runs before regridding and clears the delivered
mask. The default `"none"` preserves invalid values.

## Vertical interpolation

```python
hybrid = nvc.HybridLevels(
    a=(30000.0, 20000.0, 0.0),
    b=(0.0, 0.5, 1.0),
)
pressure = nvc.PressureLevels((500.0, 850.0))

source = nvc.CallableComponent(
    "met",
    step,
    "6h",
    exports=["ozone_mixing_ratio", "surface_pressure"],
    export_vertical={"ozone_mixing_ratio": hybrid},
)
target = nvc.CallableComponent(
    "chem",
    step,
    "6h",
    imports=["ozone_mixing_ratio"],
    import_vertical={"ozone_mixing_ratio": pressure},
)
```

The connector performs DataArray log-pressure interpolation. Hybrid sources
must publish the surface-pressure field named by `HybridLevels.ps_field`.

## Prescribed data

```python
forcing = nvc.DataComponent(
    "forcing",
    source=data_source,
    exports=["sea_surface_temperature"],
    timestep="24h",
    variable_map={"sea_surface_temperature": "sst"},
    target_grid=destination_grid,
)
```

`DataComponent` fetches at the clock start and its own cadence. It does not
need an initial-condition entry.

## Derived fields

Declare a derived field with `CellMethod`, add it to the destination imports,
then use a windowed connector:

```python
dictionary = nvc.FieldDictionary(nvc.DEFAULT_DICTIONARY)
dictionary.register(
    nvc.FieldEntry(
        "air_temperature_2m_24h_max",
        "K",
        cell_method=nvc.CellMethod(
            "air_temperature_2m", "max", np.timedelta64(24, "h")
        ),
    )
)
connector = nvc.Connector(source, target, window="24h", reduce="max")
```

Use `AccumulationMediator` when reductions need a distinct participant or
multiple sources.

## Inspect and rerun

- `driver.steps()` yields `(time, export_states)`.
- `driver.probe("source->target")` returns the last transferred Fields.
- `driver.to_xarray()` returns collected in-memory Datasets.
- `driver.reset()` rewinds time and clears records and connector state; call
  `initialize(...)` again before rerunning.

Set `collect=False` to disable in-memory collection. In that case `run()`
returns `{}`. There is no streaming output path.

## YAML

`to_yaml` stores the clock, sequence, dictionary deltas, importable component
factories, and serializable connector options. A component wrapping a live
callable or model needs a `yaml_spec` that reconstructs it from an import
path and keyword arguments.

```python
text = nvc.to_yaml(driver)
rebuilt = nvc.from_yaml(text)
rebuilt.initialize(initial_conditions)
```

Initial-condition DataArrays, custom regridder callables, and runtime state
are not serialized.
# Design and roadmap

## Purpose

nvcoupler separates coupling policy from model implementation. Components
advertise physical fields, connectors transform and transfer them, mediators
reduce them across cadences, and a Driver executes an ordered schedule.

The package borrows these boundaries from NUOPC/ESMF while remaining a
single-process Python inference runtime.

## Hard-cut DataArray design

The public boundary is `xarray.DataArray`:

- Fields store `array=...`.
- States split with `State.from_dataarray` and combine with `State.stack`.
- Callable and model interfaces accept and return DataArrays.
- Initial conditions are `dict[str, xr.DataArray]`.
- Custom regridders and vertical interpolation accept and return DataArrays.
- Collected results are in-memory `xr.Dataset` objects.

The hard cut avoids parallel compatibility layers and makes labels,
coordinates, dimensions, and attributes part of every exchange contract.

NumPy and CuPy are the supported payloads. Operations inspect
`DataArray.data` and choose the matching array namespace. This package is
inference-only; preserving differentiation graphs and training coupled models
are outside its scope.

## Key decisions

### Standard-name exchange

Connectors match standard names rather than raw model vocabulary. Aliases
belong in `FieldDictionary`; units are checked at match time. Values are not
silently converted.

### Model calls belong to adapters

Different models receive coupled fields differently. The adapter owns that
invocation while every shape remains DataArray-native:

- overwrite state variables;
- pass a conditioning DataArray;
- install an in-memory source for a model that pulls forcing.

This keeps component scheduling independent of a model's call convention.

### Grids use Earth2Studio definitions

Components use `earth2studio.grids.GridDefinition`, including `PointGrid` for
scattered locations. Grid inference and registered names are shared with the
rest of Earth2Studio.

Automatic interpolation intentionally covers the common regular-lat/lon and
point cases. Other geometries use an explicit
`xr.DataArray -> xr.DataArray` regridder, keeping ownership of specialized
grid semantics with the caller.

### Ordering defines coupling mode

There is no separate connector mode flag for lagged versus sequential
coupling. A connect before the source run is lagged; a connect after it is
sequential. The DSL therefore records the full coupling semantics.

### Derived fields are metadata

Window means, sums, maxima, and minima are represented by dictionary
`CellMethod` metadata. Windowed connectors and mediators consume that
metadata instead of parsing names.

### Results remain in memory

The Driver optionally collects cloned exports and builds xarray Datasets.
There is no second output lifecycle for streamed backends. `collect=False`
disables records for callers that manage output outside nvcoupler.

## Current limitations

- Single-process, ordered execution.
- No checkpoint/restart or ensemble-aware Driver.
- No streaming output API.
- No training or gradient-preservation API.
- Automatic spatial interpolation requires regular lat/lon sources.
- Units are checked but not converted.
- YAML cannot serialize live model objects, closures, custom adapters, custom
  regridder callables, runtime state, or initial-condition DataArrays.
- Pull-style data sources currently serve `(lat, lon)` Fields.

## Roadmap

Potential future work, subject to concrete use cases:

1. checkpoint/restart for component, connector, and mediator state;
2. broader Earth2Studio grid transformations;
3. unit conversion integrated with field metadata;
4. ensemble-aware scheduling and result organization;
5. richer serializable component factories;
6. concurrent execution where ordering dependencies permit it.

Any future extension should preserve the DataArray-only public contract and
keep coupling decisions visible in configuration.
# API reference

Public symbols are available from `earth2studio.nvcoupler` unless noted.
Public data-bearing arguments and returns use `xarray.DataArray`.

## Fields

### `Field`

```python
Field(
    array: xr.DataArray,
    standard_name: str,
    units: str,
    valid_time: np.datetime64 | None = None,
    source: str | None = None,
    mask: xr.DataArray | None = None,
    vertical: VerticalCoordinate | None = None,
)
```

One exchanged quantity. `array` cannot contain `variable`. `data` exposes the
NumPy/CuPy payload. Methods: `with_array(array)`, `to_backend("numpy" |
"cupy")`, `clone()`, and `grid_signature()`.

### `State`

```python
State(name: str, fields: Iterable[Field] = ())
State.from_dataarray(
    name, array: xr.DataArray, dictionary,
    valid_time=None, source=None, strict=True,
) -> State
state.stack(names: list[str] | None = None) -> xr.DataArray
```

A mutable mapping keyed by standard name. Also provides `add(field,
replace=True)` and `subset(names)`.

## Dictionary

```python
CellMethod(base: str, method: Literal["mean", "sum", "max", "min"], window)
FieldEntry(
    standard_name: str,
    canonical_units: str,
    description: str = "",
    aliases: frozenset[str] = frozenset(),
    cell_method: CellMethod | None = None,
)
FieldDictionary(entries: FieldDictionary | list[FieldEntry] | None = None)
```

`FieldDictionary` provides `register`, `add_alias`, `resolve`,
`standard_name`, `standard_names`, `check_units`, and `derived_from`.
`DEFAULT_DICTIONARY` is the built-in vocabulary.

## Components

```python
Component(
    name,
    timestep,
    imports=(),
    exports=(),
    dictionary=None,
    variable_aliases=None,
    export_masks=None,
    import_vertical=None,
    export_vertical=None,
    grid: str | GridDefinition | None = None,
)
```

`grid` is an `earth2studio.grids.GridDefinition` or registered grid name.
`grid_definition()` returns the explicit or inferred grid.

```python
CallableComponent(
    name,
    fn: Callable[[xr.DataArray], xr.DataArray],
    timestep,
    imports=(),
    exports=(),
    import_adapter=None,
    **component_kwargs,
)
```

```python
PrognosticComponent(
    name,
    model,
    timestep=None,
    imports=(),
    exports=None,
    import_adapter=None,
    next_input: Callable[[xr.DataArray, xr.DataArray], xr.DataArray] | None = None,
    **component_kwargs,
)
```

The model's `input_coords()` and `output_coords(...)` return DataArray
signatures; calling the model returns a DataArray. `to(device)` forwards to
the model.

```python
DiagnosticComponent(
    name, model, timestep, imports=None, exports=None, **component_kwargs
)
```

The diagnostic model consumes and returns DataArrays.

```python
DataComponent(
    name,
    source,
    exports,
    timestep,
    variable_map=None,
    target_grid: str | GridDefinition | None = None,
    device="cpu",
    **component_kwargs,
)
```

Fetches prescribed fields with earth2studio's DataArray data-source path.

### Adapters

```python
Exchange(state: xr.DataArray, imports: State, std_to_raw={}, time=None)
Exchange.inject() -> xr.DataArray
Exchange.stacked(field_order=None) -> xr.DataArray
```

`ImportAdapter` is a callable protocol:

```python
adapter(model, exchange: Exchange) -> xr.DataArray
```

Built-ins:

- `VariableOverwriteAdapter()`
- `ConditioningKwargAdapter(field_order=None,
  method="call_with_conditioning")`
- `PullAdapter(attribute="conditioning_data_source", strict_time=False,
  dictionary=None)`
- `StateDataSource(state, raw_to_std=None, strict_time=False,
  dictionary=None)`

## Grids

Grid types live in `earth2studio.grids`:

```python
from earth2studio.grids import GridDefinition, PointGrid

PointGrid(latitude, longitude, x=None)
```

`GridDefinition` is a structural protocol. `PointGrid` represents scattered
locations along dimension `x`.

## Connectors

```python
Connector(
    src,
    dst,
    fields=None,
    time_policy: Literal["constant", "linear"] = "constant",
    fill: Literal["none", "zero", "nearest"] = "none",
    regridder: Callable[[xr.DataArray], xr.DataArray] | None = None,
    sample: Literal["nearest", "bilinear"] | None = None,
    window=None,
    reduce: Literal["mean", "sum", "max", "min"] | None = None,
)
```

Methods: `match()`, `execute(time)`, and `reset()`. `last_transfer` stores
post-pipeline Fields. A PointGrid destination requires `sample=` unless a
custom regridder is supplied.

## Vertical coordinates

```python
PressureLevels(levels: tuple[float, ...])
HybridLevels(
    a: tuple[float, ...],
    b: tuple[float, ...],
    ps_field: str = "surface_pressure",
)
```

The implementation function is imported from
`earth2studio.nvcoupler.vertical`:

```python
interp_to_pressure(
    array: xr.DataArray,
    src: PressureLevels | HybridLevels,
    dst: PressureLevels,
    surface_pressure: xr.DataArray | None = None,
) -> xr.DataArray
```

## Mediators

```python
Mediator(name, timestep, imports=(), exports=(), **component_kwargs)
AccumulationMediator(name, fields: list[str], window=None, **component_kwargs)
TrailingAverageMediator(name, fields: list[str], window=None, **component_kwargs)
```

`fields` are derived standard names with dictionary `CellMethod` entries.

## Clock and sequence

```python
Clock(start, stop, dt)
RunAction(component)
ConnectAction(src, dst)
MediateAction(mediator, phase="compute")
Slot(interval, actions=[])
RunSequence(slots)
parse_run_sequence(text) -> RunSequence
derive_sequence(components, connectors=None, lagged="all") -> RunSequence
```

See [DSL and YAML](dsl_and_yaml_reference.md) for grammar and validation.

## Driver and convenience API

```python
Driver(
    components,
    sequence=None,
    clock=None,
    connectors=None,
    collect=True,
    allow_unfed_imports=False,
)
driver.initialize(ics: dict[str, xr.DataArray] | None = None) -> None
driver.run() -> dict[str, xr.Dataset]
driver.steps() -> Iterator[tuple[np.datetime64, dict[str, State]]]
driver.to_xarray() -> dict[str, xr.Dataset]
driver.reset() -> None
driver.probe("src->dst") -> dict[str, Field]
driver.describe() -> str
```

All collected output is in memory.

```python
couple(
    *components,
    start,
    stop,
    dt=None,
    connectors=None,
    collect=True,
) -> Driver
```

Auto-wires unique exporters and creates windowed transfers for compatible
derived imports.

```python
coupled(
    time,
    stop_or_nsteps,
    components,
    ics: dict[str, xr.DataArray],
    dt=None,
    collect=True,
    verbose=True,
) -> dict[str, xr.Dataset]
```

Also exported: `describe(driver)`, `describe_html(driver)`.

## YAML

```python
to_yaml(driver, path=None) -> str
from_yaml(path_or_str) -> Driver
```

The rebuilt Driver is uninitialized. Initial-condition DataArrays and custom
regridder callables are not serialized.

## Errors

All package errors derive from `CouplingError`:
`UnknownFieldError`, `UnmatchedImportError`, `UnitsMismatchError`,
`IncompatibleFieldError`, `VerticalMismatchError`, `CadenceError`,
`AmbiguousCouplingError`, and `SequenceError`.
# Run-sequence DSL and YAML reference

## Run-sequence DSL

Most systems use a sequence derived from components and connectors. Pass DSL
text to `Driver(sequence=...)` when action order is part of the configuration.

```text
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
```

### Grammar

The format is line-oriented:

- `@<interval>` opens a slot.
- `@` closes the current slot; end of input also closes it.
- `name` is `RunAction(name)`.
- `src -> dst` is `ConnectAction(src, dst)`.
- `name.phase` is `MediateAction(name, phase)`.
- `#` begins a comment; blank lines are ignored.

Intervals are strings accepted by `as_timedelta`, including `6h`, `2D`,
`90m`, and `500ms`, or `np.timedelta64` values in the Python API. Bare
numbers are rejected because their units are ambiguous. Avoid calendar month
and year units for fixed coupling intervals.

### Execution

At each clock time after `start`, a slot executes when elapsed time is a
positive multiple of its interval. Aligned slots execute in document order;
actions inside each slot execute in listed order.

- A run action advances a component and publishes exports at the current time.
- A connect action transfers the source's current exports.
- A mediate action computes the mediator's exports.

A connect before the source's run sees the previous export and is lagged. A
connect after the source's run sees the newly produced export and is
sequential.

Windowed connectors accumulate on every connect action and deliver only at
their window boundaries.

### Validation

`Driver.initialize()` validates:

- slot intervals are positive multiples of clock `dt`;
- each component or mediator runs in a slot equal to its timestep;
- action names and connector endpoints exist;
- every component appears in the sequence;
- connector fields and units match;
- every required import is fed, unless `allow_unfed_imports=True`.

`str(RunSequence)` emits parseable DSL with a final `@`. Parsing that output
reconstructs an equivalent sequence.

## YAML schema

`to_yaml(driver)` serializes configuration. `from_yaml(text_or_path)` returns
an uninitialized Driver.

```yaml
clock:
  start: '2024-01-01T00:00:00'
  stop: '2024-01-05T00:00:00'
  dt: 6h

sequence: |-
  @6h
    ocean -> atmos
    atmos
  @48h
    ocean
  @

components:
  atmos:
    class: my_package.factories.make_atmos
    kwargs:
      timestep: 6h
  ocean:
    class: my_package.factories.make_ocean
    kwargs:
      timestep: 48h

connectors:
  - src: ocean
    dst: atmos
    time_policy: constant
    fill: nearest
```

### Top-level keys

- `clock` (required): `start`, `stop`, and `dt`.
- `sequence` (required): DSL text, or a derived-sequence mapping.
- `components` (required): component name to `class` and `kwargs`.
- `dictionary` (optional): non-default `FieldEntry` definitions.
- `aliases` (optional): alias to standard-name additions.
- `connectors` (optional): connector declarations.

A graph-derived sequence is represented as:

```yaml
sequence:
  derived: true
  text: |-
    @6h
      source -> target
      source
      target
    @
```

`text` is informational. Loading re-derives the sequence from components and
connectors.

### Dictionary entries

```yaml
dictionary:
  - standard_name: air_temperature_2m_24h_max
    canonical_units: K
    description: Daily maximum near-surface temperature
    aliases: []
    cell_method:
      base: air_temperature_2m
      method: max
      window: 24h

aliases:
  t2m_daily_max: air_temperature_2m_24h_max
```

### Connector entries

Supported keys are:

- `src`, `dst`;
- optional `fields`;
- `time_policy`: `constant` or `linear`;
- `fill`: `none`, `zero`, or `nearest`;
- optional `sample`: `nearest` or `bilinear` for a PointGrid destination;
- paired `window` and `reduce` for windowed reduction.

```yaml
connectors:
  - src: atmos
    dst: ocean
    fields: [geopotential_at_1000hpa]
    time_policy: constant
    fill: none
    window: 2D
    reduce: mean
```

Custom regridder callables are not serialized. Configure those connectors in
Python.

### Component reconstruction

Each `class` is a dotted import path to a class or factory called with
`kwargs`. Components wrapping closures or live model objects require a
`yaml_spec`:

```python
component.yaml_spec = {
    "class": "my_package.factories.make_component",
    "kwargs": {"name": "atmos", "timestep": "6h"},
}
```

Accumulation mediators serialize automatically. DataArrays used as initial
conditions, model runtime state, and custom adapter objects are not part of
the YAML document.

Grid definitions are component constructor arguments. Primitive,
factory-understood grid names are suitable for YAML. Rich grid objects such
as a `PointGrid` generally require a factory that reconstructs the object
from YAML-safe keyword arguments.

### Loading and running

```python
driver = nvc.from_yaml("system.yaml")
driver.initialize({
    "atmos": atmos_initial_dataarray,
    "ocean": ocean_initial_dataarray,
})
datasets = driver.run()
```
# Errors and troubleshooting

All nvcoupler-specific errors derive from `CouplingError`. Most configuration
errors are raised during construction or `Driver.initialize()`.

## Error classes

### `UnknownFieldError`

A standard name or alias is absent from the field dictionary.

Fix a typo, register a `FieldEntry`, or pass
`variable_aliases={"raw_name": "standard_name"}` to the component.
`State.from_dataarray(..., strict=False)` skips unknown variables when that is
intentional.

### `UnmatchedImportError`

A component import has no delivery path.

Check all three:

1. another component exports the same standard name;
2. the sequence contains the required `source -> destination` action;
3. derived imports have a matching `CellMethod` and windowed connector or
   mediator.

Use `allow_unfed_imports=True` only when stale or absent forcing is deliberate.

### `UnitsMismatchError`

Matched dictionary entries disagree on canonical units. Units are normalized
for cosmetic spelling differences, but values are not converted. Align the
entries or add a converting mediator.

### `IncompatibleFieldError`

A connector cannot reconcile fields or grids. Common causes:

- explicit `fields=` are not advertised by both endpoints;
- no standard names match;
- automatic interpolation received a non-regular or non-lat/lon source;
- latitude and longitude are not trailing dimensions;
- nearest mask fill has no valid points.

For unsupported spatial layouts, pass a custom
`Callable[[xr.DataArray], xr.DataArray]` regridder.

### `VerticalMismatchError`

Vertical metadata is missing or incompatible. Check that:

- the source declares `export_vertical`;
- the destination declares supported `PressureLevels`;
- a hybrid source publishes its surface-pressure field;
- pressure levels increase from top to bottom;
- the DataArray's `level` coordinate matches the declared source levels.

### `CadenceError`

A clock span, component timestep, or sequence slot is not a positive multiple
of driver `dt`, or a component is scheduled in a slot different from its own
timestep. Choose a common divisor for `dt` and align each run action with its
component cadence.

### `AmbiguousCouplingError`

`couple()` found more than one exporter for an import and will not choose.
Build the Driver and Connector explicitly.

### `SequenceError`

The DSL is malformed, references an unknown name, leaves a component
unscheduled, or asks `derive_sequence` to order a same-cadence sequential
cycle. Correct the named action, add the missing run, or make one edge lagged.

## Direct `CouplingError` cases

### DataArray shape and field errors

- `Field.array` must be an `xr.DataArray` without a `variable` dimension.
  Split multi-variable input with `State.from_dataarray`.
- A Field mask must broadcast to the field array.
- `State.stack` requires at least one Field with identical dimensions and
  coordinates.
- `Exchange.inject` requires a `variable` state dimension and imported names
  that map to state variables.
- Conditioning several imports requires an explicit `field_order`.

### Component lifecycle

- Stateful components require a DataArray initial condition.
- `run()` requires prior `Driver.initialize(...)`.
- An exhausted Driver must be `reset()` and initialized again.
- A component may advertise only variables present in its returned
  DataArray.
- Prognostic input and output signatures must be DataArrays.
- A custom `next_input` is required when model output cannot reconstruct the
  input lead-time window.

### Connector configuration

- `window` and `reduce` must be set together.
- A windowed connector needs a destination import whose `CellMethod` matches
  base, reduction, and window.
- `sample` and `regridder` are mutually exclusive.
- A PointGrid destination needs `sample="nearest"` or `"bilinear"` unless a
  custom regridder is supplied.
- A custom regridder must return `xr.DataArray`.
- A connector cannot execute before its source has published the field.

### Pull adapter

`PullAdapter` requires a settable data-source attribute. A requested variable
must resolve to a Field in the import State. With `strict_time=True`, pulled
times must equal field `valid_time`. `StateDataSource` currently serves
exchange-shaped `(lat, lon)` fields.

### YAML

`to_yaml` needs an automatically serializable mediator or a component
`yaml_spec`. `from_yaml` requires `clock`, `sequence`, and `components`, valid
dotted import paths, valid constructor arguments, and known connector
endpoints. Initial conditions and custom regridder callables are not
serialized.

## Troubleshooting

### Output stays unchanged

Inspect `driver.describe()` and confirm the connection exists. Then use:

```python
for time, states in driver.steps():
    print(time, states)

print(driver.probe("source->destination"))
```

If the delivered `valid_time` is older than expected, the connection is
lagged. Move the connect action after the source run for sequential behavior.

### Linear time policy looks constant

Linear extrapolation needs two distinct source valid times. It holds the
latest field until that history exists. Fields with `lead_time` or `window`
dimensions also use constant behavior because one valid timestamp cannot
describe every slice.

### NaNs spread after regridding

Masked source values are not filled by default. Use `fill="nearest"` or
`fill="zero"` so invalid values are replaced before interpolation.

### Point sampling fails

Confirm the destination uses `earth2studio.grids.PointGrid`, the source has
regular `lat` and `lon` coordinates, and the connector has `sample=`. For a
different source geometry, supply a custom DataArray regridder.

### CuPy operation fails

The coupler chooses operations from the payload in `DataArray.data`. Ensure
all participating arrays use compatible NumPy or CuPy payloads and that CuPy
is installed for GPU arrays. Coordinate indexes remain host-side xarray
coordinates.

### Memory warning

Collection is entirely in memory. Shorten the run, export fewer fields, or
set `collect=False`. With collection disabled, `run()` returns `{}`.

### Rerunning a Driver

```python
driver.reset()
driver.initialize(initial_conditions)
datasets = driver.run()
```

Reset clears collected records, connector history, reductions, and probes.
