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
