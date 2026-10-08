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
