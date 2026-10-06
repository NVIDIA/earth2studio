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
