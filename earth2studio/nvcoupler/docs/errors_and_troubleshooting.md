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
