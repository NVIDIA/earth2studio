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
