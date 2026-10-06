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
