# Earth2Studio Regridder Protocol

## Goal

Provide one contract for mapping a field from one grid onto another, so models can
recommend a source on another grid for an input slot and drivers can fetch it
without knowing the method. Any regridder, built in or wrapping an external engine,
works under the same interface.

Grid geometry follows [GRID_SPEC.md](GRID_SPEC.md); this contract covers only moving
field data between two such geometries.

Status: the protocol is in `earth2studio/grids/base.py`, with `NearestRegridder`
and `BilinearRegridder` as built-in implementations. `RegriddedSource` and
`CompositeSource` live in `earth2studio.data`.

## Interface

`Regridder` is a runtime-checkable structural protocol over `xr.DataArray`.

```python
@runtime_checkable
class Regridder(Protocol):
    @property
    def source_grid(self) -> GridDefinition: ...

    @property
    def target_grid(self) -> GridDefinition: ...

    def __call__(self, x: xr.DataArray) -> xr.DataArray: ...
    def to(self, device) -> Regridder: ...
```

- `source_grid` and `target_grid` are fixed at construction.
- `__call__` maps the spatial dimensions of one field; other dimensions pass through.
- `to` optionally places precomputed state on a device ahead of use.

It is a protocol rather than an ABC because there is no shared implementation to
inherit. Wrappers around external engines conform without subclassing.

## Rules

| Rule | Requirement |
| --- | --- |
| `R1` | Indices and weights are computed at construction; `__call__` only applies them |
| `R2` | `__call__` raises `ValueError` if the spatial dims or sizes differ from `source_grid` |
| `R3` | Output dims are the non-spatial dims of `x`, in order, then `target_grid.dims` |
| `R4` | Spatial coords and grid metadata match `coord_array(..., grid=target_grid)` |
| `R5` | Output keeps the array backing (NumPy or CuPy) and device of `x` |
| `R6` | `__call__` has no observable side effects, so one instance can be shared |
| `R7` | Target points without source coverage get a documented fill value, NaN by default |
| `R8` | A regridder never fetches data |

- `R3`: non-spatial coordinates and attributes pass through. Grid-owned attributes
  (`earth2studio_grid_id`, `earth2studio_crs` and the CRS attributes) follow
  `target_grid` under `R4`, so the output handshakes against a slot declared on
  `target_grid` without further coordinate assembly.
- `R5`: a call must accept fields on any supported device, whether or not `to` was
  called. Caching device copies of precomputed state is allowed under `R6`.
- `R7`: the fill applies only where the source has no coverage. NaNs in the source
  data propagate and are never replaced by the fill.
- Floating dtypes are preserved. Other dtypes may be promoted so the fill is
  representable.

## Composition

`RegriddedSource(source, regridder, *, defer=True)` pairs a data or forecast source
with a regridder bound to its grid. The pair stands in for the source wherever one
is accepted. A model recommending a source off its slot's grid returns this pair
from `default_sources()`, and drivers fetch from it like any other source:

```python
def default_sources(self):
    return (MRMS(), RegriddedSource(GOES(), NearestRegridder(goes_grid, self.grid)))
```

- Calling the pair fetches from the source and regrids the result.
- `fetch_data` unwraps a deferred pair and regrids after temporal reduction and
  device placement (see [FETCH_DATA_SPEC.md](FETCH_DATA_SPEC.md)). With
  `defer=False` the pair is opaque, and each fetched frame is regridded before
  reduction.
- The pair forwards the source's `time_step` and `device`, and declares
  `is_forecast`.

Nothing regrids implicitly. A regridder is bound to its source's grid, so a caller
replacing the source must replace the whole pair (`R2` raises otherwise).

`CompositeSource(sources, variables)` routes each requested variable to the one
named component that provides it and joins the parts along `variable`:

- Components must all be analysis sources or all forecast sources.
- They must share every coordinate other than `variable`. A component on another
  grid is wrapped in a `RegriddedSource` onto the shared target.
- Parts with different backings are joined on the device of the first CuPy part.

## Migration

The eval recipe's tensor-native `Regridder` ABC (`recipes/eval/src/regrid.py`) and its
`NearestNeighborRegridder` and `BilinearRegridder` map onto the built-ins. Its
`RegriddedSource` and `CompositeSource` map onto the core compositions. Two
differences matter when migrating:

- The built-ins take grid definitions rather than coordinate arrays.
- The built-ins fill with NaN by default. Pass `fill_value=0` where the recipe
  relied on its bilinear default.

## Open Questions

- Should `fetch_data` also defer the regridding of components inside a
  `CompositeSource`, regridding once after the join?
