# Earth2Studio Regridder Protocol

## Goal

Provide one contract for mapping a field from one grid onto another, so models can
recommend a source on another grid for an input slot (`RegriddedSource`; see
[MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)) and drivers can fetch it without
knowing the method. This also allows arbitrary regridders to be used (internal/external
GPU accelerated flavors) under a common interface.

Grid geometry follows [GRID_SPEC.md](GRID_SPEC.md); this
contract covers only moving field data between two such geometries.

Status: the protocol is mocked up in `earth2studio/grids/base.py`, and
`RegriddedSource` in `earth2studio/data/utils.py` composes it with a source. No
regridder implementations are provided yet.

## Interface

`Regridder` is a runtime-checkable structural protocol over `xr.DataArray`, matching
the model and data source payloads. Implementations satisfy it directly.

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

- `source_grid` and `target_grid` are fixed at construction. Drivers handshake
  `target_grid` against a model slot before fetching anything.
- `__call__` maps the spatial dimensions of one field. Other dimensions pass through.
- `to` moves precomputed indices or weights to a device and returns the regridder.

A protocol rather than an ABC: the contract is a DataArray in and a DataArray out
plus two grid descriptions, with no shared implementation to inherit. Wrappers
around external engines (Earth2Grid, `earth2studio.utils.interp`) conform without
subclassing.

## Rules

| Rule | Requirement |
| --- | --- |
| `R1` | Indices and weights are computed at construction; `__call__` only applies them |
| `R2` | `__call__` raises `ValueError` if the spatial dims or sizes differ from `source_grid` |
| `R3` | Output dims are the non-spatial dims of `x`, in order, then `target_grid.dims` |
| `R4` | Spatial coords and grid metadata match `coord_array(..., grid=target_grid)` |
| `R5` | Output keeps the array backing (NumPy or CuPy) and device of `x` |
| `R6` | `__call__` has no side effects on the regridder, so one instance can be shared |
| `R7` | Target points without source coverage get a documented fill value, NaN by default |
| `R8` | A regridder never fetches data |

Under `R3`, non-spatial coordinates pass through unchanged. Attrs pass through
except grid-owned keys (`earth2studio_grid_id`, `earth2studio_crs` and the CRS
attributes), which follow `target_grid` under `R4`. `R4` means
the output handshakes against a slot declared on `target_grid` with no further
coordinate assembly.

## Composition

`RegriddedSource(source, regridder)` composes a data or forecast source with a
regridder. It forwards each request to the source and regrids the result, so it
satisfies whichever source protocol the wrapped source does, and forwards the
source's `time_step` for temporal statistics. A model recommending a source off its
slot's grid returns this composition, so drivers fetch from it like any source:

```python
source = model.default_sources()[slot] or my_source  # None: no recommendation
field = fetch_data(source, time, variable, lead_time)
```

Nothing regrids implicitly. A regridder is bound to its source's grid, so a caller
replacing the source must replace the whole composition (R2 raises otherwise).

## Migration

The eval recipe's `Regridder` ABC (`recipes/eval/src/regrid.py`) is tensor-native:
`apply(tensor, spatial_dims)` plus a `target_coords()` dictionary, with an
`apply_dataarray` CPU adapter. Its `NearestNeighborRegridder` and
`BilinearRegridder` move here as implementations of this protocol. The tensor
`apply` stays as an implementation detail behind `__call__`, which accepts CuPy
fields without a host round trip. The recipe's `RegriddedSource` moves to
`earth2studio.data` as above; its eval-specific `apply_dataarray` hook is replaced by
the protocol's `__call__`, and `fetch` awaits the wrapped source instead of
blocking on it.

## Open Questions

- Should `fetch_data(regridder=...)`, currently a reserved `str`, accept a
  `Regridder`, with strings resolving to built-in methods between the source grid
  and `target_grid`?
- Should the output dtype follow the input, or may implementations promote (for
  example to float32 to match their weights)?
