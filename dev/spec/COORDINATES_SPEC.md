# Coordinate Systems

## Goal

Replace the separate tensor and ordered-coordinate dictionary with Xarray
`DataArray` objects. A model must describe its inputs and outputs without allocating
field data, and the same contract must support one or many arrays.

This specification defines coordinate declarations and model handshakes. Grid
transformation policy and full model migration are separate work.

## Coordinate Arrays

`coord_array` creates an allocation-free `DataArray` signature. Its `dims` tuple is
the authoritative axis order, its coordinates describe fixed requirements, and its
attributes carry Earth2Studio metadata.

```python
import numpy as np

from earth2studio import coord_array

state = coord_array(
    ("batch", "lead_time", "variable", "lat", "lon"),
    coords={
        "lead_time": [np.timedelta64(0, "h")],
        "variable": ["u10m", "v10m", "tp"],
    },
    dynamic=("batch",),
    grid="fcn1",
    statistics={"tp": "sum:6h"},
)
```

The backing object stores only shape and dtype, so `state.data.nbytes == 0`. Reading
field values is invalid. Coordinate values such as variables and grid indexes remain
normal Xarray coordinates.

## Dimension Rules

- `dims` preserves and declares axis order
- Dynamic dimensions are leading wildcards whose runtime names and sizes may vary
- Fixed dimensions, sizes, and coordinate labels must match exactly
- `variable` remains a one-dimensional array of strings
- Auxiliary coordinates may depend on dimensions, such as two-dimensional latitude
  and longitude on a `y`, `x` grid

`handshake_dataarray(array, signature)` checks one array.
`handshake_dataarrays(arrays, signatures)` checks an ordered collection.

## Model Contracts

Models always use tuples, including models with one input or output. Tuple position
defines the handshake between each runtime array and its signature.

```python
import xarray as xr

from earth2studio.utils import handshake_dataarrays

class Model:
    def input_coords(self) -> tuple[xr.DataArray, ...]:
        return (state, forcing)

    def output_coords(
        self, inputs: tuple[xr.DataArray, ...]
    ) -> tuple[xr.DataArray, ...]:
        handshake_dataarrays(inputs, self.input_coords())
        return (forecast, diagnostics)

    def __call__(self, *inputs: xr.DataArray) -> tuple[xr.DataArray, ...]:
        self.output_coords(inputs)
        return forecast_array, diagnostic_array
```

This supports inputs on different coordinate systems. For example, `state` may use a
registered global grid while `forcing` uses a point dimension with auxiliary
latitude and longitude coordinates. `output_coords` must remain derivable from
coordinate signatures alone.

## Grid Metadata

Passing `grid="fcn1"` resolves the registered grid and fills its dimensions, sizes,
and index coordinates. The grid identifier is stored in `earth2studio_grid_id` and
can be resolved through the grid registry.

`fetch_data` verifies a requested grid when the source identifies its grid. A grid
mismatch reaches an explicit future regridding hook; coordinate declarations do not
select or perform regridding.

## Time Statistics

Statistics modify variables without changing their names:

```python
statistics={
    "u10m": "mean:24h",
    "tp": "sum:6h",
    "t2m": "max:-24h:0h",
}
```

`coord_array` stores the expanded representation in
`earth2studio_statistics`. `fetch_data` uses that metadata and the source cadence to
plan source times or lead times, group compatible variables, and reduce them.

## Data Flow

1. Read the model's ordered `input_coords()` signatures
2. Fetch each required `DataArray` with its matching signature as metadata
3. Validate the ordered arrays with `handshake_dataarrays`
4. Convert to the model backend internally, such as zero-copy Torch conversion
5. Return an ordered tuple of `DataArray` outputs

The public model boundary contains no separate tensor, coordinate dictionary, or API
mode switch.
