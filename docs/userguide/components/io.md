# Output Handling { #output_handling_userguide }

IO backends store the labelled DataArrays produced by workflows. Use `ZarrBackend`
as a general default, `NetCDF4Backend` for netCDF files, `XarrayBackend` for in-memory
results, `AsyncZarrBackend` for asynchronous and sharded writes, or `IceChunkBackend`
for versioned output. See the [IO API](../../modules/io.md) for backend options.

## Interface

| Method | Purpose |
| --- | --- |
| `add_array(template)` | Create arrays from dimensions, coordinates, dtype, and metadata. |
| `write(x)` | Store values at positions identified by the field's coordinate labels. |
| `flush()` | Wait for pending writes to become visible to readers. |
| `close()` | Flush, finalize the store, and release resources. |

The `earth2studio.io.base.IOBackend` protocol requires these four methods;
inheritance is optional. Reading and dictionary-like access are backend-specific
capabilities rather than requirements of the write interface.

## Core Rules

- **Declare the store first.** Templates have concrete, nonempty dimensions with
  unique labels. Their field values are never read, so allocation-free templates
  work. Writes do not create arrays or extend the declared coordinate extent.
- **Keep variable names verbatim.** Each `variable` label creates one array over the
  remaining dimensions, including labels such as `tp:sum:6h`. A template without a
  `variable` dimension must have a name, which names its single stored array.
- **Write by label.** Fields may contain subsets of stored labels in any order, but
  their dimensions must match in order. Unknown arrays or labels fail before writing.
  `AsyncZarrBackend` additionally requires full coverage of non-parallel dimensions.
- **Keep store geometry consistent.** Shared coordinates must agree; adding an
  identical template is a no-op. Auxiliary coordinates and grid metadata are retained.
  Use separate stores for outputs on incompatible grids.
- **Preserve input fields.** Backends accept NumPy, CuPy, and Torch storage, handle
  transfers themselves, and do not mutate the supplied field. Asynchronous writers
  copy the needed values before returning.
- **Finalize explicitly.** `flush` makes writes visible within an open transaction;
  `close` finalizes it, including an Icechunk commit. Closing again is harmless,
  but later writes fail. `earth2studio.run` workflows do not close your backend.

## Usage

### Declare and Write a Forecast

Given a loaded single-input, unforced model and prepared initial field `x`, plan a
store for the full output horizon. Here `times` contains the initialization times
in `x`, and `lead_times` contains all forecast leads you will write:

```python
from itertools import islice

from earth2studio.io import ZarrBackend
from earth2studio.io.utils import output_template

io = ZarrBackend("forecast.zarr")
signature = model.output_coords(model.input_coords())
template = output_template(
    signature,
    leading={"time": times},  # Resolve the dynamic leading batch prefix
    coords={"lead_time": lead_times},
)
io.add_array(template)

iterator = model.create_iterator(x)
try:
    for y in islice(iterator, nsteps):
        io.write(y)
finally:
    iterator.close()
    io.close()
```

The iterator writes forecasts only. If you also store the initial condition, include
its lead time in the template and write the appropriately selected field separately.
For chunk-producing models, each write may contain multiple forecast leads.

### Read Results

Backends with `read`, such as `XarrayBackend`, accept a template or a mapping from
every dimension to the requested labels:

```python
from earth2studio.io import XarrayBackend

memory = XarrayBackend()
memory.add_array(y)
memory.write(y)
field = memory.read(y)
memory.close()
```

Pass `device="cuda:0"` for a CuPy-backed result where supported. Stored formats can
also be opened directly with Xarray, for example:

```python
import xarray as xr

dataset = xr.open_zarr("forecast.zarr")
```

For versioned transactions, shard sizing, concurrent writers, and cloud stores,
see [Versioned and Sharded Output](../advanced/io.md).

## Create Your Own Backend

Implement the four methods in a local class. Validate templates and writes before
modifying storage, preserve coordinate and grid metadata, and make `flush` and
`close` reflect your storage system's visibility and durability guarantees. Reading
is optional; clearly document any restrictions on partial writes or concurrency.
