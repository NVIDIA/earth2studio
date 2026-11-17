# Output Handling { #output_handling_userguide }

IO backends store the labelled DataArrays produced by workflows. `AsyncZarrBackend`
is a good general-purpose default, supporting asynchronous and sharded writes.
Use `ZarrBackend` for synchronous Zarr output, `NetCDF4Backend` for netCDF files,
`XarrayBackend` for in-memory results, or `IceChunkBackend` for versioned output.
See the [IO API](../../modules/io.md) for backend options.

For Icechunk transactions, asynchronous writes, sharding, and cloud storage, see
[Advanced Output Handling](../advanced/io.md).

## Interface

| Method | Purpose |
| --- | --- |
| `add_array(template)` | Create arrays from dimensions, coordinates, dtype, and metadata. |
| `write(x)` | Store values at positions identified by the field's coordinate labels. |
| `flush()` | Wait for pending writes to become visible to readers. |
| `close()` | Flush, finalize the store, and release resources. |

The [`earth2studio.io.base.IOBackend`][interface] protocol requires these four methods;
inheritance is optional. Reading and dictionary-like access are backend-specific
capabilities rather than requirements of the write interface.

See [E2S-10: IO][io-spec] for template, label-based writing, metadata, and lifecycle
requirements.

[io-spec]: ../../specs/E2S-10_IO_SPEC.md
[interface]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/io/base.py

## Usage

### Declare and Write a Forecast

Given a loaded single-input, unforced model and prepared initial field `x`, plan a
store for the full output horizon. Here `times` contains the initialization times
in `x`, and `lead_times` contains all forecast leads you will write:

```python
from itertools import islice

from earth2studio.io import AsyncZarrBackend
from earth2studio.io.utils import output_template

io = AsyncZarrBackend(
    "forecast.zarr",
    parallel_coords={"time": times, "lead_time": lead_times},
)
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

`parallel_coords` declares the labels along axes that can be written independently.
Here, writes can select initialization times and forecast leads; each write covers
the full extent of other dimensions. Add any other independently written axes,
such as `ensemble`, to `parallel_coords`.

Close the backend when the workflow finishes; `earth2studio.run` does not close it
for you. With Icechunk, `flush` exposes pending writes within the open transaction,
while `close` commits the output.

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

## Create Your Own Backend

Implement the four methods in a local class. Validate templates and writes before
modifying storage, preserve coordinate and grid metadata, and make `flush` and
`close` reflect your storage system's visibility and durability guarantees. Reading
is optional; clearly document any restrictions on partial writes or concurrency.
