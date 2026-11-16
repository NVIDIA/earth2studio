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

## Specification

The [IO specification][io-spec] defines these compliance codes.

| Code | Requirement |
| --- | --- |
| `I1` | `add_array` never reads field values from the template. |
| `I2` | `add_array` rejects dynamic or zero-sized dimensions with `ValueError`. |
| `I3` | Each `variable` label names one array verbatim; otherwise use the template's name. |
| `I4` | Arrays share coordinates; re-adding an identical array is a no-op from any instance. |
| `I5` | Write by label; unknown labels, arrays, or dimensions raise `ValueError` first. |
| `I6` | Accept label subsets, including multiple `lead_time` entries in one write. |
| `I7` | Never modify `x`; non-blocking backends copy needed values before returning. |
| `I8` | Accept NumPy-, CuPy-, and Torch-adapter-backed arrays. |
| `I9` | After `flush`, values, coordinates, and persisted attributes round-trip. |
| `I10` | Do not persist signature markers or `earth2studio_statistics`. |
| `I11` | `close` flushes and finalizes; later writes raise and repeated closes are no-ops. |

For `I6`, backends may document restrictions: `AsyncZarrBackend` requires full
coverage of non-parallel dimensions. Write dimensions must match the stored order.
For `I10`, qualified variable labels reconstruct temporal-statistics metadata on read.
`flush` exposes transactional writes within the open transaction; `close` commits
Icechunk output. `earth2studio.run` workflows leave closing the backend to the caller.

[io-spec]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/dev/spec/IO_SPEC.md#rules

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

## Create Your Own Backend

Implement the four methods in a local class. Validate templates and writes before
modifying storage, preserve coordinate and grid metadata, and make `flush` and
`close` reflect your storage system's visibility and durability guarantees. Reading
is optional; clearly document any restrictions on partial writes or concurrency.
