# Output Handling { #output_handling_userguide }

IO backends in `earth2studio.io` write workflow outputs to memory or storage. They
take the same `xr.DataArray` fields that models and data sources produce, so
workflows write model outputs directly.

!!! note
    Backends are migrating to this interface. Until they do, `ZarrBackend`,
    `NetCDF4Backend`, `KVBackend`, `IceChunkBackend` and `AsyncZarrBackend` keep
    the previous tensor-based `add_array(coords, array_name)` and
    `write(x, coords, array_name)`.

## IO Backend Interface

```python
--8<-- "earth2studio/io/base.py:io-backend-interface"
```

- `add_array` creates the arrays a schema describes.
- `write` stores a field at the positions its coordinate labels identify.
- `flush` waits until earlier writes are visible to readers.
- `close` flushes and releases resources. `earth2studio.run` workflows do not close
  the backend for you.

Backends do not need to inherit this protocol, and many offer more, such as `read`,
`__contains__`, `__getitem__`, `__len__` and `__iter__`.

`ZarrBackend` is the recommended default. `NetCDF4Backend` writes netCDF files,
`AsyncZarrBackend` writes without blocking inference and supports sharding and
cloud stores, `XarrayBackend` keeps outputs in memory, and `IceChunkBackend` adds
versioning.

## Creating Arrays

A schema is a DataArray describing the store: its dimensions, coordinates, dtype
and metadata. Its values are never read. Each `variable` label becomes one array
over the remaining dimensions, named verbatim, including statistic labels such as
`tp:sum:6h`.

Plan the schema from a model's output coordinates with `output_schema`. It
replaces the model's dynamic leading dimensions, such as `batch`, with the run's:

```python
from earth2studio.io.utils import output_schema

# (batch, lead_time, variable, lat, lon)
signature = model.output_coords(model.input_coords())
schema = output_schema(
    signature,
    leading={"time": times},  # replaces batch
    coords={"lead_time": lead_times},  # the full forecast horizon
)
io.add_array(schema)  # one (time, lead_time, lat, lon) array per variable
```

- Schemas must be concrete: no dynamic or empty dimensions.
- Adding the same schema again does nothing, so restarted runs can call `add_array`
  unconditionally. Coordinates that conflict with the store raise a `ValueError`.
- Auxiliary coordinates, such as 2-D latitude and longitude, and grid metadata are
  stored with the arrays, so `earth2studio.grids.infer_grid` recovers the grid
  from the output.
- A schema without a `variable` dimension creates one array named after the
  schema.

## Writing to the Store

```python
for y in islice(model.create_iterator(x), nsteps):
    io.write(y)
io.close()
```

`write` locates a field by its coordinate labels. A field may hold any subset of
the store's labels, in any order, including several lead times at once. Its
dimensions must match the arrays' in order. Unknown labels or arrays raise a
`ValueError` before anything is written; writes never create arrays.

Fields may be NumPy-, CuPy- or Torch-backed, on CPU or GPU. The backend moves data
to storage itself and never modifies the field.

## Reading from the Store

Backends that can read, such as `XarrayBackend`, implement `read`, the inverse of
`write`. Pass a schema, or a mapping from every dimension to the labels you want:

```python
everything = io.read(schema)
selection = {
    "variable": ["t2m"],
    "time": times[:1],
    "lead_time": lead_times,
    "lat": lat,
    "lon": lon,
}
subset = io.read(selection, device="cuda")  # CuPy-backed; NumPy-backed on CPU
```

Any store can also be opened with Xarray, for example `xr.open_zarr(path)`.

## Versioned Output with the Icechunk Backend

`earth2studio.io.IceChunkBackend` writes to an
[Icechunk](https://icechunk.io/) repository instead of a plain Zarr store.
Icechunk adds transactional, versioned writes on top of Zarr: every array is
backed by a Zarr store as usual, but writes are only made durable when
explicitly committed, producing an immutable, named snapshot that can be
read back (or rolled back to) later. This is useful for inference campaigns
where you want a persistent, auditable history of a store's contents.

`IceChunkBackend` subclasses `ZarrBackend`, so `add_array`, `write`, `read`,
and the `__contains__`/`__getitem__`/`__len__`/`__iter__` helpers all behave
identically. The one addition is `commit`, which must be called to persist
writes:

```python
from earth2studio.io import IceChunkBackend

# `storage` may be omitted (in-memory repository), a local filesystem path,
# or an `icechunk.Storage` instance (e.g. `icechunk.s3_storage(...)`)
io = IceChunkBackend("/path/to/repo")

io.add_array(schema)
io.write(x)

# Writes are visible through `read`/`__getitem__`/`commit` immediately (each
# flushes pending writes first), but are only persisted to the Icechunk
# repository once committed
io.commit("forecast run 2024-01-01T00Z")
```

Pass `branch` to write to a named branch other than `"main"`; it is created
automatically (from the tip of `"main"`) if it does not already exist. This
requires the `icechunk` optional dependency, install with `pip install
earth2studio[data]`.

!!! note
    `write` is non-blocking by default: it submits the store write to a
    background thread and returns immediately, so the inference loop can move
    on to the next step while the previous step's write is still in flight.
    `read`, `__getitem__` and `commit` all flush pending writes first. Pass
    `blocking=True` to write synchronously instead.

### Sharding Icechunk output with the Async Zarr Backend

`IceChunkBackend` covers the common case, but `earth2studio.io.AsyncZarrBackend`
accepts any constructed Zarr store through its `store` parameter, and an
Icechunk session store is a Zarr store. Use this composition instead when you
need Zarr v3 sharding to keep the file count of a large campaign down —
`IceChunkBackend` does not support sharding:

```python
import icechunk
from earth2studio.io import AsyncZarrBackend

repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("/path/to/repo")
)
session = repo.writable_session("main")

io = AsyncZarrBackend(
    "unused",  # location comes from the store
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    store=session.store,
)
# ... write forecast steps ...
io.close()  # flush all in-flight writes BEFORE committing
session.commit("forecast run")
```

The ordering matters: `io.close()` (or `flush()`) must complete before
`session.commit()`, otherwise in-flight writes are silently excluded from the
snapshot. Committing is your responsibility here — the session is managed
outside the backend.

This composes directly with the built-in workflows. Note that
`earth2studio.run` workflows do **not** close the backend for you:

```python
import icechunk
import numpy as np
from collections import OrderedDict

from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import AsyncZarrBackend
from earth2studio.models.px import SFNO

model = SFNO.load_model(SFNO.load_default_package())
nsteps = 20
times = np.array([np.datetime64("2024-01-01")])
lead_times = np.array([np.timedelta64(6 * i, "h") for i in range(nsteps + 1)])

repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("forecast_repo")
)
session = repo.writable_session("main")

io = AsyncZarrBackend(
    "unused",
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    blocking=False,
    store=session.store,
)
run.deterministic(times, nsteps, model, GFS(), io)
io.close()  # required: run.deterministic does not close the backend
session.commit("SFNO forecast 2024-01-01T00Z")
```

Everything Icechunk provides then applies to the forecast output: reopen any
earlier snapshot with `repo.readonly_session(snapshot_id=...)`, branch with
`repo.create_branch(...)`, or point the repository at S3/GCS storage instead of
the local filesystem with `icechunk.s3_storage(...)` — the workflow code is
unchanged.

!!! note
    Committing with no new writes raises an `IcechunkError`; pass
    `commit(message, allow_empty=True)` to create an empty snapshot instead.
    Likewise, if another writer commits to the same branch first, `commit`
    raises a conflict error — see the
    [Icechunk documentation](https://icechunk.io/en/latest/) on rebasing and
    the `rebase_with` argument for resolving concurrent commits. Icechunk's
    local filesystem storage is not safe for concurrent commits; use an
    object store when multiple processes write to one repository.

## Sharding with the Async Zarr Backend

`earth2studio.io.AsyncZarrBackend` writes each forecast step as soon as it is
available, which keeps the GPU from blocking on disk IO. The cost is one file per chunk,
and because the coordinates listed in `parallel_coords` are chunked with a size of 1,
a large inference campaign can produce an enormous number of small files. This is a
common way to exhaust an inode quota on a parallel filesystem such as Lustre, and it
makes the resulting store slow to list and copy.

Zarr v3 sharding addresses this by packing many chunks into a single storage object.
Pass `shard_coords` to group chunks along one or more coordinates:

```python
io = AsyncZarrBackend(
    "output.zarr",
    parallel_coords=OrderedDict({
        "time": time,
        "lead_time": lead_time,
    }),
    # 8 lead times per shard, so 8x fewer files
    shard_coords={"lead_time": 8},
)
```

The chunk layout is unchanged, so readers still fetch a single lead time at a time. Only
the number of files on disk changes.

### Choosing a shard size

A shard is one file, so writing part of one forces Zarr to read, modify, and rewrite the
whole object. To avoid that, the backend accumulates the chunks of a shard in host
memory and writes the shard once it is complete. Three things bound the choice:

**Host memory.** Budget roughly `max_inflight_shards * 4 * shard_bytes + pool_size *
write_bytes` per process. A flushing shard costs several times its own size once Zarr's
encoded copy is counted, and with several ranks per node this applies to each of them.
For a 73 variable 721x1440 fp32 field, one lead time is about 0.3 GB, so a shard of 8
lead times is a 2.4 GB buffer and the defaults put peak usage in the tens of GB.

**Store bandwidth.** Sharded writes are slower than unsharded ones, since a shard is one
large sequential IO rather than many independent ones. `max_inflight_shards` controls how
many run at once and is the main lever for single-process performance, though raising it
stops helping once the store saturates bandwidth. Lowering it for multi-rank (distributed)
runs is usually sensible, as the ranks already supply concurrency between them.

**How fast the model produces data.** With the `AsyncZarrBackend`, none of the write cost
is visible as long as the model takes longer to produce a step than the store takes to absorb
it. Sharding is close to free in that regime. If a workflow writes more bytes per step than
the store can absorb in the time the model takes to produce them, the wall clock becomes the
IO time, and the only remedies are writing less data or a faster store.

As a rough guide, sharding a quarter degree field along `lead_time` costs a few percent
of wall clock for a proportional reduction in file count, provided the run is not already
IO bound. The best settings are problem- and system-specific, and involve tradeoffs between
speed, host memory consumption, and file count, so it is worth measuring and tuning for the
desired behavior in large inference campaigns.

### Partial shards and restarts

Shards do not need to divide evenly into your forecast length. `close()` writes out any
shard that never filled, using the array fill value for the positions that were never
supplied, which reads back exactly as an unwritten chunk would.

Writing into a shard that is already present in the store still works, but falls back to
a read-modify-write of the whole shard and logs a warning. This happens when `close()`
or `flush()` is called partway through a run and the same shards are written again
afterwards, or when restarting into an existing store that was left with incomplete
shards. To keep restarts on the fast path, align your restart boundaries with the shard
size.

Sharding composes with `zarr_codecs`, which compresses the inner chunks within each
shard, and with `chunked_coords`, which sets the chunk size of coordinates that are not
in `parallel_coords`. A shard size must always be a multiple of that coordinate's chunk
size.

### Multiple processes writing one store

!!! warning
    A shard must never contain data owned by more than one process. The backend keeps each
    shard object to a single write by buffering its chunks in host memory, but that
    guarantee holds *within* a process only. Separate ranks have separate buffers, so if two
    ranks each hold part of the same shard they will both write that shard in full and the
    later write silently discards the other's data. This is not detected and does not raise.

The rule is that the set of parallel coordinate indices a rank writes must be a union of
whole shards. In practice that makes one layout obviously correct and another
obviously fragile.

**Shard along a coordinate each rank owns entirely.** A rank running a forecast owns
every lead time of that forecast, so sharding `lead_time` is safe no matter how the
initial conditions are distributed:

```python
# Rank owns a subset of ICs, and all lead times of each
io = AsyncZarrBackend(
    "forecast.zarr",
    parallel_coords=OrderedDict({"time": all_times, "lead_time": all_lead_times}),
    shard_coords={"lead_time": 8},   # safe for any IC distribution
)
```

This is also the dimension that causes the file explosion in the first place, so it is
usually the only one worth sharding.

**Sharding along the distributed coordinate is the fragile case.** With 8 ICs across 3
ranks and `shard_coords={"time": 4}`, a contiguous block split gives rank 0 ICs 0-2 and
rank 1 ICs 3-5, so the first time shard covers ICs 0-3 and straddles two ranks. Both
buffer a partial shard, both flush it whole, and one rank's output is lost. The
read-modify-write fallback does not protect you: both ranks check for the shard before
either has written it, so both take the full overwrite path.

Such a layout is only safe when every rank's slice happens to be shard aligned, which
depends on the item count, the rank count, and the shard size all lining up. It can pass
at one rank count and silently lose data at another, so prefer the first layout.

Separately, and independent of sharding: several ranks creating the same new array at
once can race on its creation. Have one rank call `add_array` before the others begin
writing.

### Writing to cloud object storage

For cloud outputs, pass the `store` parameter instead of `fs_factory`. This routes all
writes through an [obstore](https://developmentseed.org/obstore/latest/)-backed
`zarr.storage.ObjectStore`, which uses native put and multipart-upload requests rather
than fsspec sessions. The parameter accepts a store URL, an obstore store instance, or
an already constructed zarr store:

```python
# URL form: credentials resolved from the environment
io = AsyncZarrBackend(
    "unused",  # location comes from the store
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    store="s3://my-bucket/forecasts/run-001.zarr",
    store_kwargs={"region": "us-east-1"},
)

# Instance form: full control over store construction
from obstore.store import S3Store
io = AsyncZarrBackend(
    "unused",
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    store=S3Store("my-bucket", prefix="forecasts/run-001.zarr"),
)
```

`file_name` and `fs_factory` are ignored when `store` is set. Everything else —
non-blocking writes, sharding, restarts — behaves identically; the loop pool shares a
single store instance, matching the shared state of the remote bucket.
