# Advanced Output Handling

Use this guide for versioned Icechunk output, asynchronous writes, Zarr sharding,
multi-process ownership, and cloud storage. Start with [Output Handling](../components/io.md)
for the common interface and creating output templates. Backend options are listed
in the [IO API](../../modules/io.md).

## Versioned Output with Icechunk

`IceChunkBackend` writes to an [Icechunk](https://icechunk.io/) repository instead
of a plain Zarr store. Commits create immutable snapshots, giving an inference
campaign a versioned history that can be reopened later.

The backend extends `ZarrBackend` with `commit(message)`. Its usual template,
writing, and reading methods remain available. Install the optional dependency
with `pip install 'earth2studio[data]'`.

Given a prepared output `template` and an iterable of forecast DataArrays:

```python
from earth2studio.io import IceChunkBackend

io = IceChunkBackend("forecast_repo", branch="main")
io.add_array(template)
for forecast in forecasts:
    io.write(forecast)

snapshot_id = io.commit("forecast run 2024-01-01T00Z")
io.close()
```

The `storage` argument accepts a filesystem path or an `icechunk.Storage` instance;
omitting it creates an in-memory repository. A new named branch is created from
the tip of `main` when needed.

Writes are non-blocking by default. `read`, `__getitem__`, `commit`, and `close`
flush pending writes first; use `blocking=True` for synchronous writes.
`flush()` makes pending writes visible within the open session, while `commit()`
publishes a snapshot and opens a new writable session. `close()` commits remaining
changes with a generic message and rejects later writes. Commit explicitly when
you need a descriptive message or conflict handling. Earth2Studio workflows leave
closing the backend to the caller.

Reopen a snapshot with `repo.readonly_session(snapshot_id=snapshot_id)`, or create
branches with `repo.create_branch(...)`. Icechunk also supports object storage
through constructors such as `icechunk.s3_storage(...)`.

!!! note
    An empty commit raises an `IcechunkError` unless `allow_empty=True` is supplied.
    Another writer committing to the same branch can cause a conflict; consult
    [Icechunk's documentation](https://icechunk.io/en/latest/) for rebasing and
    `rebase_with`. Local filesystem storage is not safe for concurrent commits;
    use object storage for that case.

## Asynchronous and Sharded Zarr Output

`AsyncZarrBackend` overlaps storage writes with inference. Coordinates listed in
`parallel_coords` have chunk size one, allowing separate forecast slices to be
written independently. Large campaigns can consequently produce many small files.
Zarr v3 sharding groups chunks into fewer storage objects:

```python
from collections import OrderedDict

from earth2studio.io import AsyncZarrBackend

io = AsyncZarrBackend(
    "forecast.zarr",
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    shard_coords={"lead_time": 8},
)
io.add_array(template)
for forecast in forecasts:
    io.write(forecast)
io.close()
```

Here `times` and `lead_times` contain the full output coordinate labels. Eight
lead-time chunks share one shard; readers can still select individual chunks.
Writes must cover the non-parallel dimensions in full. Define the layout before
writing, and close the backend to finish pending work.

### Choosing a Shard Size

The backend buffers chunks in host memory and writes a complete shard together,
avoiding repeated read-modify-write operations. Balance three costs:

- **Host memory:** budget approximately
  `max_inflight_shards * 4 * shard_bytes + pool_size * write_bytes` per process.
  Encoded copies and concurrent flushes add to the shard buffer itself. A
  73-variable, 721-by-1440 float32 field is about 0.3 GB per lead; eight leads
  require about 2.4 GB for one buffer before those extra costs.
- **Storage bandwidth:** `max_inflight_shards` controls concurrent shard flushes.
  Raising it helps only until bandwidth saturates. Distributed ranks already add
  concurrency, so fewer in-flight shards per rank may be appropriate.
- **Production rate:** asynchronous writing hides IO latency when storage keeps
  pace with inference. If it cannot, reduce output volume or use faster storage.

Measure the tradeoff between memory, throughput, and file count on your storage
system. Compression through `zarr_codecs` applies to the inner chunks.
`chunked_coords` sets chunk sizes for non-parallel dimensions; parallel dimensions
remain chunked at one. Each shard size must be a multiple of its chunk size.

### Partial Shards and Restarts

`close()` flushes incomplete shards, leaving unsupplied positions at the array fill
value. Writing into an existing shard falls back to read-modify-write and logs a
warning. This can happen after a mid-run `flush()` or when resuming a partial run.
Align restart boundaries with shard boundaries to retain the complete-shard fast path.

### Multiple Processes Writing One Store

!!! warning
    Each shard must belong to exactly one process. Buffers are process-local:
    two ranks writing different chunks in the same shard can each overwrite the
    whole shard, silently discarding the other's data. Read-modify-write does not
    protect against concurrent writers.

Shard along an axis each rank owns completely. For example, when ranks own distinct
initialization times and all leads for each forecast, sharding `lead_time` is safe:

```python
io = AsyncZarrBackend(
    "forecast.zarr",
    parallel_coords=OrderedDict({"time": all_times, "lead_time": all_lead_times}),
    shard_coords={"lead_time": 8},
)
```

Sharding the distributed `time` axis is safe only if every rank's assigned indexes
form whole shards. For example, splitting eight times across three ranks into
blocks 0–2, 3–5, and 6–7 crosses a shard boundary when `shard_coords={"time": 4}`.
That layout lets two ranks overwrite the same shard. Ownership must remain valid
when the number of ranks or work items changes.

Separately, concurrent creation of the same array can race: have one rank call
`add_array` before other ranks begin writing.

## Combining Icechunk and Sharding

`IceChunkBackend` does not expose sharding. Instead, pass an Icechunk session's Zarr
store to `AsyncZarrBackend` and manage the transaction yourself:

```python
import icechunk

repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("forecast_repo")
)
session = repo.writable_session("main")
io = AsyncZarrBackend(
    "unused",  # Location comes from the store
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    shard_coords={"lead_time": 8},
    store=session.store,
)
io.add_array(template)
for forecast in forecasts:
    io.write(forecast)
io.close()  # Finish all writes before committing the external session
session.commit("forecast run")
```

The same backend can be passed to an Earth2Studio workflow. Wait for `close()` or
`flush()` to complete before committing; otherwise in-flight writes may be excluded
from the snapshot. With this composition, the backend does not commit the external
session for you.

## Cloud Object Storage

Pass `store` to route writes through a Zarr store rather than `fs_factory`.
URLs and obstore instances use an [obstore](https://developmentseed.org/obstore/latest/)
backed `zarr.storage.ObjectStore`, with native put and multipart-upload operations:

```python
io = AsyncZarrBackend(
    "unused",
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    store="s3://my-bucket/forecasts/run-001.zarr",
    store_kwargs={"region": "us-east-1"},
)
```

Credentials are resolved from the environment. For explicit store configuration:

```python
from obstore.store import S3Store

io = AsyncZarrBackend(
    "unused",
    parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
    store=S3Store("my-bucket", prefix="forecasts/run-001.zarr"),
)
```

An already constructed Zarr store is also accepted. When `store` is set, `file_name`
and `fs_factory` are ignored. Non-blocking writes, sharding, and restart behavior
remain the same, with the loop pool sharing one store instance. Close the backend
after writing to finish outstanding uploads.
