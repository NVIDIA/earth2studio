(output_handling_userguide)=

# Output Handling

IO backends handle writing model outputs to disk or memory. Use them when saving
forecast results, ensemble data, or other workflow outputs.
While input data handling is primarily managed by the data sources in
{mod}`earth2studio.data`, output handling is managed by the IO backends available
in {mod}`earth2studio.io`.
These backends are designed to balance the ability for you to customize the arrays and
metadata within the exposed backend while also simplifying the design of reusable
workflows.

The key extension of the typical `(x, coords)` data structure movement throughout
the rest of the `earth2studio` code and output store compatibility is the notion of
an `array_name`. Names distinguish between different arrays within the backend and
are currently a requirement for storing `Datasets` in `xarray`, `zarr`, and `netcdf`.
This means that you must supply a name when adding an array to a store or when
writing an array. A frequent pattern is to extract one dimension of an array,
such as `"variable"` to act as individual arrays in the backend.

## IO Backend Interface

The full requirements for a standard IO backend are defined explicitly in the
`earth2studio/io/base.py`.

```{literalinclude} ../../../earth2studio/io/base.py
:lines: 24-
:language: python
```

:::{note}
IO Backends do not need to inherit this protocol; this is used to define
the required APIs. Some built-in IO backends may also offer additional functionality
that is not universally supported (and hence not required).
:::

There are two important methods that must be supported:

- `add_array`, which adds an array to the underlying store and any attached coordinates
- `write`, which explicitly stores the provided data in the backend

The `write` command can induce synchronization when the input tensor resides on the GPU
and the store.

The {mod}`earth2studio.io.kv` backend has the option for storing data on the GPU, which
can be done asynchronously.

Most stores make a conversion from PyTorch to numpy in this process, and offer several
additional utilities such as `__contains__`, `__getitem__`, `__len__`, and `__iter__`.
Refer to the implementation in {mod}`earth2studio.io.ZarrBackend`:

```{literalinclude} ../../../earth2studio/io/zarr.py
    :language: python
    :start-after: sphinx - io zarr start
    :end-before: sphinx - io zarr end
```

Common backends include {class}`earth2studio.io.ZarrBackend`,
{class}`earth2studio.io.NetCDF4Backend`, and
{class}`earth2studio.io.AsyncZarrBackend`.
Because of `datetime` compatibility, we recommend using the `ZarrBackend` as a default.

## Initializing a Store

A common data pattern seen throughout our example workflows is to initialize the
variables and dimensions of a backend using a complete `CoordSystem`, refer to
{ref}`data_userguide` for the structure. For example:

```python
# Build a complete CoordSystem
total_coords = OrderedDict(
    dict(
        'ensemble': ...,
        'time': ...,
        'lead_time': ...,
        'variable': ...,
        'lat': ...,
        'lon': ...
    )
)

# Give an informative array name
array_name = 'fields'

# Initialize all dimensions in total_coords and the array 'fields'
io.add_array(total_coords, 'fields')
```

It can be tedious to define each coordinate and dimension. However, if we have
a prognostic or diagnostic model, most of this information is already available.
Here is a robust example of such a use-case:

```python
# Set up IO backend
# assume we have `prognostic model`, `time`, and `array_name`
# Copy prognostic model output coordinates
total_coords = OrderedDict(
    {
        k: v for k, v in prognostic.output_coords(prognostic.input_coords()).items() if
        (k != "batch") and (v.shape != 0)
    }
)
total_coords["time"] = time
total_coords["lead_time"] = np.asarray(
    [total_coords["lead_time"] * i for i in range(nsteps + 1)]
).flatten()
total_coords.move_to_end("lead_time", last=False)
total_coords.move_to_end("time", last=False)
io.add_array(total_coords, array_name)
```

Prognostic models, diagnostic models, statistics, and metrics are required to have an
`output_coords` method, which maps from an input coordinate to a corresponding output
coordinate. This method is meant to simulate the result of `__call__` without having
to actually compute the forward call of the method. Review the API documentation for more details.

Another common IO use-case is to extract a particular dimension (usually `variable`) as
the array names.

```python
# A modification of the previous example:
var_names = total_coords.pop("variable")
io.add_array(total_coords, var_names)
```

## Writing to the Store

After the data arrays have been initialized in the backend, writing to those arrays
is a single line of code.

```python
x, coords = model(x, coords)
io.write(x, coords, array_name)
```

If, as above, you are extracting a dimension of the tensor to use as array names
then you can make use of {mod}`earth2studio.utils.coords.split_coords`:

```python
io.write(*split_coords(x, coords, dim="variable"))
```

For a complete workflow that uses IO backends, refer to {func}`earth2studio.run.deterministic`
or the deterministic workflow example in the gallery.

## Sharding with the Async Zarr Backend

{class}`earth2studio.io.AsyncZarrBackend` writes each forecast step as soon as it is
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

:::{warning}
A shard must never contain data owned by more than one process. The backend keeps each
shard object to a single write by buffering its chunks in host memory, but that
guarantee holds *within* a process only. Separate ranks have separate buffers, so if two
ranks each hold part of the same shard they will both write that shard in full and the
later write silently discards the other's data. This is not detected and does not raise.
:::

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

Separately, and independent of sharding: arrays are created lazily on the first write, so
several ranks writing a new array at once can race on its creation. Have one rank
establish the arrays before the others begin writing.
