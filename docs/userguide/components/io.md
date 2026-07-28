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

### Host memory is the tradeoff

A shard is one file, so writing part of one forces Zarr to read, modify, and rewrite the
whole object. To avoid that, the backend accumulates the chunks of a shard in host
memory and writes the shard once it is complete.

```text
73 variables x 721 x 1440 x fp32   = 303 MB per lead time
shard_coords={"lead_time": 8}      = 2.4 GB per shard buffer
```

The raw buffer is only part of the cost. Each shard flush that is running also holds the
copies Zarr's sharding codec makes while encoding it, and each in flight write holds the
tensor it staged out of device memory. Budget roughly:

```text
max_inflight_shards * 4 * shard_bytes  +  pool_size * write_bytes
```

Measured peak RSS is 10-13x the size of a single shard buffer at default settings, so
the 2.4 GB shard above costs roughly 30 GB resident. `max_inflight_shards` is the knob:
lower it if memory is tight.

Note that the number of *buffers* is small — with a loop over lead time there is exactly
one accumulating per array — so the memory is dominated by concurrent flushes rather than
by accumulation. Interleaving several parallel coordinates (for example batching over
`time`) increases the buffer count too, and the backend warns when many are live at once.

### Throughput

A shard is written in one large sequential IO, which is more efficient per stream than
many small chunk writes, but there are proportionally fewer of them. Sharded throughput
therefore depends on how many shard flushes run concurrently, which is what
`max_inflight_shards` controls. If sharded writes are slower than unsharded ones and
memory allows, raising it is the first thing to try.

Raising it stops helping once the filesystem saturates. On a Lustre scratch filesystem
measuring roughly 2 GB/s, sharding 8 lead times of a 73 variable quarter degree field
landed within about 15% of the unsharded write rate for 8x fewer files, and raising
`max_inflight_shards` from 4 to 8 changed nothing but memory. Measure before tuning.

Keep in mind that no amount of asynchrony can hide more IO than you have compute to
hide it behind. If a workflow writes more bytes per step than the filesystem can absorb
in the time the model takes to produce them, the wall clock is the IO time and the only
remedies are writing less data or a faster store.

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
ranks and `shard_coords={"time": 4}`, a common contiguous block split gives rank 0 ICs
0-2 and rank 1 ICs 3-5. The first time shard covers ICs 0-3 and therefore straddles two
ranks. Both buffer a partial shard, both flush it whole, and one rank's output is lost.
Note that the read-modify-write fallback does not protect you here: both ranks check for
the shard before either has written it, so both take the full overwrite path.

Such a layout is only safe when each rank's slice is shard aligned, which requires both
that the work divides evenly across ranks and that each rank's count is a multiple of the
shard size. Since work splitting commonly gives leftover items to the first few ranks,
that alignment is easy to lose. Prefer the first layout.

Separately, and independent of sharding: arrays are created lazily on the first write, so
several ranks writing a new array at once can race on its creation. Have one rank
establish the arrays before the others begin writing.
