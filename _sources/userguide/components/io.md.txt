(output_handling_userguide)=

# Output Handling

While input data handling is primarily managed by the data sources in
{mod}`earth2studio.data`, output handling is managed by the IO backends available
in {mod}`earth2studio.io`.
These backends are designed to balance the ability for users to customize the arrays and
metadata within the exposed backend while also making it easy to design reusable
workflows.

The key extension of the typical `(x, coords)` data structure movement throughout
the rest of the `earth2studio` code and output store compatibility is the notion of
an `array_name`. Names distinguish between different arrays within the backend and
are currently a requirement for storing `Datasets` in `xarray`, `zarr`, and `netcdf`.
This means that you must supply a name when adding an array to a store or when
writing an array. A frequent pattern is to extract one dimension of an array,
such as `"variable"` to act as individual arrays in the backend, see the examples below.

## IO Backend Interface

The full requirements for a standard IO backend are defined explicitly in the
`earth2studio/io/base.py`.

```{literalinclude} ../../../earth2studio/io/base.py
:lines: 24-
:language: python
```

:::{note}
IO Backends do not need to inherit this protocol; this is simply used to define
the required APIs. Some built-in IO backends also may offer additional functionality
that is not universally supported (and hence not required).
:::

There are two important methods that must be supported: `add_array`, which
adds an array to the underlying store and any attached coordinates, and `write`,
which explicitly stores the provided data in the backend.
The `write` command may induce synchronization if the input tensor resides on the GPU
and the store.
Most stores make a conversion from PyTorch to numpy in this process.
The {mod}`earth2studio.io.kv` backend has the option for storing data on the GPU, which
can be done asynchronously.

Most data stores offer several additional utilities such as `__contains__`,
`__getitem__`, `__len__`, and `__iter__`. For examples, see the implementation in
{mod}`earth2studio.io.ZarrBackend`:

```{literalinclude} ../../../earth2studio/io/zarr.py
    :language: python
    :start-after: sphinx - io zarr start
    :end-before: sphinx - io zarr end
```

Because of `datetime` compatibility, we recommend using the `ZarrBackend` as a default.

## Initializing a Store

A common data pattern seen throughout our example workflows is to initialize the
variables and dimensions of a backend using a complete `CoordSystem`. For example:

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

It can be tedious to define each coordinate and dimension, luckily if we have
a prognostic or diagnostic model, most of this information is already available.
Here is a robust example of such a use-case:

```python
# Set up IO backend
# assume we have `prognostic model`, `time` and `array_name`
# Copy prognostic model output coordinates
total_coords = OrderedDict(
    {
        k: v for k, v in prognostic.output_coords(prognostic.input_coords).items() if
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

Prognostic models, diagnostic models, statistics, and metrics are required to have a
`output_coords` method which maps from an input coordinate to a corresponding output
coordinate. This method is meant to simulate the result of `__call__` without having
to actually compute the forward call of the method. Review the API documentation for more details.

Another common IO use-case is to extract a particular dimension (usually `variable`) as
the array names.

```python
# A modification of the previous example:
var_names = total_coords.pop("variable")
io.add_array(total_coords, var_names)
```

## Writing to the store

After the data arrays have been initialized in the backend, writing to those arrays
is a single line of code.

```python
x, coords = model(x, coords)
io.write(x, coords, array_name)
```

If, as above, you are extracting a dimension of the tensor to use as array names
then you can make use of {mod}`earth2studio.utils.coords.split_coords`:

```python
io.write(*split_coords(x, coords, dim = "variable"))
```
