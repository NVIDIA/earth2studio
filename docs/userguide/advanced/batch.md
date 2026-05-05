(batch_function_userguide)=

# Batch Dimension

This section of the user guide expands on how batching is handled inside Earth2Studio.
As discussed in [data movement](data_userguide) section, there is a dedicated coordinate
axis `batch` which is commonly used in many of the model implementations.
`batch` represents a dynamic axis that can be of any size, enabling models to better
utilize compute resources.

Batch dimensions have the following rules:

- Must be able to support any nonzero size
- Must be the leading dimensions of the coordinate system
- Must be set to {py:obj}`np.empty(0)` in objects coordinate property

## Good Coordinate Definitions

`batch` is the leading dimension with a value of {py:obj}`np.empty(0)`.

```python
coords = OrderedDict(
        {
            "batch": np.empty(0),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
```

Other coordinates can have a value of {py:obj}`np.empty(0)` to denote an additional
dynamic axis but imply a required data type.
In this case, this model supports a batch but the `time` axis must be a
Numpy array of type {py:obj}`np.datetime64`. See [data movement](coordinates_userguide)
section for expected types.

```python
coords = OrderedDict(
        {
            "batch": np.empty(0),
            "time": np.empty(0),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
```

## Bad Coordinate Definitions

Batch dimension is not leading.

```python
coords = OrderedDict(
        {
            "variable": np.array(VARIABLES),
            "batch": np.empty(0),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
```

Batch dimension is not of size 0.

```python
coords = OrderedDict(
        {
            "batch": np.zeros(1),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
```

## Batch Decorator

While the use of `batch` dimension is useful for communicating a dimension that can
accept variable input sizes, it's not very convenient to manually manipulate data into a
form that matches the batch dimension.
To make using batch supporting models easier, Earth2Studio offers a utility
decorator {py:class}`earth2studio.models.batch.batch_func` which automates transforming
extra leading dimensions into a batch one.
This utility *must* be used in an object with coordinate properties.

The batch function does the following steps:

1. Squeeze leading dims into a batch dimension
2. Update batched input coordinates with a single batch index dimension
3. Execute the wrapped function and get outputs
4. Replace output batch coord with the batched input coordinates
5. Unsqueeze the leading batch coordinate into original input dimensions

Consider the following example:

```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.batch import batch_func, batch_coords


class BatchModel:
    input_coords = OrderedDict({"batch": np.zeros(0), "dim1": np.arange(2)})

    @batch_coords()
    def output_coords(
        self,
        input_coords: OrderedDict
        ) -> OrderedDict:
        return OrderedDict({"batch": np.zeros(0), "dim2": np.arange(4)})

    @batch_func()
    def __call__(self, input, coords):
        print("Model Input:", input.size(), coords)
        out = torch.cat([input, input], dim=-1)
        out_c = self.output_coords(coords).copy()
        return out, out_c


input_coords = OrderedDict(
    {"batched_dim0": np.arange(2), "batched_dim1": np.arange(3), "dim1": np.arange(2)}
)
input = torch.randn(2, 3, 2)

model = BatchModel()
print("Input:", input.size(), input_coords)
output, output_coords = model(input, input_coords)
print("Output:", output.size(), output_coords)
```

The output of the following script will be:

<!-- markdownlint-disable MD013 -->
```console
Input: torch.Size([2, 3, 2]) OrderedDict([('batched_dim0', array([0, 1])), ('bacthed_dim1', array([0, 1, 2])), ('dim1', array([0, 1]))])

Model Input: torch.Size([6, 2]) OrderedDict([('batch', array([0, 1, 2, 3, 4, 5])), ('dim1', array([0, 1]))])

Output: torch.Size([2, 3, 4]) OrderedDict([('batched_dim0', array([0, 1])), ('bacthed_dim1', array([0, 1, 2])), ('dim2', array([0, 1, 2, 3]))])
```
<!-- markdownlint-enable MD013 -->

Note that the leading two dimensions were squeezed into a single batch dimension before
the execution of the models {py:func}`BatchModel.__call__`.
The leading dimensions were then restored back while preserving the updated domain
coordinates from the model's output.

The batch decorator will also unsqueeze a batch axis to an input that is missing *only*
`batch` from the input coordinate system with no additional dimensions.
In this instance a batch size of one is implied.
For example, using the model in the example above:

```python
input_coords = OrderedDict(
    {"dim1": np.arange(2)}
)
input = torch.randn(2)

model = BatchModel()
print("Input:", input.size(), input_coords)
output, output_coords = model(input, input_coords)
print("Output:", output.size(), output_coords)
```

will execute successfully with an output of:

```console
Input: torch.Size([2]) OrderedDict([('dim1', array([0, 1]))])

Model Input: torch.Size([1, 2]) OrderedDict([('batch', array([0])), ('dim1', array([0, 1]))])

Output: torch.Size([4]) OrderedDict([('dim2', array([0, 1, 2, 3]))])
```

## Batch Dimension in IO

The IO backends require users to pre-define the output coordinate system on which the
data will be exported.
Typically, a good way to do this is to review the output coordinate system of the model,
which will typically include a `batch` dimension.
But the model won't actually return a batch dimension, thus it's a common pattern to
replace this batch dimension with whatever leading coordinate the input will have.

For example, refer to the built-in workflows.
The setup process for the IO backend is handled in a general manner by first getting the
output coordinates of the model, removing empty dimensions such as `batch` and then
prepending known leading dimensions like `time`.

```python
total_coords = prognostic.output_coords.copy()
for key, value in prognostic.output_coords.items():
    if value.shape == (0,):
        del total_coords[key]
total_coords["time"] = time
total_coords["lead_time"] = np.asarray(
    [prognostic.output_coords["lead_time"] * i for i in range(nsteps + 1)]
).flatten()
total_coords.move_to_end("lead_time", last=False)
total_coords.move_to_end("time", last=False)
var_names = total_coords.pop("variable")
io.add_array(total_coords, var_names)
```
