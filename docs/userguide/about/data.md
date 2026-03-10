(data_userguide)=

# Data Movement

:::{admonition} Keeping Data Transparent
:class: tip
"Show me your code and conceal your data structures, and I shall continue to be
mystified. Show me your data structures, and I won't usually need your code; it'll be
obvious." - *Fred Brooks*
:::

Earth2Studio aims to keep data simple and interpretable between components.
Given that this package interacts with geo-physical data, the common data-structure
inside workflows is the pairing of:

1. A PyTorch tensor (`torch.Tensor`) on the inference device holding the array data of
interest.
2. An OrderedDict of numpy arrays (`CoordSystem`) that
represents the geo-phyiscal coordinate system the tensor represents.

For example, perturbation methods operate by using a data tensor and coordinate system
to generate a noise tensor:

```{literalinclude} ../../../earth2studio/perturbation/base.py
    :lines: 25-
    :language: python
```

In later sections, users will find that most components have APIs that either generate
or interact with these two data structures.
The combination of both the data tensor and respective coordinate system provides
complete information one needs to interpret any stage of a workflow.

:::{note}
Data is **always** moved between components using unnormalized physical units.
:::

(coordinates_userguide)=

## Coordinate Systems

As previously discussed, coordinate dictionaries are a critical part of Earth-2
Inference Studio's data movement.
We wanted the coordinate object to be a fairly primitive data object that allows users
to interact with the data outside the project and keep things transparent in workflows.
Inside the package these are typed as `CoordSystem` which is defined as the following:

```python
CoordSystem = NewType("CoordSystem", OrderedDict[str, np.ndarray])
```

The dictionary is ordered because the keys correspond to the dimensions of the associated
data tensor.
Let's consider a basic example of a 2D lat-lon grid:

```python
x = torch.randn(181, 360)

coords = OrderedDict({
    "lat": np.linspace(-90, 90, 181),
    "lon": np.linspace(0, 360, 360, endpoint=False)
})
```

Much of Earth2Studio typically operates on a lat-lon grid but it's not
required to.

### Standard Coordinate Names

Earth2Studio has a dimension naming standard for its built in feature set.
We encourage users to follow similar naming schemes for compatibility between Earth-2
Inference Studio when possible and the packages we interface with.

```{list-table}
:widths: 15 40 25
:header-rows: 1

* - Key
  - Description
  - Type
* - `batch`
  - Dimension representing the batch dimension of the data. Used to denote a "free"
  dimension, consult batching docs for more details.
  - `np.empty(0)`
* - `time`
  - Time dimension, represented by numpy arrays of datetime objects.
  - `np.ndarray[np.datetime64[ns]]` (`TimeArray`)
* - `lead_time`
  - Lead time is used to denote a dimension that indexes over forecast steps.
  - `np.ndarray[np.timedelta64[ns]]` (`LeadTimeArray`)
* - `variable`
  - Dimension representing physical variable (atmospheric, surface). Earth-2
  Inference Studio has its own naming convention. See {ref}`lexicon_userguide` docs
  for more details.
  - `np.ndarray[str]` (`VariableArray`)
* - `lat`
  - Latitude coordinate array, [-90, 90] is standard
  - `np.ndarray[float]`
* - `lon`
  - Longitude coordinate array, [0, 360) is standard
  - `np.ndarray[float]`
```

:::{note}
`np.empty(0)` is used to denote variable axis in the coordinate system. Namely a
dimension in the tensor that can be of any size, greater than 0. Typically this is the
`batch` dimension.
:::

### Coordinate Utilities

The downside of using a dictionary to store coordinates is that manipulating the data
tensor and then updating the coordinate array is a manual process.
To help make this process less tedious, Earth2Studio has several utility
functions that make interacting with coordinates easier.
The bulk of these can be found in the [Earth2Studio Utilities](earth2studio.utils_api).

:::{warning}
🚧 Under construction, todo: add some example here! 🚧
:::

### Model Coordinates

Models are components where coordinate systems are essential both, for defining what
data is needed and also what data is produced. Both {ref}`prognostic_model_userguide`
and {ref}`diagnostic_model_userguide` require two functions that serve coordinate
information:

- `input_coords()` : A function that returns the expected input coordinate
system of the model. A new dictionary should be returned every time.

- `output_coords()` : A function that returns the expected output coordinate
system of the model *given* an input coordinate system. This function should also
validate the input coordinate dictionary.

For an example, consider the FourCastNet model's implementations:

```{literalinclude} ../../../earth2studio/models/px/fcn.py
    :start-after: "# sphinx - coords start"
    :end-before: "# sphinx - coords end"
    :language: python
```

The `input_coords()` function provides the coordinate spec of the input tensor expected by the
model. In the case of FourCastNet, it's expecting a input tensor of shape `[...,1,26,720,1440]`
which corresponds to `[...,lead,variables,lat,lon]`.

The `output_coords()` function provides validation and transformation of coordinates
by the model. That is, you should be able to deduce the models output data without executing
the forward pass of the model.
This function requires an input coordinate system that represent the input data it
received.
The first step is validating the `input_coords` using the coordinate handshake utils
functions.
Next the output coordinate systems are built.
This function is a place to store the complexity of a model's forward process for
other Earth2Studio components.

We encourage users to explore the coordinate transforms of models to learn more about
how they operate:

```python
from earth2studio.models.px import FCN
model = FCN(None, None, None)
input_coords = model.input_coords()
output_coords = model.output_coords(input_coords)
print("Input  lead:", input_coords['lead_time'])
print("Output lead:", output_coords['lead_time'])
output_coords = model.output_coords(output_coords)
print("Output lead 2:", output_coords['lead_time'])
```

The output of the following script will be:

```console
Input  lead: [0]
Output lead: [6]
Output lead 2: [12]
```

:::{note}
The batch dimension was not discussed intentionally. Think about it like a free dimension.
More information can be found in the {ref}`batch_function_userguide` section.
:::

## Inference on the GPU

It is beneficial to leverage the GPU for as many processes as possible.
Earth2Studio aims to get data from the data source and immediately convert
it into the tensor, coord data struction on the device.
From there, the data is kept on the GPU until the very last moment when writes are
needed to in-memory or to file.

```{figure} https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/0.1.0/docs/e2studio-data.png
:alt: earth2studio-data
:width: 600px
:align: center
```

In the figure above, that the data is first pulled from the data source as an Xarray
data array which is then then converted to a tensor.
The data remain on the device, denoted by the GPU boundary, until it needs to be written
by the IO component.

:::{admonition} Data Sources and Xarray
:class: tip
This may raise the question: Why do datasources not output directly to tensor and
coordinate dictionaries?
This is an opinionated decision due to the fact that these data sources need to store
data on the CPU regardless and can be extremely useful outside of the context of this package.
Thus they return Xarray data arrays which are is nothing more than a fancy data array
with a coordinate system attached to it!
:::
