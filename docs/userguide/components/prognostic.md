# Prognostic Models { #prognostic_model_userguide }

Prognostic models forecast future atmospheric states from initial conditions.
In Earth2Studio they provide a set of models designed to perform time
integration. For example, given a set of atmospheric fields at a particular time,
the model auto-regressively predicts the same fields (typically six hours for many models)
into the future.

The usage of prognostic models falls into two categories, which are commonly achieved
through two different APIs:

- Single time-step predictions
- Time-series predictions

The list of prognostic models that are already built into Earth2Studio can be found in
the API documentation [earth2studio.models.px](../../modules/models_px.md).
For a complete workflow example, refer to `earth2studio.run.deterministic`.

## Prognostic Interface

The full requirements for a standard prognostic model are defined explicitly in the
`earth2studio/models/px/base.py`.

```python
--8<-- "earth2studio/models/px/base.py:prognostic-model-interface"
```

!!! note
    Prognostic models do not need to inherit this protocol, this is used to define
    the required APIs. Prognostic models can maintain their internal state when using the
    iterator if necessary.

Prognostic models also tend to extend two classes:

- `earth2studio.models.px.utils.PrognosticMixin`: A utility class that
defines iterator hooks used in all the built-in models. These provide a finer level
of control over the time-series prediction of models.
- `earth2studio.models.auto.AutoModel`: Defines APIs for models that have
checkpoints that can be auto-downloaded and cached. Refer to
[AutoModels](../advanced/auto.md#automodel_userguide) for additional details.

## Prognostic Usage

### Loading a Pre-trained Prognostic

The following two commands can be used to download and load a pre-trained built
prognostic model.
More information on automatic downloading of checkpoints can be found in the
[AutoModels](../advanced/auto.md#automodel_userguide) section.

```python
from earth2studio.models.px import PrognosticModel

model_package = PrognosticModel.load_default_package()
model = PrognosticModel.load_model(model_package)
```

### Single Step Prediction

A prognostic model can be called for a single time-step using the call function.
The function takes a field DataArray with labelled coordinates (refer to
[Data Movement](../about/overview.md#data_userguide) for the structure) and
returns the predicted output.

```python
from earth2studio.utils import handshake_dataarray, handshake_time

# Assume model is an instance of a PrognosticModel
signature = model.input_coords()
x = fetch_data(source, time, signature["variable"].values, signature.lead_time.values)
# For a matching native lat/lon source, select the configured domain explicitly.
# Regrid incompatible source geometry before validation; target_grid is currently
# a pass-through, not a regridding operation.
x = x.sel(lat=signature.lat.values, lon=signature.lon.values)
handshake_time(x)
handshake_time(x, "lead_time")
lead = x.lead_time.values
handshake_dataarray(x.assign_coords(lead_time=lead - lead[-1]), signature)
x = model(x)  # Predict a single time-step
```

The standard `handshake_dim`, `handshake_coords`, and `handshake_size` utilities
accept DataArrays as well as legacy coordinate dictionaries. They inspect dimension
order, labels and sizes without reading field values. `handshake_time` validates
finite datetime/timedelta labels; `handshake_metadata` compares named attributes.
The two-argument `handshake_dataarray` compares dimensions, labels, and declared
grid ID, CRS, and statistics metadata. Normalize relative history explicitly before
comparing, as above. `handshake_nonempty` rejects unresolved axes at execution
boundaries; `output_coords()` accepts dynamic coordinate declarations for planning.

`handshake_device(array, expected_device)` checks concrete field storage against a
device supplied by the caller: NumPy is CPU and CuPy carries an exact CUDA index.
It returns `None` on a match and raises `ValueError` on a mismatch, without reading
field values, converting tensors, copying, or transferring data. Lazy arrays,
coordinate signatures, other storage types, and expected devices other than CPU
or CUDA raise `TypeError`. CPU indices normalize to `cpu`; unindexed `cuda` uses
`torch.cuda.current_device()` and therefore requires a working CUDA runtime.

Models that require input on their device must call this helper explicitly at
execution boundaries, before `x.e2s.to_torch()` or device transfers. Pass the
`.device` of an existing buffer or parameter; the helper does not inspect models.
For example, a device check in SFNO's `__call__` can use its existing
`device_buffer` before conversion:

```python
from earth2studio.utils import handshake_device

# Inside SFNO.__call__, after coordinate/time validation:
handshake_device(x, self.device_buffer.device)
tensor, _ = x.e2s.to_torch()
```

Apply the same check to iterator inputs and fields returned by input hooks before
conversion when those paths execute independently. This is an opt-in utility;
existing model entry points do not automatically enforce it. Keep device checks
out of `input_coords()` and `output_coords()`, which support allocation-free
planning signatures.

### Time-series Prediction

To predict a time-series, the `create_iterator` method can be used to create an iterable
data source to generate time-series data as the model rolls out.

```python
# Assume model is an instance of a PrognosticModel
model_iterator = model.create_iterator(x)  # x is a field DataArray
for step, x in enumerate(model_iterator):
    # Perform operations for each time-step
    # First output should always be time-step 0 (the input)
    print(x.lead_time.values)
```

## Custom Prognostic Models

To integrate your own prognostic, satisfy the interface above.
We recommend that you review the [extension examples](../../examples/index.md#extend)
examples, which walk you through implementing a custom prognostic.

## Contributing a Prognostic Model

Want to add your prognostic to the package? We are happy to work with you.
At the minimum we expect the model to abide by the defined interface and meet
the requirements set forth in our contribution guide. Typically, you are expected
to provide the weights of the model in a downloadable location that can be fetched.

[Open an issue](https://github.com/NVIDIA/earth2studio/issues) when you have an initial
implementation you would like us to review.
