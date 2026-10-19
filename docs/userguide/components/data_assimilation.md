# Data Assimilation Models { #data_assimilation_model_userguide }

Data assimilation models in Earth2Studio combine observations with a model state or
background estimate to produce an updated state. They are useful when a workflow needs
to ingest observation-like data, correct a forecast or analysis state, or prepare an
initial condition for a downstream model.

Unlike [Prognostic Models](prognostic.md#prognostic_model_userguide), data assimilation
models are not primarily responsible for rolling a forecast forward in time. Unlike
[Diagnostic Models](diagnostic.md#diagnostic_model_userguide), they are usually driven
by observation batches or state-update logic rather than deriving a single physical
field from an existing tensor state.

The list of data assimilation models that are already built into Earth2Studio can be
found in the API documentation [earth2studio.models.da](../../modules/models_da.md).

## Data Assimilation Interface

The full requirements for a standard data assimilation model are defined explicitly in
`earth2studio/models/da/base.py`.

```python
--8<-- "earth2studio/models/da/base.py:assimilation-model-interface"
```

!!! note
    Data assimilation models do not need to inherit this protocol. The protocol defines
    the APIs that built-in workflows and utilities expect.

Data assimilation models can work with tensor data, tabular observation data, or both.
For tabular inputs, models commonly use a `FrameSchema` to describe fields and
constraints. For gridded inputs, models use allocation-free `CoordinateSystem`
DataArray signatures, following the same convention described
in [Data Movement](../about/overview.md#data_userguide).

## Data Assimilation Usage

### State and Forcing Slots

As with prognostic models, `input_coords()` describes the initial state and
`forcing_coords()` describes caller-supplied conditioning, including observations.
Each returns a single signature or schema, an ordered tuple for multiple slots, or
`None` for no slots. A tuple can mix gridded signatures and DataFrame schemas.
`output_coords()` accepts initial-state signatures as separate positional arguments;
observation-only models instead use request metadata such as `request_time`.

| Model | Initial state | Forcing, in order |
| --- | --- | --- |
| `InterpEquirectangular` | None | Observation schema |
| `HealDA` | None | Conventional schema, satellite schema |
| `CorrDiffCosmoEra5SDA` | ERA5 DataArray | Observation schema |
| `StormCastSDA` | HRRR DataArray | Conditioning DataArray, observation schema |

`default_sources()` recommends providers in **input-then-forcing** order. Models
do not fetch these themselves. Callers fetch and align each gridded slot, including
regridding StormCast conditioning to the declared HRRR grid.

### Loading a Pre-trained Data Assimilation Model

Use the concrete data assimilation model class you want to run. When that class
supports automatic packages, the following pattern downloads and loads the pre-trained
weights. More information on automatic downloading of checkpoints can be found in the
[AutoModels](../advanced/auto.md#automodel_userguide) section.

```python
from earth2studio.models.da import HealDA

model_package = HealDA.load_default_package()
model = HealDA.load_model(model_package)
```

### Stateless Assimilation

The main work of a data assimilation model is the `__call__` function. It accepts one
or more observation or state inputs and returns one or more assimilated outputs.

```python
# Assume model is an instance of an AssimilationModel
analysis = model(observations)  # e.g. InterpEquirectangular
```

### Stateful Assimilation

Some assimilation workflows need to process a sequence of observation batches while
maintaining internal state. For those workflows, use `create_generator`.

```python
# Assume model is an instance of an AssimilationModel
generator = model.create_generator()
next(generator)  # Prime without assimilating: yields None or the initial state

for observations in observation_batches:
    analysis = generator.send(observations)

generator.close()
```

Pass initial-state slots to `create_generator`; each `send` supplies only the
forcing slots for the next analysis. A single forcing value is sent directly;
multiple slots are sent as a tuple, preserving optional entries with `None`.
For example, StormCastSDA uses:

```python
generator = model.create_generator(initial_state)
state = next(generator)
# conditioning matches state's valid time; observations target state + one hour
state = generator.send((conditioning, observations))
# next_conditioning must match the updated state window
state = generator.send((next_conditioning, None))
generator.close()
```

HealDA receives `(conv_obs, sat_obs)` and needs at least one observation batch.
CorrDiff receives one observation DataFrame (or `None`) per time in its initial
ERA5 input and stops after those times are consumed. Interpolation receives one
DataFrame per analysis. DataFrame `request_time` metadata identifies requested
analyses where the model requires it. Consult each model's time-window contract.

Single outputs and output signatures are returned directly; only multiple outputs
are tuples. When migrating older DA code, replace `init_coords()` with
`input_coords()` and move observation-schema queries to `forcing_coords()`.

## Custom Data Assimilation Models

To integrate your own data assimilation model, satisfy the interface above and keep the
input and output schemas explicit. The model should advertise initialization
requirements with `input_coords()`, observations and conditioning with
`forcing_coords()`, and produced outputs with `output_coords()`. Declare fixed named
execution parameters in input-then-forcing order. Use `.e2s.to_torch()` and
`from_torch()` at gridded execution boundaries for torch-based computation.

We recommend reviewing the [extension examples](../../examples/index.md#extend), which
show the style expected for adding custom Earth2Studio components.

## Contributing a Data Assimilation Model

Want to add a data assimilation model to the package? We are happy to work with you.
We expect the model to abide by the defined interface and meet the requirements set
forth in our contribution guide. Typically, you are expected to provide any required
weights or assets in a downloadable location that can be fetched.

[Open an issue](https://github.com/NVIDIA/earth2studio/issues) when you have an initial
implementation you would like us to review. If you are aware of an existing model and want
us to implement it, [open a feature request](https://github.com/NVIDIA/earth2studio/issues)
and we will get it triaged.
