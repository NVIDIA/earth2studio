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

`AssimilationModel` defines independent execution (`__call__`), `input_coords`,
`output_coords`, `default_sources`, and `to`. It does not require
an iterator. `StatefulAssimilationModel` extends it with `forcing_coords`, `initialize`, `step`, and
`create_iterator` for continuing a trajectory with explicit state.

```python
--8<-- "earth2studio/models/da/base.py:stateful-assimilation-model-interface"
```

Data assimilation models can work with gridded DataArrays, tabular DataFrames, or both,
including their outputs.
For tabular inputs, models commonly use a `FrameSchema` to describe fields and
constraints. For gridded inputs, models use allocation-free `CoordinateSystem`
DataArray signatures, following the same convention described
in [Data Movement](../about/overview.md#data_userguide).

## Data Assimilation Usage

### Background and Forcing Slots

`input_coords()` describes all arguments of independent models, including observations.
For stateful models it describes initialization inputs; `forcing_coords()` declares
additional caller-supplied step conditioning, including observations.
Each returns a single signature or schema, an ordered tuple for multiple slots, or
`None` for no slots. A tuple can mix gridded signatures and DataFrame schemas.
`output_coords()` accepts input signatures/schemas as separate positional arguments.

| Model | Inputs, in order | Forcing, in order | Execution |
| --- | --- | --- | --- |
| `InterpEquirectangular` | Observation schema | — | Independent calls |
| `HealDA` | Conventional schema, satellite schema | — | Independent calls |
| `CorrDiffCosmoEra5SDA` | ERA5 DataArray, observation schema | — | Independent calls |
| `StormCastSDA` | HRRR DataArray | Conditioning, observations | Independent or stateful |
| `StormCastConusSDA` | HRRR DataArray | Conditioning, observations | Independent or stateful |

A background input does not imply stateful execution: CorrDiff analyzes each
supplied ERA5 background independently. Stateful continuation is an explicit
capability, not something inferred from `input_coords()`.

`default_sources()` recommends providers in **input-then-forcing** order. Models
do not fetch these themselves. Callers fetch and align each gridded slot to the
declared signature.

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

### Independent Analyses

The main work of a data assimilation model is the `__call__` function. It accepts one
or more observation or state inputs and returns one or more assimilated outputs.

```python
# Assume model is an instance of an AssimilationModel
analysis = model(observations)  # e.g. InterpEquirectangular
```

```python
# Observation-only model, e.g. InterpEquirectangular
for observations in observation_batches:
    analysis = model(observations)

# CorrDiff: callers supply each independent background and observation batch.
for background, observations in cases:
    analysis = corrdiff(background, observations)
```

HealDA receives `(conv_obs, sat_obs)` and needs at least one observation batch.
CorrDiff accepts an observation DataFrame or `None` alongside the background.
Interpolation receives one DataFrame per analysis. Consult each model's documentation
for its time-window and metadata requirements.

### Stateful Assimilation

`StatefulAssimilationModel` follows the prognostic explicit-state convention.
`initialize` accepts initial fields followed by initial forcing and returns the
**first computed analysis** and model-defined state. `step` accepts the previous
output, new forcing, and state, returning the next analysis and state. For StormCastSDA:

```python
analysis, state = model.initialize(x0, conditioning0, obs0)
analysis, state = model.step(analysis, conditioning1, None, state=state)
```

The pair `(analysis, state)` is the complete continuation. `step` does not mutate
either argument; retain both to checkpoint or branch a trajectory.

Alternatively, `create_iterator` manages that pair for the caller:

```python
iterator = model.create_iterator(x0, conditioning0, obs0)
analysis = next(iterator)  # First computed analysis, one hour after x0
analysis = iterator.send((conditioning1, None))  # Next analysis without observations
iterator.close()
```

There is no priming yield or initial-state yield: every yield is a computed output.
Initial forcing belongs in `create_iterator`; later sends supply forcing for the
next step in `forcing_coords()` order. StormCastSDA always requires conditioning,
even when optional observations are `None`. Thus subsequent `next(iterator)` or
`send(None)` cannot replace `send((conditioning, None))`. Conditioning is aligned
to the input valid time, and observations to the output valid time one hour later.

Return one output directly, or multiple objects in declared order (`return y1, y2`).
Independent models declare observations in `input_coords()`; only stateful models
declare step conditioning and observations in `forcing_coords()`.

## Custom Data Assimilation Models

To integrate your own data assimilation model, satisfy the interface above and keep the
input and output schemas explicit. Declare arguments with `input_coords()`,
additional stateful conditioning with `forcing_coords()`, and produced outputs
with `output_coords()`. Declare fixed named
execution parameters in input-then-forcing order. Implement the stateful extension
only when outputs and explicit state continue a trajectory; independent models need
no iterator. Output declarations use `CoordinateSystem` for DataArrays and
`FrameSchema` for DataFrames. Use `.e2s.to_torch()` and
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
