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
constraints. For tensor inputs, models use the same `CoordSystem` convention described
in [Data Movement](../about/overview.md#data_userguide).

## Data Assimilation Usage

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
analysis, = model(observations)
```

### Stateful Assimilation

Some assimilation workflows need to process a sequence of observation batches while
maintaining internal state. For those workflows, use `create_generator`.

```python
# Assume model is an instance of an AssimilationModel
generator = model.create_generator()
generator.send(None)  # Prime the generator

for observations in observation_batches:
    analysis, = generator.send(observations)

generator.close()
```

## Custom Data Assimilation Models

To integrate your own data assimilation model, satisfy the interface above and keep the
input and output schemas explicit. The model should advertise initialization
requirements with `init_coords()`, accepted inputs with `input_coords()`, and produced
outputs with `output_coords()`.

We recommend reviewing the [extension examples](../../examples/index.md#extend), which
show the style expected for adding custom Earth2Studio components.

## Contributing a Data Assimilation Model

Want to add a data assimilation model to the package? We are happy to work with you.
We expect the model to abide by the defined interface and meet the requirements set
forth in our contribution guide. Typically, you are expected to provide any required
weights or assets in a downloadable location that can be fetched.

Open an issue when you have an initial implementation you would like us to review. If
you are aware of an existing model and want us to implement it, open a feature request
and we will get it triaged.
