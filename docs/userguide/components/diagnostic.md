# Diagnostic Models { #diagnostic_model_userguide }

Diagnostic models in Earth2Studio provide a set of models that are independent of time,
focused on predicting new or modified values.
For example, given an instantaneous set of atmospheric fields, a diagnostic model can
predict a new field, such as precipitation.
These models differ from
[Prognostic Models](prognostic.md#prognostic_model_userguide) because they do not
perform time integration (stepping a model forward through multiple time steps).
Although statistics and metrics could be considered diagnostic in a broad sense,
Earth2Studio reserves the term for models (numerical or AI) that predict or
calculate derived physical quantities.
The term is not for standard mathematical reductions used in analysis.

The list of diagnostic models that are already built into Earth2Studio can be found in
the API documentation [earth2studio.models.dx](../../modules/models_dx.md).

## Diagnostic Interface

The full requirements for a standard diagnostic model are defined explicitly in the
`earth2studio/models/dx/base.py`.

```python
--8<-- "earth2studio/models/dx/base.py:25:79"
```

!!! note
    Diagnostic models do not need to inherit this protocol, this is only used to define
    the required APIs.

Diagnostic models also tend to extend one class:

* `earth2studio.models.auto.AutoModel`: Defines APIs for models that have
checkpoints that can be auto-downloaded and cached. Refer to the
[AutoModels](../advanced/auto.md#automodel_userguide) guide for additional
details.

## Diagnostic Usage

### Loading a Pre-trained Diagnostic

The following two commands can be used to download and load a pre-trained built
diagnostic model.
More information on automatic downloading of checkpoints can be found in the
[AutoModels](../advanced/auto.md#automodel_userguide) section.

```python
from earth2studio.models.dx import DiagnosticModel

model_package = DiagnosticModel.load_default_package()
model = DiagnosticModel.load_model(model_package)
```

### Prediction

The main work of diagnostic models is the `__call__` function, which takes in
a data tensor and coordinate system (refer to
[Data Movement](../about/overview.md#data_userguide) for the structure) and
returns the primary output.

```python
# Assume model is an instance of a DiagnosticModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = model(x, coords)  # Predict a single time-step
```

## Custom Diagnostic Models

To integrate your own diagnostic, satisfy the interface above.
We recommend reviewing the [extension examples](../../examples/index.md#extend)
examples, which walk you through implementing a custom diagnostic model.

## Contributing a Diagnostic Model

Want to add your diagnostic to the package? We are happy to work with you.
We expect the model to abide by the defined interface and meet
the requirements set forth in our contribution guide. Typically, you are expected
to provide the weights of the model in a downloadable location that can be fetched.

Open an issue when you have an initial implementation you would like us to review. If
you are aware of an existing model and want us to implement it, open a feature request
and we will get it triaged.
