(diagnostic_model_userguide)=

# Diagnostic Models

Diagnostic models in Earth2Studio provide a set of models that are independent of time,
focused on predicting new or modified values.
For example, given an instantaneous set of atmospheric fields, a diagnostic model can
predict a new field, such as precipitation.
These models differ from {ref}`prognostic_model_userguide` because they do not perform
time integration.
Calculations such as statistics or metrics could fall into a diagnostic classification,
but we distinguish that diagnostic models are in fact models (numerical, AI) used
to predict and calculate derived physical processes or quantities.
They are not standard mathematical calculations or reductions for the purpose of analysis.

The list of diagnostic models that are already built into Earth2Studio can be found in
the API documentation {ref}`earth2studio.models.dx`.

## Diagnostic Interface

The full requirements for a standard diagnostic model are defined explicitly in the
`earth2studio/models/dx/base.py`.

```{literalinclude} ../../../earth2studio/models/dx/base.py
:lines: 25-
:language: python
```

:::{note}
Diagnostic models do not need to inherit this protocol, this is only used to define
the required APIs.
:::

Diagnostic models also tend to extend one class:

1. {class}`earth2studio.models.auto.AutoModel`: Defines APIs for models that have
checkpoints that can be auto downloaded and cached. See the {ref}`automodel_userguide`
guide for additional details.

## Diagnostic Usage

### Loading a Pre-trained Diagnostic

The following two commands can be used to download and load a pre-trained built
diagnostic model.
More information on automatic downloading of checkpoints can be found in the
{ref}`automodel_userguide` section.

```python
from earth2studio.models.dx import DiagnosticModel

model_package = DiagnosticModel.load_default_package()
model = DiagnosticModel.load_model(model_package)
```

### Prediction

The main work of diagnostic models is the {func}`__call__` function, which takes in
a data tensor with coordinate system and returns the primary output.

```python
# Assume model is an instance of a DiagnosticModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = model(x, coords)  # Predict a single time-step
```

## Custom Diagnostic Models

Integrating your own diagnostic is easy, just satisfy the interface above.
We recommend that you review the {ref}`extension_examples` examples, which will
step you through the process of implementing your own diagnostic model.

## Contributing a Diagnostic Model

Want to add your diagnostic to the package? Great, we will be happy to work with you.
At the minimum we expect the model to abide by the defined interface and meet
the requirements set forth in our contribution guide. Typically, you are expected
to provide the weights of the model in a downloadable location that can fetched.

Open an issue when you have an initial implementation you would like us to review. If
you're aware of an existing model and want us to implement it, open a feature request
and we will get it triaged.
