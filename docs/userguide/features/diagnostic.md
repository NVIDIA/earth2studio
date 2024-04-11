(diagnostic_model_userguide)=

# Diagnostic Models

Diagnostic models in Earth2Studio provides a set of models that are independent of time,
focused on predicting new or modified values.
For example, given a instanteous set of atmospheric fields a diagnostic model may
predict a new field such as precipitation.
These models differ from {ref}`prognostic_model_userguide` since they do not perform
time integration.
Calculations such as statistics or metrics could fall into a diagnostic classification,
but we make the distinction that diagnostic models are in fact models used to predict
physical processes. Not standard mathematical calculations / reductions the purpose of
analysis of those physics.

The list of diagnostic models that are already built into Earth2studio can be found in
the API documentation {ref}`earth2studio.models.dx`.

## Diagnostic Interface

The full requirements for a standard diagnostic model our defined explicitly in the
`earth2studio/models/dx/base.py`.

```{literalinclude} ../../../earth2studio/models/dx/base.py
    :lines: 25-
    :language: python
```

:::{note}
Diagnostic models do not need to inherit this protocol, this is simply used to define
the required APIs.
:::

Diagnostic models also tend to extend one class:

1. {class}`earth2studio.models.auto.AutoModel`: Defines APIs for models that have
checkpoints that can be auto downloaded and cached. See the Automodel guide for
additional details.

## Diagnostic Prediction

The work horse of diagnostic models is the {func}`__call__` function which takes in
a data tensor with coordinate system and returns the primary output.

```python
# Assume model is an instance of a DiagnosticModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = model(x, coords)  # Predict a single time-step
```

## Custom Diagnostic Models

Integrating your own diagnostic is easy, just satisfy the inferface above.
We recommend users have a look at the custom prognostic example which will step users
through the simple process of implementing you're own diagnostic model for your personal
needs in the {ref}`extension_examples` examples.

## Contributing a Diagnostic Models

Want to add your diagnostic to the package? Great, we will be happy to work with you.
At the minimum we expect the model to abide by the defined interface as well as meet
the requirements set forth in our contribution guide. Typically we also expect users
to provide the weights of the model in a downloadable location that can be progmatically
fetched.

Open an issue when you have an initial implementation you would like us to review. If
you're aware of an existing model and want us to implemented it, open a feature request
and we will get it triaged.
