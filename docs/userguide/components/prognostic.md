(prognostic_model_userguide)=

# Prognostic Models

Prognostic models in Earth2Studio provide a set of models designed to perform time
integration. For example, given a set of atmospheric fields at a particular time,
the model auto-regressively predicts the same fields 6 hours into the future.

The usage of prognostic models falls into two categories which are commonly achieved
through two different APIs:

1. Single time-step predictions
2. Time-series predictions

The list of prognostic models that are already built into Earth2Studio can be found in
the API documentation {ref}`earth2studio.models.px`.

## Prognostic Interface

The full requirements for a standard prognostic model are defined explicitly in the
`earth2studio/models/px/base.py`.

```{literalinclude} ../../../earth2studio/models/px/base.py
:lines: 25-
:language: python
```

:::{note}
Prognostic models do not need to inherit this protocol, this is simply used to define
the required APIs. Prognostic models can maintain their internal state when using the
iterator if necessary.
:::

Prognostic models also tend to extend two classes:

1. {class}`earth2studio.models.px.utils.PrognosticMixin`: A utility class that
defines iterator hooks used in all the built-in models. These provide a finer level
of control over the time-series prediction of models.
2. {class}`earth2studio.models.auto.AutoModel`: Defines APIs for models that have
checkpoints that can be auto downloaded and cached. See the {ref}`automodel_userguide`
guide for additional details.

## Prognostic Usage

### Loading a Pre-trained Prognostic

The following two commands can be used to download and load a pre-trained built
prognostic model.
More information on automatic downloading of checkpoints can be found in the
{ref}`automodel_userguide` section.

```python
from earth2studio.models.px import PrognosticModel

model_package = PrognosticModel.load_default_package()
model = PrognosticModel.load_model(model_package)
```

### Single Step Prediction

A prognostic model can be called for single time-step using the call function.

```python
# Assume model is an instance of a PrognosticModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = model(x, coords)  # Predict a single time-step
```

### Time-series Prediction

To predict a time-series, the create generator API can be used to create an iterable
data source to generate time-series data as the model rolls out.

```python
# Assume model is an instance of a PrognosticModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
model_iterator = model.create_iterator(x, coords)  # Create iterator for time integration
for step, (x, coords) in enumerate(model_iterator):
    # Perform operations for each time-step
    # First output should always be time-step 0 (the input)
```

## Custom Prognostic Models

Integrating your own prognostic is easy, just satisfy the interface above.
We recommend users have a look at the custom prognostic example which will step users
through the simple process of implementing their own prognostic model for their personal
needs in the {ref}`extension_examples` examples.

## Contributing a Prognostic Model

Want to add your prognostic to the package? Great, we will be happy to work with you.
At the minimum we expect the model to abide by the defined interface as well as meet
the requirements set forth in our contribution guide. Typically, users are expected
to provide the weights of the model in a downloadable location that can be fetched.

Open an issue when you have an initial implementation you would like us to review.
