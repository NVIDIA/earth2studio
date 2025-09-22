(analysis_model_userguide)=

# Analysis Models

Analysis models in Earth2Studio provides a set of models that serve as inference sinks
and pipeline endpoints.
Unlike {ref}`diagnostic_model_userguide` and {ref}`prognostic_model_userguide` which
transform and return data, analysis models consume data without returning modified
tensors or coordinates.

Analysis models are designed for terminal operations in inference pipelines, such as:

- Complex validation and scoring functions
- Data export to downstream processing systems
- Statistical analysis and metric computation
- Model evaluation and performance assessment
- Data archival and storage operations
- Custom inference utilities and hooks

These models differ fundamentally from diagnostic and prognostic models as they do not
produce data for further pipeline stages, instead serving as endpoints that process,
analyze, or consume the input data.

The list of analysis models that are already built into Earth2studio can be found in the
API documentation {ref}`earth2studio.models.ax`.

## Analysis Interface

The full requirements for a standard analysis model are defined explicitly in the
`earth2studio/models/ax/base.py`.

```{literalinclude} ../../../earth2studio/models/ax/base.py
:lines: 25-
:language: python
```

:::{note}
Analysis models do not need to inherit this protocol, this is simply used to define the
required APIs.
The key distinguishing feature is that the `__call__` method returns `None`, making
these models inference sinks rather than data sources or transformers.
:::

## Analysis Usage

### Execution

Similar to diagnostic models, analysis models implement a {func}`__call__` function
which takes in a data tensor with coordinate system and performs analysis operations
without returning anything.

```python
# Assume model is an instance of an AnalysisModel
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
model(x, coords)  # Perform analysis - returns None
```

Since analysis models return `None`, they are typically used at the end of inference
pipelines or in conjunction with other models:

```python
# Example: Analysis model used after prognostic prediction
from earth2studio.models.px import PrognosticModel
from earth2studio.models.ax import ValidationModel

# Load models
prognostic = PrognosticModel.load_model(prognostic_package)
validator = ValidationModel.load_model(validation_package)

# Generate prediction and analyze
x, coords = prognostic(input_tensor, input_coords)
validator(x, coords)  # Validate prediction (no return value)
```

## Custom Analysis Models

Integrating your own analysis model is easy, just satisfy the interface above.
The key requirement is that the `__call__` method returns `None` and performs whatever
analysis, validation, or downstream processing operations are needed.

We recommend users have a look at the {ref}`extension_examples` examples, which will
step users through the simple process of implementing their own analysis model.

```python
class CustomAnalysisModel:
    def __init__(self):
        # Initialize your analysis model
        pass

    def __call__(self, x: torch.Tensor, coords: CoordSystem) -> None:
        # Perform analysis operations
        # Save results, compute metrics, validate data, etc.
        # No return statement needed (returns None implicitly)
        pass

    def input_coords(self) -> CoordSystem:
        # Return expected input coordinate system
        return CoordSystem(...)

    def to(self, device):
        # Move model to device if needed
        return self
```

## Contributing Analysis Models

Want to add your analysis model to the package? Great, we will be happy to work with
you.
At the minimum we expect the model to abide by the defined interface and meet the
requirements set forth in our contribution guide.

Open an issue when you have an initial implementation you would like us to review.
If you're aware of an existing analysis model and want us to implement it, open a
feature request and we will get it triaged.
