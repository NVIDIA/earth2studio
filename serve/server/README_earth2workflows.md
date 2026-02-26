# Write Earth2Studio workflows that are easily served as REST APIs

## `Earth2Workflow` class

The `Earth2Workflow` interface allows writing Pythonic Earth2Studio workflows that are natural to
run using a pure Python interface and can be automatically given a REST API interface when running
under the inference server. This minimizes boilerplate code and allows a smooth transition from
development to operations.

`Earth2Workflow` extends the `Workflow` interface, providing a default implementation for managing
the REST API. Workflows that require more customization of the API features can still use the
lower-level `Workflow` directly.

### Sample `Earth2Workflow` implementation

This is a simplified example demonstrating the key concepts. For complete working examples, see the
`example_workflows` directory.

```python
from api_server.workflow import workflow_registry, Earth2Workflow
from earth2studio.io import IOBackend
from datetime import datetime
from earth2studio.data import GFS
from earth2studio.models.px import FCN
from earth2studio import run


@workflow_registry.register
class ExampleEarth2Workflow(Earth2Workflow):
    name = "example_earth2_workflow"
    description = "example workflow"

    def __init__(self):
        super().__init__()
        self.model = FCN.from_pretrained()
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 20,
    ):
        run.deterministic(
            start_time, num_steps, self.model, self.data, io
        )
```

### Requirements for `Earth2Workflow` subclasses

Initialization of models and other permanent resources (e.g. data sources) should go in `__init__`
and the workflow logic should be in the `__call__` function. Then, you can easily run these
workflows on Python:

```python
from earth2studio.io import ZarrBackend

workflow = ExampleEarth2Workflow()
io = ZarrBackend("output.zarr")
workflow(io=io)
```

Implementing the workflow as a subclass of `Earth2Workflow` automatically provides it with a REST
API schema. It gets the following features:

- The keyword arguments of `__call__` are automatically converted into a REST API schema and
  inference requests to the `name` attribute (in the example above, `"example_earth2_workflow"`) are
  forwarded by the API server to `__call__`. There are a few requirements/considerations:
  - `__call__` must have the argument `io: IOBackend`. The server will pass an output backend (whose
    type is determined by the server settings).
  - The other arguments of `__call__` must:
    - have type hints (these are used to generate the schema).
    - have types that are supported by Pydantic and are JSON serializable. Additionally,
      `datetime` or `timedelta` arguments, including such data types nested inside data structures,
      are supported. This covers most commonly used argument types.
      Two examples of types that are _not_ supported are NumPy arrays and PyTorch tensors.
      Define these as lists instead.
      See the [Pydantic
      documentation](https://docs.pydantic.dev/dev/api/standard_library_types/) for
      information about supported types.
  - Arguments with a default value are optional for the API client while those without a default
    value are mandatory.
- `__init__` is run once per inference worker and the initialized resources (assigned to `self`)
  then persist over multiple requests. The keyword arguments of `__init__` are converted into a
  configuration that can be modified from the service config file.
- If `name` is not set, it will default to the name of the class (in this case,
  `ExampleEarth2Workflow`).

### API schema

The Pydantic model representing the `__call__` method of the workflow can be found in the
`Parameters` attribute of the class:

```python
import pprint

pprint.pp(ExampleEarth2Workflow.Parameters.model_json_schema())
```

prints

```python
{'properties': {'start_time': {'default': ['2024-01-01T00:00:00'],
                               'items': {'format': 'date-time',
                                         'type': 'string'},
                               'title': 'Start Time',
                               'type': 'array'},
                'num_steps': {'default': 20,
                              'title': 'Num Steps',
                              'type': 'integer'}},
 'title': 'ExampleEarth2WorkflowParameters',
 'type': 'object'}
```

The Pydantic model for the configuration parameters can be found in the `Config` attribute of the
class. In this case, `__init__` has no arguments, so this model is empty.

## Registration

The `@workflow_registry.register` decorator causes the workflow to be automatically added into the
API registry when run under the API server. The class-level `name` and `description` attributes are
used for registration. As with any `Workflow`, your `Earth2Workflow` must be in the directory
indicated by the `WORKFLOW_DIR` environment variable in order to be discovered by the API server.

## REST API Usage

To execute the sample workflow above over the REST API, start the inference server, then:

```bash
curl -X POST "http://localhost:8000/v1/infer/example_earth2_workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "start_time": ["2024-01-01T00:00:00"],
      "num_steps": 10
    }
  }'
```

Querying the status of the workflow and retrieving the results is explained in
[README_workflows.md](README_workflows.md).
