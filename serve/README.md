# Quickstart guide

## Developer quickstart

Developers who have Earth2Studio installed on a gpu-enabled system can easily get started with the
inference platform as follows.
For developers who prefer to test using a container with requirements pre-installed,
please refer to the section [Container Builds](#container-builds) below.

* Install redis

  ```bash
  apt update && apt install redis
  ```

* Install requirements for the inference server

  ```bash
  cd server
  pip install -r requirements.txt
  ```

* The default Dockerfile CMD starts up the inference server.

* Check health

  ```bash
  curl localhost:8000/health
  ```

### Creating and testing a custom workflow locally

* Use the Earth2Workflow base class to develop the inference workflows.
  Examples are shown in the files: server/example_workflows/deterministic_earth2_workflow.py.

An example of a locally tested custom_workflow is shown below.

```python
"""
Deterministic Workflow Custom Pipeline

This pipeline implements the deterministic workflow from examples/01_deterministic_workflow.py
as a custom pipeline that can be invoked via the REST API.
"""

from datetime import datetime
from typing import Literal

from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import IOBackend
from earth2studio.models.px import DLWP, FCN
from earth2studio.serve.server import Earth2Workflow, workflow_registry


@workflow_registry.register
class DeterministicEarth2Workflow(Earth2Workflow):
    """
    Deterministic workflow with auto-registration
    """

    name = "deterministic_earth2_workflow"
    description = "Deterministic workflow with auto-registration"

    def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
        super().__init__()

        if model_type == "fcn":
            package = FCN.load_default_package()
            self.model = FCN.load_model(package)
        elif model_type == "dlwp":
            package = DLWP.load_default_package()
            self.model = DLWP.load_model(package)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 20,
    ):
        """Run the deterministic workflow pipeline"""

        run.deterministic(start_time, num_steps, self.model, self.data, io)

print("initializing ")
model = DeterministicEarth2Workflow()
print("calling model")
from earth2studio.io import ZarrBackend
io = ZarrBackend()
model(io)
```

It is run as follows without needing to start redis etc.

```bash
python serve/server/example_workflows/custom_workflow.py
```

* Refer to these READMEs [Earth2Workflow](./server/README_earth2workflows.md),
  [Workflow](./server/README_workflows.md)

## Container builds

The Earth2Studio parent directory contains Dockerfiles that let you build the inference service
for deployment onto Lepton.AI.

### Inference Container

The inference container can be built from the [Dockerfile](./Dockerfile).

Alternatively, the prebuilt container images can be used from the
[NGC registry][ngc-registry] after onboarding.

<!-- markdownlint-disable-next-line MD013 -->
[ngc-registry]: https://registry.ngc.nvidia.com/orgs/dycvht5ows21/containers/earth2studio-scicomp/tags

## Lepton.AI onboarding

Please talk to your NVIDIA contact or TAM to get onboarded onto the Lepton.AI cluster.

## Lepton.AI deployment

Please see the [deployment guide](DEPLOY.md) for instructions on how to set up the inference
service on your Lepton.AI endpoint.

## Using the inference service

Once you set up your inference endpoint, you may either call the services directly through REST
APIs or you may use the client SDK.

## Writing custom inference workflows

You may port more [predefined examples](../examples) or write your own custom workflows using the
[custom workflows](server/README_workflows.md) guide.
