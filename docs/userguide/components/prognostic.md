# Prognostic Models { #prognostic_model_userguide }

Prognostic models advance atmospheric or other physical fields through time. A model
can produce one forecast chunk or continue a trajectory using explicit state.
Browse the built-in models in the [prognostic API](../../modules/models_px.md).

## Interface

Models exchange labelled `xarray.DataArray` fields in physical units. Coordinate
methods describe those fields without allocating their values; see the
[Coordinates and Metadata](coordinates.md).

For a model with one input and no forcing, the interface is:

| Method | Purpose |
| --- | --- |
| `input_coords()` | Declare initial fields, history, variables, and grid. |
| `forcing_coords()` | Declare external conditioning; `None` when unforced. |
| `output_coords(input_coords)` | Validate coordinates and plan the output. |
| `default_sources()` | Recommend data providers for the caller. |
| `model(x)` | Compute an independent forecast chunk. |
| `initialize(x)` | Return the first forecast and continuation state: `(y, state)`. |
| `step(y, state=state)` | Return the next forecast and updated state. |
| `create_iterator(x)` | Manage state internally and yield forecast chunks. |
| `to(device)` | Place model computation on the requested device. |

Multi-input models take separate, named arguments in declared order. Forcing follows
the initial inputs in `initialize` and the previous outputs in `step`, with `state`
last. One output is returned directly; multiple outputs form an ordered tuple.
The protocol is defined in `earth2studio.models.px.base.PrognosticModel`; inheritance
from it is not required. `stochastic` identifies stochastic models, which expose
`set_rng(seed, reset=True)` for reproducible rollouts.

See [E2S-06: Model Contract][model-spec] for the full coordinate, execution, state,
randomness, and domain-selection requirements.

[model-spec]: ../../specs/E2S-06_MODEL_CONTRACT_SPEC.md

## Usage

### Load and Predict

Use a concrete model class to load pretrained weights. This example requires SFNO's
optional dependencies, weights, and sufficient device memory; see
[AutoModels](../advanced/auto.md#automodel_userguide).

```python
import numpy as np

from earth2studio.data import GFS, fetch_data
from earth2studio.grids import infer_grid
from earth2studio.models.px import SFNO

device = "cuda:0"
model = SFNO.load_model(SFNO.load_default_package()).to(device)
signature = model.input_coords()
x = fetch_data(
    GFS(),
    np.array(["2024-01-01T00"], dtype="datetime64[ns]"),
    signature.coords["variable"].values,
    lead_time=signature.coords["lead_time"].values,
    device=device,
    target_grid=infer_grid(signature),
)
planned = model.output_coords(x)  # Coordinate validation, without inference
y = model(x)
```

### Continue a Forecast

Use explicit state when you need to save or branch a trajectory:

```python
y, state = model.initialize(x)             # First forecast
y, state = model.step(y, state=state)      # Second forecast
```

For a simple unforced rollout, use the iterator instead:

```python
from itertools import islice

iterator = model.create_iterator(x)
for y in islice(iterator, 4):
    print(y.lead_time.values)  # SFNO: 6h, 12h, 18h, 24h
iterator.close()
```

For forced models, pass the initial forcing window to `create_iterator`, then use
`iterator.send(forcing)` for each subsequent step, or send a tuple for multiple
forcing slots. Supply the newest required frames; required time-varying forcing is
not fetched or silently reused. Static forcing without a `lead_time` dimension is
supplied only at initialization.

`PrognosticMixin` provides iterator hooks: `rear_hook` runs before every yield,
including the first, and `front_hook` runs before subsequent steps. Both affect the
continuing trajectory. Hooks do not run in `model(x)`, `initialize`, or `step`;
perturb the initial input explicitly when needed. For stochastic models, call
`set_rng` before a new rollout; reseeding does not change an existing explicit state.

## Wrap Your Own Prognostic Model

Keep a wrapper in your own project, such as `my_project/models/my_model.py`, and pass
its instance to Earth2Studio workflows just like a built-in model. The wrapper
adapts your existing inference code to the interface above, handling coordinates,
physical-unit conversion, device placement, and explicit continuation state.

The **`earth2studio-create-prognostic` agentic skill** provides a model contract,
wrapper skeleton, method templates, and validation guidance. Give your coding agent
the [skill directory][prognostic-skill] from a checkout matching your Earth2Studio
version, and ask it to load `SKILL.md` and the referenced files. Provide:

- Your working inference script, weights, configuration, and dependency requirements.
- Input/output variable names and units, grid, normalization, history, and forecast step.
- Required forcing or static data, device support, and stochastic behavior.
- A small reference input and expected output for numerical comparison.

For example:

```text
Use the earth2studio-create-prognostic skill to wrap my existing model in
my_project/models/my_model.py for local use with Earth2Studio. Follow the model
contract and relevant rules from this checkout. Scope the work to my project;
skip package exports, dependency extras, changelog, and pull-request steps.

Reference inference: <script>; weights/configuration: <paths>.
Variables/units/grid: <details>; history and forecast step: <details>.
Forcing, normalization, devices, and randomness: <details>.

Preserve the reference inference behavior. Validate coordinate-only planning,
single-call and multi-step execution, input immutability, state replay, and
numerical agreement with the reference. Show a short local usage example.
```

Use explicit named parameters for every input and forcing slot. Build signatures
with `coord_array` and `coord_array_like`; for PyTorch models, use
`x.e2s.to_torch()` and `from_torch()` at field boundaries. `PrognosticMixin` supplies
private call/iterator helpers, but your wrapper must expose its public methods.
Automatic weight downloading is optional: a wrapper can accept an already loaded
model and local assets.

The skill's conformance checks help validate the interface. Also compare a short
rollout with your reference implementation: matching shapes alone does not establish
that normalization, timing, or predictions are correct. See the
[extension examples](../../examples/index.md#extend) for complete integrations.

[prognostic-skill]: https://github.com/NVIDIA/earth2studio/tree/1.0.0-rc/skills/earth2studio-create-prognostic
