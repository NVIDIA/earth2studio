# Diagnostic Models { #diagnostic_model_userguide }

Diagnostic models transform physical fields in a single call: for example, deriving
precipitation, downscaling a forecast, or generating samples conditioned on a state.
They do not continue an autoregressive trajectory. Browse the
[diagnostic API](../../modules/models_dx.md) for available models.

## Interface

| Method | Purpose |
| --- | --- |
| `input_coords()` | Declare one input signature or an ordered tuple of signatures. |
| `output_coords(...)` | Validate input coordinates and declare the output signatures. |
| `default_sources()` | Recommend providers in input-slot order. |
| `model(x)` | Transform a field; multiple inputs are separate named arguments. |
| `to(device)` | Place model computation on the requested device. |

Fields are `xarray.DataArray` objects; signatures follow the
[Data and Metadata](coordinates.md).
Return one output directly or multiple outputs as a tuple. The protocol is
[`earth2studio.models.dx.base.DiagnosticModel`][interface]; inheritance is optional.
The `stochastic` flag and, for stochastic models, `set_rng(seed, reset=True)` describe
randomness. Diagnostic models require no forcing, state, iterator, or hook interface.

See [E2S-06: Model Contract][model-spec] for diagnostic-model requirements, including
coordinate planning, output ownership, randomness, and optional domain selection.

[model-spec]: ../../specs/E2S-06_MODEL_CONTRACT_SPEC.md
[interface]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/models/dx/base.py

## Usage

Load pretrained models through their concrete class, for example:

```python
from earth2studio.models.dx import PrecipitationAFNO

model = PrecipitationAFNO.load_model(PrecipitationAFNO.load_default_package())
model = model.to("cuda:0")
```

This requires the model's optional dependencies and weights; see
[AutoModels](../advanced/auto.md#automodel_userguide). Given a field `x` prepared on
the model's device with the required variables and grid:

```python
planned = model.output_coords(x)  # Validate and inspect output coordinates
precipitation = model(x)
```

When chaining models, inspect the diagnostic's `input_coords()` and explicitly select
or regrid the prognostic output to match it. A common grid does not guarantee matching
variable order, history, or accumulation windows.

## Wrap Your Own Diagnostic Model

Implement the interface in a local module, with fixed named arguments such as
`__call__(self, atmosphere, ocean)`. Use `coord_array` for declarations and
`coord_array_like` when preserving an input's geometry. For PyTorch computation,
convert at the boundary with `.e2s.to_torch()` and `from_torch()`.

The [earth2studio-create-diagnostic skill][diagnostic-skill] includes wrapper and
validation guidance. Ask your agent to adapt it to a standalone wrapper in your
project, supplying your inference script, weights, variable/unit definitions, and
input/output grids. Validate against reference predictions as well as the declared
interface. The [prognostic customization guide](prognostic.md#wrap-your-own-prognostic-model)
shows how to scope an agent request to local use.

[diagnostic-skill]: https://github.com/NVIDIA/earth2studio/tree/1.0.0-rc/skills/earth2studio-create-diagnostic
