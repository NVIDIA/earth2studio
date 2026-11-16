# Data Assimilation Models { #data_assimilation_model_userguide }

Data assimilation models turn observations, optionally combined with a background
field, into an analysis. Use them to prepare initial conditions or update a forecast
with new measurements. The [assimilation API](../../modules/models_da.md) includes
both independent analysis models and models that continue a stateful trajectory.

## Interface

| Method | Purpose |
| --- | --- |
| `input_coords()` | Declare initial inputs, including observations for independent models. |
| `output_coords(...)` | Validate input declarations and plan the output. |
| `default_sources()` | Recommend providers in input-then-forcing order. |
| `model(...)` | Produce an independent analysis from separate, named inputs. |
| `to(device)` | Place model computation on the requested device. |

`AssimilationModel` defines this independent interface. `StatefulAssimilationModel`
adds `forcing_coords()`, `initialize(...)`, `step(..., state=state)`, and
`create_iterator(...)`, following the [prognostic](prognostic.md) explicit-state
convention. For these models, observations supplied during a rollout are forcing.
The protocols are in [`earth2studio.models.da.base`][interface]; inheritance is optional.

Each input or output slot is either a gridded DataArray described by a
`CoordinateSystem`, or a DataFrame described by a `FrameSchema`. A schema maps column
names to representative NumPy arrays: empty arrays leave values unconstrained;
nonempty arrays specify allowed values. Multiple slots form an ordered tuple, which
can mix gridded and tabular declarations. Models expose `stochastic` and `set_rng`;
seeding is a no-op for deterministic models.

See [E2S-07: Data Assimilation Contract][da-spec] for slot ordering, observation
schemas and timing, and independent and stateful execution requirements.

[da-spec]: ../../specs/E2S-07_DA_MODEL_CONTRACT_SPEC.md
[interface]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/models/da/base.py

## Usage

### Choose the Execution Pattern

| Model | Inputs | Forcing | Execution |
| --- | --- | --- | --- |
| `InterpEquirectangular` | Observations | — | Independent |
| `HealDA` | Conventional, satellite observations | — | Independent |
| `CorrDiffCosmoEra5SDA` | ERA5 background, observations | — | Independent |
| `StormCastSDA` | HRRR background | Conditioning, observations | Independent or stateful |
| `StormCastConusSDA` | HRRR background | Conditioning, observations | Independent or stateful |

For example, load HealDA through its concrete class, with its optional dependencies
and weights installed. Given observation frames prepared to match its two schemas:

```python
from earth2studio.models.da import HealDA

model = HealDA.load_model(HealDA.load_default_package())
analysis = model(conv_obs, sat_obs)
```

HealDA requires at least one observation batch. CorrDiff accepts
`model(background, observations)`, where observations may be `None`. Use
[`fetch_dataframe` and `fetch_dataframes`](datasources.md#tabular-observations) to
prepare observation requests and retain request-time metadata.

### Continue a Stateful Analysis

For a loaded StormCastSDA model and prepared fields:

```python
analysis, state = model.initialize(x0, conditioning0, obs0)
analysis, state = model.step(analysis, conditioning1, None, state=state)
```

Or let the iterator retain state:

```python
iterator = model.create_iterator(x0, conditioning0, obs0)
analysis = next(iterator)  # First computed analysis, one hour after x0
analysis = iterator.send((conditioning1, None))  # Next step without observations
iterator.close()
```

Conditioning remains required when observations are absent: `next(iterator)` or
`send(None)` cannot replace `send((conditioning, None))`. StormCastSDA aligns
conditioning to the input valid time and observations to the output time, one hour
later. StormCastConusSDA aligns both to the output valid time and filters observations
using the wrapped model's `time_tolerance`.

## Wrap Your Own Assimilation Model

Keep a wrapper in your own project. Declare gridded signatures and observation
schemas, use fixed named execution arguments, and implement the stateful extension
only when outputs and explicit state continue a trajectory. Validate observation
timing and optional-input behavior as well as numerical agreement with your
reference analysis.

The [earth2studio-create-assimilation skill][assimilation-skill] provides contract
and validation guidance. Ask your agent to adapt it to a local wrapper, supplying
your inference code, observation schema, background/forcing requirements, and
reference cases. See the [prognostic customization guide](prognostic.md#wrap-your-own-prognostic-model)
for an example agent request scoped to your project.

[assimilation-skill]: https://github.com/NVIDIA/earth2studio/tree/1.0.0-rc/skills/earth2studio-create-assimilation
