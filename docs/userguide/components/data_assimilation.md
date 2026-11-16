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
The protocols are in `earth2studio.models.da.base`; inheritance is optional.

Each input or output slot is either a gridded DataArray described by a
`CoordinateSystem`, or a DataFrame described by a `FrameSchema`. A schema maps column
names to representative NumPy arrays: empty arrays leave values unconstrained;
nonempty arrays specify allowed values. Multiple slots form an ordered tuple, which
can mix gridded and tabular declarations. Models expose `stochastic` and `set_rng`;
seeding is a no-op for deterministic models.

## Specification

The [assimilation contract][da-spec] defines the following requirements without
numbered compliance codes.

- **Honor slot order and schemas.** Supply each input separately, matching its grid
  or observation columns, units, and metadata. Keep optional observations in their
  declared position as `None` when the model permits them to be absent.
- **Align observation times.** Observation windows and conditioning times are
  model-specific. A background field alone does not imply stateful execution.
- **Fetch outside the model.** `default_sources()` only recommends providers.
  Callers fetch gridded and tabular data, align them, and supply required metadata
  such as the observation request time.
- **Keep planning separate.** `output_coords` validates declarations without reading
  field values or running assimilation. Actual outputs must match that plan.
- **Preserve the continuation.** Stateful initialization returns the first computed
  analysis and state. Every iterator yield is a computed output. `step` consumes
  `(analysis, state)` plus new forcing without mutating inputs or state, allowing
  the saved pair to be replayed or branched.
- **Use fixed signatures.** Declare named slots with backgrounds before forcing and
  `state` last, positional-or-keyword. Independent models declare all arguments in
  `input_coords`; stateful models additionally declare `forcing_coords`.
- **Keep rollout state explicit.** Outputs and serializable state fully determine
  continuation. Required forcing is never implicitly reused; hooks run only in the iterator.
- **Isolate randomness.** Calls and new trajectories advance a local stream.
  `set_rng` resets it; `reset=False` initializes only an absent stream. Seeded execution
  preserves global RNG state, and reseeding leaves live trajectories unaffected.
- **Declare analysis timing.** Stateful gridded outputs carry `lead_time`. Step cadence
  is the last output lead minus the later of the first input slot's last lead and zero.
  Drivers request tabular forcing at the analysis valid time and pass `None` for no rows.

[da-spec]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/dev/spec/DA_MODEL_CONTRACT_SPEC.md

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
