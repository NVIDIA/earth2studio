# Assimilation and stateful assimilation — PR #1261

## Decision

`AssimilationModel` describes independent analyses: `__call__`, input/forcing/output
coordinate hooks, `default_sources`, and `to`. `StatefulAssimilationModel` extends
it with `initialize`, `step`, and `create_iterator`, following the prognostic
explicit-state contract. Both protocols are structurally runtime-checkable.

This supersedes the earlier cycling-only iterator proposal. An iterator can stream
independent batches; its presence alone does not establish trajectory semantics.
A background input also does not imply statefulness.

Reference: [Peter's review comment](https://github.com/NVIDIA/earth2studio/pull/1261#discussion_r4331626720).

## Model classification

| Model | AssimilationModel | StatefulAssimilationModel |
| --- | --- | --- |
| HealDA | Yes | No |
| Interpolation | Yes | No |
| CorrDiff SDA | Yes | No |
| StormCast SDA | Yes | Yes |

HealDA, interpolation, and CorrDiff expose independent calls. Their artificial
`create_generator` wrappers are removed. CorrDiff backgrounds are caller-owned;
successive analyses do not depend on preceding outputs.

StormCast advances the previous analysis by one hour, consuming caller-supplied
conditioning and optional observations. Its serializable continuation state holds
RNG snapshots so replay and interleaved trajectories do not depend on mutable
model-global random state.

## Execution contract

- `__call__` and `initialize` accept background slots followed by forcing slots.
  Concrete methods have fixed named arguments.
- `initialize` returns the first computed analysis plus explicit state.
- `step` accepts previous outputs, new dynamic forcing, then state. It returns
  new outputs and state without modifying caller inputs or prior state. State is
  positional-or-keyword in concrete implementations, as in prognostics.
- `create_iterator` takes the initialization arguments and first yields the
  computed analysis. Subsequent sends provide dynamic forcing in slot order.
  There is no initial-condition or `None` priming yield.
- Slots support DataArrays and DataFrames. Optional observations are preserved
  as `None`; required gridded conditioning remains mandatory. Outputs support
  DataArrays or DataFrames, with coordinate signatures or FrameSchema declarations.
- Source recommendations are input-then-forcing ordered. Models never fetch data.

```python
# Independent analyses
for background, observations in cases:
    analysis = corrdiff(background, observations)

# Explicit StormCast continuation
analysis, state = stormcast.initialize(x0, conditioning0, obs0)
analysis, state = stormcast.step(analysis, conditioning1, None, state)

# Equivalent managed continuation
iterator = stormcast.create_iterator(x0, conditioning0, obs0)
analysis = next(iterator)
analysis = iterator.send((conditioning1, None))
```

For StormCast, conditioning is valid at the input time and observations at the
output time one hour later. `None` disables observations but does not waive
conditioning. Analysis time, observation windows, and required metadata retain
their model-specific contracts; no new universal metadata container is introduced.

## Integration

The protocols are exported from `earth2studio.models.da`. Callers, examples, API
generation, user documentation, and DA implementation rules follow the split.
The evaluation runner describes its observation-only input restrictions explicitly
rather than inferring statefulness from background-slot presence.

Runtime protocol checks establish member presence; behavior is verified separately
through independent-call and StormCast continuation tests, including first-yield
semantics, replay, ownership, optional observations, and required conditioning.
