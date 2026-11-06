# Assimilation Model Contract

The public interfaces are `AssimilationModel` and `StatefulAssimilationModel` in
[base.py](../../../earth2studio/models/da/base.py); read their docstrings with this
file. Implementation rules live in
[e2s-013](../../../.agents/rules/e2s-013-assimilation-models.mdc), user-facing
usage in the [DA user guide](../../../docs/userguide/components/data_assimilation.md),
and 0.x porting notes in the data assimilation section of
[MIGRATION.md](../../../MIGRATION.md). Runtime protocol membership checks method
presence only, and there is no DA conformance checker: verify behavior in the
model's native tests.

## Two protocols

| Protocol | Adds | Use when |
| --- | --- | --- |
| `AssimilationModel` | `__call__`, `input_coords`, `output_coords`, `default_sources`, `to`, `stochastic`, `set_rng` | Every analysis is an independent call |
| `StatefulAssimilationModel` | `forcing_coords`, `initialize`, `step`, `create_iterator` | Outputs plus explicit state continue a trajectory |

Statefulness is a capability of the model, never inferred. A background input or
a caller looping over times does not make a model stateful (CorrDiff takes each
ERA5 background as an independent input). Do not add artificial iterators to
independent models. Current wrappers: HealDA, HealDAv2, interpolation (independent,
deterministic); CorrDiff-COSMO SDA (independent, stochastic); StormCast SDA and
StormCast-CONUS SDA (stateful, stochastic).

## Slots

- A slot is one argument. A gridded slot is a `CoordinateSystem` (allocation-free
  DataArray from `coord_array`/`coord_array_like`) and is passed as a DataArray; a
  tabular slot is a `FrameSchema` and is passed as a pandas or cuDF DataFrame.
  Never use legacy coordinate dictionaries.
- Declare one slot directly, several as an ordered tuple, none as `None`. Tuples
  may mix gridded and tabular slots, in inputs, forcing and outputs. Slot order
  is public API; keep optional slots in place and pass `None` for them.
- `FrameSchema` maps column names to representative arrays: an empty array
  leaves the column's values unconstrained, a non-empty one lists the allowed
  values (typically `time`, `lat`, `lon`, `observation`, `variable`).
- Gridded slots declare auxiliary geographic coordinates and any dynamic leading
  dimensions explicitly. Configure domain, grid and variable sets on the
  instance before querying signatures, and keep them fixed for a run.
- `input_coords()` describes every argument of an independent model (observations
  included), or the initialization inputs of a stateful one. `forcing_coords()`
  (stateful only) describes caller-supplied conditioning and observations. Only
  stateful models declare forcing. Optional slots still have a declaration.
- `output_coords(*input_slots, **kwargs)` takes the input signatures as separate
  arguments (fixed named parameters on concrete models); extra keywords are
  model-specific (for example `request_time`). It returns a `CoordinateSystem`
  or `FrameSchema` directly, or an ordered tuple that may mix both. Outputs are
  real analyses, never a priming `None`. Invalid coordinates raise `ValueError`.
- Concrete `__call__`, `initialize`, `step` and `create_iterator` take fixed named
  parameters, one per slot, in input-then-forcing order, not `*x`. Return a
  single output directly and several in declared order (`return y1, y2`).
- `default_sources()` follows the same order, input slots then forcing slots:
  one source directly, a tuple with `None` for unspecified slots, or top-level
  `None` for no recommendation. Models never fetch; callers fetch and align data
  to each slot's grid, variables and time window. Drivers read recommendations
  with `earth2studio.models.utils.recommended_sources`.

## Time, windows and cadence

- Gridded `lead_time` is relative to the analysis time and increases. Forcing
  lead times are relative to the state being processed. Models document their
  observation windows (and any required DataFrame `attrs`, such as
  `request_time`) because there is no machine-readable window declaration yet;
  filter with `filter_time_range` and validate columns early with
  `validate_observation_fields`.
- Drivers request DataFrame forcing at the analysis valid time, the end of the
  output being produced, and pass `None` when the source returns no rows.
  Example: StormCast SDA takes conditioning at the input valid time and
  observations at the output valid time, one hour later.
- Stateful gridded outputs carry `lead_time`, and output offsets are added to the
  final input lead time, never a constant. The **step interval** is the output's
  last lead time minus the later of the first input slot's last lead time and
  zero. An input ending at zero steps by the output's last lead time. A
  background ending before zero (say `-9h` with a first output at `+3h`) is
  brought up to the analysis time by `initialize` and then steps by 3 h; no
  initial condition is published for it. A model whose `step` advances by the
  full gap from its background must declare inputs ending at zero.
- Models never return the initial condition; drivers take it from the inputs
  (`earth2studio.models.px.utils.initial_condition`) when they publish it.
- A model computing several output lead times per step declares them all in
  `output_coords()` and returns them together; one step is still one computation
  and one yield.

## Stateful continuation

- `initialize(*inputs, *forcing)` runs one core computation and returns
  `(first_analysis, state)`. It takes the whole forcing window and any static
  forcing (gridded forcing without `lead_time`), which it stores in the state.
- `step(*outputs, *new_forcing, state)` takes the previous output slots, new
  dynamic forcing only (static slots omitted, optional observations kept as
  `None`) and the state, and returns `(next_analysis, next_state)`. Gridded
  forcing supplies the final declared lead shifted by the output being advanced;
  older frames live in the state. Declare `state` as a positional-or-keyword
  parameter after the arrays; generic callers use `state=state`.
- `(output, state)` is the complete continuation. State is serializable (arrays,
  tensors, scalars, DataArrays, dataclasses, tuples of these, or `None`), holds
  history, statics, latents and RNG position (a counter or snapshot, not a live
  `torch.Generator`), and does not duplicate the output unless the published
  output is not a faithful next input. `step` modifies neither argument, and the
  same arguments replay to the same result, stochastic draws included, so
  trajectories can be checkpointed, restored or branched with one instance.
- `__call__` equals `initialize(...)[0]`: it keeps no state and applies no hooks.
- `create_iterator(*inputs, *forcing)` yields the output of `initialize` first
  (never an initial state or priming `None`; `n` analyses take `n` yields). A
  value sent at a yield is forcing for the next step: one slot directly, several
  as a tuple in declared order, static slots omitted, optional slots as `None`
  entries. `next(it)` is `send(None)`, valid only when no required new forcing
  is needed. Required conditioning is never fetched or reused implicitly (StormCast
  SDA raises `ValueError` on `send(None)`), but observations may be `None`.
- Missing required forcing raises `ValueError`; an observation-only model may
  require at least one non-`None` observation batch (HealDA does).
- Execution hooks, if a model offers them, are iterator-only: the rear hook edits
  every yielded analysis starting with the first, the front hook runs before each
  later `step`, and their results feed the recurrence. `__call__`, `initialize`
  and `step` apply none. Do not use `PrognosticMixin` or its batching decorators.

## Randomness

- Declare `stochastic: bool`. Every model implements `set_rng(seed, reset=True)`;
  deterministic models make it a no-op.
- Stochastic models keep a model-local generator. Each call or new trajectory
  advances it; `reset=True` restarts it from `seed`, `reset=False` creates it
  only if absent. Seeding and seeded execution leave global RNG state unchanged
  (use a local generator, or the scoped `earth2studio.models.utils.fork_rng`
  across the numerical call only, never across an iterator yield). Never-seeded
  models may draw from global RNG.
- A live trajectory continues from its own state and is unaffected by reseeding
  the model; reseeding changes only new calls and new `initialize` calls.
- Runners seed each stochastic model per member with `set_rng(seed, reset=True)`
  before a rollout or call. Keep sampling isolated per model instance; models
  using a scoped global fork are not safe to run concurrently in one process.

## Ownership and execution boundary

- Inputs are borrowed: never modify values, coordinates, attrs or encoding of a
  DataArray, nor a DataFrame's rows or `attrs`. Clone borrowed tensors before
  in-place writes. Outputs belong to the model and live on its device.
- Validate gridded inputs with `handshake_dataarray` against the declared
  signature (plus `handshake_time` and `handshake_nonempty` where needed).
- Convert gridded fields with `x.e2s.to_torch()` and `from_torch(output, signature)`;
  use `dfseries_to_torch` for DataFrame columns (pandas or cuDF) without
  forcing GPU data through host arrays. Use `@torch.inference_mode()` unless
  gradients are needed (for example DPS guidance) and note why in a comment.
- Handle `GeneratorExit` in iterators that need cleanup.

## Construction and loading

Subclass `torch.nn.Module` (and `AutoModelMixin` for checkpoint-backed models);
register a `device_buffer` and expose `device`; register other device-following
tensors as buffers. Use the `OptionalDependencyFailure("<extra>")` import guard
with the extra declared in `pyproject.toml`, and `@check_optional_dependencies()`
on the class and `load_model`. `load_default_package` returns a `Package` with
a pinned checkpoint; resolve files with `package.resolve`, call `.eval()` on
networks and keep `load_model` options minimal. Spatial-domain selection, where
offered, is a `DomainAdaptable.with_region(region)` view, not an in-place setter.

## Using assimilation models in workflows

- Independent analyses are called directly, in a caller-owned loop over times.
  `PrognosticRunner` and `CoupledRunner` do not drive them. To initialize a
  prognostic model from one, write a `DataSource` adapter that fetches the
  observation slots, calls the model and returns the analysis (the model still
  never fetches); no such adapter ships yet.
- Only stateful models can be a `CoupledRunner` component (see the
  `earth2studio-couple-models` skill). That requires a gridded first input slot
  with lead times, gridded outputs with a `variable` dimension and shared output
  lead times, and observation slots that tolerate `None` (an empty source result
  arrives as `None`). DataFrame slots are fetched from sources, never fed by
  connectors. The coupler calls `initialize` and `step` directly, so iterator
  hooks do not apply.

## Porting a 0.x wrapper

Remove `create_generator` (use independent calls or `initialize`/`step`/forecast-first
`create_iterator`), replace `init_coords()` with `input_coords()` plus
`forcing_coords()`, replace CorrDiff's `.seed` with `set_rng`, convert coordinate
dictionaries to signatures, return one analysis directly (not an always-tuple),
drop priming `None` yields, and add `default_sources()` and `stochastic`.
