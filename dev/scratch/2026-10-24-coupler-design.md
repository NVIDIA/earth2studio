# Coupler design report (1.0.0-rc)

Status: research + initial lightweight implementation in `earth2studio/coupler/`.
Reference only the `1.0.0-rc` branch. Related: PR #1229 (nvcoupler, draft).

## Goal

Connect prognostic models that feed each other (StormScope GOES -> MRMS, an
atmosphere and an ocean at different cadences) and produce an ordinary
`earth2studio.run.Runner`, so `Pipeline(coupled_runner, output=...)` works with
no changes to `earth2studio.run`. It must be small enough that an agent can use
it from the model signatures alone.

## What the 1.0 contracts already give us

The model protocol already carries what a coupler would otherwise invent:

| Coupler need | Existing mechanism |
| --- | --- |
| Which fields a model produces / needs | `output_coords()["variable"]`, `forcing_coords()["variable"]` |
| Native timestep | `output_coords lead_time[-1] - input_coords lead_time[-1]` |
| Grid compatibility | non-variable dims/coords of the signatures |
| Feed new data between steps | `create_iterator(...).send(forcing)` |
| Initial conditions / external forcing | `default_sources()` / `recommended_sources`, `fetch_data` |
| Output streams, resume, IO, distribution | `Runner` protocol + `Pipeline` |

So the design has no component wrapper, no field dictionary and no new data type.
The component *is* the `PrognosticModel`; the variable name (lexicon) *is* the
field identity; the runner protocol *is* the driver interface.

## Implementation

`earth2studio/coupler/runner.py`: `CoupledRunner` and `Link`.

```python
runner = CoupledRunner(
    {"goes": goes, "mrms": mrms},            # name -> PrognosticModel
    sources={"goes": goes_src, "mrms": mrms_src},   # optional: default_sources()
    links=[...],                              # optional: overrides
)
Pipeline(runner, output={"goes": zarr_a, "mrms": zarr_b}).run(items)
runner.describe()                              # dry-run schedule + data flow
```

- **Probe.** At construction every model is probed once: step, variables, grid,
  forcing slots, sources. Nothing runs and nothing is allocated.
- **Auto-wire.** A time-varying forcing slot is fed by the single other model
  that produces all its variables. Zero producers means the slot uses a source
  (error if none); two means error, resolved with a `Link`. Static slots always
  use sources.
- **Link(src, dst, slot=None, reduce=None, regridder=None)** is the whole
  declarative surface. `reduce=None` holds the producer's latest output;
  otherwise it is an injected `DataArray -> DataArray` over the producer outputs
  in the consumer's step window (`window_mean` ships). `regridder` is an injected
  `grids.Regridder`; grids are compared with `infer_grid` + `same_geometry` and
  a mismatch without a bound regridder is a construction error.
- **Clock.** The coupling window is the slowest model's step (override with
  `window=`); each model must divide it. One runner step = one window; faster
  models publish several lead times in it (the Runner protocol allows streams
  to skip steps or vary).
- **Semantics** (validated against the `PrognosticRunner` iterator protocol):
  - `last`: the consumer, about to step from `t`, receives the producer's newest
    output valid at or before `t` (held if the producer is slower). Equal
    cadences give StormScope's behavior: MRMS at lead `L` gets GOES at `L`
    (the producer advances first).
  - `mean`: the consumer receives the mean of the producer's outputs in
    `(t, t + step]` (producer must be faster, and is scheduled first; cycles of
    `mean` links are rejected). This is the DLESyM / SamudrACE "time-mean atmosphere
    forces the ocean" pattern.
  - Initial windows for a wired slot come from the producer's own initial
    condition when it covers the window (GOES x -> MRMS conditioning), else
    from the slot's source.
- **Streams** are named after the models and declared in `output_coords`;
  IC is published first, then per window. `supports_member_batching=False`,
  matching the execution contract's phase 3 note.
- Fails loud: unknown models, unfed slots, grid mismatch, non-multiple windows,
  ambiguous producers, missing producer output.

Not in scope here (intentionally): ensembles/member batching, mid-item resume,
multi-slot prognostics, diagnostics inside the coupled loop, conservative
regridding, unit conversion table.

## Lighthouse cases

1. **StormScope GOES + MRMS**. Equal cadence, same HRRR grid, GOES variables
   produced by `StormScopeGOES.output_coords` are exactly `StormScopeMRMS`'s
   `forcing_coords` variables, so `CoupledRunner({"goes": goes, "mrms": mrms})`
   with default sources needs no links. Replaces the hand-written loops in
   `examples/04_nowcasting/03_stormscope_goes_example.py` and
   `recipes/eval/src/pipelines/stormscope.py`. Test mocks mimic the exact
   `send` ordering (`test/coupler/test_coupler.py`). Needs verification with real
   weights (integration check 7 in the execution contract).
2. **Atmosphere + ocean at different cadences** (DLESyM-like: 6 h atmosphere, 48 h ocean).
   `links=[Link("atmos", "ocean", reduce=window_mean), Link("ocean", "atmos")]`:
   atmosphere runs 8 steps per window with SST held from the last ocean output;
   ocean steps once with the 48 h mean atmosphere. Today `DLESyM` in the repo is
   a single fused model; this works once atmosphere and ocean are split into two
   slots/models (execution contract integration check 5). Mock test covers values.

## Mapping to nvcoupler (PR #1229)

| nvcoupler | Here |
| --- | --- |
| `PrognosticComponent` / `CallableComponent` / `DiagnosticComponent` | The model itself |
| `DataComponent` | A source bound to a slot (existing) |
| `Connector(time_policy, window, reduce)` | `Link(reduce=None/callable)` |
| `Connector(regridder, fill, vertical)` | `Link(regridder=...)`; fill/vertical dropped |
| `Mediator` (windowed reductions) | `reduce="mean"` |
| `FieldDictionary` / standard names | Lexicon variable names |
| Run sequence / lag vs sequential | `last` vs `mean` + scheduling order |
| `Driver` + `Clock` | `CoupledRunner` (a `Runner`) + window |
| `couple()` auto-wiring | Auto-wiring in constructor |
| YAML DSL, `collect`, in-memory Datasets | Dropped: `Pipeline`/IO handle output |

Kept from nvcoupler: DataArray-native boundaries, ordering-based coupling
semantics, name-based auto-wiring, up-front validation. Dropped: components,
mediators synthesis, vertical interpolation, mask fill, dictionary, DSL, its own
output model.

## Survey of existing coupling frameworks

(Subagent reads of docs/code; items marked unverified were not read.)

- **legoESM** (JAX monolithic ESM): coupling is hard-coded sequential/lagged in
  drivers; the declared `Interface(form, transfer, regrid, coupling)` bricks are
  unused scaffolding. Useful: dt-weighted `FluxAccumulator` with residual
  carry, identity-remap fast path, fail-loud on missing/non-finite fields (silent
  zero defaults produced perpetual polar night), integer coupling-interval
  checks. Lesson: do not over-build interfaces before use.
- **ESMF/NUOPC**: Driver/Model/Mediator/Connector, advertise -> match by standard
  name -> realize. Lag vs sequential is purely connector position in the run
  sequence; child must advance exactly one parent step (integer multiple);
  no built-in averaging (mediators do it). Borrowed: name matching, probe before
  run, integer-multiple check. Skipped: phases, IPD, PET lists, namespaces.
- **OASIS3-MCT / YAC**: declarative couple record (src/dst, period,
  time_reduction instant/average/accumulate/min/max, lag, scale/offset,
  interpolation fallback stack); models just put/get with their own date;
  YAC requires coupling period to be an integer multiple of both timesteps. Borrowed: reduction
  vocabulary, period validation, lag as previous value, transform per link.
  Skipped: MPI, weight files, restart files.
- **ClimaCoupler.jl / MUSCLE3**: ClimaCoupler sub-cycles to a coupling dt and uses
  a flux accumulator for slow surface models; MUSCLE3 uses ports/conduits with
  strict-causality bridges, no many-to-one. Borrowed: sub-cycling by cadence
  ratio, accumulate-then-push for slow components, graph validation at
  construction (one producer per input).
- **preCICE / CMEPS / SamudrACE / DLESyM paper**: preCICE serial-explicit =
  sequential, parallel-explicit = lagged, window loop with subcycling, only the last
  sample is exchanged by default; CMEPS averages fast-period fluxes before
  mapping (linear ops, average on source grid, map once); SamudrACE averages 20
  six-hourly atmosphere flux steps into one 5-day ocean step, atmosphere forced by
  SST held for the window; DLESyM (96 h cycle) feeds time-averaged atmosphere
  fields over 48 h windows to the ocean. These define the `mean` and held-`last`
  semantics above. Alignment of averaging windows with ocean timestamps matters
  (SamudrACE Text S1): the window here is `(t, t + step]`.

## Open questions

1. Implicit/iterative coupling and checkpoint of mean accumulators: not needed yet.
2. Ensembles: per-member coupled state needs member batching in the runner; the
   contract lists it as open. Seeding is in place (`_seed_model` per model).
3. Which slot of DLESyM-like fused models is split into separate models? Needs
   a model-side change; the coupler cannot split a fused model.
4. `last` with a faster producer returns only its newest frame; a consumer wanting
   several frames needs a lead window wider than one (supported by the
   probe: labelled by the slot's lead_time) but this is untested with real models.
5. Regridding between differing grids: the user injects a bound `Regridder`
   (`grids.resolve_regridder` needs the `utils` extra, so the coupler does not
   build one itself); auto-selection could be added.
8. Only single-input/single-output-slot models with one output lead per step are
   accepted (clear errors otherwise); fused multi-lead models such as DLESyM need
   to be split first. `data_requests` lists the source fetches only.
6. GPU: frames stay on the device the models use; `to(device)` forwards to models.
7. Real-weights verification of StormScope equivalence to `eval` pipeline.

## Review follow-ups (implemented)

- Delivered frames are checked (held output not stale; a reducing window has
  exactly `consumer step / producer step` frames) before relabelling.
- Regridder output is validated against the slot's spatial coordinates.
- An explicit source for a slot wins over auto-wiring.
- Each model and diagnostic seeds from `seed + crc32(name)`, so stochastic
  models draw independent noise.
- Reducers `window_last/min/max/sum` ship alongside `window_mean`;
  `check_finite=True` raises on NaN/inf delivered between models.
- `Link(window_start=True)` holds the producer as of the window start;
  `Link(scale=, offset=)` converts units; a units mismatch (when both signatures
  carry `attrs["units"]`) raises unless a scale/offset is given.
- `CoupledRunner(diagnostics={stream: (model, diagnostic)})` publishes
  diagnostic streams.
- Deferred: a dict of initial-condition arrays, ensembles, checkpointing,
  multi-slot/multi-lead models, interpolation, automatic/conservative regridding,
  weighted and per-variable reductions.

## Revision: providers instead of links (supersedes the Link API above)

`Link`, the cycle checker, `lookahead`/`producers` and the `window_*` helpers
were removed: they amounted to a small DAG framework. What feeds a forcing slot
is now an injected function, `provider(history, start, step) -> DataArray`.

- `CoupledRunner(models, sources, providers={(model, slot): provider}, ...)`.
- Provider builders are plain functions: `latest(src)` (hold), `reduced(src, fn)`
  (any xarray reduction over `(start, start+step]`; waits via `NotReady`) and
  `then(provider, *fns)` (regridder, unit conversion, masking).
- `match_by_variable(models, skip)` is a pure function returning the auto-wiring as a
  plain dict of `latest` providers; the runner uses it for slots without a provider
  or an explicit source, and it can be called and edited by hand.
- Scheduling is one rule: the furthest-behind model steps next unless its
  provider raises `NotReady`; if none can step it raises "deadlocked".
- Delivered fields are checked against the slot grid and (optionally) for
  finiteness at delivery, replacing construction-time grid/units checks; units,
  scale/offset and `window_start` are expressed as ordinary functions.
- Removed from the earlier follow-ups: reducer helpers, `Link.scale/offset/
  window_start/regridder`, the units attr check.
