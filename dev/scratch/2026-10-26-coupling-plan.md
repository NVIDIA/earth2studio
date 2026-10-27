<!-- markdownlint-disable MD013 -->
# Coupler Integration Implementation Plan

> Working plan for landing a coupler on `1.0.0-rc` that runs on the new model,
> regrid, time-statistics and execution contracts. Supersedes the structure of
> PR #1229 (`nvcoupler`) while keeping its intent: components exchange fields
> through connectors, a scheduler sequences them, and validation is eager. The
> spec written in Task 1 becomes normative; this document is working context.

**Status (2026-10-26):** Tasks 1–4 and 6 implemented on
`pzharrington/coupler-integration`; Task 5 (GPU real-model checks) not started.
Deviations from this plan, all reflected in `dev/spec/COUPLING_SPEC.md`:

- Scheduling is dataflow: a component waiting on a frame published later (for
  example inside a producer's multi-lead chunk) runs at a later tick instead of
  erroring at its due tick. Only chunks that never run by the final lead time
  raise.
- Sparse observations are `FrameSchema` forcing slots fetched with
  `fetch_dataframe` at the analysis time; an empty frame becomes `None`. Gridded
  optional slots were dropped.
- Components may have several output slots, one stream each (`name.i`), as
  long as the slots share lead times. Every model uses the runner's device;
  per-model device staging is deferred.
- `data_requests` lists only `fetch_data` calls, per the execution contract.
- There are no per-protocol adapters: `_Node` is one record of a model's parsed
  declarations, and the scheduler branches on `stateful` where the kinds differ.
- Package code is about 1,500 lines with docstrings and license headers.

**Goal:** Couple existing model instances at their own cadences, on their own
grids, with declared time reductions, and run the result under `Pipeline` like
any other `Runner`.

**Architecture:** Three public names in `earth2studio.coupling`: `Connector`
(a frozen edge declaration), `CoupledRunner` (validation, derived schedule and
execution of one work item), and `CouplingError`. The runner holds each
model's `(y, state)` and a valid-time-keyed buffer of published fields, and
pulls each model's inputs through connectors just before it runs. No component
class, field wrapper, dictionary, mediator, clock class, run sequence, YAML,
DSL or application layer.

**Stack:** Python, xarray, NumPy/CuPy, Torch, pytest; existing `earth2studio.run`,
`earth2studio.grids`, `earth2studio.utils.time_statistics`.

## Sources consulted

- `dev/spec/MODEL_CONTRACT_SPEC.md`, `DA_MODEL_CONTRACT_SPEC.md`,
  `EXECUTION_CONTRACT_SPEC.md`, `REGRID_SPEC.md`, `TIME_STATISTICS_SPEC.md`,
  `FETCH_DATA_SPEC.md`.
- `earth2studio/run/{runner,pipeline,_fields,work}.py` and
  `dev/examples/{06_runner,09_pipeline}.py`.
- PR #1229 (`nvcoupler-rebased-1.0.0-rc`): `driver.py`, `sequence.py`,
  `clock.py`, `connector.py`, `component.py`, `api.py`, `mediator.py`, and
  pzharrington's review comments (summarized below).
- `earth2-impact-modeling`: `docs/requirements/coupling-infrastructure.md`
  (CR1–CR7), `docs/two-way.md`, `docs/application-api.md`,
  `docs/vs-diagnostic.md`, `docs/implementation.md`, `docs/applications/*`,
  `examples/*.py`, talk notes.

## What the review asked for, and how this plan answers it

| Review point | Plan |
| --- | --- |
| Components are existing model protocols; no `Component` ABC | `CoupledRunner` takes `Mapping[str, PrognosticModel \| DiagnosticModel \| StatefulAssimilationModel]`, adapted to a private node layer (see Components). |
| Drop `PullAdapter`, `StateDataSource`, `ConditioningKwargAdapter` | Inputs are fed positionally through `initialize`/`step`/`__call__`; nothing is injected into a model. |
| Driver satisfies `Runner`; `steps()` the only output path; drop `collect`, `to_xarray`, `coupled()` | `CoupledRunner` implements `to`, `output_coords`, `data_requests`, `run_item`, and nothing else beyond `describe()`. Standalone use is `run_item(WorkItem(...))`, as for `PrognosticRunner`. |
| Connectors are declarations; driver owns state; same-pair collisions must not be silent | `Connector` is a frozen dataclass keyed by `(src, dst, slot)`; duplicates raise. Buffers and reductions live in the per-item run. |
| Use `earth2studio.grids.Regridder`, not a local alias | `Connector.regridder: Regridder \| None`; `R2`/`R4` checked at construction; grids compared with `same_geometry`. |
| Time stats via qualified labels; drop `CellMethod`, mediators | A slot labelled `tp:sum:24h` is reduced by the runner with `apply_time_statistic` over buffered base frames. |
| Use the lexicon, not a parallel vocabulary | Matching is by E2S variable name; lexicon units are the exchange convention (see Units). |
| Metadata on the DataArray, not a `Field` wrapper | Published fields are the model's outputs; validity is the `lead_time` coordinate plus the item time. |
| No YAML, no text DSL; `RunSequence`/`derive_sequence` semi-public | Dropped. Ordering is derived from declarations; `describe()` shows it. See Why no `RunSequence`. |
| Narrow public API; one `CouplingError(ValueError)` | `__all__ = ["Connector", "CoupledRunner", "CouplingError"]`. |
| Defer `vertical.py` | Not ported. |
| Multi-lead-time chunks, ensembles, irregular cadence, devices | Buffer keyed by valid time handles chunks; leading dims pass through connectors; irregular cadence is a source wrapper; one `to(device)` with per-model staging. |
| Rename to `earth2studio.coupling`; docs to `dev/spec` + `dev/examples`; `testing.py` into fixtures | Done as laid out below. |

## Design

### Components

A component is a name bound to a model. The coupler never subclasses or
mutates a model, so every component also runs outside the coupler and under
its conformance checker. From the model alone the runner reads:

- **Exports:** output slots, published after every call, plus the initial
  condition from `initial_condition(x)` at the item time for stateful models.
- **Imports:** `forcing_coords()` slots for stateful models, `input_coords()`
  slots for diagnostics. A slot is *connected* when a `Connector` targets it;
  otherwise it is *sourced* and fetched like `PrognosticRunner` fetches
  forcing. There is no auto-matching.
- **Cadence:** one rule for all kinds. A component whose outputs carry
  `lead_time` steps every `output lead_time[-1] - input lead_time[-1]`. Any
  other component fires when its required inputs gain a frame.
- **Grid:** `infer_grid()` on each slot; `same_geometry` decides whether a
  connector needs a regridder.

The scheduler and buffers talk to a private `_Node` (input slots, output
slots, `due(t)`, `fire(t, *x)`), with one adapter per protocol:

| Protocol | Fires | Adapter behavior |
| --- | --- | --- |
| `PrognosticModel` | At its cadence | `initialize` on first fire, `step` after; forcing window rules from the model contract |
| `DiagnosticModel` | When its inputs arrive | Stateless call |
| `StatefulAssimilationModel` | At its cadence if outputs carry `lead_time`, else when its forcing arrives | `initialize` then `step`; an optional slot receives `None` when no frame is available at the required time |

Where the shapes from the impact docs and the repo land:

| Shape | Example | Lands as |
| --- | --- | --- |
| Stateful, own cadence, several forcings | Soil bucket, hydrology, fire spread, crop stage | `PrognosticModel` |
| Stateful, corrected by sparse observations | Soil bucket nudged by 72 h SMAP | `StatefulAssimilationModel` with an optional slot |
| Stateless, any output geometry | Wind power at farms, damage at `asset_id`, point downscaling | `DiagnosticModel`; point geometry via a sampling `Regridder` |
| Transform between components | Unit conversion, derived variables, masking | `DiagnosticModel` placed between producer and consumer |
| Static context | DEM, land cover, exposure table, embeddings | A source on an initialization-only static slot |
| Running statistic on the producer's axis | Running max, degree-days | `PrognosticModel` whose state is the accumulator |
| Event or ragged products | TC tracks, fire perimeters, first-passage times | `StatefulAssimilationModel` with gridded forcing and `FrameSchema` outputs |
| Cross-member analysis | Loss exceedance curve | Not a component: `Pipeline` `Scorer` or post-processing |
| External process models | BMI-wrapped Fortran hydrology, WRF-SFIRE | Prognostic shape, but cannot meet `P19` or `P21`; follow-up |

TC trackers fit the stateful assimilation shape: `initialize(frame)` returns
the first detections and a state holding the path buffer, and
`step(tracks, frame, state=state)` returns the new track points and the
updated state. Migrating `TCTrackerWuDuan` and `TCTrackerVitart` moves their
mutable `path_buffer` into explicit state. What is missing is outside the
coupler: `Runner.run_item` yields only DataArrays, so a DataFrame-valued
stream needs an execution-contract extension. Both are follow-ups. The
"assimilation" name is a stretch for a tracker, but the protocol is exactly
"explicit state, gridded or tabular slots and outputs", so a second protocol
is not justified.

### Connectors

```python
@dataclass(frozen=True)
class Connector:
    src: str                        # producing component
    dst: str                        # consuming component
    slot: int = 0                   # dst forcing (or diagnostic input) slot
    variables: Mapping[str, str] | None = None  # dst label -> src variable
    regridder: Regridder | None = None          # src grid -> dst slot grid
    time: Literal["exact", "hold"] = "exact"
```

Resolution of a connector, given the dst slot's required valid times:

1. Take the slot's variable labels. A qualified label `name:method:window`
   maps to base `name` plus a reduction. `variables` renames when the producer
   uses a different label.
2. For each required valid time `T`, `exact` needs frames at `T`; `hold` takes
   the latest frame at or before `T`. A reduction selects frames in
   `[T - w, T)` and calls `apply_time_statistic`.
3. Apply `regridder` (the slot grid is its `target_grid`), then `_map_field`
   onto the slot signature.

Any other transform is a `DiagnosticModel` component between the two models.
That keeps the connector a pure function of the buffer and moves custom code
under an existing contract and conformance checker. Buffer retention per
producer is the maximum over its consumers of window plus lag.

### Scheduling

The clock is `dt = gcd(cadences)` stepping from `item.time + dt` through
`item.time + final_lead_time`. At each time `t`:

1. The *due* set is every component whose `_Node.due(t)` is true.
2. A due component is *ready* when every connected slot can be resolved from
   the buffer for its required valid times. Required valid times follow the
   model contract: `initialize` needs the whole forcing window at
   `item.time + forcing lead_time`; `step` needs the window's final lead time
   shifted by each lead time of the current `y`.
3. Ready components run in declaration order; publishing may make others
   ready. If due components remain and none is ready, raise `CouplingError`
   naming the cycle and the fix: `time="hold"` on one edge, or a forcing
   declaration at an earlier lead time.

Lagged versus sequential coupling is derived from what models declare. A slot
needing frames valid at `t` from a producer also due at `t` is sequential. A
slot needing earlier frames, or using `time="hold"`, is lagged. DLESyM's shape
falls out directly: the atmosphere's SST slot at lead `0h` reads the ocean's
previous chunk, and the ocean's `z1000:mean:48h` slot reads the atmosphere's
fresh chunk.

`initialize` is scheduled like a step, at the first `t` where the component's
forcing is resolvable. The soil model needs `tp:sum:24h` valid at `+24h`, so
it initializes at `t0 + 24h` after four atmosphere steps and publishes its
initial condition at `t0`. Frames valid at or before `item.time` always come
from a source, the slot's `default_sources()` entry unless overridden; later
frames come from the connector. This is how StormScope MRMS takes its GOES
history from the GOES archive and later frames from the GOES model.

Validation is eager at `CoupledRunner` construction: unknown names, duplicate
`(src, dst, slot)`, labels the producer does not export, renames between
different units once the units table exists, windows not divisible by
producer cadence, grid mismatch without a regridder, regridder grid mismatch
(`R2`/`R4`), and cycles. `describe(final_lead_time)` dry-runs the same
scheduler without models and returns the plan as text, so the preview cannot
drift from execution.

### The runner

```python
class CoupledRunner:
    supports_member_batching = False
    def __init__(self, components, connectors=(), sources=None, seed=0): ...
    def describe(self, final_lead_time) -> str
    def to(self, device) -> CoupledRunner
    def output_coords(self, final_lead_time) -> Mapping[str, CoordinateSystem]
    def data_requests(self, item) -> tuple[DataRequest, ...]
    def run_item(self, item) -> Iterator[Mapping[str, xr.DataArray]]
```

- `sources` maps `(component, slot)` or `component` to a source for input
  slots and sourced slots; defaults come from `recommended_sources`.
- One stream per component, named by it. A stream's schema is
  `_signature_for_stream(output_coords, leads)` with the component's own leads
  through the final lead time. `X1`–`X4` hold as for `PrognosticRunner`.
- Each clock time yields the components that published at it, each with a
  leading `time` dimension labelled `item.time`.
- Seeding follows `X5`: every stochastic component is seeded with
  `member_seed(seed, item.time, member_id)` before `initialize`.
- `to(device)` moves every model; fields are staged to each model's device at
  the call boundary, so a CPU-only impact model couples to GPU models.
- Leading dimensions other than `time` pass through connectors untouched, so
  member batching later needs only IC batching and the capability flag.

`Pipeline` is the only composition and execution layer above the runner. The
impact docs' `Application` sugar is not planned: `CoupledRunner` construction
is already the declaration, and a second front door would compete with
`Pipeline` for the same role.

### Out of scope for the first PR

Interpolation in time, conservative regridding, unit conversion, vertical
interpolation, masks beyond regridder `fill_value` and NaN propagation,
DataFrame payloads through connectors, DataFrame-valued streams, member
batching, mid-item checkpoint, iterator hooks on coupled models, and splitting
the fused DLESyM wrapper. Each is listed under Follow-ups.

## Why no `RunSequence`

An explicit sequence would earn its place only by expressing something the
declarations cannot. Each candidate reason fails:

- **Lagged versus sequential choice.** Expressed per edge with `time="hold"`
  or an earlier forcing lead, which is local and survives adding components.
- **Sub-cycling.** Fully determined by cadences.
- **Deterministic tie-breaks.** Declaration order already fixes them.
- **Inspection and debugging.** `describe()` shows the derived order without
  letting users author one that disagrees with the data dependencies.
- **Concurrency.** A linear sequence hides which components are independent;
  the derived graph exposes it for later stream-parallel execution.
- **NUOPC familiarity.** A documentation concern, met by `describe()` output.

`RunSequence`, `derive_sequence`, and the text DSL are not ported.

## Units

Name-only matching is sound when both sides honor the lexicon, because a
lexicon name implies its unit: data sources already convert to lexicon units
through their lexicon modifiers, and built-in models emit them. It fails in
these cases:

- **A component departs from lexicon units.** A user's bucket model reading
  `tp` in mm instead of m is silently wrong by 1000×.
- **Renames across different quantities.** `variables={"tp": "tpf"}` maps a
  flux in kg m-2 s-1 onto an accumulation in m.
- **Reductions over rates.** `tpf:sum:24h` sums a rate; `sum` is only
  meaningful for accumulations.
- **Names outside the lexicon.** Impact outputs such as `damage_fraction`
  match by exact string with no semantic check.
- **Equivalent quantities do not match.** `tp06` and `tp:sum:6h` need an
  explicit `variables` rename.

A units table is cheap. Of 388 lexicon entries, 348 end in a parenthesised
unit, so a script can generate `E2STUDIO_UNITS`; about 65 entries need hand
attention (40 with no unit, about 25 whose parenthetical is a wavelength),
plus normalization such as `0 - 1` versus `%`. One test keeps the table in
step with the vocabulary. Estimate: about one day plus review.

It enables two coupler checks that are otherwise impossible: a rename between
names with different units is an error that suggests a converting diagnostic,
and `sum` over a unit containing `s-1` is an error that suggests `mean`. It
cannot catch a component that departs from lexicon units, since models do not
declare slot units. The larger payoff is CF `units` attributes in IO backends,
which are not written today. Recommendation: land the table as a separate
lexicon PR and enable the two checks when it exists. The coupling spec states
that connected slots exchange lexicon units.

## Tasks

Work on a branch from `1.0.0-rc`. Follow `dev/README.md`: spec first, backing
code, numbered example, tests. Run Python with `uv run` or `.venv/bin/python`.
No CHANGELOG edit. Open the PR with the `developer-open-refactor-pr` skill.

### Task 1: Spec

**Files:** `dev/spec/COUPLING_SPEC.md` (new), `EXECUTION_CONTRACT_SPEC.md`
(edit), `DA_MODEL_CONTRACT_SPEC.md` (edit).

- [ ] Write `COUPLING_SPEC.md` from the Design section: goal, component
  derivation and cadence rule, `Connector` and resolution rules, scheduling
  rules, runner interface and stream rules, rule table `C1`–`Cn`, out of
  scope, and a short lineage paragraph linking #1229 and the impact-modeling
  docs. Match the register of `REGRID_SPEC.md`.
- [ ] In `EXECUTION_CONTRACT_SPEC.md`, update the coupled-runner paragraph and
  the illustrative `Pipeline(driver.to_runner(...))` line to
  `Pipeline(CoupledRunner(components, connectors), output=...)`, and point
  phase 3 at the new spec.
- [ ] In `DA_MODEL_CONTRACT_SPEC.md`, add the cadence rule for stateful models
  and how a source reports "no observation at this time" for optional slots.

### Task 2: Buffer and connector

**Files:** `earth2studio/coupling/__init__.py`, `errors.py`, `_buffer.py`,
`connector.py`; `test/coupling/conftest.py`, `test_connector.py`.

- [ ] `CouplingError(ValueError)` with a did-you-mean helper, trimmed from
  `nvcoupler/errors.py`.
- [ ] `_buffer.py`: one producer's frames by valid time, with `select`,
  `latest_at_or_before`, `window`, and `retain_after`; backing agnostic;
  carries leading dims. Tests with hand-built multi-lead chunks.
- [ ] `connector.py`: the dataclass, label splitting with `time_statistics`
  helpers, required valid times per the model contract, and resolution:
  exact or hold, reductions via `apply_time_statistic`, `regridder`,
  `_map_field`. Re-reducing accumulations (`tp:sum:6h` into `tp:sum:24h`) is
  allowed when the window is a multiple of the producer cadence.
- [ ] `conftest.py`: tiny CPU models reusing the add-one pattern from
  `test/run/test_runner.py`: an unforced 6 h producer, a forced 24 h consumer
  with a `tp:sum:24h` slot on a coarser grid, a 1 h consumer with a `hold`
  edge, a diagnostic, and a stateful assimilation model with an optional slot.
  Each passes its conformance checker in a test.

### Task 3: Nodes, scheduler and runner

**Files:** `earth2studio/coupling/_nodes.py`, `_schedule.py`, `runner.py`;
`test/coupling/test_schedule.py`, `test_runner.py`, `test_pipeline.py`.

- [ ] `_nodes.py`: `_Node` and the three adapters.
- [ ] `_schedule.py`: the scheduler as a pure function over nodes plus a
  "frames available" predicate, shared by `describe()` and `run_item`. Cycle
  error text names the components and both fixes.
- [ ] `runner.py`: construction-time validation, per-item run state, IC and
  pre-`item.time` frames fetched through `DataRequest`s, `output_coords`,
  `data_requests`, `run_item`, `to`, `describe`, `X5` seeding, device staging.
- [ ] Tests: the hand-computable atmosphere/ocean toy from #1229 example 01
  (`z(96h) = 19.2336`, `sst(96h) = 2.180147`, 48 h means `[4.2, 13.8147]`),
  lagged versus sequential derivation, `hold`, a diagnostic in the chain, a
  late-initializing consumer, an optional slot fed `None`, multi-lead chunks,
  a regridded edge, cycle and duplicate errors, `data_requests` completeness,
  and `Pipeline` writing one `XarrayBackend` per component.
- [ ] Equivalence: a forced single model run as two coupled components
  reproduces `PrognosticRunner` on the fused form.

### Task 4: Examples

**Files:** `dev/examples/10_coupling.py`, `dev/examples/README.md`.

- [ ] One example, CPU only, no downloads. A synthetic 6 h atmosphere feeds
  the impact docs' soil bucket, written as a ~50-line
  `StatefulAssimilationModel` with `tp:sum:24h` and `t2m:mean:24h` forcing and
  an optional `swvl1` observation slot that is `None` between 72 h
  retrievals. Print `describe()`, run one item with `run_item`, then run
  several initial times under `Pipeline` with per-component stores. Close with
  a two-model feedback loop showing the cycle error and the `time="hold"` fix.
- [ ] Type-annotated and docstringed (mypy and interrogate apply).

### Task 5: Real-model checks (GPU, may land as a follow-up PR)

- [ ] StormScope GOES to MRMS as two components with default sources, compared
  against the explicit loop in `MODEL_CONTRACT_SPEC.md` and the eval
  `stormscope` pipeline. Use `Region` views to keep it small.
- [ ] FCN plus `PrecipitationAFNO` feeding the example's soil model at 0.25°,
  reproducing the impact docs' 15-day run shape.

### Task 6: Verification and PR

```bash
.venv/bin/python -m pytest test/coupling -q --tb=short
.venv/bin/python -m pytest test/run -q --tb=short
.venv/bin/python dev/examples/10_coupling.py
make format && make lint
```

Rule identifiers in `COUPLING_SPEC.md` must match error messages and test
names. PR title prefix `[E2S 1.0]`, label `1.0.0`, base `1.0.0-rc`.

## Size estimate

| Module | Lines |
| --- | --- |
| `errors.py` | 40 |
| `_buffer.py` | 100 |
| `connector.py` | 200 |
| `_nodes.py` | 150 |
| `_schedule.py` | 200 |
| `runner.py` | 300 |
| tests | 800 |
| example | 250 |

Roughly 1,000 lines of package code against about 4,500 in #1229's
`earth2studio/nvcoupler/`.

## Follow-ups

- **DataFrame streams and TC trackers:** extend `Runner.run_item` and
  `Pipeline` outputs to DataFrame-valued streams, then migrate the trackers to
  `StatefulAssimilationModel`. Useful without the coupler.
- **DataFrame payloads through connectors:** the valid-time buffer extends to
  `pd.DataFrame` frames, for observation slots fed by another component.
- **Units table:** `E2STUDIO_UNITS` in the lexicon, then the two connector
  checks and CF `units` attributes in IO backends.
- **Time interpolation:** `time="interpolate"` when a fast consumer, such as
  1 h hydrology under 6 h forcing, needs more than `hold`.
- **Member batching:** batch ICs along `ensemble` and set
  `supports_member_batching = True` (execution spec open decision 2).
- **Mid-item checkpoint:** `(y, state)` per component plus buffers are
  serializable; needs a runner capability (execution spec open decision 3).
- **Hooks:** per-component perturbation through `front_hook`/`rear_hook`
  attributes, once a perturbation use case arrives.
- **Impure models:** a relaxed conformance tier waiving `P19` and `P21`, with
  a runner capability saying the item cannot be checkpointed, once a BMI or
  external-process model is in hand.
- **Split DLESyM:** two wrappers sharing weights, atmosphere with an SST slot
  at `0h`, ocean with `z1000:mean:48h`/`ws10m:mean:48h` slots; equivalence
  test against the fused model.
- **Conservative regridder** under the `Regridder` protocol, the blocker named
  by `two-way.md` for flux exchange.
