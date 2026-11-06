---
name: earth2studio-couple-models
description: >
  Use when running two or more Earth2Studio models together so one feeds
  another (e.g. StormScope GOES to MRMS, atmosphere and ocean at different
  cadences, a weather model driving an impact model): wires components and
  connectors into a CoupledRunner, checks the schedule, and runs it under
  Pipeline. Not for single-model forecasts, new model wrappers, or data sources.
argument-hint: The models to couple, and which outputs feed which slots (optional)
---

# Couple Models

`earth2studio.coupling.CoupledRunner` runs named components, each at its own
cadence and on its own grid. `Connector` edges pass fields between them. The
schedule is derived from what each model declares, and a coupled run is a
`Runner`, so `Pipeline` writes one store per stream. Components are ordinary
models. Do not wrap or subclass them, and do not add pair-specific code to the
runner: express the exchange with connectors. Every component still runs alone.

Use `PrognosticRunner` instead when only one model runs and its forcing comes
from data sources.

## Components

| Component | Runs | Calls |
| --- | --- | --- |
| Prognostic model | Every cadence | `initialize(*inputs, *forcing)`, then `step(*outputs, *forcing, state=state)` |
| `StatefulAssimilationModel` | Every cadence | As above; DataFrame slots may be `None` |
| Diagnostic model | At each valid time its inputs arrive | `model(*inputs)` |

- Outputs are gridded with a `variable` dimension. A stateful component's output
  slots share their lead times; a model whose outputs advance at different
  cadences (fused DLESyM) is split into components first.
- Cadence = last output lead minus the later of the first input slot's last lead
  and zero. Output chunk `k` covers the output leads plus `k` cadences and is due
  at its last lead; chunk 0 comes from `initialize`. A consumer whose forcing
  reduces a day of another's output initializes once that day exists.
- An independent assimilation model (HealDA, interpolation, CorrDiff) is not a
  component: it has no `initialize`/`step`, and a diagnostic needs a connected
  input to know when to run, so its DataFrame slots are rejected. Use it behind
  a `DataSource` adapter that fetches the observation slots, calls the model and
  returns the analysis (no such adapter ships yet); the model itself never
  fetches. Only `StatefulAssimilationModel` couples.

## What connectors feed

- **Imports** are a stateful component's time-varying gridded forcing slots, or
  a diagnostic's input slots. A connector feeds the labels its `variables`
  names, or every label of the slot without `variables`. Several connectors may
  feed one slot from different producers if their labels differ. Labels no
  connector feeds are fetched from the slot's source.
- Input slots, static forcing (no `lead_time`) and DataFrame slots are always
  fetched from sources, never fed by connectors. Static slots are fetched once,
  at initialization; DataFrame slots are requested at the end of the chunk being
  produced, and an empty frame arrives as `None`.
- A diagnostic input slot without `lead_time`, as most declare, takes the call's
  valid time as a leading dimension, as under `DiagnosticRunner`.
- **State slots**: a stateful component's input slot whose same-position output
  slot declares its variables on the same grid. A `state=True` connector
  overwrites the variables it names there before each call, for models that
  carry a coupled field such as SST in their own state: in the inputs
  `initialize` receives and in the output each `step` advances. Values the
  producer cannot supply at or before the item time are kept, not fetched. The
  overwrite is not published; streams carry each model's own output.
- **Resolution** per required valid time `T` and label: split the label
  (`tp:sum:24h` is base `tp`, modifier `sum:24h`) and map the base through
  `variables` to the producer's variable. Plain label with `exact`: the
  producer's frame at `T`. Plain label with `hold`: the latest frame at or
  before `T` published on an *earlier* tick. Qualified label: the producer's
  frames in `[T - w, T)` at its frame spacing, reduced with `apply_time_statistic`
  (the frame at `T` is excluded). If the producer cannot supply it and `T` is at
  or before the item time, a stateful consumer takes it from the slot's source;
  for a window ending later whose missing frames are at or before the item time
  (the producer publishes no initial condition), those early frames of the base
  variable come from the source and are reduced together with the producer's.
  Otherwise the consumer waits: frames after the item time always come from the
  producer. Then frames are staged onto the consumer's device, regridded, and
  mapped onto the slot signature with its statistics metadata.
- Required times follow the model contract: `initialize` takes the whole forcing
  window; `step` takes the window's final lead shifted by each lead of the output
  it advances; a diagnostic takes its valid time.
- Connectors select, reduce and regrid; they never convert. Derived variables,
  units and vertical levels are a diagnostic component between the models
  (`DerivedWS`, `DerivedPressureLevels`), which publishes its own stream.
  Connected slots exchange lexicon units; nothing checks units across a
  connector, so check them yourself.

## Scheduling and streams

- Ticks are the lead times at which some stateful component's chunk is due.
  Before the first tick every stateful component fetches its inputs and holds
  them as frames, and publishes an initial condition for each output slot that
  declares its same-position input slot's variables on the same grid and whose
  input ends at zero. State slots come first under the model contract, so
  output-only slots (a diagnostic on another grid) and outputs of a background
  that ends before zero start at the first chunk.
- At each tick components run in declaration order whenever their imports
  resolve, repeating until nothing more can run. A consumer waiting on a frame
  published later runs later; a diagnostic runs once per valid time at which all
  its `exact` inputs exist (its frame spacing is the least common multiple of its
  producers').
- Lag and sequence are not declared, they follow from the edges: an `exact` slot
  needing a frame at its producer's valid time runs after it; a `hold` slot, or
  a slot declaring an earlier lead, lags. A loop of `exact` edges is a cycle.
- Construction dry-runs the schedule over two periods of the cadences' least
  common multiple beyond the latest first chunk, so cycles and missing sources
  fail there. The same scheduler drives `describe`, `output_coords`,
  `data_requests` and `run_item`, so a plan cannot drift from a run. Old frames
  are dropped as the run advances.
- Each output slot is one stream, named after the component when it has one
  output slot, else `name.0`, `name.1`; names must be unique. A stateful slot
  publishes its initial condition (when it has one) and every chunk; a diagnostic
  slot publishes at every call. Streams carry the producing model's grid and
  the item's `time`. `data_requests` lists `fetch_data` calls only, not
  DataFrame fetches.
- Seeding: each stochastic component is seeded per item with `set_rng(...,
  reset=True)`, from `seed` offset by the CRC-32 of its name, so components draw
  independent noise. An item carries at most one member
  (`supports_member_batching = False`; `work_items(..., member_batch=1)`).

## Workflow

1. **Read the declarations, not the docs.** For each model, print
   `input_coords()`, `forcing_coords()` and `output_coords(input_coords())`, as
   `inspect` in the [template](references/coupling-template.py) does.
   Record for each model:
   - its kind: stateful (`initialize`/`step`: prognostic or stateful
     assimilation) or diagnostic,
   - its cadence (see Components; a background input ending before zero does not
     lengthen the step),
   - each slot's variables, including qualified labels such as `tp:sum:24h`,
   - its lead times and grids, and which slots are DataFrames or static.

   Then list, slot by slot, which producer variable feeds which consumer slot.
   Anything left over comes from a source.
2. **Choose a connector for each fed slot** using the
   [patterns](references/connector-patterns.md):
   - `exact` for the same valid time or a reduction, `hold` for a lag or to
     break a feedback loop;
   - `variables` for renames, or to feed part of a slot so several producers
     (and the slot's source) can share it;
   - a regridder whenever grids differ;
   - `state=True` for fields the model carries in its own state;
   - a diagnostic component between models for any conversion: derived
     variables (`DerivedWS`), units, or vertical levels
     (`DerivedPressureLevels`). Connectors never convert.

   State the timing you chose for each edge and why.
3. **Build the runner** from the [template](references/coupling-template.py):
   components in a dict, connectors, `sources` in input-then-forcing slot order
   (`None` keeps a recommendation, a single source covers the first input slot,
   more sources than slots is an error), and optionally `devices` and `seed`.
   Construction validates everything; fix each `CouplingError` using the
   tables below.
4. **Check the schedule** with `print(runner.describe(final_lead_time))` over a
   few cadences of the slowest component, before running anything. Each row is
   a call: when it is due, what it outputs, and where each connected slot reads
   from. `[forcing[0] <- land at +6h from +0h]` means the slot needed +6h and
   got the +0h frame (a `hold` lag); `from +0h..+18h` is a reduction window.
   Confirm every lag and window is the intended one.
5. **Run one item**, `runner.run_item(WorkItem(time, final_lead_time))`, and
   sanity-check a few values per stream. Then run
   `Pipeline(runner, output={stream: backend, ...}, device=...)`.
6. **Test** with hand-computable toys that mirror the real cadences, labels and
   grids. Follow the [testing template](references/testing-template.py): assert
   exact published values, the lead times each component was forced at, the
   `describe` rows, and that wrong wiring raises `CouplingError`. Tests with real
   weights are marked `@pytest.mark.package`.
7. **Verify** with the commands below, and report anything you did not run.

```bash
uv run python skills/earth2studio-couple-models/references/coupling-template.py
uv run pytest <your tests> -q
make format && make lint
```

## Rules and errors

`CouplingError` is a `ValueError`; its message names the rule it broke and
suggests close names for misspelled components and variables. Rules C6, C9 and
C10 describe runtime behavior rather than errors.

| Rule | Requirement | Typical cause / fix |
| --- | --- | --- |
| `C1` | Components are prognostic, stateful assimilation or diagnostic models with gridded outputs; a stateful component's output slots share one `lead_time` | Outputs not gridded or at different leads: split the model |
| `C2` | Stateful components step at their declared cadence; a diagnostic runs when its inputs arrive and needs at least one `exact` connector | Feed it with an `exact` edge |
| `C3` | Connectors target a time-varying gridded forcing slot or a diagnostic input slot; several may feed one slot if their labels differ. State connectors target state slots and may share one if their variables differ | Two connectors feed one label, or a bad `slot=`: name each connector's labels in `variables` |
| `C4` | Every fed label's base, after `variables`, is published by exactly one producer output slot; `variables` names declared bases | Label unknown or published by two output slots: rename with `variables=` or pick the right producer |
| `C5` | Without a regridder, each producer slot read and the consumer slot have the same geometry; with one, its source grid matches each slot read and its target grid the consumer slot | `BilinearRegridder(src_grid, dst_grid)` built from `infer_grid` |
| `C6` | Required valid times follow the model contract; DataFrame slots are requested at the chunk's end, static slots only initially | (behavior) |
| `C7` | `exact` waits for frames at each required time; `hold` takes the latest earlier-published frame and applies to plain labels only | `hold` on a qualified label: use `exact`, or declare the slot at an earlier lead |
| `C8` | Qualified labels reduce producer frames over `[T - w, T)`; the window must divide by the producer's frame spacing | Fix the window or the producer's cadence |
| `C9` | Frames at or before the item time that the producer does not hold come from the slot's source, for stateful consumers only, including early frames of a later window; state connectors keep the model's own values instead | (behavior) |
| `C10` | An empty DataFrame from a source is passed as `None` | (behavior) |
| `C11` | A stateful chunk due by the final lead time that cannot run raises, naming the cycle or the missing frame; construction checks this first | Cycle of `exact` edges, or a frame that never arrives: make one edge `hold`; check cadences and the final lead |
| `C12` | One uniquely named stream per output slot; `describe`, `output_coords` and `data_requests` come from the scheduler that runs the item | Two output slots resolve to the same stream name: rename a component |

Other construction errors: a slot with no source and no connector (`needs a
source`), more sources than slots, a source or device for an unknown component,
a connector naming an unknown component, `state=True` without `variables`, a
component that is neither stateful nor diagnostic, and a runner with no stateful
component.

## Common Mistakes

- **Expecting list order to sequence the models.** Order follows the edges; dict
  order only breaks ties between components ready at the same tick.
- **Using `hold` where `exact` was meant.** This silently lags by a step.
  Check the `describe` row.
- **Reductions use the window `[T - w, T)`.** The frame at `T` is excluded.
- **Expecting an initial condition in every stream.** Streams whose outputs
  differ from the inputs, or whose input ends before zero, start at the first
  forecast.
- **Regridding the wrong way round.** The source grid is the producer's output
  grid and the target is the consumer's slot grid.
- **Converting inside a connector.** Connectors only select, reduce and
  regrid; put conversions in a diagnostic component.
- **Coupling a fused multi-cadence model.** Split it into components first.
- **Feeding a DataFrame slot through a connector.** Observation slots come from
  sources; an empty result reaches the model as `None`.
- **Batching ensemble members.** One member per item.

## Reference Map

| Reference | Read when |
| --- | --- |
| [Coupling template](references/coupling-template.py) | Always; runnable CPU starting point |
| [Connector patterns](references/connector-patterns.md) | Choosing timing, variables, grids, units, state, sources, devices |
| [Testing template](references/testing-template.py) | Writing tests for a coupling |
| [Coupling package](../../earth2studio/coupling/) | Docstrings for `Connector` and `CoupledRunner` |
| [DA protocols](../../earth2studio/models/da/base.py) | What a stateful assimilation component must declare |
| [Run package](../../earth2studio/run/) | `Runner`, `WorkItem`, `Pipeline` |
| [Coupling tests](../../test/coupling/) | Worked toys: observations, state overwrite, chunks, seeding, devices |
| [Migration notes](../../MIGRATION.md) | Model contract changes behind the slot and state rules |
