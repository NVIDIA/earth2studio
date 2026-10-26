<!-- markdownlint-disable MD013 -->
# Earth2Studio Coupling

## Goal

Run several models together, each at its own cadence and on its own grid, with
fields passed between them, and run the result under `Pipeline` like any other
`Runner`. Components are ordinary models: the coupler never wraps, subclasses or
mutates them, so every component also runs alone and under its conformance
checker.

Status: `earth2studio.coupling` exports `Connector`, `CoupledRunner` and
`CouplingError`. Tests are in `test/coupling/`; the tutorial is
`dev/examples/10_coupling.py`.

Lineage: this replaces the `nvcoupler` prototype of PR #1229 and keeps its
intent (connectors between components, a derived schedule, eager validation and
a readable plan) while building on the model, regrid, time-statistics and
execution contracts. The impact-modeling requirements it serves are in the
`earth2-impact-modeling` docs (CR1–CR7).

## Interface

```python
from earth2studio.coupling import Connector, CoupledRunner

runner = CoupledRunner(
    {"weather": weather_model, "soil": soil_model},
    [Connector("weather", "soil", regridder=to_soil)],
    sources={"weather": weather_ic, "soil": (soil_ic, None, retrievals)},
)
print(runner.describe(np.timedelta64(72, "h")))
Pipeline(runner, output={"weather": w, "soil": s}).run(items)
```

```python
@dataclass(frozen=True)
class Connector:
    src: str
    dst: str
    slot: int = 0
    variables: Mapping[str, str] | None = None  # dst base label -> src variable
    regridder: Regridder | None = None
    time: Literal["exact", "hold"] = "exact"

class CoupledRunner:  # satisfies earth2studio.run.Runner
    supports_member_batching = False
    def __init__(self, components, connectors=(), sources=None, seed=0): ...
    def describe(self, final_lead_time) -> str: ...
    def to(self, device) -> CoupledRunner: ...
    def output_coords(self, final_lead_time) -> Mapping[str, CoordinateSystem]: ...
    def data_requests(self, item) -> tuple[DataRequest, ...]: ...
    def run_item(self, item) -> Iterator[Mapping[str, xr.DataArray]]: ...
```

`sources` maps a component to one source per input-then-forcing slot, in the
order of `default_sources()`; a single source covers the first input slot and
`None` entries fall back to the model's recommendation. `CouplingError` is a
`ValueError`.

## Components

| Protocol | Runs | Calls |
| --- | --- | --- |
| `PrognosticModel` | Every cadence | `initialize(*x, *forcing)`, then `step(*y, *forcing, state=state)` |
| `StatefulAssimilationModel` | Every cadence | As above; DataFrame slots may be `None` |
| `DiagnosticModel` | At each valid time its inputs arrive | `model(*inputs)` |

A component may have several gridded output slots. A stateful component's
output slots share their lead times; models whose outputs advance at different
cadences, such as fused DLESyM, are split into components. The cadence is the
outputs' last lead time minus the first input slot's last lead time. Output chunk `k` covers the output lead times
plus `k` cadences and is due at its last lead time; chunk 0 comes from
`initialize`. A component whose forcing reduces a day of another's output
therefore initializes once that day exists, with no special case.

Imports are a stateful component's time-varying gridded forcing slots, or a
diagnostic's input slots. A connector feeds an import; every other slot is
fetched from its source. Input slots, static forcing and DataFrame slots are
always fetched.

## Connectors and resolution

For each call, every import needs a set of valid times, following the model
contract: `initialize` takes the whole forcing window; `step` takes the window's
final lead time shifted by each lead time of the output it advances; a
diagnostic takes the valid time it is called for. For each required time `T`
and slot label:

1. Split the label: `tp:sum:24h` is base `tp` with modifier `sum:24h`. Map the
   base through `variables` to the producer's variable, which exactly one of
   the producer's output slots publishes.
2. Plain label, `exact`: the producer's frame at `T`. Plain label, `hold`: the
   latest frame at or before `T` that the producer published on an earlier
   tick. Qualified label: the producer's frames in `[T - w, T)` at its frame
   spacing, reduced with `apply_time_statistic`.
3. If the producer cannot supply it and `T` is at or before the item time, a
   stateful consumer fetches it from the slot's source. Otherwise the consumer
   waits.
4. Regrid producer frames with the connector's regridder, then map onto the
   slot signature with the slot's temporal-statistics metadata.

Any other transform between models, such as unit conversion or derived
variables, is a `DiagnosticModel` component placed between them.

## Scheduling

Ticks are the lead times at which some stateful component's chunk is due.
Before the first tick, every stateful component fetches its inputs, holds them
as frames at their lead times, and publishes an initial condition for each
output slot whose same-position input slot has the same variables. State slots
come first under the model contract, so output-only slots start at the first
chunk. At each tick, components run in
declaration order whenever their imports resolve, repeating until nothing more
can run. A component whose import waits on a frame published later runs later;
a diagnostic runs once per valid time at which all its exact inputs exist.

Lagged and sequential coupling are not declared. An `exact` slot needing a
frame at the same valid time as its producer's output runs after it; a `hold`
slot, or one declaring an earlier lead time, lags. A loop of `exact` edges
cannot run and is reported as a cycle.

Construction dry-runs the schedule over two periods of the cadences' least
common multiple beyond the latest first chunk. The same scheduler drives
`describe`, `output_coords`, `data_requests` and `run_item`, so a plan cannot
drift from a run. Frames older than the oldest one any consumer read in that
dry run, plus the longest cadence, are dropped as the run advances.

## Streams

Each output slot yields one stream: named after the component when it has one
output slot, else `name.0`, `name.1` and so on. Stream names must be unique. A
stateful slot publishes its initial condition, when it has one, and every chunk;
a diagnostic slot publishes at every call.
Streams carry the producing model's grid and the item's `time`, as `X1`–`X4`
require. Per-member seeding follows `X5`.

## Independent assimilation models

An `AssimilationModel` without `initialize` and `step`, such as HealDA, computes
one analysis per call from observation DataFrames, with no lead times. It is not
a component today: as a diagnostic it would have no connected input to tell it
when to run, so the runner rejects its DataFrame slots. It fits elsewhere:

- **As a source.** Initializing a prognostic model from HealDA means its input
  slot's source is an analysis. A `DataSource` adapter fetches each observation
  slot at the requested times, calls the model and returns the analysis, so it
  serves `PrognosticRunner` and `CoupledRunner` alike, behind `RegriddedSource`
  when grids differ. Models still never fetch (`P22`); the adapter does.
- **As a component, later.** Publishing analyses on a schedule during a run
  needs components that run on a fixed cadence from sources alone.
- **Cycling is different.** An analysis that replaces a prognostic model's state
  mid-run, with a forecast as background, is a `StatefulAssimilationModel`, or
  warm-start work items (execution contract open decision 1).

## Rules

| Rule | Requirement |
| --- | --- |
| `C1` | Components are prognostic, stateful assimilation or diagnostic models with gridded output slots; a stateful component's slots carry the same `lead_time` |
| `C2` | Stateful components step at their declared cadence; diagnostics run when their inputs arrive and need at least one `exact` connector |
| `C3` | One connector per `(dst, slot)`, targeting a time-varying gridded forcing slot or a diagnostic input slot |
| `C4` | Every slot label's base, after `variables`, is published by exactly one producer output slot; renames name declared bases |
| `C5` | Without a regridder each producer slot read and the consumer slot have the same geometry; with one, its source grid matches each slot read and its target grid the consumer slot |
| `C6` | Required valid times follow the model contract; DataFrame slots are requested at the chunk's end, static slots only initially |
| `C7` | `exact` waits for frames at each required time; `hold` takes the latest earlier-published frame and applies to plain labels only |
| `C8` | Qualified labels reduce producer frames over `[T - w, T)`; the window must divide by the producer's frame spacing |
| `C9` | Frames at or before the item time that the producer does not hold come from the slot's source, for stateful consumers only |
| `C10` | An empty DataFrame from a source is passed as `None` |
| `C11` | A stateful chunk due by the final lead time that cannot run raises, naming the cycle or the missing frame; construction checks this first |
| `C12` | One uniquely named stream per output slot; `describe`, `output_coords` and `data_requests` come from the scheduler that runs the item; `data_requests` lists `fetch_data` calls only, not DataFrame fetches |

## Out of scope or future work

- Time interpolation, conservative regridding, unit conversion and vertical
  interpolation.
- A `DataSource` adapter over independent assimilation models, with per-member
  seeding when they are stochastic.
- Components that run on a fixed cadence from sources alone, such as
  observation-only analyses.
- DataFrame payloads through connectors, and DataFrame-valued streams.
- Output slots that advance at different lead times; split them, as for DLESyM.
- Member batching, mid-item checkpoints and iterator hooks on coupled models.
- Per-model devices: every model and fetch uses the runner's device.

## Open decisions

1. **Units.** Connected slots exchange lexicon units. A lexicon units table
   would let connectors reject renames across units and sums over rates.
2. **DataFrame streams.** `Runner.run_item` yields DataArrays only, which blocks
   tabular products such as TC tracks as coupled outputs.
