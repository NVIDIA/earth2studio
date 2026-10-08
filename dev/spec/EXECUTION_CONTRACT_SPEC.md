# Execution Contracts (draft)

Goal: one user-facing execution API, `Pipeline`, for single-model and coupled
runs, without pulling coupling mechanics into it.

## Shape

- `Pipeline` supervises **work**: distribution across ranks, member grouping,
  resume, output filtering and regridding, scoring, and IO. One concrete class,
  never subclassed; variation points are constructor arguments.
- A `Runner` executes **one work item**: it fetches inputs, steps models, and
  yields outputs. Built-in runners cover the common cases; anything else is a
  small hand-written class.

Rule of thumb: "what does model B receive from model A, and when?" belongs to the
runner, or the coupler behind it. "Which rank runs this item, where does its
output go, and what happens if it crashes?" belongs to `Pipeline`.

```python
model = SFNO.load_model(SFNO.load_default_package())
Pipeline.from_model(model, GFS()).run(items)

# Coupled (illustrative): the coupler's Driver supplies the runner.
Pipeline(driver.to_runner(ics={...})).run(items)
```

`earth2studio.run` never imports the coupler; coupled runners live with it.

## Runner

`earth2studio.run.runner`:

```python
class Runner(Protocol):
    supports_member_batching: bool
    def to(self, device) -> Runner
    def output_coords(self, horizon) -> Mapping[str, CoordSystem]
    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None
    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]
```

- **Built once, run per item.** A runner binds sources, never data; initial
  conditions and forcing are fetched inside `run_item`.
- **Streams.** Each yielded mapping is one step, holding one output per stream
  that published at that step. Streams separate outputs with different grids or
  cadences, such as DLESyM's atmosphere and ocean.
- **Schemas.** `output_coords(horizon)` gives each stream's coordinates for one
  item, including `lead_time` but excluding `time` and `ensemble`, which
  `Pipeline` adds. Runs may stop early; the schema is an upper bound.
- **Requests.** `data_requests(item)` describes every `fetch_data` call as a
  `DataRequest` without fetching, for predownload. `()` means no inputs; `None` means not knowable
  upfront, which disables complete predownload but not execution.
- **Members.** `WorkItem.member_ids` carries a member group. A runner with
  `supports_member_batching = False` gets groups of one. Runners seed stochastic
  models with `set_rng` before each rollout, from a seed derived from the item's
  time and member ID; a value sent into the model iterator is already forcing.

## Pipeline (planned)

Upstreamed from `recipes/eval/src/pipelines/base.py` with typed constructor
arguments instead of a Hydra config. The loop is today's, with `run_item` moved
onto the runner:

```python
class Pipeline:
    def __init__(self, runner, *, output=None, scorer=None, output_variables=None,
                 regridder=None, progress=None, member_batch=1): ...

    @classmethod
    def from_model(cls, prognostic, source, *, diagnostics=None, **kwargs):
        return cls(PrognosticRunner(prognostic, source, diagnostics=diagnostics), **kwargs)

    def run(self, items):
        self.output.prepare(self.runner, items)       # via runner.output_coords
        for item in self.distribute(items):           # this rank's share
            if self.progress.is_done(item):
                continue
            for step in self.runner.run_item(item):
                for stream, data in step.items():      # filter, regrid, score, write
                    self._emit(item, stream, data)
            self.output.flush()
            self.progress.mark_done(item)
```

Resume is item-granular, as in every recipe today. `OutputManager` needs one
store per stream; that is the main IO change.

## Built-in runners

**`PrognosticRunner(prognostic, source=None, *, forcing=None, diagnostics=None)`**
in `run`. Sources default to the model's `default_sources()`. The runner drives
`rollout_iterator`, which yields forecasts only, so it publishes the initial
condition itself as the first step, matching `run.deterministic`. Declared
forcing is fetched and sent at every step, and appears in `data_requests`. The
prognostic stream is `forecast`; each diagnostic adds a stream under the name the
caller gives it. Member batching is not yet supported.

**`DiagnosticRunner`** (planned) applies diagnostics directly to source data at
each item's time, replacing the diagnostic-only paths in recipes.

**Coupled runner**, with the coupler. Wraps the impact-modeling `Driver`. Per
item it builds a driver for the item's window, fetches each component's initial
condition from bound sources, calls `initialize`, and yields each step of
`Driver.steps()` as component-keyed streams. The Driver alone sequences
components and resolves exchanges. This needs a few Driver changes:

- a per-item clock, by factory or by retargeting on `reset`;
- initial conditions fetched from sources rather than passed as tensors;
- no ensemble support at first, so `supports_member_batching = False`.

Its streams are named after components.

## Not part of this boundary

Explicit model state, `initialize`/`step` protocols, components, ports, and
schedules belong to the model contract and the coupler. Mid-item checkpointing
can become an optional runner capability once models expose explicit state.

## Phasing

1. **Done.** `Runner`, `WorkItem`, `DataRequest`, and `PrognosticRunner`.
2. **Pipeline.** Upstream `Pipeline`, work distribution, `OutputManager`, and
   progress markers. Replace the eval `forecast`, `diagnostic`, and `dlesym`
   pipelines with `PrognosticRunner` configurations.
3. **Coupled runner.** Express StormScope GOES and MRMS as Driver components and
   replace the eval `stormscope` pipeline.
4. **Workflows.** Make `run.deterministic`, `run.diagnostic`, and `run.ensemble`
   thin wrappers over `Pipeline.from_model`.

## Open decisions

1. **Dependent work items**, such as warm-start cycling: support in v1 or
   declare out of scope?
2. **Ensembles:** where perturbations are configured, likely a runner constructor
   argument, and member batching for coupled runs.
3. **Async fetching:** if needed, change runner and `Pipeline` loop together.

## Integration checks

1. `PrognosticRunner` matches `run.deterministic` — `test/run/test_runner.py`.
2. A hand-written runner stops early and returns `None` data requests —
   `dev/examples/06_runner.py`.
3. `Pipeline` drives `PrognosticRunner` and a coupled runner identically, writing one
   store per stream.
4. Split DLESyM through the coupled runner reproduces the fused model.
5. The StormScope coupled runner matches the eval `stormscope` pipeline.
6. Resume survives a world-size change.
