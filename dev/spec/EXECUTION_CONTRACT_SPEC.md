# Execution Contracts (draft)

Goal: one user-facing execution API, `Pipeline`, for single-model and coupled
runs, without pulling coupling mechanics into it.

## Shape

- `Pipeline` supervises **work**: distribution across ranks, resume, output
  filtering and regridding, scoring, and IO. One concrete class, never
  subclassed; variation points are constructor arguments.
- A `Runner` executes **one work item**: it fetches inputs, steps models, and
  yields outputs. Built-in runners cover the common cases; anything else is a
  small hand-written class.

Rule of thumb: "what does model B receive from model A, and when?" belongs to the
runner, or the coupler behind it. "Which rank runs this item, where does its
output go, and what happens if it crashes?" belongs to `Pipeline`.

```python
model = SFNO.load_model(SFNO.load_default_package())
items = work_items(times, horizon=np.timedelta64(240, "h"))
output = OutputManager(lambda stream, template: ZarrBackend(f"out/{stream}.zarr"))
progress = ProgressStore("out/.progress")
Pipeline.from_model(model, GFS(), output=output, progress=progress).run(items)

# Coupled (illustrative): the coupler's Driver supplies the runner.
Pipeline(driver.to_runner(ics={...}), output=output).run(items)
```

`earth2studio.run` never imports the coupler; coupled runners live with it.

### Modules

Each module has one owner, so the implementation tracks do not conflict:

| Module | Contents |
| --- | --- |
| `run/runner.py` | `WorkItem`, `DataRequest`, `Runner`, `PrognosticRunner`, `DiagnosticRunner` |
| `run/work.py` | `work_items`, `distribute_work`, `member_seed`, `ProgressStore`, rank helpers |
| `run/output.py` | `OutputManager` |
| `run/pipeline.py` | `Pipeline`, `Scorer` |

`run/pipeline.py` adds the public exports to `earth2studio.run`; the other modules
are imported by path until then.

## Work items

A `WorkItem` is the unit `Pipeline` distributes, runs, and marks done:

```python
@dataclass(frozen=True)
class WorkItem:
    time: np.datetime64          # reference time: initialization, or first valid time
    horizon: np.timedelta64      # last lead time to produce; 0 runs one step
    member_ids: tuple[int, ...] = ()  # ensemble members; empty when deterministic
```

Items are immutable and hashable. Callers build them, usually with `work_items`;
runners only read them. The horizon must be a multiple of the runner's step.

```python
def work_items(
    times: Sequence[np.datetime64],
    horizon: np.timedelta64,
    members: int | Sequence[int] | None = None,
    member_batch: int = 1,
) -> list[WorkItem]: ...
```

`work_items` orders items time-major. With `members`, it splits each time's
members into consecutive groups of `member_batch`, so a group never spans two times.
`members=None` gives deterministic items.

An item list is valid when it has no duplicate `(time, horizon, member)` units and
either every item has members or none does.

## Runner

`earth2studio.run.runner`:

```python
class Runner(Protocol):
    supports_member_batching: bool
    def to(self, device) -> Runner
    def output_coords(self, horizon) -> Mapping[str, CoordinateSystem]
    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None
    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]
```

- **Built once, run per item.** A runner binds sources, never data; initial
  conditions and forcing are fetched inside `run_item`.
- **Streams.** A stream is a named sequence of outputs sharing one schema:
  variables, grid, and lead-time pattern. `Pipeline` writes each to its own
  store. Each yielded mapping is one step, holding one output per stream that
  published at it. Outputs with different grids or cadences, such as a coupled
  atmosphere and ocean, need separate streams. Names follow the producing model
  or component and stay stable across runs.
- **Requests.** `data_requests(item)` describes every `fetch_data` call as a
  `DataRequest` without fetching, for predownload. `()` means no inputs; `None`
  means not knowable upfront, which disables complete predownload but not
  execution.
- **Members.** `WorkItem.member_ids` carries a member group. A runner with
  `supports_member_batching = False` gets groups of at most one; `Pipeline` raises
  `ValueError` before running otherwise. A value sent into the model iterator is
  already forcing.

### Stream schemas

`output_coords(horizon)` returns one planning signature per stream: an
allocation-free DataArray built with `coord_array`, as models return from
`output_coords` (see [MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)).

- The fixed dimensions are `lead_time`, holding every lead time an item with this
  horizon can publish, followed by the field dimensions, such as `variable` and the
  grid's dimensions.
- Grid metadata (`earth2studio_grid_id`, CRS attributes), auxiliary coordinates such
  as curvilinear latitude and longitude, dtype and user attributes are those of the
  fields the stream yields.
- `time` and `ensemble` are never fixed dimensions; they may appear only in the
  dynamic prefix, as in a model signature's `(batch, time)`.
- Runs may stop early; the schema is an upper bound.

`Pipeline` concretizes each schema for a store with
`output_template(signature, leading={"ensemble": members, "time": times})`, which
replaces the dynamic prefix, or prepends when there is none (see
[IO_SPEC.md](IO_SPEC.md)). Grid metadata therefore reaches `add_array` without any
coordinate assembly.

### Yielded outputs

Each output a runner yields for stream `s`:

- has dimensions `time`, with the single label `item.time`, then the fixed
  dimensions of `s`'s schema in order;
- has `lead_time` labels drawn from the schema's `lead_time`, and labels equal to
  the schema's along every other fixed dimension;
- carries no other dynamic dimension, except a leading `ensemble` labeled
  `item.member_ids` when the runner batches members.

A diagnostic that drops `lead_time` has it restored by its runner before yielding.

### Seeding

A runner seeds every stochastic model (`stochastic = True`) once per member,
immediately before that member's rollout or first call:

```python
model.set_rng(member_seed(self.seed, item.time, member_id), reset=True)
```

A deterministic item seeds as member `0`. `seed` is a runner constructor argument,
by default `0`, and runners that seed expose it as `runner.seed` so it can enter
the progress manifest. Runners never touch global RNG state, which the model
contract keeps unperturbed (`P14`, `D10`).

```python
def member_seed(base: int, time: np.datetime64, member_id: int) -> int: ...
```

`member_seed` is the eval recipe's derivation, unchanged so existing eval results
reproduce: FNV-1a 64 over `struct.pack(">qqq", base, seconds, member_id)`, where
`seconds` is `time` as integer seconds since the epoch, reduced modulo `2**63`. It
depends only on its arguments, never on batching, world size, rank or process.

## Pipeline

`earth2studio.run.pipeline`, upstreamed from
`recipes/eval/src/pipelines/base.py` with typed constructor arguments instead of a
Hydra config, and `run_item` moved onto the runner.

```python
StreamSetting: TypeAlias = T | Mapping[str, T | None] | None

class Pipeline:
    def __init__(
        self,
        runner: Runner,
        *,
        output: OutputManager | None = None,
        scorer: Scorer | None = None,
        output_variables: StreamSetting[Sequence[str]] = None,
        regridder: StreamSetting[Regridder] = None,
        progress: ProgressStore | None = None,
        device: torch.device | str | None = None,
    ) -> None: ...

    @classmethod
    def from_model(
        cls, prognostic, source=None, *, forcing=None, diagnostics=None, seed=0,
        **kwargs,
    ) -> Pipeline:
        runner = PrognosticRunner(prognostic, source, forcing, diagnostics, seed=seed)
        return cls(runner, **kwargs)

    def run(self, items: Sequence[WorkItem]) -> None: ...
```

At least one of `output` and `scorer` is required. `device` defaults to
`cuda:{LOCAL_RANK}` when CUDA is available, otherwise CPU.

### Per-stream settings

`output_variables` and `regridder` apply per stream:

- a mapping keys settings by stream name; streams it omits, or maps to `None`, are
  unfiltered or not regridded;
- a single value is shorthand for `{"forecast": value}`, the prognostic stream of
  `PrognosticRunner`;
- a key that names no stream raises `ValueError` when `run` starts.

Filtering precedes regridding, so unwanted variables are never regridded. A
regridder's `source_grid` must equal the grid of its stream's schema, and the
stored schema takes its `target_grid` as `R4` describes (see
[REGRID_SPEC.md](REGRID_SPEC.md)); a mismatch raises `ValueError` before any item
runs.

### Run loop

```python
def run(self, items):
    rank, world_size = rank_info()
    items = self._validate(items)                  # identical on every rank
    self.runner.to(self.device)
    horizon = max(item.horizon for item in items)
    schemas = self._stored_schemas(self.runner.output_coords(horizon))
    if self.progress is not None:
        self.progress.open(self._manifest(schemas))
    if self.output is not None:
        self.output.prepare(schemas, items)       # stores span all items
    remaining = self._remaining(items)            # rank 0 lists, then broadcasts
    error = None
    try:
        for item in distribute_work(remaining, rank, world_size):
            self.scorer and self.scorer.begin_item(item)
            for step in self.runner.run_item(item):
                for stream, x in step.items():     # filter, regrid, score, write
                    self._emit(item, stream, x)
            self.output and self.output.flush()
            self.scorer and self.scorer.finish_item(item)
            self.progress and self.progress.mark_done(item)
    except Exception as e:
        error = e
    self.output and self.output.close()           # collective
    raise_if_any_failed(error)                    # collective
```

Without a `ProgressStore`, `remaining` is every item. `_emit` filters the variables,
applies the stream's regridder, passes the result to `scorer.update`, adds an
`ensemble` dimension labeled `item.member_ids` when the store has one and `x` does
not, then writes to the stream's backend.

- **Stores span all items.** Schemas and stores are planned from the full item list,
  not the remaining items, so a resumed run, on any world size, writes into the
  same stores. The `lead_time` extent comes from the largest horizon; shorter
  items leave fill values.
- **Identical items.** Every rank passes the same items in the same order. `run`
  compares a digest of the list across ranks and raises `ValueError` on a mismatch.
- **Durability order.** An item is marked done only after its outputs are flushed
  and its scorer has finished it. A crash leaves the item unmarked, and a resumed
  run repeats it in full; label-addressed writes make the repeat idempotent.
- **Idle ranks.** A rank with no items still joins every collective: `open`,
  `prepare` and `close`.
- **Failures.** A rank that fails stops running items but still closes its output
  and joins a final reduction of a failure flag across ranks. The failing rank
  re-raises its error and the others raise `RuntimeError`, so no rank waits
  forever at a barrier. This holds with or without an `OutputManager`.

### Scorer

```python
class Scorer(Protocol):
    def begin_item(self, item: WorkItem) -> None: ...
    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None: ...
    def finish_item(self, item: WorkItem) -> None: ...
```

A minimal hook for online scoring. `update` receives what would be written: after
filtering and regridding, before the `ensemble` dimension is added. Metrics,
verification data and any communication between ranks, such as gathering an
ensemble group, stay in the scorer, which lives with the recipe. A scorer that
needs durable state persists it in `finish_item`.

## OutputManager

`earth2studio.run.output`. Owns one IO backend per stream, and the distributed
lifecycle around them.

```python
BackendFactory: TypeAlias = Callable[[str, xr.DataArray], IOBackend | None]

class OutputManager:
    def __init__(
        self, backends: Mapping[str, IOBackend | None] | BackendFactory
    ) -> None: ...
    @property
    def backends(self) -> Mapping[str, IOBackend]: ...
    def prepare(
        self, schemas: Mapping[str, CoordinateSystem], items: Sequence[WorkItem]
    ) -> None: ...
    def write(self, stream: str, x: xr.DataArray) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

- **Backends.** A factory receives the stream name and its concrete template, so
  a backend can derive its layout from it, as `AsyncZarrBackend` needs. A mapping
  supplies backends already built. Either may give `None` for a stream, which is
  then not written. A stream with neither a backend nor an explicit `None` raises
  `ValueError` in `prepare`, so outputs are never dropped silently.
- **`prepare`** concretizes each schema with `output_template`, using the sorted
  unique item times and, when items carry members, the sorted union of their
  `member_ids`. Rank 0 builds its backends and calls `add_array`; after a barrier
  the other ranks build theirs and call `add_array` again, which validates the
  existing store and is otherwise a no-op (`I4`). A factory is therefore called on
  every rank, rank 0 first. Overwriting an existing store is the backend's choice,
  made at construction.
- **`write`** forwards to the stream's backend; a stream without one is skipped.
- **`flush`** flushes every backend (`I9`).
- **`close`** is collective. Every rank closes its backends (`I11`), then barrier,
  including after a local failure. It is idempotent. Closed backends stay readable
  through `backends`, which is how `run.*` wrappers return results.

### Shard ownership

Every write targets the positions of one item: its `time`, its `member_ids` (when
the store has `ensemble`), and any `lead_time` labels. Two ranks race when one
storage chunk holds positions of items they own. `time` and `ensemble` are the
distributed dimensions.

In a distributed run (`world_size > 1`), `prepare` checks every backend:

- A backend with `write_chunks()` returning a mapping is a shared store. Along
  `time`, each chunk must hold one label. Along `ensemble`, each chunk must lie
  within one item's member group, for every group. Other dimensions are
  unconstrained, since an item writes all their chunks itself.
- A backend with `write_chunks()` returning `None` keeps process-local storage, as
  `XarrayBackend` does. Each rank's store then holds only its own items.
- A backend without `write_chunks` raises `ValueError`, because its safety is
  unknown. `NetCDF4Backend` is rejected this way.

```python
def write_chunks(self) -> Mapping[str, int] | None: ...
```

`write_chunks` is an optional backend method outside the IO protocol. It gives, per
stored dimension, the size of the smallest unit two writers cannot share: the
shard size for sharded stores, otherwise the chunk size. It is valid after
`add_array`. `ZarrBackend` (and so `IceChunkBackend`) and `AsyncZarrBackend`
implement it as part of the `OutputManager` work. Single-process runs skip the
check.

## Progress

`earth2studio.run.work.ProgressStore` records which items are done, so a run can
resume.

```python
class ProgressStore:
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        tags: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None: ...
    def open(self, manifest: Mapping[str, Any]) -> None: ...
    def remaining(self, items: Sequence[WorkItem]) -> list[WorkItem]: ...
    def is_done(self, item: WorkItem) -> bool: ...
    def mark_done(self, item: WorkItem) -> None: ...
    def clear(self) -> None: ...
```

- **Always written.** When a `Pipeline` has a `ProgressStore`, it marks every
  finished item, whether or not the run is itself a resume.
- **Keyed on item identity.** A marker records one unit: `(time, horizon,
  member_id)`, with no member for a deterministic item. `mark_done` writes one
  marker per member of the item, and an item is done when all its units are. Keys
  derive from the values alone, never from item order, rank, world size or member
  grouping, so a resumed run may change any of those. A partly done group runs
  again in full.
- **Atomic markers.** Each marker is one file under `path`, written to a temporary
  name and renamed, so a crash never leaves a marker without its data. `path`
  must be visible to every rank.
- **Listed once.** Only rank 0 calls `remaining`; `Pipeline` broadcasts the result,
  so ranks always agree on the work left.
- **Manifest.** `open` is collective. Rank 0 writes the manifest when none exists,
  and raises `ValueError` on every rank when one exists and differs, naming the
  differing keys. `overwrite=True` clears markers and the manifest first.

`Pipeline` builds the manifest; it is JSON:

| Key | Value |
| --- | --- |
| `version` | Manifest format version, `1` |
| `streams` | Per stream: the stored schema's dims, non-spatial labels, grid ID and dtype |
| `seed` | `runner.seed`, or `null` when the runner has none |
| `tags` | The store's `tags`, such as model name and checkpoint version |

Model identity cannot be detected reliably, so callers put it in `tags`; a change
in variables, grid, lead-time extent or seed is caught without them.

Progress and stores are cleared separately. Overwriting a store without clearing
its progress skips items whose data is gone; `overwrite` on both is the fresh-run
idiom.

## Distribution

`earth2studio.run.work`:

```python
def distribute_work(items: Sequence[T], rank: int, world_size: int) -> list[T]: ...
def rank_info() -> tuple[int, int]: ...  # (rank, world_size); (0, 1) if uninitialized
def rank0_first(fn: Callable[[], T]) -> T: ...  # rank 0 runs fn, barrier, then the rest
def broadcast(obj: T | None) -> T: ...  # rank 0's object on every rank
def barrier() -> None: ...
def raise_if_any_failed(error: Exception | None) -> None: ...  # all ranks agree
```

- **Split.** `distribute_work` gives each rank one contiguous, balanced slice: the
  first `len(items) % world_size` ranks get one extra item. An idle rank gets `[]`.
  Item groups are never split, since a group is one item.
- **Process group.** The helpers use `torch.distributed` directly, and are no-ops
  when it is not initialized. Core never initializes it. Launch with `torchrun`
  and call `torch.distributed.init_process_group`, or physicsnemo's
  `DistributedManager.initialize()`, which initializes the same process group;
  physicsnemo is never required.
- **Collectives.** `broadcast` uses `broadcast_object_list`; the final failure flag
  uses `all_reduce`. With the NCCL backend, tensors move to the rank's
  CUDA device.

## Built-in runners

**`PrognosticRunner(prognostic, source=None, forcing=None, diagnostics=None, seed=0)`**
in `run`. Sources default to the model's `default_sources()`. The runner drives
`create_iterator`, which yields forecasts only, so it publishes the initial
condition itself as the first step, matching `run.deterministic`. Declared forcing
is fetched initially and its time-varying slots are sent at every step; all of it
appears in `data_requests`. The prognostic stream is `forecast`; each diagnostic
adds a stream under the name the caller gives it. Member batching is not yet
supported.

**`DiagnosticRunner(diagnostics, source=None, step=None, seed=0)`** in `run`
applies diagnostics directly to source data, as in downscaling an analysis. Step
`k` fetches each diagnostic's inputs at the item's time plus `k * step`, so a
`ForecastSource` supplies lead times and a `DataSource` supplies later valid times.
Without `step`, an item is one step at lead time zero. Each diagnostic's stream is
named by the caller, and sources default to each diagnostic's `default_sources()`.

**Coupled runner**, with the coupler. Wraps the impact-modeling `Driver`. Per
item it builds a driver for the item's window, fetches each component's initial
condition from bound sources, calls `initialize`, and yields each step of
`Driver.steps()` as component-keyed streams. The Driver alone sequences
components and resolves exchanges. This needs a few Driver changes:

- a per-item clock, by factory or by retargeting on `reset`;
- initial conditions fetched from sources rather than passed as tensors;
- no ensemble support at first, so `supports_member_batching = False`.

Its streams are named after components.

## Rules

| Rule | Requirement |
| --- | --- |
| `X1` | `output_coords` returns one signature per stream; `time`/`ensemble` are never fixed dims |
| `X2` | Stream signatures carry the yielded fields' grid metadata and auxiliary coordinates |
| `X3` | Yielded outputs have dims `time` (one label, `item.time`), then the schema's fixed dims |
| `X4` | Yielded `lead_time` labels lie within the schema; other fixed labels equal it |
| `X5` | A runner seeds each stochastic model with `member_seed` and `reset=True` per member |
| `X6` | `member_seed` depends only on base seed, time and member ID |
| `X7` | Every rank passes identical items; a mismatch raises `ValueError` before running |
| `X8` | Stores are planned from all items, so resume on any world size reuses them |
| `X9` | `distribute_work` gives balanced contiguous slices; idle ranks still join collectives |
| `X10` | Per-stream settings accept a mapping or a `forecast` shorthand; unknown keys raise |
| `X11` | A regridder's `source_grid` must match its stream; mismatches raise before running |
| `X12` | Each stream has a backend or an explicit `None`; otherwise `prepare` raises |
| `X13` | Rank 0 creates stores before other ranks open them; reopening is a no-op (`I4`) |
| `X14` | Distributed shared stores chunk `time` by 1 and `ensemble` within member groups |
| `X15` | Distributed runs reject backends without `write_chunks` |
| `X16` | Progress markers are keyed on `(time, horizon, member_id)` and written atomically |
| `X17` | An item is marked done only after its outputs flush and its scorer finishes it |
| `X18` | Rank 0 lists remaining items and broadcasts them |
| `X19` | A manifest mismatch raises `ValueError` unless the progress store overwrites |
| `X20` | A failure on any rank raises on every rank at the end of `run`, never hanging |

Tests live in `test/run/`. Rules other than `X7`, `X9`'s collectives, `X13`, `X18`
and `X20` are checked single-process; those are checked under `torchrun` with two
CPU ranks on gloo (integration check 6).

## Not part of this boundary

Explicit model state, `initialize`/`step` protocols, components, ports, and
schedules belong to the model contract and the coupler. Mid-item checkpointing
can become an optional runner capability built on `(y, state)` (open decision 3).
Perturbation of initial conditions is not yet placed (open decision 2).

## Phasing

1. **Done.** `Runner`, `WorkItem`, `DataRequest`, `PrognosticRunner`, and
   `DiagnosticRunner`.
2. **Pipeline.** In parallel against this spec: `run/work.py`, the runner
   follow-up (DataArray stream schemas, seeding, removing
   `_yields_initial_condition`), and `run/output.py`. Then `run/pipeline.py`.
   Replace the eval `forecast`, `diagnostic`, and `dlesym` pipelines with built-in
   runners; fused DLESyM runs under the single-slot `PrognosticRunner`.
3. **Coupled runner.** Express StormScope GOES and MRMS as Driver components and
   replace the eval `stormscope` pipeline.
4. **Workflows.** Make `run.deterministic`, `run.diagnostic`, and `run.ensemble`
   thin wrappers over `Pipeline.from_model`.

## Resolved decisions

- **Distributed layer:** `torch.distributed` directly; physicsnemo optional.
- **Scoring in core:** the minimal `Scorer` hook above; metrics and rank
  communication stay in recipes.
- **Member grouping** moves from the `Pipeline` constructor (`member_batch`) to
  `work_items`, since groups are part of item identity.

## Open decisions

1. **Dependent work items**, such as warm-start cycling: support in v1 or
   declare out of scope?
2. **Ensembles:** where perturbations are configured, likely a runner constructor
   argument, and member batching for coupled runs.
3. **Mid-item resume:** must `run.*` keep step-level checkpoint resume? If so,
   runners need an optional mid-item checkpoint built on `(y, state)`. Blocks
   phase 4 only.
4. **Async fetching:** if needed, change runner and `Pipeline` loop together.

## Integration checks

1. `PrognosticRunner` matches `run.deterministic` — `test/run/test_runner.py`.
2. A hand-written runner stops early and returns `None` data requests —
   `dev/examples/06_runner.py`.
3. `Pipeline` drives `PrognosticRunner` and `DiagnosticRunner` identically,
   writing one store per stream; later, a coupled runner too.
4. A forced model (StormCast-shaped mock) and fused DLESyM run through
   `PrognosticRunner` unchanged.
5. Split DLESyM through the coupled runner reproduces the fused model.
6. Resume survives a world-size change — `torchrun`, two CPU ranks, gloo.
7. The StormScope coupled runner matches the eval `stormscope` pipeline.
