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
output = {"forecast": ZarrBackend("out/forecast.zarr")}
progress = ProgressStore("out/.progress")
Pipeline.from_model(model, GFS(), output=output, progress=progress).run(items)

# Coupled (illustrative): the coupler's Driver supplies the runner.
Pipeline(driver.to_runner(ics={...}), output=output).run(items)
```

`earth2studio.run` never imports the coupler; coupled runners live with it.

### Modules

One owner per module, so implementation tracks do not conflict:

| Module | Contents |
| --- | --- |
| `run/runner.py` | `WorkItem`, `DataRequest`, `Runner`, `PrognosticRunner`, `DiagnosticRunner` |
| `run/work.py` | `work_items`, `distribute_work`, `member_seed`, `ProgressStore`, rank helpers |
| `run/output.py` | `OutputManager` (internal) |
| `run/pipeline.py` | `Pipeline`, `Scorer` |

`run/pipeline.py` adds the public exports to `earth2studio.run`; until then, import
the other modules by path.

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

Items are ordered time-major. Each time's members are split into consecutive
groups of `member_batch`, so no group spans two times; `members=None` gives
deterministic items. A valid item list repeats no `(time, horizon, member)` unit,
and either every item has members or none does.

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
- **Members.** `WorkItem.member_ids` carries a member group. Without
`supports_member_batching`, a runner gets at most one member per item, or
`Pipeline` raises `ValueError` before running. A value sent into the model
iterator is already forcing.

### Stream schemas

`output_coords(horizon)` returns one planning signature per stream, an
allocation-free `coord_array` like a model's `output_coords` (see
[MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)):

- Fixed dimensions are `lead_time`, covering every lead time the horizon allows,
then the field dimensions (`variable` and the grid's).
- Grid metadata, auxiliary coordinates (such as curvilinear latitude and longitude),
dtype and attributes match the fields the stream yields.
- `time` and `ensemble` may appear only in the dynamic prefix, as in a model's
`(batch, time)`, never as fixed dimensions.
- The schema is an upper bound; runs may stop early.

`Pipeline` makes each store's template with
`output_template(signature, leading={"ensemble": members, "time": times})`, which
replaces the dynamic prefix or prepends to a signature without one (see
[IO_SPEC.md](IO_SPEC.md)). Grid metadata reaches `add_array` with no coordinate
assembly.

### Yielded outputs

Each output yielded for a stream:

- has dimensions `time`, labeled `item.time` alone, then the schema's fixed
dimensions in order;
- takes its `lead_time` labels from the schema's, and matches the schema's labels
on every other fixed dimension;
- has no other dynamic dimension, except a leading `ensemble` labeled
`item.member_ids` when the runner batches members.

### Seeding

Before each member's rollout or first call, a runner seeds every stochastic model:

```python
model.set_rng(member_seed(self.seed, item.time, member_id), reset=True)
```

Deterministic items seed as member `0`. `seed` is a runner constructor argument,
by default `0`; seeding runners expose it as `runner.seed` for the progress
manifest. Runners never touch global RNG state (`P14`, `D10`).

```python
def member_seed(base: int, time: np.datetime64, member_id: int) -> int: ...
```

`member_seed` keeps the eval recipe's derivation, so eval results reproduce:
FNV-1a 64 over `struct.pack(">qqq", base, seconds, member_id)`, with `time` in
integer seconds since the epoch, modulo `2**63`. It depends only on its arguments.

## Pipeline

`earth2studio.run.pipeline`, upstreamed from `recipes/eval/src/pipelines/base.py`
with typed arguments in place of a Hydra config and `run_item` moved onto the
runner.

```python
class Pipeline:
    def __init__(
        self,
        runner: Runner,
        *,
        output: Mapping[str, IOBackend] | None = None,
        scorer: Scorer | None = None,
        output_variables: Mapping[str, Sequence[str]] | None = None,
        regridder: Mapping[str, Regridder] | None = None,
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

`output` or `scorer` is required. `device` defaults to `cuda:{LOCAL_RANK}` when
CUDA is available, otherwise CPU.

### Per-stream settings

`output`, `output_variables` and `regridder` map stream names to settings. A stream
left out keeps all its variables, keeps its grid, or is not written, respectively;
`run` logs unwritten streams once. A key naming no stream raises `ValueError` when
`run` starts, catching typos such as `"forcast"`.

Variables are filtered before regridding. A regridder's `source_grid` must equal its
stream's grid, and the stored grid becomes `target_grid` (`R4`, see
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
    outputs = OutputManager(self.output or {})
    outputs.prepare(schemas, items)               # stores span all items
    remaining = self._remaining(items)            # rank 0 lists, then broadcasts
    error = None
    try:
        for item in distribute_work(remaining, rank, world_size):
            self.scorer and self.scorer.begin_item(item)
            for step in self.runner.run_item(item):
                for stream, x in step.items():     # filter, regrid, score, write
                    self._emit(item, stream, x)
            outputs.flush()
            self.scorer and self.scorer.finish_item(item)
            self.progress and self.progress.mark_done(item)
    except Exception as e:
        error = e
    outputs.close()                               # collective
    raise_if_any_failed(error)                    # collective
```

Without a `ProgressStore`, every item remains. `_emit` filters variables, regrids,
calls `scorer.update`, adds `ensemble` (labeled `item.member_ids`) if the store has
it and `x` lacks it, then writes.

- **Stores span all items.** Schemas and stores come from the full item list, not
the remaining items, so a resume at any world size reuses them. The `lead_time`
extent follows the largest horizon; shorter items leave fill values.
- **Identical items.** Every rank passes the same items in the same order; `run`
compares digests and raises `ValueError` on a mismatch.
- **Durability.** An item is marked done only after its outputs flush and its
scorer finishes it. An interrupted item reruns in full; label-addressed writes make
that idempotent.
- **Idle ranks** still join every collective: `open`, `prepare` and `close`.
- **Failures.** A failing rank stops running items, closes its output and joins a
final failure-flag reduction. It re-raises its error and the other ranks raise
`RuntimeError`, so none hangs at a barrier, with or without `output`.

### Scorer

```python
class Scorer(Protocol):
    def begin_item(self, item: WorkItem) -> None: ...
    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None: ...
    def finish_item(self, item: WorkItem) -> None: ...
```

A minimal online-scoring hook. `update` receives what would be written, after
filtering and regridding and before `ensemble` is added. Metrics, verification
data and cross-rank communication, such as gathering an ensemble group, stay in the
scorer, implemented by recipes or externally. A scorer persists any durable state
in `finish_item`.

## Output

`earth2studio.run.output.OutputManager` is internal; `Pipeline` wraps its `output`
mapping in one.

```python
class OutputManager:
    def __init__(self, backends: Mapping[str, IOBackend]) -> None: ...
    def prepare(
        self, schemas: Mapping[str, CoordinateSystem], items: Sequence[WorkItem]
    ) -> None: ...
    def write(self, stream: str, x: xr.DataArray) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

- `prepare` builds each template from the sorted unique item times and, for
ensembles, the sorted union of `member_ids`. Rank 0 calls `add_array`; after a
barrier, the other ranks call it too, which validates the store and otherwise does
nothing (`I4`).
- `write` forwards to the stream's backend, skipping streams without one.
- `flush` flushes every backend (`I9`).
- `close` is collective and idempotent: every rank closes its backends (`I11`),
even after a local failure, then barriers. Closed backends stay readable, so
`run.*` wrappers can still return the caller's backend.

Callers build backends on every rank before `run`, creating or deleting storage on
rank 0 only:

```python
rank, _ = rank_info()
output = rank0_first(
    lambda: {"forecast": ZarrBackend(path, backend_kwargs={"overwrite": rank == 0})}
)
```

`AsyncZarrBackend` needs its parallel dimensions' labels at construction, so its
caller supplies the item times. Deriving them from the first `add_array` template,
an open IO question, would remove that step.

### Shard ownership

An item writes one `time`, its `member_ids` along `ensemble`, and any `lead_time`
labels. Ranks race when a storage chunk holds positions of items owned by different
ranks, so `time` and `ensemble` are the distributed dimensions; other dimensions are
safe, since one item writes all their chunks.

With `world_size > 1`, `prepare` checks known backends through their existing
attributes, with no IO protocol or backend changes:

| Backend | Check |
| --- | --- |
| `ZarrBackend`, `IceChunkBackend` | `time` chunks of 1; `ensemble` chunks within member groups |
| `AsyncZarrBackend` | `time`/`ensemble` in `parallel_coords`; no shard spans items |
| `XarrayBackend`, `KVBackend` | Process-local: each rank holds only its own items |
| Others, including `NetCDF4Backend` | Unknown: shared storage may be corrupted |

A failed check raises `ValueError`; the last two rows warn once instead. Default Zarr
chunking passes. Single-process runs skip the check.

## Progress

`earth2studio.run.work.ProgressStore` records finished items, so runs can resume.

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

- **Always written.** Every finished item is recorded, not only in resumed runs.
- **Unit keys.** Progress records `(time, horizon, member_id)` units, with no member
for deterministic items; an item is done when all its units are. Keys ignore item
order, rank, world size and member grouping, so a resume may change any of them.
A partly done group reruns in full.
- **Per-rank logs.** Each rank appends to its own log under `path`, named by run
attempt and rank, so no file has two writers. `mark_done` appends one line per item
(time, horizon, members) and fsyncs; a torn last line is ignored, so no record
outlives its data. `path` must be shared by all ranks.
- **Bounded file count.** In `open`, before any writes, rank 0 compacts existing
logs into one `done` log: write a temporary file, rename it, delete the merged logs.
The directory then holds the manifest, the compacted log and one log per rank,
however large the campaign. An interrupted compaction only duplicates lines.
- **Listed once.** Only rank 0 calls `remaining`, and `Pipeline` broadcasts the
result.
- **Manifest.** `open` is collective. Rank 0 writes the manifest if none exists; if
one differs, every rank raises `ValueError` naming the differing keys.
`overwrite=True` clears the logs and manifest first.

`Pipeline` builds the manifest as JSON:

| Key | Value |
| --- | --- |
| `version` | Manifest format version, `1` |
| `streams` | Per stream: the stored schema's dims, non-spatial labels, grid ID and dtype |
| `seed` | `runner.seed`, or `null` |
| `tags` | The store's `tags`, such as model name and checkpoint version |

Changes in variables, grid, lead-time extent or seed are caught automatically;
model identity is not detectable, so callers record it in `tags`.

Progress and stores are cleared independently: overwriting a store but not its
progress skips items whose data is gone. A fresh run overwrites both.

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

- **Split.** Each rank gets one contiguous, balanced slice; the first
`len(items) % world_size` ranks get one extra item, and idle ranks get `[]`. A
member group is one item, so it is never split.
- **Process group.** Helpers call `torch.distributed` directly and are no-ops when
it is uninitialized. Core never initializes it: launch with `torchrun` and call
`init_process_group`, or physicsnemo's `DistributedManager.initialize()`, which
sets up the same group. physicsnemo is optional.
- **Collectives.** `broadcast` uses `broadcast_object_list` and the failure flag uses
`all_reduce`. Under NCCL, tensors go to the rank's CUDA device.

## Built-in runners

`PrognosticRunner(prognostic, source=None, forcing=None, diagnostics=None, seed=0)`
in `run`. Sources default to the model's `default_sources()`. The runner drives
`create_iterator`, which yields forecasts only, so it publishes the initial
condition itself as the first step, matching `run.deterministic`. Declared forcing
is fetched initially and its time-varying slots are sent at every step; all of it
appears in `data_requests`. The prognostic stream is `forecast`; each diagnostic
adds a stream under the name the caller gives it. Member batching is not yet
supported.

`DiagnosticRunner(diagnostics, source=None, step=None, seed=0)` in `run`
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
| `X1` | `output_coords` gives one signature per stream; `time`/`ensemble` are never fixed |
| `X2` | Stream signatures carry the yielded fields' grid metadata and auxiliary coordinates |
| `X3` | Outputs have dims `time` (`item.time` only), then the schema's fixed dims |
| `X4` | Output `lead_time` labels lie within the schema's; other fixed labels equal it |
| `X5` | Runners seed each stochastic model per member: `member_seed`, `reset=True` |
| `X6` | `member_seed` depends only on base seed, time and member ID |
| `X7` | Every rank passes identical items, or `run` raises `ValueError` first |
| `X8` | Stores are planned from all items, so resume at any world size reuses them |
| `X9` | `distribute_work` slices evenly and contiguously; idle ranks join collectives |
| `X10` | Per-stream mappings raise `ValueError` for unknown stream names |
| `X11` | A regridder's `source_grid` must match its stream, or `run` raises first |
| `X12` | Streams absent from `output` are not written; `run` logs them once |
| `X13` | Rank 0 creates stores before other ranks open them; reopening is a no-op (`I4`) |
| `X14` | Distributed Zarr stores share no chunk or shard between items |
| `X15` | Distributed runs warn once per backend they cannot check |
| `X16` | Progress keys `(time, horizon, member_id)` in per-rank logs; torn lines ignored |
| `X17` | Items are marked done only after outputs flush and the scorer finishes |
| `X18` | Rank 0 lists remaining items and broadcasts them |
| `X19` | A manifest mismatch raises `ValueError` unless the progress store overwrites |
| `X20` | A failure on any rank raises on every rank at the end of `run`, never hanging |

Tests live in `test/run/`. `X7`, `X9`'s collectives, `X13`, `X18` and `X20` run
under `torchrun` with two CPU ranks on gloo (integration check 6); the rest run
single-process.

## Not part of this boundary

Explicit model state, `initialize`/`step` protocols, components, ports, and
schedules belong to the model contract and the coupler. Mid-item checkpointing
could become an optional runner capability built on `(y, state)` (open decision 3).
Where initial-condition perturbation lives is undecided (open decision 2).

## Phasing

1. **Done.** `Runner`, `WorkItem`, `DataRequest`, `PrognosticRunner`, and
`DiagnosticRunner`.
2. **Pipeline.** In parallel against this spec: `run/work.py`, the runner
follow-up (DataArray stream schemas, seeding, removing
`_yields_initial_condition`), and `run/output.py`; then `run/pipeline.py`. Replace
the eval `forecast`, `diagnostic` and `dlesym` pipelines with built-in runners;
fused DLESyM runs under the single-slot `PrognosticRunner`.
3. **Coupled runner.** Express StormScope GOES and MRMS as Driver components and
replace the eval `stormscope` pipeline.
4. **Workflows.** Make `run.deterministic`, `run.diagnostic`, and `run.ensemble`
thin wrappers over `Pipeline.from_model`.

## Resolved decisions

- **Distributed layer:** `torch.distributed` directly; physicsnemo optional.
- **Scoring:** core has only the `Scorer` hook; metrics and rank communication stay
in recipes.
- **Member grouping** moves from the `Pipeline` constructor to `work_items`, since
groups are part of item identity.
- **Plain mappings** for per-stream settings and backends: no shorthand types, no
backend factory.
- **Shard safety without IO changes:** known backends are checked through existing
attributes; others get a warning.

## Open decisions

1. **Dependent work items**, such as warm-start cycling: support in v1 or
declare out of scope?
2. **Ensembles:** where perturbations are configured, likely a runner constructor
argument, and member batching for coupled runs.
3. **Mid-item resume:** must `run.*` keep step-level checkpoint resume? If so,
runners need an optional mid-item checkpoint built on `(y, state)`. Blocks phase
4 only.
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
