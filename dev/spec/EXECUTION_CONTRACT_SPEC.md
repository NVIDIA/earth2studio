# Execution Contracts (draft)

Goal: one user-facing execution API, `Pipeline`, for single-model and coupled
runs, without pulling coupling mechanics into it.

## Shape

- `Pipeline` supervises **work**: distribution across ranks, resume, output
  filtering and regridding, scoring, and IO. One concrete class, never
  subclassed; variation points are constructor arguments.
- A `Runner` executes **one work item**: it fetches inputs, steps models, and
  yields outputs. Built-in runners cover the common cases.

"What does model B receive from model A, and when?" belongs to the runner, or the
coupler behind it. "Which rank runs this item, where does its output go, and what
happens if it crashes?" belongs to `Pipeline`.

```python
model = SFNO.load_model(SFNO.load_default_package())
items = work_items(times, final_lead_time=np.timedelta64(240, "h"))
output = {"forecast": ZarrBackend("out/forecast.zarr")}
progress = ProgressStore("out/.progress")
Pipeline.from_model(model, GFS(), output=output, progress=progress).run(items)

# Coupled: CoupledRunner is a Runner; one stream per component.
Pipeline(CoupledRunner(components, connectors), output=output).run(items)
```

`earth2studio.run` never imports the coupler; coupled runners live with it.

| Module | Contents |
| --- | --- |
| `run/runner.py` | `DataRequest`, `Runner`, `PrognosticRunner`, `DiagnosticRunner` |
| `run/work.py` | `WorkItem`, `work_items`, `member_seed`, `ProgressStore`, rank helpers |
| `run/output.py` | `OutputManager` (internal) |
| `run/pipeline.py` | `Pipeline`, `Scorer`; adds the public exports to `earth2studio.run` |

`run/work.py` is a leaf: it imports nothing else from `run`.

## Work items

```python
@dataclass(frozen=True)
class WorkItem:
    time: np.datetime64          # reference time: initialization, or first valid time
    final_lead_time: np.timedelta64      # last lead time to produce; 0 runs one step
    member_ids: tuple[int, ...] = ()  # ensemble members; empty when deterministic

def work_items(
    times: Sequence[np.datetime64],
    final_lead_time: np.timedelta64,
    members: int | Sequence[int] | None = None,
    member_batch: int = 1,
) -> list[WorkItem]: ...
```

Callers build items, usually with `work_items`; runners only read them. The
final lead time must be a multiple of the runner's step.

- `work_items` orders items time-major in the caller's `times` order and splits
  each time's members into groups of `member_batch`. Times and final lead times are
  normalized to seconds, so items hash by instant.
- Every item writes lead times from zero, so a valid list repeats no
  `(time, member)` whatever the final lead times, and either every item has members or
  none does. `work_items` and `OutputManager.prepare` raise `ValueError` otherwise.

## Runner

```python
class Runner(Protocol):
    supports_member_batching: bool
    def to(self, device) -> Runner
    def output_coords(self, final_lead_time) -> Mapping[str, CoordinateSystem]
    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None
    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]
```

- **Built once, run per item.** A runner binds sources, never data; inputs are
  fetched inside `run_item`.
- **Streams.** A stream is a named sequence of outputs with one schema
  (variables, grid, lead-time pattern), written to its own store. Each yielded
  mapping is one step, holding one output per stream that published at it.
  Different grids or cadences need separate streams. Names follow the producing
  model or component.
- **Requests.** `data_requests(item)` describes every `fetch_data` call without
  fetching, for predownload. `()` means no inputs; `None` means not knowable
  upfront.
- **Members.** Without `supports_member_batching`, a runner gets at most one
  member per item, or `Pipeline` raises `ValueError` before running. A runner
  that batches gives every model input of a multi-member item a leading
  `ensemble` dimension labeled `item.member_ids`, repeating slots shared by all
  members, so models receive the batch as an extra dynamic leading dimension.
- **Source members.** Members of one ensemble may read different data, e.g. four
  seeded branches from each IFS ENS member. A built-in runner given
  `source_member`, a function from a work item's member ID to a source member
  label, fetches every slot whose source is an ensemble source (see
  [FETCH_DATA_SPEC.md](FETCH_DATA_SPEC.md#ensemble-members)) for the item's
  members: initial conditions and forcing alike, the distinct labels in one
  request, recorded in `DataRequest.member`. Other slots, and every slot
  without `source_member`, are shared by all members. An item without member
  IDs raises `ValueError`. With one member per item, the fetched member's field
  reaches the model without an `ensemble` dimension; in a batch, members sharing
  a source member share its one fetched field.

### Stream schemas and outputs

`output_coords(final_lead_time)` returns one allocation-free `coord_array` signature per
stream (see [MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)). Its fixed
dimensions are `lead_time`, covering every lead the final lead time allows, then
`variable` and the grid's. `time` and `ensemble` may appear only in the dynamic
prefix. The schema carries the yielded fields' grid metadata, auxiliary
coordinates, dtype and attributes, and is an upper bound: runs may stop early.
`Pipeline` makes each store's template with
`output_template(signature, leading={"ensemble": members, "time": times})` (see
[IO_SPEC.md](IO_SPEC.md)).

Each yielded output has dimensions `time`, labeled `item.time` alone, then the
schema's fixed dimensions, plus a leading `ensemble` labeled `item.member_ids`
when the runner batches members. Its `lead_time` labels lie within the schema's;
other fixed labels equal the schema's.

### Seeding

Before each item's rollout or first call, a runner seeds every stochastic model
with `model.set_rng(member_seed(self.seed, item.time, member_id), reset=True)`,
where `member_id` is the item's first member. Deterministic items seed as member
`0`.

Models draw one random stream per call, so a batch is seeded once and its
members share that stream: a stochastic member's output depends on which members
share its batch. `member_batch=1` seeds every member alone, reproducing each
member exactly whatever the other members. Seeding each batch element on its own
(`set_rng` taking one seed per element, migrated model by model) is later work.
`Pipeline` records the member groups in the progress manifest, so a resume
cannot regroup or reorder members. `seed` is a constructor argument (default
`0`) exposed as `runner.seed` for the progress manifest. Runners never touch
global RNG state (`P14`, `D10`).

A runner given a `perturbation` applies it to the fetched initial condition
before publishing or rolling it out, one member at a time: it calls
`perturbation.set_rng(member_seed(self.seed, item.time, member_id), reset=True)`
and perturbs that member's field alone, in the layout it has without a batch.
A perturbation without `set_rng` runs inside a Torch RNG fork seeded the same
way. Unlike a stochastic model's draws, a member's perturbation therefore does
not depend on its batch. Perturbations that run models, read their own sources
or perturb members together (bred vectors, `LaggedEnsemble`) raise
`ValueError` at construction.

`member_seed(base, time, member_id)` keeps the eval recipe's derivation, so eval
results reproduce: FNV-1a 64 over `struct.pack(">qqq", base, seconds, member_id)`,
modulo `2**63`.

## Pipeline

Upstreamed from `recipes/eval/src/pipelines/base.py`, with typed arguments in
place of a Hydra config and `run_item` moved onto the runner.

```python
class Pipeline:
    def __init__(
        self,
        runner: Runner,
        output: Mapping[str, IOBackend] | None = None,
        scorer: Scorer | None = None,
        output_variables: Mapping[str, Sequence[str]] | None = None,
        regridder: Mapping[str, Regridder] | None = None,
        progress: ProgressStore | None = None,
        device: torch.device | str = "auto",
        sync: Callable[[WorkItem | None], None] | None = None,
    ) -> None: ...

    @classmethod
    def from_model(
        cls, prognostic, source=None, forcing=None, diagnostics=None, seed=0, **kwargs
    ) -> Pipeline: ...  # wraps a PrognosticRunner

    def run(self, items: Sequence[WorkItem]) -> None: ...
```

At least one of `output` and `scorer` is required; `output=None` writes nothing,
for scorer-only runs. `device="auto"`, the default, picks `cuda:{LOCAL_RANK}` when
CUDA is available, otherwise CPU.

`output`, `output_variables` and `regridder` map stream names to settings. A
stream left out is not written (logged once), keeps all its variables, or keeps
its grid. Variables are filtered before regridding; a regridder's `source_grid`
must equal its stream's grid, and the stored grid becomes `target_grid` (`R4`,
see [REGRID_SPEC.md](REGRID_SPEC.md)).

### Run loop

```python
def run(self, items):
    rank, world_size = rank_info()
    items = self._validate(items)                  # identical on every rank
    self.runner.to(self.device)
    final_lead_time = max(item.final_lead_time for item in items)
    schemas = self._stored_schemas(self.runner.output_coords(final_lead_time))
    if self.progress is not None:
        manifest = self._manifest(schemas)
        attempt = broadcast(self.progress.setup(manifest) if rank == 0 else None)
        self.progress.open(manifest, rank, attempt)
    outputs = OutputManager(self.output or {})
    rank0_first(lambda: outputs.prepare(schemas, items))  # stores span all items
    if world_size > 1:
        outputs.check_distributed()
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
    close_error = outputs.close()                 # returns, never raises
    raise_if_any_failed(error or close_error)     # collective
```

`_emit` filters, regrids, calls `scorer.update`, adds `ensemble` if the store has
it and `x` lacks it, then writes.

- **Stores span all items**, not the remaining ones, so a resume at any world
  size reuses them. Shorter final lead times leave fill values.
- **Durability.** An item is marked done only after its outputs flush and its
  scorer finishes. An interrupted item reruns in full; label-addressed writes make
  that idempotent.
- **Collectives.** Only `Pipeline` runs collectives; `OutputManager` and
  `ProgressStore` are rank-local and are passed the rank where they need it. Idle
  ranks join every collective. A failing rank stops, closes its output and joins
  the final failure-flag reduction; it re-raises its error and the others raise
  `RuntimeError`. A close failure counts as that rank's failure. Setup steps that
  can fail on some ranks only, such as placing the runner on its device, progress
  setup and store creation on rank 0, reduce a failure flag the same way, so no
  rank waits on one that failed.

### Scorer

```python
class Scorer(Protocol):
    def begin_item(self, item: WorkItem) -> None: ...
    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None: ...
    def finish_item(self, item: WorkItem) -> None: ...
```

A minimal online-scoring hook. `update` receives what would be written, before
`ensemble` is added. Metrics, verification data and cross-rank communication stay
in the scorer; it persists durable state in `finish_item`. Hooks run only on the
rank that owns the item, so they never run collectives over the pipeline's
ranks; cross-rank reductions happen in `sync` or after `run` returns.

### Sync

`sync` is a per-round hook every rank joins, for coordination during a run such
as reducing a scorer's accumulators. When it is set, ranks run their items in
lockstep rounds: round `r` runs each rank's `r`-th item, then every rank calls
`sync` with that item, or `None` once it has none left, so `sync` may run
collectives. Both phases of a round reduce a failure flag, so a failure stops
every rank in the same round. Items are marked done only after their round's
`sync` returns. Without `sync`, ranks run their items independently.

## Output

`OutputManager` is internal; `Pipeline` wraps its `output` mapping in one.

```python
class OutputManager:
    def __init__(self, backends: Mapping[str, IOBackend]) -> None: ...
    def prepare(
        self, schemas: Mapping[str, CoordinateSystem], items: Sequence[WorkItem]
    ) -> None: ...
    def check_distributed(self) -> None: ...
    def write(self, stream: str, x: xr.DataArray) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> Exception | None: ...
```

- `prepare` builds templates from the sorted unique item times and the sorted
  union of `member_ids` and adds them to every backend. `Pipeline` runs it on
  rank 0 first; on the other ranks it only validates (`I4`).
- `write` skips streams without a backend; `flush` flushes all (`I9`).
- `close` is idempotent: it closes every backend (`I11`) and returns the first
  error instead of raising, so no rank skips the failure-flag reduction. Closed
  backends stay readable.

Callers build backends on every rank, creating storage on rank 0 only:

```python
rank, _ = rank_info()
output = rank0_first(
    lambda: {"forecast": ZarrBackend(path, backend_kwargs={"overwrite": rank == 0})}
)
```

`AsyncZarrBackend` needs its parallel labels at construction, so its caller
supplies the item times.

### Distributed writes

Ranks write different `time` and `ensemble` positions, so with `world_size > 1`
every store must use blocks of 1 along both. After `prepare`,
`check_distributed` reads each backend's optional `write_blocks` (see
[IO_SPEC.md](IO_SPEC.md)) and raises `ValueError` for a wider block. Backends
without `write_blocks`, process-local or unknown, warn once per type. Both Zarr
backends report it; default `ZarrBackend` chunking passes, `chunks={}` does not.

## Progress

`ProgressStore` records finished items, so runs can resume.

```python
class ProgressStore:
    def __init__(
        self,
        path: str | os.PathLike,
        tags: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None: ...
    def setup(self, manifest: Mapping[str, Any]) -> int: ...  # one rank; returns attempt
    def open(
        self, manifest: Mapping[str, Any], rank: int = 0, attempt: int | None = None
    ) -> None: ...
    def remaining(self, items: Sequence[WorkItem]) -> list[WorkItem]: ...
    def is_done(self, item: WorkItem) -> bool: ...
    def mark_done(self, item: WorkItem) -> None: ...
    def clear(self) -> None: ...
```

- **Unit keys.** Every finished item is recorded as `(time, final_lead_time, member_id)`
  units; an item is done when all its units are. Keys ignore order, rank and
  world size, so a resume may change any of them; a partly done group reruns in
  full. Grouping is fixed by the manifest's `member_groups`.
- **Setup.** Rank 0 calls `setup`, which clears the store if `overwrite`, merges
  existing logs into one and writes the manifest if none exists. It returns the
  run attempt, which `Pipeline` broadcasts.
- **Open.** Every rank then calls `open` with its rank and the attempt. It raises
  `ValueError` naming the differing keys if the stored manifest differs, and
  loads the recorded progress.
- **Logs.** Each rank appends to its own log, named by attempt and rank.
  `mark_done` appends a line and fsyncs; torn or unparseable lines are skipped.
  `path` must be shared by all ranks.
- **Listed once.** Only rank 0 calls `remaining`; `Pipeline` broadcasts it.

`Pipeline` builds the manifest as JSON: `version` (`2`), `streams` (per stream:
dims, non-spatial labels, grid ID and dtype, and whether it is written), `seed`
(`runner.seed` or `null`), `member_groups` (the distinct ordered member groups
of the items, `[[]]` without members), and `tags`. The store merges its `tags` in, so a tag
change is a mismatch. Model identity is not detectable, so callers record it in
`tags`.

Progress and stores are cleared independently: overwriting a store but not its
progress skips items whose data is gone.

## Distribution

```python
def distribute_work(items: Sequence[T], rank: int, world_size: int) -> list[T]: ...
def rank_info() -> tuple[int, int]: ...  # (rank, world_size); (0, 1) if uninitialized
def rank0_first(fn: Callable[[], T]) -> T: ...  # rank 0 runs fn, barrier, then the rest
def broadcast(obj: T | None) -> T: ...  # rank 0's object on every rank
def barrier() -> None: ...
def raise_if_any_failed(error: Exception | None) -> None: ...  # all ranks agree
```

Each rank gets one contiguous, balanced slice; idle ranks get `[]`. Helpers call
`torch.distributed` directly and are no-ops without a process group. Core never
initializes one: launch with `torchrun` and call `init_process_group`, or use
physicsnemo's `DistributedManager`, which is optional.

## Built-in runners

`PrognosticRunner(prognostic, source=None, forcing=None, diagnostics=None, seed=0,
source_member=None, perturbation=None)`
drives `create_iterator` and publishes the initial condition as the first step,
matching `run.deterministic`, when the input ends at zero and the output declares
every input variable on the input's grid; otherwise the streams start at the
first forecast. Each step advances by the model's step interval (see
[MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md#lead-time)). Sources
default to `default_sources()`. Declared forcing is fetched initially and its
time-varying slots are sent every step. The prognostic stream is `forecast`;
each diagnostic adds a stream under the caller's name.

`DiagnosticRunner(diagnostics, source=None, step=None, seed=0, source_member=None)` applies
diagnostics directly to source data. Step `k` fetches inputs at the item's time
plus `k * step`; without `step`, an item is one step at lead time zero.

The **coupled runner**, `earth2studio.coupling.CoupledRunner`, lives with the
coupler ([COUPLING_SPEC.md](COUPLING_SPEC.md)). Per item it fetches each
component's initial condition from bound sources, runs components at their own
cadences, and yields component-named streams.

All three batch members (`supports_member_batching = True`) and take
`source_member` (see **Source members** above).

## Rules

| Rule | Requirement |
| --- | --- |
| `X1` | `output_coords` gives one signature per stream; `time`/`ensemble` are never fixed |
| `X2` | Stream signatures carry the yielded fields' grid metadata and auxiliary coordinates |
| `X3` | Outputs have dims `time` (`item.time` only), then the schema's fixed dims |
| `X4` | Output `lead_time` labels lie within the schema's; other fixed labels equal it |
| `X5` | Each item seeds stochastic models once, from its first member's `member_seed` |
| `X6` | Perturbations draw each member alone from its `member_seed` (base seed, time, member) |
| `X7` | Every rank passes identical items, or `run` raises `ValueError` first |
| `X8` | Stores are planned from all items, so resume at any world size reuses them |
| `X9` | `distribute_work` slices evenly and contiguously; idle ranks join collectives |
| `X10` | Per-stream mappings raise `ValueError` for unknown stream names |
| `X11` | A regridder's `source_grid` must match its stream, or `run` raises first |
| `X12` | Streams absent from `output` are not written; `run` logs them once |
| `X13` | Rank 0 creates stores before other ranks prepare them; re-adding is a no-op (`I4`) |
| `X14` | Distributed stores write blocks of 1 along `time` and `ensemble` |
| `X15` | Distributed runs warn once per backend type without `write_blocks` |
| `X16` | Progress keys `(time, final_lead_time, member_id)` in per-rank logs; bad lines skipped |
| `X17` | Items are marked done only after outputs flush, the scorer finishes and `sync` returns |
| `X18` | Rank 0 lists remaining items and broadcasts them |
| `X19` | A manifest mismatch raises `ValueError` unless the progress store overwrites |
| `X20` | A run or close failure on any rank raises on every rank at the end of `run` |
| `X21` | With `sync`, every rank calls it once per round, idle ranks with `None` |
| `X22` | With `source_member`, slots read from ensemble sources fetch the item's mapped members |

Tests live in `test/run/`. `X7`, `X9`'s collectives, `X13`, `X18`, `X20` and `X21` run
under `torchrun` with two CPU ranks on gloo; the rest run single-process.

## Out of scope

Explicit model state, `initialize`/`step`, components, ports and schedules belong
to the model contract and the coupler. Mid-item checkpointing could become an
optional runner capability built on `(y, state)` (open decision 3).

## Phasing

1. **Done.** `Runner`, `WorkItem`, `DataRequest` and the built-in runners.
2. **Pipeline.** Done: `run/work.py`, the runner follow-up, `run/output.py` and
   `run/pipeline.py`. Next: replace the eval `forecast`, `diagnostic` and
   `dlesym` pipelines with built-in runners.
3. **Coupled runner.** `CoupledRunner` is done
   ([COUPLING_SPEC.md](COUPLING_SPEC.md)), and `dev/examples/11_stormscope_coupling.py`
   runs the Flash StormScope GOES and MRMS models as coupled components (GPU
   check pending). Next: compare against the eval `stormscope` pipeline and
   replace it.
4. **Workflows.** `run.deterministic`, `run.diagnostic` and `run.ensemble` become
   thin wrappers over `Pipeline.from_model`.

## Resolved decisions

- `torch.distributed` directly; physicsnemo optional.
- Core has only the `Scorer` and `sync` hooks; metrics and rank communication stay
  in recipes.
- Member grouping lives in `work_items`, since groups are part of item identity.
- Perturbations are a runner argument (`perturbation=`, per component for
  `CoupledRunner`), drawn per member under `member_seed`.
- Plain mappings for per-stream settings and backends.
- Only `Pipeline` runs collectives; output and progress stores are rank-local.
- Distributed stores need blocks of 1 along `time` and `ensemble`, which backends
  report through an optional `write_blocks`.

## Open decisions

1. **Dependent work items**, such as warm-start cycling: v1 or out of scope?
2. **Ensembles:** per-element seeding of stochastic batches; bred vectors (model
   runs, own source, `+/-` pairs) and lagged ensembles in the runners.
3. **Mid-item resume:** must `run.*` keep step-level checkpoint resume? If so,
   runners need an optional checkpoint built on `(y, state)`. Blocks phase 4 only.
4. **Async fetching:** if needed, change runner and `Pipeline` loop together.

## Integration checks

1. `PrognosticRunner` matches `run.deterministic` — `test/run/test_runner.py`.
2. A hand-written runner stops early and returns `None` data requests —
   `dev/examples/06_runner.py`.
3. `Pipeline` drives `PrognosticRunner` and `DiagnosticRunner` identically,
   writing one store per stream — `test/run/test_pipeline.py`; `CoupledRunner`
   too — `test/coupling/test_pipeline.py`.
4. A forced model (StormCast-shaped mock) and fused DLESyM run through
   `PrognosticRunner` unchanged.
5. Split DLESyM through the coupled runner reproduces the fused model.
6. Resume survives a world-size change — `torchrun`, two CPU ranks, gloo,
   `test/run/test_pipeline.py`.
7. The StormScope coupled runner matches the eval `stormscope` pipeline.
