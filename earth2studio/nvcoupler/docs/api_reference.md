# API reference

Lookup page for everything `earth2studio.nvcoupler` exports (`import
earth2studio.nvcoupler as nvc`). Signatures are copied from the source;
behavior notes are one to three lines — for usage and rationale see
[user_guide.md](user_guide.md) and [concepts.md](concepts.md). Common
argument types: `TimeLike = str | np.datetime64`,
`DeltaLike = str | np.timedelta64` (string forms in
[dsl_and_yaml_reference.md](dsl_and_yaml_reference.md#interval-strings-as_timedelta)),
`CoordSystem` = earth2studio's ordered coordinate dict.

## field

### `Field` (dataclass)

```python
Field(
    data: torch.Tensor,
    coords: CoordSystem,
    standard_name: str,
    units: str,
    valid_time: np.datetime64 | None = None,
    source: str | None = None,
    mask: torch.Tensor | None = None,
    vertical: VerticalCoordinate | None = None,
)
```

One exchanged quantity: a torch tensor plus its coordinates and canonical
identity. Raises `CouplingError` at construction if `coords` contains a
`"variable"` dimension (a Field is one variable) or if `data.ndim` disagrees
with `len(coords)`. `mask` is boolean, True = valid. Methods:
`to(device) -> Field`, `clone() -> Field`, `grid_signature() -> tuple`
(hashable spatial-grid key used for regridder caching).

### `State`

```python
State(name: str, fields: Iterable[Field] = ())
```

A `MutableMapping[str, Field]` keyed by standard name; every component owns an
import and an export State. `state[key] = field` enforces
`key == field.standard_name` (`CouplingError` otherwise); missing keys raise a
`KeyError` listing the present fields.

- `add(field: Field, replace: bool = True) -> None` — `replace=False` raises
  `CouplingError` on duplicates.
- `subset(names: Iterable[str]) -> State`
- `to(device: Any) -> State`
- `as_tensor(names: list[str] | None = None) -> tuple[torch.Tensor, CoordSystem]`
  — stacks fields along a new `"variable"` axis inserted before the first
  spatial dim; all selected fields must share identical coords. Default order
  is `sorted(fields)`.
- `State.from_tensor(name, x, coords, dictionary, valid_time=None, source=None, strict=True) -> State`
  (classmethod) — splits a multi-variable tensor into Fields, resolving raw
  variable names through the dictionary; unknown names raise
  `UnknownFieldError` unless `strict=False` (then they are skipped).

## dictionary

### `CellMethod` (frozen dataclass)

```python
CellMethod(base: str, method: Literal["mean", "sum", "max", "min"], window: np.timedelta64)
```

Machine-readable "I am `method` of `base` over `window`" tag for derived
fields; drives windowed `Connector`s, `AccumulationMediator`, and `couple()`'s
windowed-connector synthesis. Unsupported methods raise `ValueError`.

### `FieldEntry` (frozen dataclass)

```python
FieldEntry(
    standard_name: str,
    canonical_units: str,
    description: str = "",
    aliases: frozenset[str] = frozenset(),
    cell_method: CellMethod | None = None,
)
```

One dictionary entry. `aliases` accepts any iterable (coerced to frozenset).

### `FieldDictionary`

```python
FieldDictionary(entries: FieldDictionary | list[FieldEntry] | None = None)
```

Registry resolving standard names and aliases to entries; constructing from
another `FieldDictionary` copies it (the standard way to extend the default).

- `register(entry: FieldEntry) -> None` — re-registering a standard name
  replaces it; a name that is already an alias raises `ValueError`.
- `add_alias(standard_name: str, alias: str) -> None` — raises
  `UnknownFieldError` for unknown standard names, `ValueError` when the alias
  is taken or collides with a standard name.
- `resolve(name: str) -> FieldEntry` — raises `UnknownFieldError` (with
  did-you-mean suggestions) for unknown names.
- `standard_name(name: str) -> str`, `standard_names() -> list[str]`,
  `__contains__(name: str) -> bool`
- `check_units(standard_name: str, units: str, *, src: str, dst: str) -> None`
  — raises `UnitsMismatchError` when normalized units disagree with canonical
  (checked, not converted).
- `derived_from(standard_name: str) -> CellMethod | None`

### `DEFAULT_DICTIONARY`

Module-level `FieldDictionary` with the curated v1 vocabulary (earth2studio
surface/pressure-level variables plus the built-in derived window fields).

## clock

### `Clock`

```python
Clock(start: TimeLike, stop: TimeLike, dt: DeltaLike)
```

Driver clock stepping `start → stop` inclusive at `dt`. Raises `ValueError`
for non-positive `dt` or `stop <= start`, and `CadenceError` when the span is
not a multiple of `dt`. Iterating yields each time **after** `start` (the IC
lives at `start`, step 0). Properties `current`, `step_index`, `n_steps`;
methods `elapsed()`, `done()`, `advance()` (raises `StopIteration` past
`stop`), `reset()`, `times()` (all `n_steps + 1` times including `start`).

The module also provides (not re-exported at package level, import from
`earth2studio.nvcoupler.clock`): `as_datetime(t)`, `as_timedelta(d)`,
`fmt_timedelta(d)`, `is_multiple(interval, dt)`.

## component

### `Exchange` (frozen dataclass) — one step's coupling bundle

```python
Exchange(
    x: torch.Tensor,
    coords: CoordSystem,
    imports: State,
    std_to_raw: Mapping[str, str] = {},
    time: np.datetime64 | None = None,
)
```

Everything an `ImportAdapter` needs for one coupled model step: the
component's current state tensor and coords, the import `State` (already
subset to the fields actually delivered), the standard-name →
raw-model-variable map, and the valid time the step advances to. Two
accessors cover the common delivery shapes, both pure torch and
autograd-safe:

- `inject() -> torch.Tensor` — returns `x` with each imported field
  overwritten into its matching variable slice (clone + `index_copy_`,
  autograd-intact). Requires a `"variable"` dim in the state coords and that
  every imported field is a state variable (`CouplingError` otherwise).
- `stacked(field_order=None, *, who="Exchange.stacked") -> tuple[torch.Tensor, CoordSystem]`
  — stacks the imports into one tensor with a leading `"variable"` dim. With
  more than one import, `field_order=` is mandatory (`CouplingError`).

### `ImportAdapter` (runtime-checkable `Protocol`) — the model-invocation contract

```python
class ImportAdapter(Protocol):
    def __call__(
        self, model: Any, exchange: Exchange
    ) -> tuple[torch.Tensor, CoordSystem]: ...
```

An adapter **owns the model call**: it receives the model/step-fn and the
step's `Exchange` and must return the stepped `(x, coords)`. Implementations
must be autograd-safe (no in-place mutation of tensors that may carry grad).
Any callable matching this signature can be passed as `import_adapter=`.

### `VariableOverwriteAdapter`

```python
VariableOverwriteAdapter()
```

Default adapter: `model(exchange.inject(), exchange.coords)` — overwrite
matching variable slices, then step (see `Exchange.inject` for the
requirements and errors).

### `ConditioningKwargAdapter`

```python
ConditioningKwargAdapter(field_order: list[str] | None = None, method: str = "call_with_conditioning")
```

Stacks imports (`exchange.stacked(field_order)`) and calls
`model.<method>(x, coords, conditioning=..., conditioning_coords=...)`
(StormScope pattern). With more than one import, `field_order=` is mandatory
— channel order cannot be inferred (`CouplingError`).

### `ExtraTensorAdapter`

```python
ExtraTensorAdapter(field_order: list[str] | None = None, kwarg: str | None = None)
```

Stacks imports and calls `model(x, coords, coupling)` positionally, or
`model(x, coords, **{kwarg: coupling})` when `kwarg` is given (DLESyM /
PhysicsNeMo 4-tensor pattern). Same `field_order` rule as above.

### `PullAdapter` (from `pull.py` — see the [pull section](#pull) below)

The fourth built-in adapter, for models that fetch forcing internally
(StormCast-style) instead of accepting it as an argument.

### `Component` (abstract base)

```python
Component(
    name: str,
    timestep: DeltaLike,
    imports: Iterable[str] = (),
    exports: Iterable[str] = (),
    dictionary: FieldDictionary | None = None,
    variable_aliases: Mapping[str, str] | None = None,
    export_masks: Mapping[str, torch.Tensor] | None = None,
    import_vertical: Mapping[str, Any] | None = None,
    export_vertical: Mapping[str, Any] | None = None,
)
```

NUOPC_Model analog. `imports`/`exports` may be aliases (resolved to standard
names via the dictionary); `variable_aliases` maps raw model variable names to
standard names and registers them as aliases. Class attribute
`requires_ic: bool = True` (subclasses that can initialize without an
`(x, coords)` pair set it False). Phases:
`advertise() -> tuple[list[str], list[str]]`,
`realize(clock: Clock) -> None` (raises `CadenceError` when `timestep` is not
a multiple of `clock.dt`), abstract `initialize(x, coords)` and
`run(time)`, `finalize()`. Helpers: `should_run(time) -> bool` (a temporary
shim — cadence gating lives in the Driver's slot alignment; slated for
deletion), `grid_coords() -> CoordSystem | None`,
`resolve_std_to_raw(coords) -> dict[str, str]`,
`publish(x, coords, valid_time) -> None` (splits a model output tensor into
export Fields, attaching declared masks/verticals; raises `CouplingError` if
an advertised export is missing from the output variables).

### `CallableComponent`

```python
CallableComponent(
    name: str,
    fn: StepFn,                      # (x, coords) -> (x, coords)
    timestep: DeltaLike,
    imports: Iterable[str] = (),
    exports: Iterable[str] = (),
    import_adapter: ImportAdapter | None = None,
    **kwargs,                        # forwarded to Component
)
```

Wraps a plain step function — the entry point for synthetic components and
non-ML models. `initialize` publishes the IC at `clock.start` so lagged
coupling has t0 data. Property `state -> tuple[torch.Tensor, CoordSystem]`.

### `PrognosticComponent`

```python
PrognosticComponent(
    name: str,
    model: Any,                      # earth2studio PrognosticModel
    timestep: DeltaLike | None = None,
    imports: Iterable[str] = (),
    exports: Iterable[str] | None = None,
    import_adapter: ImportAdapter | None = None,
    next_input: NextInputFn | None = None,   # (prev_x, prev_coords, out, out_coords) -> (x, coords)
    **kwargs,
)
```

Wraps an earth2studio `PrognosticModel`, calling it directly (not via
`create_iterator`) so imports can be injected between steps. `timestep`
defaults to the model's output/input lead-time difference; `exports` default
to the output variables resolvable through the dictionary. Published exports
have singleton batch/time/lead_time dims squeezed away. The default
`next_input` handles single-window models and raises `CouplingError` for
multi-window ones (supply the hook). Also: `to(device)`, property `state`.

### `DataComponent`

```python
DataComponent(
    name: str,
    source: Any,                     # earth2studio DataSource
    exports: Iterable[str],
    timestep: DeltaLike,
    variable_map: Mapping[str, str] | None = None,   # std name -> raw source name
    interp_to: CoordSystem | None = None,
    device: Any = "cpu",
    **kwargs,
)
```

Prescribed forcing from a data source via `fetch_data` at its own cadence
(`requires_ic = False`). `initialize()` with no arguments fetches at
`clock.start`; an explicit `(x, coords)` pair (with a `"variable"` dim) is
published as-is instead. Raw source names resolve through `variable_map`,
then `variable_aliases`, then the dictionary's aliases.

### `DiagnosticComponent`

```python
DiagnosticComponent(
    name: str,
    model: Any,                      # earth2studio DiagnosticModel
    timestep: DeltaLike,
    imports: Iterable[str] | None = None,
    exports: Iterable[str] | None = None,
    **kwargs,
)
```

Stateless single-step transform (`requires_ic = False`): stacks its imports in
the model's raw variable order, adds singleton dims the model expects, calls
`model(x, coords)`, publishes the outputs. Imports/exports default to the
model's `input_coords()`/`output_coords()` variables resolved through the
dictionary (every input must resolve, or `UnknownFieldError`). Missing
imports at run time raise `CouplingError` pointing at the run sequence.
Also `to(device)`.

## connector

### `Regridder` — the spatial-regrid callable contract

```python
Regridder = Callable[[torch.Tensor], torch.Tensor]
```

A regridder maps a tensor whose **trailing dims are the source spatial dims**
to the same tensor on the destination grid (leading batch/window dims are
preserved): `tensor[..., H, W] -> tensor[..., H', W']`, or for HEALPix
`[..., face, h, w]` layouts the trailing three. It must be pure torch (no
numpy round-trip) to keep autograd intact. When passed as
`Connector(regridder=...)` it is applied to every field of that connector,
and the output coords are rebuilt from the destination's `grid_coords()`.
Build HEALPix ones with `earth2grid`, as `models/px/dlesym.py` does.

### `Connector`

```python
Connector(
    src: Component,
    dst: Component,
    fields: list[str] | None = None,
    time_policy: Literal["constant", "linear"] = "constant",
    fill: Literal["none", "zero", "nearest"] = "none",
    regridder: Regridder | None = None,
    window: DeltaLike | None = None,
    reduce: Literal["mean", "sum", "max", "min"] | None = None,
)
```

Moves matched fields `src.exports -> dst.imports` through the pipeline
time policy → vertical → mask fill → spatial regrid. `fields=None` matches
the intersection of advertised exports/imports; an explicit list must be in
both (`IncompatibleFieldError` otherwise, also raised when the intersection
is empty). `match() -> list[str]` also unit-checks every field
(`UnitsMismatchError`). `execute(time: np.datetime64) -> None` performs the
transfer (raises `CouplingError` if the source has not produced a field yet);
`last_transfer: dict[str, Field]` holds the most recent delivery (see
`Driver.probe`); `reset()` clears per-run exchange state (history, running
reduction, probes). `time_policy="linear"` extrapolates from the two most
recent exports and falls back to constant (with one warning) for fields
carrying a `lead_time`/`window` dim. Auto-regrid requires regular 1D lat/lon
source grids; identical grids pass through as identity; differing HEALPix
`face` grids require `regridder=` (`IncompatibleFieldError`).

`window`/`reduce` must be set together (`CouplingError` otherwise) and make
this a **windowed connector**: each `execute` folds the source exports into a
trailing running reduction, and delivery happens only at execute times
aligned to `window`. Matching pairs each source export `base` with a
destination import whose dictionary entry carries
`CellMethod(base, reduce, window)` — the delivered Field carries that
*derived* standard name; no matching derived import raises `CouplingError`
(the coupler never invents names), and `match()` returns base names plus
derived names. The window origin is the `valid_time` of the first execute's
source field (the clock start under lagged coupling). Mid-window the
destination's previous import is untouched; `time_policy` does not apply on
the windowed path. This is the preferred replacement for a single-source
`AccumulationMediator`.

## pull

Pull-pattern coupling for models that fetch their own forcing via
`fetch_data(self.conditioning_data_source, ...)` inside `__call__`
(StormCast is the canonical case). The pull path crosses `fetch_data`'s
xarray/numpy boundary, so pull-coupled components are **inference-only** —
no autograd through the exchange.

### `StateDataSource`

```python
StateDataSource(
    state: State,
    raw_to_std: Mapping[str, str] | None = None,
    strict_time: bool = False,
    dictionary: FieldDictionary | None = None,   # defaults to DEFAULT_DICTIONARY
)
```

An in-memory object satisfying the earth2studio DataSource protocol:
`__call__(time, variable) -> xr.DataArray` with dims
`(time, variable, lat, lon)`, built from the State's Fields (which must be
exchange-shaped `(lat, lon)`; other dims raise `CouplingError`). Requested
names resolve in order: **(1)** a standard name already in the State,
**(2)** the `raw_to_std` map (built by `PullAdapter` from the Exchange's
`std_to_raw`, which only covers state variables), **(3)** dictionary
fallback — a raw/alias name (e.g. `u10m`, `t2m`) is resolved to its standard
name through `dictionary` and looked up in the State. No hit raises
`CouplingError` naming the held fields. The source is a **snapshot view**:
every requested time receives whatever the connector last delivered;
`strict_time=True` raises `CouplingError` when a requested time differs from
a served field's `valid_time`.

### `PullAdapter`

```python
PullAdapter(
    attribute: str = "conditioning_data_source",
    strict_time: bool = False,
    dictionary: FieldDictionary | None = None,
)
```

`ImportAdapter` for pull-pattern models. Each call sets
`model.<attribute>` to a fresh `StateDataSource` over `exchange.imports`
(with `raw_to_std` inverted from `exchange.std_to_raw`, and
`strict_time`/`dictionary` forwarded), then returns
`model(exchange.x, exchange.coords)` unchanged — the model's own fetch
receives this step's coupled forcing. A model without the attribute raises
`CouplingError` pointing at `ConditioningKwargAdapter` for
argument-style conditioning. `strict_time=False` (the default) serves the
snapshot and lets the run sequence own cadence alignment — a sequential
connect before the pulling component's run guarantees fresh forcing.

## vertical

### `PressureLevels` (frozen dataclass)

```python
PressureLevels(levels: tuple[float, ...])
```

Constant pressure levels in hPa, ordered top to bottom (increasing, else
`ValueError`). Method `pressure_pa() -> np.ndarray`.

### `HybridLevels` (frozen dataclass)

```python
HybridLevels(a: tuple[float, ...], b: tuple[float, ...], ps_field: str = "surface_pressure")
```

Hybrid sigma-pressure levels `p_k = a_k + b_k * p_s` (`a` in Pa, `b`
dimensionless, top to bottom). Validates equal lengths and strict pressure
monotonicity across the plausible surface-pressure range [500, 1100] hPa
(`ValueError`). `ps_field` names the surface-pressure export a connector
pulls from the source automatically; `len(hybrid)` is the level count.

The interpolation kernel `interp_to_pressure(x, coords, src, dst, ps=None)`
lives in `earth2studio.nvcoupler.vertical` (not re-exported); connectors call
it for you. Linear in log-pressure, clamped at column ends, differentiable.

## mediator

### `Mediator` (base class)

```python
Mediator(name: str, timestep: Any, imports=(), exports=(), **kwargs)
```

`Component` whose import `State` forwards every delivered field to
`accumulate(field)`; when scheduled by a `MediateAction`, `run(time)` calls
`compute(time)`, which must populate `export_state`. Subclass and implement
both to build custom reductions (including unit conversions —
[the v1 remedy for mismatched units](errors_and_troubleshooting.md)).
`requires_ic = False`.

### `AccumulationMediator`

```python
AccumulationMediator(name: str, fields: list[str], window: Any = None, **kwargs)
```

Windowed running reduction (O(1) memory in window length). Each entry of
`fields` must be a *derived* dictionary entry carrying a `CellMethod`
(`CouplingError` otherwise); the cell method supplies the base import, the
reduction (mean/sum/max/min), and the window, which becomes the mediator's
timestep unless `window=` overrides it (`CouplingError` when fields disagree
on windows and no override is given). Duplicate deliveries with the same
`valid_time` are ignored; `compute` with zero samples raises `CouplingError`.
After each compute, `samples_last_window: dict[str, int]` reports the counts
and the accumulators reset. For one source feeding one destination, prefer
the windowed connector (`Connector(..., window=, reduce=)`), which shares the
same accumulator core; mediators are the multi-source / custom-reduction
generalization.

### `TrailingAverageMediator`

```python
TrailingAverageMediator(name: str, fields: list[str], window: Any = None, **kwargs)
```

`AccumulationMediator` restricted to mean reductions (`CouplingError` for
non-mean fields) — the exact semantics of DLESyM's ocean coupling and
PhysicsNeMo's `TrailingAverageCoupler`.

## sequence

### Action dataclasses (frozen)

```python
RunAction(component: str)
ConnectAction(src: str, dst: str)
MediateAction(mediator: str, phase: str = "compute")
```

`Action = RunAction | ConnectAction | MediateAction`; `str()` of each emits
its DSL line.

### `Slot` / `RunSequence`

```python
Slot(interval: np.timedelta64, actions: list[Action] = [])
RunSequence(slots: list[Slot])
```

`Slot.interval` is coerced through `as_timedelta` (interval strings accepted).
`RunSequence` methods: `components_run() -> set[str]`,
`connections() -> list[ConnectAction]`,
`validate(components: dict, dt: DeltaLike) -> None` (raises `SequenceError` /
`CadenceError`; full rule list in
[dsl_and_yaml_reference.md](dsl_and_yaml_reference.md#validation)), and
`__str__` emitting round-trippable DSL.

### `parse_run_sequence`

```python
parse_run_sequence(text: str) -> RunSequence
```

Parses the DSL (grammar in
[dsl_and_yaml_reference.md](dsl_and_yaml_reference.md#grammar)); raises
`SequenceError` with the offending line number.

### `derive_sequence`

```python
derive_sequence(
    components: dict[str, Component],
    connectors: Iterable[Connector | tuple[str, str]] | None = None,
    lagged: set[tuple[str, str]] | Literal["all"] = "all",
) -> RunSequence
```

Derives the canonical run sequence from the coupling graph: one slot per
distinct component cadence, fast to slow. Within a slot: lagged connects
delivered at this cadence (sorted by component declaration order), then each
mediator's compute followed by its outgoing connects, then component runs
topologically ordered over the sequential (non-lagged) edges, each run
followed by its outgoing sequential connects. `lagged="all"` (default) is the
NUOPC-explicit shape; edges not in the `lagged` set are sequential, and a
cycle of sequential edges among same-cadence components raises
`SequenceError` telling you to mark one edge lagged. Unknown endpoint names
raise `SequenceError` with did-you-mean suggestions. This is what
`Driver(sequence=None)` and `couple()` call.

## driver

### `Driver`

```python
Driver(
    components: dict[str, Component],
    sequence: RunSequence | str | None = None,
    clock: Clock | None = None,
    connectors: list[Connector | tuple[str, str]] | None = None,
    collect: bool = True,
    io: dict[str, IOBackend] | None = None,
    allow_unfed_imports: bool = False,
)
```

Executes a coupled system declared as components + connections. With
`sequence=None` (the default, declarative form) the schedule is derived from
the coupling graph via `derive_sequence(components, connectors)` at
construction; an explicit `RunSequence` or DSL string overrides it (the
escape hatch for sequential coupling or hand-tuned action order). The
attribute `sequence_derived: bool` records which path was taken. `clock`
keeps its positional slot but omitting it raises an actionable
`CouplingError`. `connectors` accepts `Connector` instances or bare
`(src, dst)` name tuples (auto-built into default Connectors; unknown names
raise `CouplingError`); with an explicit sequence, any `ConnectAction`
without a prebuilt connector still gets a default `Connector(src, dst)`.
`io=` streams each component's exports to an earth2studio `IOBackend` (one
array per export field, leading `time` axis over the component's ring times
including t0; NaN-initialized), independent of `collect`. Unknown `io` keys
raise `CouplingError`.

- `initialize(ics: dict[str, tuple[torch.Tensor, CoordSystem]] | None = None) -> None`
  — validates the sequence, matches connectors (units checks), rejects unfed
  imports (`UnmatchedImportError`, downgraded to a warning by
  `allow_unfed_imports=True`), warns on unconsumed exports and oversized
  in-memory collection, then realizes and initializes every component.
  Components with `requires_ic=False` (mediators, data components,
  diagnostics) need no `ics` entry; missing ICs for the rest raise
  `CouplingError`.
- `run() -> dict[str, xr.Dataset]` — runs to `clock.stop` under
  `torch.inference_mode()`; returns `to_xarray()` when `collect=True`, else
  `{}`.
- `steps() -> Iterator[tuple[np.datetime64, dict[str, State]]]` — yields
  `(time, {component: export State})` after every driver step (inference
  mode); the notebook-inspection path.
- `rollout(n_steps: int) -> dict[str, State]` — advances `n_steps` keeping
  the autograd graph when grad is enabled (the coupled fine-tuning entry
  point); raises `CouplingError` when fewer steps remain.
- `reset() -> None` — rewinds the clock, clears records, and calls
  `Connector.reset()` on every connector (time-policy history, probes,
  windowed running reductions and window origins); `initialize(ics)` must be
  called again before running. Running an exhausted clock raises
  `CouplingError`.
- `probe(connector: str) -> dict[str, Field]` — last exchanged fields on a
  connector addressed as `"src->dst"` (`KeyError` listing known names).
- `to_xarray() -> dict[str, xr.Dataset]` — collected exports, one Dataset per
  component with a leading `time` axis of that component's ring times.
- `describe() -> str` / `_repr_html_()` — delegate to `api.describe` /
  `api.describe_html`.

## api

### `couple`

```python
couple(
    *components: Component,
    start: TimeLike,
    stop: TimeLike,
    dt: DeltaLike | None = None,
    connectors: list[Connector] | None = None,
    collect: bool = True,
) -> Driver
```

Auto-wires components into a ready-to-initialize `Driver`: every import is
matched to its unique exporter by standard name (`AmbiguousCouplingError` for
several, `UnmatchedImportError` for none). A derived import whose base field
someone exports becomes a windowed
`Connector(src, dst, fields=[base], window=cm.window, reduce=cm.method)`; an
`AccumulationMediator` (named `med_<shortest alias>`) is synthesized only
when the `(src, dst)` pair already carries a plain transfer, which a windowed
connector cannot share. A user-prebuilt windowed connector for the pair is
honored. The run sequence is derived from the graph (`sequence_derived=True`)
in the canonical lagged layout, one slot per cadence. `dt` defaults to the
GCD of the component timesteps. Returns an *uninitialized* driver.

### `coupled`

```python
coupled(
    time: TimeLike,
    stop_or_nsteps: TimeLike | int,
    components: Sequence[Component] | dict[str, Component],
    ics: dict[str, tuple],
    dt: DeltaLike | None = None,
    collect: bool = True,
    verbose: bool = True,
) -> dict
```

One call from initial conditions to `dict[str, xarray.Dataset]`:
`couple(...)` + `initialize(ics)` + a tqdm-wrapped run. An integer
`stop_or_nsteps` means that many driver (`dt`) steps.

### `describe` / `describe_html`

```python
describe(driver: Driver) -> str
describe_html(driver: Driver) -> str
```

Terraform-plan-style preview (text / self-contained Jupyter HTML) of
components, connectors (fields, policies, lagged/sequential mode, slot), and
the run sequence. The mode column is per exchange: `sequential` iff the
source ran (or the mediator computed) earlier in the same slot — the
destination consumes state produced this iteration — else `lagged`. Works
before `initialize()` — only advertised names, the sequence, and the clock
are consulted.

## config

### `to_yaml`

```python
to_yaml(driver: Driver, path: str | os.PathLike | None = None) -> str
```

Serializes a `Driver` to YAML text (also written to `path` when given).
Hand-written sequences serialize as plain DSL text; derived sequences as
`sequence: {derived: true, text: <DSL>}` (the text is informational —
`from_yaml` re-derives). Windowed connectors carry `window`/`reduce` keys.
Raises `CouplingError` for any component that is neither an
`AccumulationMediator` nor carries a `yaml_spec` attribute. Schema and rules
in [dsl_and_yaml_reference.md](dsl_and_yaml_reference.md#the-yaml-schema).

### `from_yaml`

```python
from_yaml(path_or_str: str | os.PathLike) -> Driver
```

Builds an **uninitialized** `Driver` from YAML text or a file path (a
newline-free string naming an existing file is read as a path). Components
are rebuilt by importing each `class` path and calling it with `kwargs`;
failures raise `CouplingError` naming the path. A `sequence` mapping with
`derived: true` re-derives the schedule from components + connectors
(deterministic, so round-trips reproduce identical runs); a mapping without
it raises `CouplingError`. Windowed connectors are rebuilt from their
`window`/`reduce` keys. Call `driver.initialize(ics)` afterwards.

## errors

All configuration errors derive from `CouplingError`; messages name the
components/fields involved and the concrete fix. See
[errors_and_troubleshooting.md](errors_and_troubleshooting.md) for triggers
and remedies.

| Exception | Bases | Raised when |
|---|---|---|
| `CouplingError` | `Exception` | Base class; also raised directly for generic misconfiguration (missing ICs, unproduced exports, bad `yaml_spec`, ...) |
| `UnknownFieldError(name, candidates)` | `CouplingError` | A name is not a registered standard name or alias (did-you-mean suggestions) |
| `UnmatchedImportError(component, field, available_exports)` | `CouplingError` | An advertised import that nothing exports/delivers |
| `UnitsMismatchError(field, src, src_units, dst, dst_units)` | `CouplingError` | Matched fields disagree on (normalized) units |
| `IncompatibleFieldError` | `CouplingError` | A connector cannot reconcile matched fields (grid layout, mask fill, missing regridder) |
| `VerticalMismatchError` | `CouplingError` | Vertical coordinates cannot be reconciled (missing `vertical`, missing surface pressure, non-monotone hybrid levels) |
| `CadenceError(what, interval, dt)` | `CouplingError` | An interval is not a positive multiple of the reference dt |
| `AmbiguousCouplingError(field, importer, exporters)` | `CouplingError` | `couple()` found several exporters for one import |
| `SequenceError` | `CouplingError` | Run-sequence parse or validation failure |

## dlesym_split

### `split_dlesym`

```python
split_dlesym(dlesym: Any, dictionary: FieldDictionary | None = None)
    -> tuple[DLESyMAtmosComponent, DLESyMOceanComponent]
```

Re-exposes a constructed `earth2studio.models.px.DLESyM`'s internal atmos and
ocean sub-models as two components exchanging SST and 48 h window-mean
coupling fields through explicit connectors (identity transfers on the shared
HEALPix grid). Both components step the full 96 h parent cadence and call the
parent's own coupling/insolation methods. Unknown DLESyM variables and the
window-mean entries are auto-registered on a private copy of
`DLESYM_DICTIONARY`. **Honest limitation:** this module has only been
exercised against structural mocks — the real-weights equivalence gate
(`test/nvcoupler/test_dlesym_weights_equivalence.py`) has not been run.

### `build_dlesym_driver`

```python
build_dlesym_driver(
    dlesym: Any,
    start: Any,
    stop: Any,
    dictionary: FieldDictionary | None = None,
    collect: bool = True,
) -> Driver
```

`split_dlesym` plus a `Driver` whose 96 h run sequence reproduces the native
`DLESyM._forward` ordering (SST lagged across steps, window means sequential
within a step). `stop - start` must be a multiple of 96 h. Initialize both
halves with the same DLESyM-layout IC:
`driver.initialize({"atmos": (x, coords), "ocean": (x, coords)})`.

### `DLESYM_DICTIONARY`

Module-level `FieldDictionary` copy of the default — the extension point for
non-default DLESyM vocabularies.

## testing

Deterministic toy components — public API for downstream tests and the docs'
own snippets. A two-component system with the DLESyM cadence structure whose
values are hand-computable from spatially constant ICs (the executable spec in
`test/nvcoupler/test_driver.py` relies on this).

```python
ATMOS_GRID = (32, 64)
OCEAN_GRID = (16, 32)

grid_coords(nlat: int, nlon: int) -> CoordSystem
```

Regular lat/lon coords, 90 → −90, 0 → 360 (endpoint excluded).

```python
fake_atmos(gain: torch.Tensor | float = 1.0, timestep: str = "6h") -> CallableComponent
```

`"atmos"`: imports `sea_surface_temperature`, exports
`geopotential_at_1000hpa`; update `z ← z + 1 + gain·0.1·sst`. Pass
`gain=torch.tensor(1.0, requires_grad=True)` to test gradient flow across the
exchange.

```python
fake_ocean(gain: torch.Tensor | float = 1.0, timestep: str = "48h", with_mask: bool = False) -> CallableComponent
```

`"ocean"`: imports `geopotential_at_1000hpa_48h_mean`, exports
`sea_surface_temperature`; update `sst ← sst + gain·0.01·z48m`.
`with_mask=True` attaches a land mask (northern half invalid) to the SST
export.

```python
atmos_ic(z0: float = 0.0, sst0: float = 2.0) -> tuple[torch.Tensor, CoordSystem]
ocean_ic(sst0: float = 2.0, z48m0: float = 0.0) -> tuple[torch.Tensor, CoordSystem]
```

Spatially constant initial conditions on the matching grids, with `variable`
coords `["z1000", "sst"]` / `["sst", "z48m"]`.

Minimal end-to-end use (executed):

```python
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

driver = nvc.couple(fake_atmos(), fake_ocean(), start="2024-01-01", stop="2024-01-05")
print(driver.describe())            # plan preview, incl. the synthesized windowed connector
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
datasets = driver.run()             # dict[str, xr.Dataset]
```

See `examples/09_nvcoupler/`
([README](../../../examples/09_nvcoupler/README.rst)) for the five worked
examples built on these toys.
