# Earth2Studio Model Contract

## Goal

Define prognostic and diagnostic coordinate, interface, iterator, hook, ownership,
RNG and domain-selection semantics for model-independent execution.

This document is the source of truth for the model contract. Prognostic models follow
the explicit-state protocol in Model Interface: wrappers implement `initialize` and
`step`, and may delegate explicit `__call__`/`create_iterator` methods to private
`PrognosticMixin` helpers. Where
wrappers or the checker do not match yet, this spec governs and Migration lists the
gap.

`earth2studio.models.conformance` checks the rules below and reports their
identifiers. Model tests cover execution, batching, metadata, hooks, ownership and
checkpoint continuation. DA's companion interfaces are defined in
`earth2studio/models/da/base.py` and the
[DA user guide](../../docs/userguide/components/data_assimilation.md); see Open Questions
for remaining generic DA planning concerns.

## Coordinate Systems

`CoordinateSystem` is an `xarray.DataArray` created by `coord_array()` that stores
only shape and dtype, never field values, even when every dimension is concrete.
Read labels from `.coords`, order from `.dims` and lengths from `.sizes`; field
`.values` are unsupported. Dimension and auxiliary coordinate arrays still occupy
memory.

`input_coords()` is a declaration. Dynamic dimensions form an explicitly marked,
zero-sized leading prefix (`earth2studio_dynamic_dims`). Any model may declare them,
under any name (`batch`, `time` or model-specific axes); this is not a
regional/global distinction. All other dimensions, labels, auxiliary coordinates and
declared grid/CRS/statistics metadata must match. Concrete inputs may have any
leading batch dimensions, or none. Fixed trailing dimension order is authoritative,
and a zero-sized fixed dimension is not a wildcard. Resolve multiple dynamic
dimensions together or right to left: concretizing `time` in `(batch, time, ...)`
keeps `batch` dynamic, but concretizing only `batch` leaves a non-leading wildcard
and is rejected. Dimensions are never silently reordered, and wildcards never become
fixed zero-length axes.

Configure flexible shapes and variable sets on the model instance before querying
`input_coords()`: a model supporting arbitrary crops is configured with its domain,
and an observation model with optional channels with its available variables. The
declaration then has concrete spatial coordinates and labels, so fetching and output
planning are unambiguous. Keep the configuration fixed for a run; changing it
requires new signatures and a new fetch/output plan. Models with a selectable spatial
domain implement `DomainAdaptable` (see Domain selection), which returns a new view
per region instead of reconfiguring a model in place. Constructors or model-specific
setters configure other flexible declarations; a shared variable-selection interface
is an open question, not a method callers can rely on. An unresolved
wildcard never requests an unspecified region or variable set. Model-specific
validation (patch-size multiples, supported channel combinations) supplements the
coordinate handshakes.

`output_coords(input_coords)` validates and resolves outputs without touching field
data, so rollouts can be planned before allocation. `coord_array_like(input,
replacements)` preserves leading dimensions, dtype, name, metadata and unaffected
coordinates. Replacing `lead_time` or `variable` drops dependent auxiliaries (e.g.
`valid_time`, per-variable units); recompute them explicitly if needed. Temporal
statistics are declared only in qualified labels such as `tp:sum:6h`:
`coord_array()` and `coord_array_like()` derive the `earth2studio_statistics`
attribute from them, recomputing it when variables are replaced. Spatial
replacements need `coord_array(grid=...)` with a new grid definition;
`coord_array_like()` rejects them to prevent stale grid metadata.

### Grid-backed signatures

Pass a registered name or `GridDefinition` to `coord_array(grid=...)` instead of
repeating grid axes in each model. The helper attaches grid metadata,
`earth2studio_crs` when the definition has a CRS, and `earth2studio_grid_id` for
registered names. Projected, curvilinear and point signatures include geographic
coordinates; HEALPix signatures use index coordinates. Handshakes validate
geographic coordinates as well as axes.

- `StormScopeGOES`, `StormScopeMRMS`: checkpoint `CurvilinearGrid` on `y, x`;
  output offsets are added to the final input lead time.
- `StormCastCONUS`: cropped `ProjectedGrid` with registered HRRR CRS on `y, x`;
  output advances the final input lead time by one hour.
- `PrecipitationAFNO`: registered `latlon-0.25deg-south-pole-excluded` grid
  (720 × 1440) on `lat, lon`; output is `tp:sum:6h` with derived statistics
  metadata.

Models declaring relative lead-time history subtract the final input lead time
before checking the declared window. Beforehand, `lead_time` must be an explicit,
nonempty 1-D timedelta coordinate without `NaT`; datetime and numeric labels are
rejected. Output planning accepts dynamic declarations and concrete DataArrays
without mutating either.

Validation uses the public handshakes in `earth2studio.utils.coords`, which also
accept legacy coordinate dictionaries:

- `handshake_dim`: dimension order, including unlabelled axes; a tuple checks the
  complete ordered dimensions.
- `handshake_size`: dimension sizes.
- `handshake_coords`: coordinate dimensions and label values, without Xarray
  alignment or attached auxiliaries; `subset=True` checks labels before an explicit
  model-specific selection.
- `handshake_dataarray(input, signature)`: the three above plus grid ID, CRS and
  statistics metadata. Relative-history models normalize lead times before calling
  it.
- `handshake_time`: explicit, nonempty, finite datetime/timedelta labels, optionally
  interval alignment or a minimum offset; `dimension=False` for auxiliary/scalar
  validity times.
- `handshake_metadata`: explicitly named attributes.
- `handshake_nonempty(input)`: rejects unresolved zero-sized axes at execution
  boundaries.

Grid-type-specific attributes are not part of the generic handshake, and no check
materializes field values or infers execution mode from field storage. Dynamic
leading axes stay wildcards for planning, and generic leading axes need no labels;
models validate temporal labels with `handshake_time`, using `allow_dynamic=True` for
dynamic planning axes. Model methods keep their history/output transformations and call the
standard handshakes for validation.

### Execution boundary

The public `PrognosticModel` and `DiagnosticModel` protocols (see Model Interface)
take and return DataArrays: one positional argument per input slot, with multiple
outputs grouped in a tuple. **All exported prognostic and diagnostic wrappers take and return
DataArrays.** Fields are NumPy-backed on CPU or CuPy-backed on CUDA; wrappers
document whether inputs must be on the model device or are moved at the core
boundary. At the Torch boundary, `.e2s.to_torch()` and
`from_torch(tensor, signature)` convert; the latter keeps all coordinates and output
metadata without materializing the signature. Outputs omit signature kind/schema/dynamic attributes
and keep user metadata, grid ID/CRS and applicable statistics. Precipitation changes
the variable to `tp:sum:6h`, which declares its statistics. FCN requires finite
timedelta lead times and advances six hours, including from nonzero offsets.

`batch_func` dispatches DataArrays to `.e2s.batch()`/`.e2s.unbatch()`, which pack
arbitrary leading dimensions, handle an existing `batch` dimension and insert a
singleton when there are none. Batch labels, batch-only auxiliaries and spatial
auxiliaries survive even when the core drops attrs; output variable counts may
change. Mixed batch/fixed auxiliary coordinates are unsupported. The core must keep
the packed batch dimension's size and label order; reordering is rejected to prevent
mislabeled output.

`PrognosticMixin` hooks take and return `y` in the original leading dimensions and
run only during iteration; `clear_hooks()` restores identity hooks. Checkpoints save
`(y, state)` (`P21`); level-two checkpoints store each field tensor separately from
dimensions, coordinate values/attrs, name, attrs and encoding. Restarts call `step` on
the saved pair, so they yield the step after the saved state rather than repeating
it.

A daily-mean model can declare two consecutive inputs with start-of-day timestamps:
`mean:0h:24h` for ordinary channels and `mean:1h:25h` for hourly interval-ending `tp`
and `ttr`, with matching statistics metadata. Its initial condition is the latest
input day, and its iterator yields daily predictions. Hooks see one prediction;
front- and rear-hook edits feed the next rolling state (see
Open Questions). Fields move to the model device at the Torch/ONNX boundary. Input
preparation and units follow `TIME_STATISTICS_SPEC.md` (calendar-day means).

The legacy `CoordSystem` (`OrderedDict[str, np.ndarray]`) remains for statistics,
perturbations, private numerical helpers and, until backends migrate
([IO_SPEC.md](IO_SPEC.md)), tensor-based IO; convert explicitly with
`.e2s.to_torch()` and `from_torch()`. `fetch_data()` returns one field DataArray,
not a tensor/coordinate pair. Random/Random_FX keep their data-source API, and
assimilation protocols are outside this migration. Regional prognostic wrappers also
use native DataArrays. StormCastCONUS's stored `ProjectedGrid` is its geometry
source of truth: geographic auxiliaries derive from the cropped grid declaration,
native axes from `input_coords()["y"]`/`["x"]`, and read-only `hrrr_y`/`hrrr_x` from
those coordinates.

Runtime protocol membership checks method presence, not signatures, so it does not
detect the execution API or whether `initialize`/`step` are implemented.
`models.conformance` checks native DataArray signatures and the explicit-state
execution API; dictionary signatures and initial-condition-first iterators fail. See
`dev/examples/03_coordinate_signatures.py` for signature planning and
`dev/examples/04_xarray_model_execution.py` for DataArray execution.

## Model Interface

`earth2studio/models/px/base.py` and `earth2studio/models/dx/base.py` declare the
protocols. `PrognosticMixin` supplies `forcing_coords()` (no forcing),
`default_sources()` (no recommendation), `stochastic = False`, hooks, and private
`_default_call`/`_default_create_iterator` helpers. It has no public execution stubs.
Wrappers declare and document explicit execution signatures,
optionally delegating to the private helpers. The protocol is the requirement; inheriting the
mixin or deriving these methods from the primitives is optional. Direct
implementations must satisfy the same behavioral rules.

A single-DataArray protocol cannot express forced or stateful rollouts. StormCast and
StormScope fetched conditioning from a model-owned `conditioning_data_source` that
callers could not configure and the GOES+MRMS rollout had to bypass. Atlas kept its
latent in the generator frame, so chaining `__call__` was wrong. The interface
therefore has tuple signatures, declared forcing, an explicit-state transition and
recommended sources that models never fetch.

```python
class PrognosticModel(Protocol):
    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]: ...
    def forcing_coords(
        self,
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...] | None: ...
    def output_coords(
        self, *input_coords: CoordinateSystem
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...]: ...
    def initialize(
        self, *x: xr.DataArray
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]: ...
    def step(
        self, *y: xr.DataArray, state: Any
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]: ...
    def default_sources(
        self,
    ) -> (
        DataSource | ForecastSource | tuple[DataSource | ForecastSource | None, ...] | None
    ): ...
    def __call__(
        self, *x: xr.DataArray
    ) -> xr.DataArray | tuple[xr.DataArray, ...]: ...
    def create_iterator(self, *x: xr.DataArray) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        xr.DataArray | tuple[xr.DataArray, ...] | None,
        None,
    ]: ...
```

### Argument slots

Each signature is a slot supplied as a separate positional DataArray (`P17`).
Initial arguments `*x` concatenate input slots and forcing slots, if any exist.
Step arguments `*y` concatenate preceding output slots and time-varying forcing
slots, if any exist. This concatenates argument sequences, not array contents.
Within each sequence, declared slot order is preserved. Simple models use `x`
and `y`; complex models may name individual parameters descriptively.

**Wrappers must declare explicit execution signatures** (`P24`, `D11`): a fixed
number of named parameters, one per declared array slot, rather than `*args`, `*x`
or `*y`. For prognostic wrappers this applies to `__call__`, `initialize`, `step`
and `create_iterator`; for diagnostics it applies to `__call__`. The protocols use
`*x` and `*y` only to describe different fixed arities across models, not a variable
number of inputs to any one wrapper.

Generic forwarding helpers may use variadic arguments internally, but wrappers must
expose explicit signatures instead of inheriting a variadic execution method as their
public interface. A wrapper may delegate its explicitly declared method to a mixin
helper. Generic callers may unpack a slot tuple when calling a wrapper; this does not
make the wrapper's signature variadic.

| Described by | Contents |
| --- | --- |
| `input_coords()` | Initial fields, fetched once at initialization |
| `forcing_coords()` | External fields: full window at first, then newest frames |
| `output_coords()` | Outputs, fed back into the next `step` |
| the model | Everything else a rollout needs; see Explicit state |

One output is returned directly; multiple outputs form a tuple. Callers unpack
that tuple for the next step, but pass a single DataArray directly (splatting a
DataArray would iterate its leading dimension). State is keyword-only in the
variadic protocol; concrete implementations must declare it as a fixed
positional-or-keyword parameter after the arrays. Generic callers use its keyword:
`model.step(y, f, state=state)` or `model.step(*outputs, f, state=state)`.

- **Forcing is declared, not inferred.** StormCast declares its conditioning in
  `forcing_coords()` with plain labels such as `u10m`, though `u10m` is also a state
  output on the same grid.
- **Forcing works like the inputs.** `initialize` takes the whole window and each
  `step` only the newest frames, as with `x` and `y`. The model keeps older frames in
  its state, so callers never assemble windows.
- **Caller-supplied statics are forcing slots without `lead_time`.** `initialize`
  stores them in the state; later steps omit these slots. Statics shipped
  with the checkpoint stay in the wrapper.
- **Models without forcing** return `None` from `forcing_coords()` (the mixin
  default) and take no additional forcing arguments.

Slots are positional, not keyed:

- **Slot order is public API and append-only.** A signature's `.name` is for display
  only.
- **Automation matches by content.** Pipelines and couplers match providers to slots
  by variable, grid and valid time. Position only aligns tuples within one model.

### Splitting rules

- **Slots within a group split when their coordinates differ.** DLESyM declares its
  atmosphere and ocean as separate input and output slots, because one step yields
  more atmosphere lead times than ocean ones.
- **No two output slots share identical non-variable coordinates** (`P18`). A
  diagnostic on the state grid and lead times joins the state output slot.
- **Output slots holding state variables come first, in input-slot order.**
  Output-only slots (a diagnostic on another grid, say) follow them.

### Lead times

`input_coords()` and `forcing_coords()` declare windows with lead times relative to
initialization. To start at `t0`, fetch `x` and the forcing window at `t0` plus their
declared lead times and pass both to `initialize`. Each `step` then needs only the
newest forcing frames: the window's final lead time, shifted by each lead time of
`y`, which `y` carries. `step` selects forcing by lead time, so extra frames (the
whole shifted window, say) are harmless.

StormCast declares `[0h]` for both, taking conditioning at `t0`, `t0 + 1h`, and so
on. StormScopeMRMS declares a multi-frame GOES window and takes one new frame per
step. ACE needs SST and insolation at the target time, so it declares `[0h, 6h]` and
takes the `t + 6h` frame per step.

### Explicit state

`initialize(*x)` runs the first core computation from the input window and
forcing window. It returns `y`, the first forecast, and a model-defined state holding
everything else needed for a rollout: older input and forcing frames, statics,
latents, noise states and the RNG position. `step(*y, state=state)` returns
the next forecast and state. Wrappers may delegate an explicitly declared
`__call__` or `create_iterator` to the mixin's private helpers, or implement the same
contract directly. Weights,
configuration, cached statics and the `set_rng` seed stay on the model.

For a two-frame model, `initialize([x(-6h), x(0h)])` returns `y = x(+6h)` and keeps
`x(0h)` in the state; the first `step` returns `x(+12h)` and keeps `x(+6h)`.

Models do not return the initial condition. Drivers that publish it take it from the
inputs with `initial_condition(x)` (`earth2studio.models.px.utils`), which reduces
each input slot to its final lead time. `initialize` and `step` stay separate methods:
`initialize` takes the input group and the whole forcing window, creates the state and
draws from the model's RNG stream, whereas `step` takes the output group and newest
forcing, and is pure in its state.

- **`(y, state)` is the complete continuation.** Checkpoints save both, so any
  rollout can be snapshot, restored or branched without another model instance.
- **`step` is pure** (`P19`). It modifies neither `y` nor `state`, and the same
  arguments and seed give the same result.
- **State is serializable** (`P21`): arrays, tensors, scalars, DataArrays,
  dataclasses and tuples of these, or `None`. Store RNG position as a counter or
  generator-state tensor, not a `torch.Generator`. No base class is required.
- **State does not duplicate `y`,** unless published outputs are not a faithful next
  input (clipped, cast or post-processed); then the model keeps its own copy.
- **`y` always matches `output_coords()`,** including diagnostics, output-only slots
  and multi-lead-time chunks. `step` builds the next window by selecting input
  variables by label and lead times from the end, never by position: the window is the
  state's older frames followed by `y`, keeping the last `N` lead times.
- **Models never fetch** (`P22`). Missing forcing raises `ValueError`; the derived
  methods check slot counts before calling `initialize` or `step`; model-specific
  implementations validate the arrays against the declared coordinates.
- **`initialize` and `step` are each one core computation,** so `initialize` is
  as expensive as a `step`.
- **One step is one core computation and one yield.** Models computing several lead
  times per call (DLWP, Aurora1p5, SamudrACE, InterpModAFNO) declare them all in
  `output_coords()` and return them together, with forcing windows covering the
  chunk. Every sent value then feeds a real advance, the front hook runs once per
  step, snapshots never fall mid-chunk, and each yield costs one core call.

Example state with model-specific internal and RNG state:

```python
@dataclass(frozen=True)
class AtlasState:
    history: xr.DataArray
    latent: torch.Tensor
    rng_step: int

def initialize(self, x):
    window = self._validate(x)
    state = AtlasState(window.isel(lead_time=[0]), self._encode(window), 0)
    return self._advance(window.isel(lead_time=[-1]), state)

def step(self, y, state):
    return self._advance(y, state)

def _advance(self, latest, state):
    out, latent = self._forward(state.history, latest, state.latent, state.rng_step)
    return out, AtlasState(latest, latent, state.rng_step + 1)
```

### Derived iteration with `send`

`create_iterator` yields forecasts only: its first yield is the output of
`initialize`, and `nsteps` forecasts take `nsteps` yields. The initial forcing window
is consumed by `initialize`, so a value sent to the iterator is always the forcing for
the next `step`, as a single DataArray or a tuple of time-varying forcing slots
in declared order.
Static slots are omitted after initialization. `next(it)` is `send(None)`, so
models requiring no new forcing keep plain loops.

```python
# Single-input, single-output, unforced wrapper.
def __call__(self, x):
    return self._default_call(x)

def create_iterator(self, x):
    return self._default_create_iterator(x)
```

The helpers validate input and forcing slot counts. The iterator publishes the
rear-hook result and feeds it into the next step. Both hooks receive the payload
directly, without copies. In-place front-hook edits also modify the previously
yielded payload; hooks can return new arrays when that is undesirable.

```python
# Unforced
for y in islice(sfno.create_iterator(x), nsteps): ...

# Forced: the driver fetches conditioning alongside initial conditions
it = stormcast.create_iterator(x_hrrr, fetch(stormcast.forcing_coords(), t0))
y = next(it)
for _ in range(nsteps - 1):
    y = it.send((fetch(stormcast.forcing_coords(), t0 + y.lead_time),))

# Publishing the initial condition is the driver's choice
io.write(initial_condition(x))
for y in islice(sfno.create_iterator(x), nsteps):
    io.write(y)

# Explicit loop over two state slots
(atm, ocn), state = model.initialize(x_atm, x_ocn, forcing)
for _ in range(nsteps - 1):
    (atm, ocn), state = model.step(atm, ocn, forcing, state=state)

# Coupled: GOES output conditions MRMS; neither model owns a data source
# Concrete implementations declare state as a fixed positional-or-keyword parameter.
y_goes, s_goes = goes.initialize(x_goes)
y_mrms, s_mrms = mrms.initialize(x_mrms, x_goes)  # GOES window
for _ in range(nsteps - 1):
    y_mrms, s_mrms = mrms.step(y_mrms, y_goes, s_mrms)  # newest frame
    y_goes, s_goes = goes.step(y_goes, s_goes)
```

StormScope GOES and MRMS also support `create_iterator`. GOES has no forcing slot;
MRMS declares its GOES history through `forcing_coords()`, so a coupled loop sends
each new GOES forecast to the MRMS iterator:

```python
goes_it = goes.create_iterator(x_goes)
mrms_it = mrms.create_iterator(x_mrms, x_goes)
y_mrms, y_goes = next(mrms_it), next(goes_it)
for _ in range(nsteps - 1):
    y_mrms = mrms_it.send(y_goes)  # GOES frame at y_mrms's lead time
    y_goes = next(goes_it)
```

The same GOES window initializes both models, and each iteration's `y_goes` has the
lead time of `y_mrms`: the newest frame MRMS needs. MRMS keeps older GOES frames in
its state, so the coupler holds no history. This assumes equal cadences; otherwise
the coupler aligns lead times.

Hooks take and return `y`, which always matches `output_coords()`. The front hook
runs before every `step` and its edits feed the rollout, so per-step perturbation and
noise injection work as before. The front hook does
not run before `initialize`: perturb `x` to edit the initial window. The rear hook
edits forecasts before publication, starting with the first forecast; its returned
output also feeds the next front hook and `step`.

### Default sources

`default_sources()` returns a `DataSource | ForecastSource` directly for a single
slot, a tuple of sources or `None` for multiple slots, or `None` for no recommendations.
A `None` tuple entry means no recommendation for that slot.
Tuple entries are ordered by `input_coords()` slots followed by
`forcing_coords()` slots (`P23`); the slot
signature remains the requirement. It is required for prognostic and diagnostic
models (the prognostic mixin recommends nothing). Drivers
read either through `earth2studio.models.utils.recommended_sources(model)`, fetch,
and handshake the result
against the slot.

```python
def default_sources(self):
    grid = infer_grid(self.input_coords())
    return (
        RegriddedSource(MRMS(), NearestRegridder("mrms-conus-0.01deg", grid)),
        RegriddedSource(GOES(scan_mode="C"), NearestRegridder("goes-east-conus", grid)),
    )
```

A regridder is bound to its source's grid, so the two are recommended as one
provider. Callers may keep the default or supply any provider matching the slot,
such as a pre-regridded archive.

### Diagnostic models

`DiagnosticModel.__call__(*x)` takes one positional DataArray per input slot in
`input_coords()` order, and returns a single DataArray or a tuple in output-slot
order. Wrappers must declare fixed, named input parameters (`D11`); the protocol's
variadic notation only describes differing arities across wrappers. Generic callers
use declared slot order, not parameter names. `output_coords` likewise takes one
separate positional signature/DataArray per input slot, with fixed named parameters
on concrete wrappers. It returns one signature or a tuple of output signatures.
Slot order is append-only, automation matches by content, and slots split only when
coordinates differ. A diagnostic defines `default_sources()`, returning a source
for a single input slot, a tuple in input-slot order, or `None` for no recommendations.
Forcing, `initialize`, `step` and hooks do not apply.

### Domain selection

A model whose spatial domain can be chosen implements the optional `DomainAdaptable`
protocol (`earth2studio.models.base`) in addition to `PrognosticModel` or
`DiagnosticModel`. Generic callers detect it with `isinstance`; fixed-domain models do
not implement it. Design notes and wrapper guidance are in
[flexible_domain.md](../scratch/flexible_domain.md).

```python
@runtime_checkable
class DomainAdaptable(Protocol):
    def with_region(self, region: Region, /) -> Self: ...
```

A loaded model starts on its default domain, usually its full native footprint, and
`with_region` is the only way to change it. A model with no natural full domain, such
as one that builds a local projection anywhere on the globe, ships a documented
default region so that its signatures are concrete (`P2`, `D2`).

`with_region` returns a **view**: a new model instance with its own coordinate
signatures that shares the loaded weights and statics. The model selects the grid of
every input, forcing and output slot itself, so multi-grid models keep their slots
consistent. Selection uses each grid's own logic: a crop model selects and snaps cells
from its supported footprint, and a globe-wide patch model builds a local grid at its
native spacing and shape. A model rejects region kinds its grids cannot select with
`ValueError`; grid-native selections such as HEALPix faces remain model-specific until
a region kind is added for them. Model-specific settings such as CorrDiff's halo and
input margin are configuration of the model a view is made from, not protocol
arguments.

Each view's `input_coords()`, `forcing_coords()` and `output_coords()` are
authoritative. An internal run grid may be larger than a declared grid, as when
CorrDiff computes a halo and trims it. A run binds one view and keeps it fixed (see
Coordinate Systems). Views run safely in sequence, and concurrently only where the
wrapper documents it.

#### Regions

`earth2studio.grids.Region` describes geography independently of any model grid. It
is a frozen value type built through classmethods that validate and normalize it:

```python
Region.box(west, south, east, north)
Region.band(south, north)
Region.rectangle(lat, lon, half_width_m, half_height_m=None, bearing_deg=0.0)
Region.circle(lat, lon, radius_m)
```

| `kind` | `params` |
| --- | --- |
| `"box"` | `(west, south, east, north)` |
| `"band"` | `(south, north)` |
| `"rectangle"` | `(lat, lon, half_width_m, half_height_m, bearing_deg)` |
| `"circle"` | `(lat, lon, radius_m)` |

- `box` uses CRS84 longitude/latitude, with edges along meridians and parallels.
  `west > east` crosses the antimeridian; `west == east` is a zero-width box, not
  the whole globe.
- `band` spans every longitude between two parallels, such as the tropics, a polar
  cap (`band(60, 90)`) or the whole globe (`band(-90, 90)`). It has no east or west
  edge, so coverage checks treat longitude as continuous.
- `rectangle` lies in the local azimuthal-equidistant plane centered at `(lat, lon)`;
  `half_height_m=None` makes a square. `bearing_deg` is the clockwise angle from true
  north to the local positive *y* axis, with positive *x* to its right; at a pole,
  `lon` is the reference meridian. A model accepting `rectangle` must honor its center
  and orientation, or raise `ValueError`.
- `circle` holds every point within geodesic distance `radius_m` of `(lat, lon)`, a
  circle in the same local plane. At a pole, `lon` is irrelevant and normalized to 0.

Direct construction applies the same validation and normalization as the
classmethods, so equal requests compare and hash equal. Validation covers finite
values, coordinate ranges and positive half-widths and radii. Longitude is normalized to
`[-180, 180)` and bearing to `[0, 360)`; in-range values are kept exactly and
coordinates are never rounded, which could change the requested footprint. A `box`
spanning a nonzero multiple of 360° of longitude is rejected rather than collapsed to
zero width; use `band` instead. Masks, vertical extent and variables are not part of
a region. Further kinds (polygons, tracks) are additive.

Coverage checks are model-specific: on projected and curvilinear grids, selecting
cell centers inside a geographic box does not guarantee coverage. `subset_indexers()`
keeps its center-selection meaning. A crop does not retain the parent's registered
grid ID unless its geometry is identical (see [GRID_SPEC.md](GRID_SPEC.md)).

## Rules

### Prognostic

These rules apply to native DataArray execution and explicit continuation state.

| Rule | Requirement |
| --- | --- |
| `P1` | Structurally satisfies `PrognosticModel` |
| `P2` | Allocation-free DataArray declarations with explicit dynamic leading dimensions |
| `P3` | `input_coords()["lead_time"]` is relative, strictly increasing, and ends at zero |
| `P4` | `output_coords()` treats its argument as read-only |
| `P5` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `P6` | Shifting input `lead_time` by an offset shifts output `lead_time` by the same offset |
| `P7` | `create_iterator()` yields forecasts only, the first being `initialize`'s output |
| `P8` | The 1st yield matches the coordinate system `output_coords()` declared |
| `P9` | Every forecast matches its planned coordinates and structural metadata |
| `P10` | `create_iterator()` applies both hooks; `__call__`, `initialize` and `step` apply none |
| `P11` | The model declares a boolean `stochastic` attribute |
| `P12` | A stochastic model implements `set_rng(seed, reset=True)` |
| `P13` | Seeding determines a rollout, and different seeds give different rollouts |
| `P14` | After `set_rng()`, seeding and stepping leave global RNG state unperturbed |
| `P15` | Stepping the model does not modify its input tensor or coordinate system |
| `P16` | Model advances do not overwrite earlier yields; explicit in-place hook edits are allowed |
| `P17` | One positional DataArray per slot; fields precede forcing, in declared order |
| `P18` | No two output slots share identical non-variable coordinates |
| `P19` | `step` modifies neither `y` nor `state`; replaying `(y, state)` reproduces it |
| `P20` | Call, initialization forecast and first iterator yield agree without hooks |
| `P21` | The state is serializable, and a saved `(y, state)` round-trips |
| `P22` | The model never fetches; missing forcing raises `ValueError` |
| `P23` | `default_sources()` follows the single/tuple/None contract in Default sources |
| `P24` | Wrappers declare fixed, named execution parameters (see Argument slots) |

### Diagnostic

| Rule | Requirement |
| --- | --- |
| `D1` | Structurally satisfies `DiagnosticModel` |
| `D2` | Allocation-free DataArray declarations with explicit dynamic leading dimensions |
| `D3` | `output_coords()` treats its argument as read-only |
| `D4` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `D5` | `__call__` matches declared coordinates and structural metadata |
| `D6` | `__call__` does not modify its input tensor or coordinate system |
| `D7` | The model declares a boolean `stochastic` attribute |
| `D8` | A stochastic model implements `set_rng(seed, reset=True)` |
| `D9` | Seeding determines the output, and different seeds give different output |
| `D10` | After `set_rng()`, seeding and calling leave global RNG state unperturbed |
| `D11` | Wrappers declare fixed, named `__call__` parameters (see Argument slots) |

### Domain

These rules apply only to models implementing `DomainAdaptable`.

| Rule | Requirement |
| --- | --- |
| `G1` | `with_region(region)` returns a view covering the region, or raises `ValueError` |
| `G2` | Views leave the parent and sibling views' signatures, parameters and results unchanged |
| `G3` | A view reaches the full supported footprint, with the original model's validation |
| `G4` | Domain-dependent prognostic state is rejected by a view on an incompatible domain |

`G1` coverage is geometric: the usable area of every output slot contains the region,
and a `rectangle` request's center and orientation are honored. It does not guarantee
skill or the absence of edge artifacts. `G2` holds for sequential use with equivalent
seeding; for example, seeding before each rollout of A in an A, B, A sequence
reproduces A, and the root `state_dict` is unchanged.

## Lead Time

Declared input `lead_time` is relative to analysis time and ends at zero (e.g.
`[-6h, 0h]`). Output offsets are added to the *final input* lead time, never a
constant, allowing nonzero starts and resumed rollouts; shifting every input lead
time shifts output equally (`P6`). `forcing_coords()` follows the same convention.

## Iteration

`create_iterator()` yields complete forecasts only, starting with the output of
`initialize`; `nsteps` forecasts take `nsteps` yields, and no yield is a partial
step. A value sent at a yield is the forcing for the next `step`. Drivers that publish
the initial condition take it from `initial_condition(x)`, which reduces each input
slot to its final lead time.

## Hooks

**Hooks belong to the iterator (`P10`).** `__call__`, `initialize` and `step` apply
none, and neither hook sees the initial condition. `create_iterator` applies the rear
hook to every forecast, starting with the output of `initialize`, and the front hook
before every `step`: front hook → `step` → rear hook → yield.

- `front_hook` transforms `y` just before the model computes from it; its edits feed
  the rollout.
- `rear_hook` transforms each forecast before it is yielded; its returned output
  is both published and fed into the next advance.

A step computing several lead times yields them together, so each hook runs once per
chunk, and `P8`/`P9` hold because `output_coords()` declares the whole chunk.
Aurora1p5 declares six hourly outputs per core advance, so both hooks run once per
six-hour cycle.

Each hook takes and returns `y` (a DataArray, or a tuple for multi-slot outputs) and
is a single callable slot. Callers compose transformations explicitly, with no
registration chain, so ordering stays visible at the assignment site and drivers
cannot silently interleave transformations with caller hooks. `clear_hooks()`
restores identity hooks. Single-step callers can transform inputs and outputs
directly; iterator hooks add access to the fields fed back between steps (see
`examples/02_medium_range/02_model_perturbation_hook.py`). The front hook reaches
only `y`; history buffers and recurrent state are reached by editing `state` in an
explicit `initialize`/`step` loop.

GraphCast, GenCast and WeatherNext apply both hooks to public DataArrays while
preserving native recurrence when numerical inputs are unchanged.

## Ownership of Tensors

A model borrows its input and owns its output. None of `__call__`, `initialize`,
`step` and `create_iterator()` may modify caller input tensors or coordinates (`P15`,
`D6`),
and model advances must not overwrite earlier yields (`P16`); views are allowed only
if their buffers will not be overwritten by the model. Hooks receive the payload
directly and may explicitly edit it in place, including a previously yielded array.
Callers retaining snapshots with such hooks must copy them before advancing.
The motivating failure was
`stormcast` overwriting the caller's initial condition
([issue #1133](https://github.com/NVIDIA/earth2studio/issues/1133), fixed in
PR #1134). `AsyncZarrBackend` still copies every non-blocking write defensively;
model-level ownership guarantees address the underlying buffer-lifetime problem.

The reverse also holds: the `y` a model returns or yields feeds the next `step`, so
callers must not modify it in place. An in-place edit, such as a unit conversion in
an IO backend, would silently change the rollout; edits meant to feed back belong in
the front hook.

Legacy `batch_func` rebuilds coordinates but does not protect tensor inputs:
`_compress_batch` uses `unsqueeze`/`flatten` views, so internal writes reach callers.

## Stochasticity

`P11`–`P14` and `D7`–`D10` require a readable boolean `stochastic` declaration and,
when true, `set_rng(seed: int, reset: bool = True) -> None`. Undeclared models
default to `False`, supplied by `PrognosticMixin`; diagnostics need no base class.
The declaration lets drivers plan ensembles before execution.

The seed is the first positional argument, supporting seed-only core APIs.
`reset=True` replaces the generator; `reset=False` initializes it only if absent,
otherwise ignoring the seed and keeping the trajectory. Ensemble drivers reseed each
member with `reset=True` before `initialize`; components that seed lazily may call
`set_rng(fallback_seed, reset=False)` without clobbering driver seeding, whereas
resetting to the same seed before each rollout would repeat identical noise.
Never-seeded models fall back to the global RNG rather than failing.

The same seed must reproduce every rollout step (`P13`) or diagnostic output (`D9`)
exactly, and different seeds must differ. A model declaring `stochastic=False` whose
results vary for the same input also fails these rules.

### RNG isolation

After `set_rng()`, neither seeding nor stepping/calling may leave global RNG state
perturbed (`P14`, `D10`). Use a local `torch.Generator` for every draw, a functional
PRNG key, or `torch.random.fork_rng()` around global seeding and execution. Forking
supports external packages without generator injection (e.g. `aifs2ens`/`anemoi`);
pass `devices` explicitly, since the default forks all visible CUDA devices and
warns. Unseeded global-RNG draws remain allowed.

For a core that accepts only global seeding, store the seed without drawing and
isolate each step, seeding from the position held in the state (see RNG with explicit
state):

```python
def set_rng(self, seed: int, reset: bool = True) -> None:
    if reset or self._seed is None:
        self._seed = seed

# Seeded step:
with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
    torch.manual_seed(mix(self._seed, state.rng_step))
    out = self.core_model(...)
```

Isolation keeps other models, perturbations and dataloaders from having their streams
reset. Reproducibility checks cannot catch this interference: a model may reproduce
perfectly alone while destroying independence in a cascade. The rule constrains the
observable effect, not the mechanism.

Forking isolates sequential calls, not concurrent ones: `fork_rng` saves and restores
process-global state, so concurrent threads can interleave fork windows. Drivers must
not run a global-RNG component concurrently with another stochastic component in the
same process.

### RNG with explicit state

The seed stays on the model; each rollout's stream position lives in its state.
`initialize` draws the rollout's starting point from the model's stream, so new
rollouts still advance it and only `set_rng(..., reset=True)` restarts it. Each
`step` derives its draws from the state alone, so replaying a saved state is exact
(`P19`):

| Mechanism | Per-step stream |
| --- | --- |
| Forked global | `fork_rng(devices=...)`, then `manual_seed(mix(seed, rng_step))` |
| Local generator | generator rebuilt from, or restored to, the state's position |
| Functional key | `fold_in(key, rng_step)`, both held in the state |

Calling `set_rng` mid-rollout affects only later `initialize` calls, never a live
trajectory.

### Seeding is the only entry point

Constructors and `load_model()` take no `seed`: load the model, then call
`model.set_rng(seed)`. Calls and new iterators advance that stream; only an explicit
reset restarts it.

### Current wrappers

Each wrapper owns its RNG implementation. FCN3 delegates to its backend's RNG API;
CorrDiff passes sample seeds to its diffusion backend; StormScope and NSRDB use
explicit noise generators; GenCast and WeatherNext advance functional JAX keys;
DLESyM keeps its native local generator. Backends without a generator API run inside
a wrapper-local Torch RNG fork (`earth2studio.models.utils.fork_rng`) that saves and
resumes model-owned CPU/CUDA RNG state between sampling calls. Seeds initialize
streams once per reset and sampling advances them; forks cover only the relevant
numerical calls and never span iterator yields. Aurora also resets its cached noise,
and DiagnosticWrapper dispatches seeding to its stochastic components.

## Conformance

`earth2studio.models.conformance.check_prognostic_contract(model)` evaluates every
applicable probe before failing, reporting collected violations and returning
unevaluated rules with reasons. `rollout=False` retains declarations/planning
(`P1`–`P6`), stochastic declarations and seeding (`P11`, `P12`, the seeding half of
`P14`), slot/signature checks (`P17`, `P18`, `P24`) and sources (`P23`).
`check_diagnostic_contract(model, forward=False)` retains `D1`–`D4`, `D7`, `D8`,
`D11` and the seeding half of `D10`. Behavioral `reset=False` checks need execution.

Probes concretize dynamic dimensions for every declared slot and use each
signature's `.shape`, auxiliary coordinates and grid/statistics metadata.
Execution and planning both pass separate positional arguments in input-slot order.
Pseudo-random probes expose input mutation; models rejecting unphysical data need
realistic initial-condition fixtures. Generic probes cannot prove numerical
correctness, semantic source ordering or absence of internal fetching.

Every forecast in the probed rollout is checked against independently planned
coordinates. The checker rebases the declared input history at the previous planned
output's final lead before planning the next forecast, supporting multi-frame
histories and multi-output steps without trusting yielded coordinates. Calls and
yields must preserve declared grid, CRS and statistics metadata and omit signature
kind/schema/dynamic attributes; user attributes may change through hooks.

### Enforcement

Model creation skills produce `test_<model>_conformance` from existing mock-weight
fixtures, without real weights or network access. `test/models/test_conformance.py`
tests the checker; `test/models/test_model_conformance.py` introspects
`earth2studio.models.px`/`dx` and requires every class to conform or be explicitly
exempt with a pinned reason. Backend-dependent execution needs its optional
dependencies; a dependency skip is not evidence of conformance.

## Migration

The checker implements this contract. Wrappers and consumers migrate separately:

- **Wrappers.** The mixin supplies hooks, defaults and private helpers, not public
  execution methods. Concrete wrappers define `__call__`, `initialize`, `step`
  and `create_iterator`; missing methods fail structural `P1`.
- **Protocol annotations.** `PrognosticModel` now declares variadic DataArray
  arguments and single-or-tuple output signatures. Migrated wrappers declare explicit
  fixed signatures (`P24`, `D11`) without inheriting variadic execution methods.
  Wrappers and callers in
  `earth2studio.run`, perturbations, `dxwrapper`, `interpmodafno` and recipes must
  migrate together; structural protocol membership does not validate signatures.
- **One iterator API.** `create_iterator` remains the public name and is not
  deprecated. Migrated implementations yield forecasts only. Coordinate the
  transition with consumers that currently count `nsteps + 1` yields, so their
  forecasts retain the correct lead-time labels.
- **Chunked yields.** Unmigrated DLWP, Aurora1p5, SamudrACE and InterpModAFNO yield
  one lead time at a time and declare `front_hook_interval`, the number of yields per
  front-hook call (DLWP: 2, front hook → compute +6h and +12h → rear hook and yield
  each). Migrated, they yield whole chunks and `front_hook_interval` is removed.
- **Conformance.** The checker rejects initial-condition-first iteration and probes
  `P17`–`P24`, `D11` and multiple slots. Signature checks follow decorators that
  preserve the wrapped method's declared signature. Unmigrated wrappers fail the
  checker until they implement the new execution and declaration requirements.
- **Diagnostics.** Add a `DiagnosticMixin` supplying `stochastic = False` and a
  `default_sources()` recommending nothing, inherited by every diagnostic wrapper.
  `default_sources()` then becomes a required `DiagnosticModel` member, as for
  `PrognosticModel`, and drivers drop their `getattr` fallbacks for both attributes.
- **Domain selection.** Add `DomainAdaptable`, `Region` and checks for `G1`–`G4`.
  `StormScopeGOES` and `StormScopeMRMS` implement it for box regions: Flash variants
  crop the HRRR footprint with context, and baseline variants return full-footprint
  views, which cover any supported region. `CorrDiffCosmoEra5`,
  `StormCastCONUS`/`StormCast` and `cBottleSR` adopt the protocol additively,
  keeping their constructor limits, `set_domain` and
  `set_super_resolution_window` as adapters; see
  [flexible_domain.md](../scratch/flexible_domain.md).
- **Examples and drivers.** `earth2studio.run`, `examples/` and `dev/examples/` call
  `create_iterator` and count `nsteps + 1` yields; they move to `nsteps` yields,
  writing `initial_condition(x)` where they publish the initial condition.

## Open Questions

- Should `P13`'s exact comparison allow a configurable tolerance for nondeterministic GPU
  kernels that make deterministic models appear stochastic?
- An opaque, unseedable dependency would need a separate `seedable` declaration and
  would skip `P13` while keeping `P14`. No current wrapper needs this split.
- `P14`/`D10` cover only models implementing `set_rng`; widen them if a deterministic
  model that reseeds globally appears.
- DA now separates independent `AssimilationModel` calls from the extending
  `StatefulAssimilationModel` lifecycle (`initialize`, `step`, `create_iterator`).
  Its first yield is the computed analysis; there are no priming yields. Explicit
  state, replay, and ownership follow the prognostic conventions, extended to
  DataFrame slots and optional observations. `send(None)` only permits advancement
  when no new required forcing is needed; StormCast requires conditioning even
  without observations. Tabular outputs use `FrameSchema`.

  Remaining generic DA planning questions are analysis-time resolution for
  observation-only models, machine-readable observation windows and optionality
  (including HealDA's at-least-one constraint). RNG follows the shared convention
  in [the DA contract](DA_MODEL_CONTRACT_SPEC.md). Model-specific metadata/window
  contracts remain authoritative until those are standardized.
- Do drivers accept yields with several `lead_time` entries? IO backends do
  ([IO_SPEC.md](IO_SPEC.md)); chunked steps for DLWP, Aurora1p5, SamudrACE and
  InterpModAFNO depend on drivers as well.
- A hook editing `y` can leave derived private state (Atlas's latent) stale.
  Should `step` re-derive from `y`, or document what it ignores?
- Front hooks no longer see frames older than `y`, nor the initial condition, so
  they cannot apply fresh noise to a whole history window at once. Is this a
  limitation for history-window perturbation workflows?
- Should components declare their RNG mechanism (e.g. `rng = "local" | "global"`) so
  drivers can run global-RNG components serially?
