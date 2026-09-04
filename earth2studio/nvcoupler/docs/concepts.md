# Concepts

The deep conceptual reference for nvcoupler. The [package README](../README.md)
is the overview; this page pins down the exact semantics of each abstraction.
For task-oriented recipes see the [user guide](user_guide.md), for the DSL and
YAML grammar the [DSL and YAML reference](dsl_and_yaml_reference.md), for
signatures the [API reference](api_reference.md), and for every error class the
[errors and troubleshooting page](errors_and_troubleshooting.md). Rationale for
these designs lives in [design and roadmap](design_and_roadmap.md).

## Field and State

A `Field` is exactly one physical quantity: a torch tensor, its earth2studio
`CoordSystem`, a canonical identity (`standard_name`, `units`), an optional
`valid_time`, `source` (provenance), `mask`, and `vertical` metadata. Two rules
are enforced at construction:

- **One variable per Field.** A `"variable"` dimension in `coords` raises
  `CouplingError` immediately — multi-variable tensors are split with
  `State.from_tensor`.
- **Coords insertion order is dimension order.** `data.ndim` must equal
  `len(coords)`, and the *i*-th key of the coords dict describes the *i*-th
  tensor axis. There is no name-based reordering anywhere in the framework.

```python
from collections import OrderedDict
import numpy as np
import torch
import earth2studio.nvcoupler as nvc

coords = OrderedDict(
    {"lat": np.linspace(90.0, -90.0, 8), "lon": np.linspace(0.0, 360.0, 16, endpoint=False)}
)
sst = nvc.Field(
    data=torch.full((8, 16), 290.0),
    coords=coords,
    standard_name="sea_surface_temperature",
    units="K",
    valid_time=np.datetime64("2024-01-01"),
)
# Field('sea_surface_temperature' [K], lat: 8, lon: 16, valid_time=2024-01-01)
```

A `State` is a mutable mapping of Fields keyed by standard name (the key must
equal the field's `standard_name`, enforced on `__setitem__`). Every component
owns an import State and an export State; Connectors move Fields between them.
`State.from_tensor` splits a tensor along `"variable"`, resolving raw model
names through a `FieldDictionary`; `State.as_tensor(names)` stacks Fields back
along a new `"variable"` axis inserted immediately before the first spatial
dimension (earth2studio's `batch, time, lead_time, variable, spatial...`
convention). All stacked fields must share identical coords — regrid through a
Connector first.

```python
x = torch.zeros(2, 8, 16)
tensor_coords = OrderedDict({"variable": np.array(["z1000", "sst"]), **coords})
state = nvc.State.from_tensor(
    "demo", x, tensor_coords, nvc.DEFAULT_DICTIONARY,
    valid_time=np.datetime64("2024-01-01"),
)
sorted(state)   # ['geopotential_at_1000hpa', 'sea_surface_temperature']
y, ycoords = state.as_tensor(["sea_surface_temperature", "geopotential_at_1000hpa"])
```

Field data is torch end to end — never round-tripped through numpy inside the
framework — so autograd graphs survive every exchange.

### The physical-units exchange contract

Fields cross the coupling seam in **physical units**. A model that trains on
normalized inputs normalizes internally: the DLESyM split adapter
([`dlesym_split.py`](../dlesym_split.py)) is the reference implementation —
each half denormalizes its outputs (`data * scale + center`) before publishing
and renormalizes imports (`(data - center) / scale`) before calling its U-Net.
Because both directions use the same per-variable constants, the round trip is
exact up to float rounding, and `test/nvcoupler/test_dlesym_split.py` asserts
the reconstructed coupling tensor matches the parent's own math. The payoff:
any component can consume any other's exports without knowing its
normalization statistics.

Units are **checked, not converted** (see the dictionary section below); no
scaling ever happens silently in a Connector.

### valid_time semantics — and an honest caveat

`valid_time` is the single instant the data is valid for. Components stamp it
when publishing (`initialize` publishes at `clock.start`, so lagged coupling
has data at t0; each `run(time)` publishes at `time`), and Connector time
policies reason about it.

The caveat: some exported Fields are not instantaneous. A field carrying a
`lead_time` or `window` dimension (e.g. the DLESyM atmos component's
window-mean exports, which have a leading `window` axis over the ocean's 48 h
chunks) represents *many* times but carries one `valid_time` stamp — the end
of the producing step's window. That is fine for `"constant"` transfers, but a
single-timestamp extrapolation is ill-defined for such fields, so the
Connector's `"linear"` policy detects `lead_time`/`window` dims and **falls
back to `"constant"` for that field**, logging a one-time warning per field.

### Masks

`mask` is an optional boolean tensor broadcastable to `data`, with
**True = valid** (e.g. ocean points for SST; the toy ocean in
[`testing.py`](../testing.py) marks the northern half of its grid as land with
`False`). Masks are declared per export via a component's `export_masks=` and
consumed by the Connector's fill stage; after filling, the transferred field's
mask is cleared (`None`) since every point is then valid.

### Grid signatures

`Field.grid_signature()` returns a hashable tuple of
`(dim, shape, values.tobytes())` for every spatial dim
(`level, face, lat, lon, hpx, height, width, y, x`). Connectors key their
lazily built regridders and mask fillers on it, so the KDTree/interpolation
setup cost is paid once per distinct grid (and, for fillers, per distinct
mask), not per step — the `dxwrapper.py` caching pattern.

## FieldDictionary

Connectors match exports to imports by **standard name**, never by raw model
variable strings. A `FieldDictionary` maps both standard names and aliases to
`FieldEntry(standard_name, canonical_units, description, aliases, cell_method)`.
Lookup is case-sensitive; an alias maps to exactly one standard name (remapping
raises), an alias may not collide with a standard name, and re-registering a
standard name replaces its entry. `DEFAULT_DICTIONARY` ships a curated v1
vocabulary (earth2studio surface variables, the pressure-level fields used by
the coupled models in this repo, and the mediator-produced derived fields);
components each get a private *copy*, extended by their `variable_aliases=`
(raw model name → standard name).

### Units: checked, not converted

`normalize_units` collapses cosmetic differences before comparing: lowercase,
strip `**`/`^` exponent markers and spaces (behaviorally aligned with
`earth2studio.lexicon.earthmover.normalize_units`), then apply a synonym table
(`m/s` ≡ `m s-1`, `kelvin` ≡ `K`, `mm` ≡ `kg m-2` for precipitation depth,
the dimensionless family `1`/`(0-1)`/`fraction` ≡ `""`, `celsius` ≡ `degC`,
...). Values are **never converted** — a genuine disagreement raises
`UnitsMismatchError` at connector match time, naming both components and
suggesting the fix (convert in a Mediator or align the dictionary entries).

```python
from earth2studio.nvcoupler.dictionary import normalize_units

normalize_units("m s**-1") == normalize_units("m/s") == "m s-1"   # True

nvc.DEFAULT_DICTIONARY.check_units(
    "sea_surface_temperature", "degC", src="ocean", dst="atmos"
)  # raises UnitsMismatchError: 'ocean' exports 'degC' but 'atmos' expects 'K'
```

### CellMethod: derived fields as first-class dictionary entries

A time-reduced field (a 48 h precipitation sum, a 24 h temperature max) is a
dictionary entry carrying a machine-readable `CellMethod(base, method, window)`
with `method` in `mean | sum | max | min` — the declaration "I am *method* of
*base* over *window*". No suffix string-parsing anywhere.

Three consumers read it:

- **Windowed `Connector`s** — `Connector(src, dst, window=..., reduce=...)`
  pairs each source export `base` with a destination import whose entry
  carries `CellMethod(base, reduce, window)` and delivers under that derived
  name (see [the connector pipeline](#the-connector-pipeline)).
- **`AccumulationMediator`** — constructed with the *derived* names, it
  resolves each entry, takes the base field as its import, the method as its
  running reduction, and the (common) window as its timestep unless
  `window=` overrides it. A derived name without a `cell_method` raises.
- **`couple()`** — when a component imports a derived field nobody exports,
  auto-wiring checks the entry's cell method: if some unique component exports
  the *base* field, it wires a windowed
  `Connector(base-exporter, importer, fields=[base], window=cm.window,
  reduce=cm.method)`. Only when that `(src, dst)` pair already carries a
  plain transfer (the importer also imports the base directly, or a second
  derived field rides the same pair) is an `AccumulationMediator` synthesized
  and wired `base-exporter -> mediator -> importer` instead; a user-prebuilt
  windowed connector for the pair is honored. No cell method or no base
  exporter raises `UnmatchedImportError`; multiple base exporters raise
  `AmbiguousCouplingError`.

```python
entry = nvc.DEFAULT_DICTIONARY.resolve("total_precipitation_48h_sum")
entry.cell_method
# CellMethod(base='total_precipitation_6h', method='sum', window=numpy.timedelta64(48,'h'))

med = nvc.AccumulationMediator("med", ["total_precipitation_48h_sum"])
med.import_names   # ['total_precipitation_6h']
med.export_names   # ['total_precipitation_48h_sum']
med.timestep       # 48 hours (from the cell method's window)
```

## Clock and the cadence model

Three intervals coexist and must nest:

1. **Driver dt** — the `Clock`'s step, the finest granularity at which
   anything can happen.
2. **Component timesteps** — each component declares its own cadence as a
   plain `timestep`; the Driver runs it only in run-sequence slots whose
   interval equals that timestep, and those slots execute only on clock
   steps aligned with it (slot alignment). There is no per-component alarm
   object and no offset mechanism — cadence is purely timestep alignment.
3. **Slot intervals** — each run-sequence `@interval` slot executes on driver
   steps aligned with it.

The validation rules, all raising `CadenceError` at construction or
`initialize` time (never mid-rollout):

- `Clock(start, stop, dt)`: `stop - start` must be a positive whole multiple
  of `dt`.
- `Component.realize(clock)`: the component's timestep must be a positive
  whole multiple of `dt`.
- `RunSequence.validate`: every slot interval must be a multiple of `dt`, and
  every component (or mediator) scheduled in a slot must have a timestep
  **equal** to that slot's interval. Components never scheduled anywhere raise
  `SequenceError`.

```python
import numpy as np

clock = nvc.Clock("2024-01-01", "2024-01-03", "6h")
clock.n_steps                       # 8
timestep = np.timedelta64(48, "h")  # a 48 h component on this 6 h clock
[(t - clock.start) % timestep == np.timedelta64(0) for t in clock.times()]
# [True, False, False, False, False, False, False, False, True]

nvc.Clock("2024-01-01", "2024-01-02", "7h")   # raises CadenceError (24 h span, 7 h dt)
```

Iterating a Clock yields times *after* start: the initial condition lives at
`start` (the caller's step 0), and the first yielded time is `start + dt` —
mirroring `earth2studio.run` where the iterator's 0th output is the IC. Every
timestep is trivially aligned at `start` itself (elapsed time zero), but no
slot executes there — the driver only runs actions at times strictly after
`start`; that is why `initialize` seeds export states at t0.

Clocks are **one-shot**. Running past `stop` raises, and calling
`run()`/`steps()`/`rollout()` on an exhausted driver raises a `CouplingError`
telling you to call `driver.reset()` — which rewinds the clock, clears
collected records, connector history, and IO state — and then
`driver.initialize(ics)` again before rerunning. A reset-and-reinitialized run
reproduces the original exactly (asserted in `test_driver.py`).

## Components and the NUOPC phase lifecycle

Every component walks the NUOPC phases, driven by the Driver:

1. **`advertise()`** — return the import and export standard-name lists
   (declared at construction; names are resolved through the dictionary, so
   aliases work).
2. **`realize(clock)`** — validate the component's timestep against the
   driver dt and attach the shared clock.
3. **`initialize(x, coords)`** — set internal state from an initial condition
   and *seed the export state at `clock.start`*, so lagged coupling has data
   at t0.
4. **`run(time)`** (repeated) — advance one component timestep; exports become
   valid at `time`.
5. **`finalize()`** — cleanup hook (no-op by default).

### The requires_ic contract

`requires_ic` (class attribute, default `True`) tells the Driver whether
`initialize` needs an `(x, coords)` pair. During `Driver.initialize(ics)`:
components present in `ics` get their entry; components with
`requires_ic = False` (mediators, `DataComponent`, `DiagnosticComponent`) are
initialized with no arguments — a DataComponent fetches at `clock.start`, a
DiagnosticComponent just records its grid, a Mediator does nothing; a
`requires_ic = True` component missing from `ics` raises `CouplingError`
naming it. `requires_ic = False` components still *accept* an explicit IC
(a DataComponent publishes it instead of fetching; a DiagnosticComponent
pushes it through the model once to seed t0 exports for lagged chains).

### The four component kinds

- **`CallableComponent`** — wraps a plain `fn(x, coords) -> (x, coords)` step
  function. The entry point for synthetic components and **non-ML models**
  (process-based hydrology, crop models, anything Python-callable). Owns an
  `(x, coords)` state; imports are injected through its ImportAdapter each run.
- **`PrognosticComponent`** — wraps an earth2studio `PrognosticModel`.
  Timestep defaults to the model's output-minus-input lead time; exports
  default to the output variables resolvable through the dictionary. It steps
  the model directly (not via `create_iterator`) so imports can be injected
  between steps. Models with multi-window sliding inputs need a
  `next_input(prev_x, prev_coords, out, out_coords)` hook; the default handles
  single-window models and raises with instructions otherwise. Publishing is
  **exchange-shaped**: singleton `batch`/`time`/`lead_time` dims are squeezed
  so exported Fields carry plain spatial coords like everyone else's (the seam
  tests in `test/nvcoupler/test_seams.py` exist precisely for this).
- **`DataComponent`** — prescribed forcing from an earth2studio `DataSource`
  (`requires_ic = False`). Instead of stepping a model it fetches its export
  variables at its own cadence via `fetch_data`. Swapping a modeled ocean for
  observed SST is a one-line component substitution; connectors, mediators,
  and the run sequence are untouched.
- **`DiagnosticComponent`** — wraps an earth2studio `DiagnosticModel`
  (`requires_ic = False`): a stateless per-run transform that stacks its
  imported Fields in the model's raw variable order, conforms singleton dims
  the model expects, calls it, and publishes the outputs. Imports/exports
  default to the model's own `input_coords()`/`output_coords()` variables.

`Mediator` subclasses (below) are the fifth participant type; they run in the
sequence like components but compute reductions rather than stepping models.

## ImportAdapters: who owns the model call

Real models disagree on how coupled forcing arrives, so the **adapter — not
the component — owns the model invocation**. Each of the four built-ins maps
to a real-world coupling pattern:

| Adapter | Call shape | Real-world pattern |
|---|---|---|
| `VariableOverwriteAdapter` (default) | overwrite matching variable slices of `x`, then `model(x, coords)` | prescribed forcing as a state channel |
| `ConditioningKwargAdapter` | `model.call_with_conditioning(x, coords, conditioning=..., conditioning_coords=...)` | StormScope |
| `ExtraTensorAdapter` | `model(x, coords, coupling)` (or a named kwarg) | DLESyM / PhysicsNeMo 4-tensor |
| `PullAdapter` | install a `StateDataSource` on `model.conditioning_data_source`, then `model(x, coords)` | StormCast |

An adapter receives the model and an `Exchange` — a frozen bundle of the
step's state tensor and coords, the delivered import `State`, the
standard-name → raw-variable map, and the step time — and returns the stepped
`(x, coords)`. The two `Exchange` accessors cover the common delivery shapes:
`exchange.inject()` (variable-slice overwrite) and `exchange.stacked()`
(channel-stacked forcing tensor).

`VariableOverwriteAdapter` requires the imported field to be a state variable
of the model (resolved through `std_to_raw` aliases); `Exchange.inject`
clones the state tensor and uses `index_copy_` on the clone, keeping autograd
intact. It raises with a pointer to the other adapters when the model has no
`"variable"` dim or the import is not a state channel.

The stacking adapters (`ConditioningKwargAdapter`, `ExtraTensorAdapter`)
require an **explicit `field_order=[...]`** whenever more than one field is
imported. Silently stacking in alphabetical order is forbidden by design:
models are channel-order-sensitive, and a permuted conditioning tensor *runs
without error and predicts garbage* — the worst failure mode in ML systems.
With a single import the order is trivially inferable and `field_order` may be
omitted.

### The pull pattern: masquerade, not argument

The fourth delivery shape exists because some models (StormCast is the
canonical case) offer *no* argument to deliver forcing through: the model
calls `fetch_data(self.conditioning_data_source, time, variables, ...)`
inside its own `__call__`, and the only injection point it exposes is that
settable data-source attribute. So `PullAdapter`'s job is a **masquerade
rather than an argument**: before each step it sets the attribute to a
`StateDataSource` — a tiny in-memory object satisfying the DataSource
protocol that answers the model's fetches from the component's import State —
then calls `model(x, coords)` unchanged. The model runs its unmodified
production fetch path (fetch → interpolate → concatenate) believing it is
reading GFS; it is reading the coupler. There is precedent for exactly this
masquerade in earth2studio's serve workflows: `stormcast_conus_workflow.py`
stages a full conditioning forecast to temp files and replays it through an
`InferenceOutputSource`; the shim is the same trick minus the staging — the
"source" is this step's live exchange.

The cadence-alignment contract: a `StateDataSource` is a snapshot view —
whatever the connector last delivered is what *every* requested time
receives. Alignment is the run sequence's job: a **sequential** connect
placed before the pulling component's run in the same slot
(`global`, then `global -> stormcast`, then `stormcast`) guarantees the
served fields are fresh at the pulled time. `strict_time=True` turns that
contract into a check, raising when the model pulls times that do not match
the served fields' `valid_time`.

Honest limitation: the pull path runs through the model's own
`fetch_data`/xarray machinery, so field data crosses a numpy boundary and the
autograd graph is severed there — pull-coupled components are
**inference-only** (no gradients through the exchange). The push-pattern
adapters above keep autograd intact.

Adapters (other than `PullAdapter`, per the boundary just described) must be
autograd-safe (no in-place mutation of tensors that may carry
grad); any object matching the `ImportAdapter` protocol can be passed as
`import_adapter=`.

## The Connector pipeline

`Connector(src, dst)` transfers the intersection of src's advertised exports
and dst's advertised imports, or an explicit `fields=[...]` (each of which
must appear in *both* lists, else `IncompatibleFieldError`). An empty match
raises. Units are checked against the dictionary at match time. Every
`execute(time)` runs each matched field through four stages, in this order:

### 1. Time policy

The connector keeps a **2-deep history** per field, `(previous, latest)`,
rotated only when a genuinely *new* export arrives (different `valid_time`) —
re-seeing the same export on repeated executes (a slow source polled by a fast
slot) must not collapse the extrapolation baseline.

- `"constant"` (default): deliver the latest export as-is — the destination
  holds the source's last state between updates (PhysicsNeMo
  `ConstantCoupler` behavior).
- `"linear"`: extrapolate from the two most recent exports toward the current
  time: `data + (data - prev) * dt_ahead / dt_hist`, restamping `valid_time`
  to `time`. Falls back to constant when there is no previous export, when
  either `valid_time` is missing, when history is non-increasing, when
  `dt_ahead == 0` — and, permanently with a one-time warning, for fields
  carrying a `lead_time`/`window` dimension (see the valid_time caveat above).

### 2. Vertical interpolation

Triggered only when the **destination** declares `import_vertical` for this
field *and* it differs from the field's `vertical` metadata. v1 supports
interpolation onto `PressureLevels` only (destination wanting anything else
raises `VerticalMismatchError`, as does a source field with no vertical
metadata). For `HybridLevels` sources (`p_k = a_k + b_k * p_s`) the connector
pulls the surface-pressure field named by `HybridLevels.ps_field` from the
source's exports automatically, raising with a concrete fix ("add it to the
source's export list") if absent. Interpolation is linear in log-pressure via
`torch.searchsorted` + gathers (differentiable in values), clamped at the
column ends. Models that encode levels in variable names (`z500`, `t850`)
never touch this stage.

### 3. Mask fill

Applies only when the field carries a mask and `fill != "none"`:

- `"zero"`: invalid points become 0.
- `"nearest"`: each invalid point takes its nearest valid neighbor
  (great-circle metric via a unit-sphere KDTree; pure gather, so
  differentiable) — the principled version of DLESyM's SST NaN-interpolation
  hack. The filler is cached per (grid signature, mask bytes). A mask with no
  valid points raises.

Fill runs **before** regridding by design: interpolating first would bleed
invalid (e.g. land) values into valid ocean points near the coast.

### 4. Spatial regrid

The selection ladder, evaluated per field:

1. **Identity fast path** — every spatial dim of the field exists in the
   destination grid with an equal coordinate array (works for lat/lon and for
   identical HEALPix `face/height/width` grids): pass through untouched. Only
   taken when no user regridder is set.
2. **Point target** — the destination advertises a `"point"` dim (its
   `points=` is a `PointSet`, a scattered set of sample locations rather than
   a mesh — stations, sites, arbitrary query coordinates; see
   [`points`](#points-a-scattered-sample-location-grid) below). Handled
   before the HEALPix guard since a point destination has no `face`/`lat`/
   `lon` mesh of its own to compare against.
3. **HEALPix guard** — a `face` dim on either side with *differing* grids and
   no user regridder raises `IncompatibleFieldError` pointing at
   `regridder=` (build one with `earth2grid`, as `models/px/dlesym.py` does).
4. **User regridder** — a `regridder=` callable on the connector overrides
   everything, including a point destination: it is applied to the trailing
   spatial dims of any layout, and the output coords are rebuilt from the
   destination grid. The contract:
   `tensor[..., *src_spatial] -> tensor[..., *dst_spatial]`, operating on the
   trailing spatial axes and preserving leading (batch/window/...) axes.
5. **Auto bilinear** — both grids must expose 1D `lat`/`lon`, the source must
   be *regular* (equally spaced), and the field's trailing two dims must be
   `(lat, lon)`; otherwise `IncompatibleFieldError` with the `regridder=`
   escape hatch. Uses `earth2studio.utils.interp.latlon_interpolation_regular`
   (edge clamping stands in for extrapolation), built lazily and cached per
   source grid signature.

A destination with no grid of its own (`grid_coords()` is `None` — mediators)
skips regridding entirely: fields pass through on the source grid, and
reduction happens there.

### `points`: a scattered sample-location grid

A component constructed with `points=PointSet(lat=..., lon=...)` targets N
arbitrary locations instead of a mesh — `grid_coords()` reports
`{"point": labels}` (`labels` is `PointSet.names` if given, else an integer
index) in place of whatever the component's own state coords happen to be.
Delivering to it needs `Connector(..., sample="nearest" | "bilinear")`:

- `"bilinear"` reuses the mesh regridder's kernel, reshaping the N
  destination points as a degenerate `[N, 1]` mesh — one bilinearly
  interpolated value per point. Same regularity requirement as auto bilinear
  above.
- `"nearest"` is a great-circle nearest-neighbor lookup via a unit-sphere
  KDTree (the same construction as the mask filler's nearest-fill), gathering
  one source grid cell per point.

Both are built lazily and cached per `(source grid signature, point set
signature, method)`. `sample=` and `regridder=` are mutually exclusive
(`CouplingError`); a point destination with neither set raises `CouplingError`
at `execute()` rather than silently picking one — as does a `"point"`-dim
destination with no `points=` registered on it (reachable if a component
hand-builds coords carrying a `"point"` key without going through `points=`),
and a source without a regular 1D lat/lon grid (`IncompatibleFieldError`).
This is the primitive downscaling-style applications need to sample a dense
forecast (or a static context raster) down to station or site coordinates.

After the pipeline the field lands in `dst.import_state`, and a copy of the
reference is kept in `connector.last_transfer` for `driver.probe("src->dst")`.

### Windowed connectors: window= and reduce=

Setting `window=` and `reduce=` **together** (either alone raises
`CouplingError`) turns the connector into a windowed reduction — the
preferred path for simple fast→slow coupling that needs "the trailing 48 h
mean" rather than the instantaneous field. The semantics:

- **Matching is by CellMethod, not name intersection.** Each source export
  `base` is paired with a destination import whose dictionary entry carries
  `CellMethod(base, reduce, window)`, and the delivered Field carries that
  *derived* standard name (`geopotential_at_1000hpa` in,
  `geopotential_at_1000hpa_48h_mean` out). A missing derived import raises —
  the coupler never invents names. `match()` returns both the consumed base
  names and the delivered derived names.
- **Every `execute` folds the source export into a running reduction**
  (`"mean" | "sum" | "max" | "min"`; one accumulator per field, duplicate
  `valid_time`s ignored) — the same accumulator core the mediators use.
- **Delivery happens only at execute times aligned to `window`**; mid-window
  the destination's previous import stands untouched. The alignment origin is
  the `valid_time` of the first execute's source field — under lagged
  coupling (connector before the source's run in the slot) that is the clock
  start, so the first delivery lands exactly one window after t0, with no
  driver hook.
- `time_policy` does not apply on the windowed path; the spatial pipeline
  (vertical → fill → regrid) still runs on each delivery.

```python
from earth2studio.nvcoupler.testing import fake_atmos, fake_ocean

conn = nvc.Connector(fake_atmos(), fake_ocean(), window="48h", reduce="mean")
conn.match()
# ['geopotential_at_1000hpa', 'geopotential_at_1000hpa_48h_mean']
```

A windowed connector replaces a single-source `AccumulationMediator` plus its
two connects and its `med.compute` slot action. Reach for a Mediator (below)
when several sources feed one reduction or the reduction needs custom code.

## Mediators

A `Mediator` is the multi-source, generalized form of the windowed-reduction
machinery — it shares the same running-accumulator core as windowed
connectors. It sits between cadences as its own participant: its import state
forwards every arriving field to `accumulate(field)`, and when its slot runs
the driver calls `compute(time)` (the `med.compute` action), which must
populate the export state.

`AccumulationMediator(name, [derived_names])` implements windowed reductions:

- **Running, O(1) memory.** Reductions are running torch ops — `add` for
  mean/sum, `torch.maximum`/`torch.minimum` — so memory is one accumulator
  per derived field regardless of window length. Mean divides by the sample
  count at compute time. Gradients flow through mean/sum; max/min propagate to
  the extremal sample.
- **One base, many derived.** Several derived fields may reduce the same base
  import (the 24 h max *and* 24 h mean of t2m accumulate from each delivered
  t2m field); the mediator imports each base once and fans deliveries out.
- **Duplicate dedup.** A field arriving with the same `valid_time` as the last
  accumulated one (two connectors feeding the mediator, or a re-executed slot)
  is ignored rather than double-counted.
- Compute with zero samples raises `CouplingError` ("is a connector feeding
  this mediator in a faster slot?"); after compute the accumulators clear, so
  each window is independent (trailing, non-overlapping).

`TrailingAverageMediator` is the mean-only restriction — the exact semantics
of DLESyM's ocean forcing and PhysicsNeMo's `TrailingAverageCoupler`; non-mean
fields raise at construction.

When one source feeds one destination, prefer the
[windowed connector](#windowed-connectors-window-and-reduce) — it produces
the same numbers from the same accumulator core with fewer moving parts.

## Coupling semantics: ordering is the coupling mode

There is no `lagged=True` flag on connectors. Whether coupling is lagged
(NUOPC-explicit) or sequential is **purely the position of the connect action
relative to the source's run in the same slot**: a connect executed before
the source runs delivers its previous export; after, the export just
produced. The driver executes a slot's actions strictly in order and adds no
hidden exchanges. (`derive_sequence`'s `lagged=` parameter is not a flag on
the exchange — it just selects where the connect is placed.)

A minimal pair, hand-checkable: `src` increments its `t2m` state by 1 each
step; `dst` copies its imported `t2m` into its `d2m` export.

```python
lagged = """
@6h
  src -> dst     # before dst runs: dst sees src's PREVIOUS export
  src
  dst
@
"""
sequential = """
@6h
  src
  src -> dst     # after src runs: dst sees the export just produced
  dst
@
"""
```

After one 6 h step from `t2m = 0`: the lagged system's `d2m` is **0.0** (the
t0 export), the sequential system's is **1.0** (the export produced in the
same step). Derived sequences (`Driver(sequence=None)`, `couple()`,
`derive_sequence(lagged="all")`) always generate the lagged shape — connects
precede runs — with one exception: mediator deliveries follow their
`med.compute` in the same slot and are therefore sequential. `describe()`
labels each connect `lagged` or `sequential` in its plan table (sequential
iff the source ran or computed earlier in the same slot). Run
[`examples/09_nvcoupler/02_lagged_vs_sequential.py`](../../../examples/09_nvcoupler/02_lagged_vs_sequential.py)
to see the divergence over a real rollout, and see the
[DSL reference](dsl_and_yaml_reference.md) for the full grammar.
