# Errors and troubleshooting

Every configuration error in nvcoupler derives from `CouplingError`
(`earth2studio/nvcoupler/errors.py`) and is designed to fire at
`Driver.initialize()` — not mid-rollout — naming the components, the field,
and the concrete fix. This page catalogs every exception class with its
actual raise sites, a representative message (quoted fragments are exact, so
you can grep for them), root causes, and fixes; a
[troubleshooting section](#troubleshooting-non-error-failure-modes) covers
the failure modes that are warnings or silent by design. Recipes for doing
things right the first time are in the [user guide](user_guide.md); the
class-by-class API is in the [API reference](api_reference.md).

The hierarchy: `UnknownFieldError`,
`UnmatchedImportError`, `UnitsMismatchError`, `IncompatibleFieldError`,
`VerticalMismatchError`, `CadenceError`, `AmbiguousCouplingError`, and
`SequenceError` all subclass `CouplingError`; a number of checks raise plain
`CouplingError` directly.

## UnknownFieldError

A name could not be resolved in the field dictionary.

Raised from:

- `FieldDictionary.resolve()` / `standard_name()` — any lookup of an
  unregistered name, including component `imports=`/`exports=` lists at
  construction time and `DiagnosticComponent`'s default import resolution.
- `FieldDictionary.add_alias()` — aliasing to a standard name that does not
  exist.
- `State.from_tensor(strict=True)` — a `variable` coordinate value the
  dictionary does not know.

Message:

```text
Field name 'sea_surface_temp' is not a registered standard name or alias.
Did you mean: 'sea_surface_temperature', 'surface_pressure'? Register it with
FieldDictionary.register(FieldEntry(...)) or add an alias with
FieldDictionary.add_alias(...).
```

Root causes: a typo (the did-you-mean list usually nails it), or a genuinely
new field. Fix: correct the spelling, or register a
`FieldEntry(standard_name, canonical_units, ...)` — for model-vocabulary
names, prefer `variable_aliases={"raw": "standard_name"}` on the component,
which registers the alias for you. Lookups are case-sensitive.

## UnmatchedImportError

A component advertises an import that nothing delivers. Two distinct raise
sites with the same message shape:

1. **`couple()` auto-wiring** (`api._synthesize_mediators`): no component
   exports the imported field, and it is not a derived (CellMethod) entry
   whose base field someone exports.
2. **`Driver._check_unfed_imports` at `initialize()`**: components *do*
   export the field, but no connector in *your* run sequence delivers it —
   you forgot a `src -> dst` line. Without this check the component would
   silently run the whole simulation on its stale initial-condition forcing.

Message (case 2 — note the exporter is listed, which tells you the connect
line to add):

```text
Component 'atmos' imports 'sea_surface_temperature' but no component exports
it. Did you mean: 'sea_surface_temperature'? Available exports: atmos exports
geopotential_at_1000hpa; ocean exports sea_surface_temperature; med exports
geopotential_at_1000hpa_48h_mean. Add an alias, a Mediator producing the
derived field, or a DataComponent supplying it from a data source.
```

Fixes, in order of likelihood: add the missing `ocean -> atmos` line to the
run sequence; add an alias so the exporter's name resolves to the same
standard name; register a CellMethod entry (so `couple()` can synthesize a
mediator) or add one explicitly; supply the field from a `DataComponent`.
Deliberately unfed imports are opt-in via
`Driver(..., allow_unfed_imports=True)`, which downgrades case 2 to a
warning (grep `no connector in the run sequence delivers it`).

## UnitsMismatchError

Raised from `FieldDictionary.check_units()`, called by `Connector.match()`
for every matched field — so it fires at `initialize()`.

```text
Field 'sea_surface_temperature': 'ocean' exports units 'degC' but 'atmos'
expects 'K'. Unit conversion is not performed in v1 — convert in a Mediator
or align the FieldDictionary entries.
```

Root cause: the two components' dictionaries carry different
`canonical_units` for the same standard name (units are normalized before
comparison — `m s**-1`, `m s^-1`, and `M/S` compare equal, as do
`degC`/`celsius` and `(0-1)`/`dimensionless`). nvcoupler checks units, never
converts values. Fix: make the entries agree, or insert a converting
Mediator. This is an honest v1 limitation (no pint).

## IncompatibleFieldError

A connector could not reconcile matched fields. Raise sites in
`connector.py`:

- `match()` — the explicit `fields=[...]` list contains names not in both
  endpoints: `fields ['air_temperature_2m'] are not in both 'ocean' exports
  (...) and 'atmos' imports (...)`.
- `match()` — nothing matches at all: `Connector atmos->atmos: no fields
  match — 'atmos' exports [...], 'atmos' imports [...]`. Usually an alias or
  advertisement problem, or a connector between the wrong pair.
- `_build_latlon_regridder` — `Auto regrid requires a regular 1D source
  lat/lon grid; pass a custom regridder=...` for curvilinear/unstructured
  sources.
- `_apply_regrid` — `source and destination HEALPix 'face' grids differ —
  pass a custom regridder=` (identical face grids pass through as identity;
  differing ones need e.g. an `earth2grid`-built callable, see
  `models/px/dlesym.py`).
- `_apply_regrid` — `auto regrid needs lat/lon on both grids` when either
  side lacks lat/lon spatial dims, and `must have (lat, lon) as trailing
  dims` when the field's dim order puts something after lon.
- `_build_mask_filler` — `Mask fill impossible: no valid source points`
  (`fill="nearest"` with an all-False mask).

Fixes: pass `regridder=` on the Connector for anything the bilinear
regular-lat/lon kernel cannot handle; reorder dims so lat/lon trail; check
the fields list against `src.advertise()` / `dst.advertise()`.

## VerticalMismatchError

Source and destination vertical coordinates cannot be reconciled. Only
components declaring `import_vertical`/`export_vertical` (fields with a
`level` dim) ever see these. Raise sites:

In `connector._apply_vertical`:

- destination declares `import_vertical` but the incoming field has no
  `vertical` metadata: `'chem' expects 'ozone_mixing_ratio' on
  PressureLevels(...), but the source field has no vertical coordinate` —
  add `export_vertical={...}` on the source.
- destination wants something other than pressure levels: `only
  interpolation onto PressureLevels is supported in v1`.
- hybrid source without surface pressure in the source's exports:
  `hybrid->pressure interpolation of 'ozone_mixing_ratio' needs
  'surface_pressure' in 'met' exports — add it to the source's export list`.

In `vertical.py` (`interp_to_pressure` / `_log_source_pressure`):

- `coords have no 'level' dim` — you called the interpolation on a field
  without a level axis.
- data `level` coordinate does not match the declared `PressureLevels`
  source: `does not match the declared PressureLevels source ... Reorder the
  data so levels increase top to bottom, or fix the source component's
  export_vertical declaration`.
- `'level' coord length N != tensor level size M`.
- `Non-positive pressure from hybrid coefficients`, and `Hybrid levels
  a + b * ps are not strictly increasing along the level axis` — unphysical
  ps values or misordered coefficients at interpolation time.

Related `ValueError`s at construction: `PressureLevels` rejects
non-increasing levels; `HybridLevels` rejects `a`/`b` of unequal length and
coefficients that produce `non-increasing pressures` anywhere in the
plausible surface-pressure range [50000, 110000] Pa. Order everything top to
bottom (increasing pressure).

## CadenceError

A time interval does not divide cleanly. All three raise sites share the
suffix `is not a positive multiple of the driver clock dt ... Choose a
driver dt that divides every component timestep (typically their GCD).`

- `Component.realize()` — component timestep vs clock dt (a 5 h component on
  a 6 h clock).
- `RunSequence.validate()` — a slot interval vs clock dt, and a component or
  mediator scheduled in a slot that does not equal its own timestep:
  `Component 'atmos' (timestep 21600000000000 nanoseconds) scheduled in a
  slot of interval 43200000000000 nanoseconds ...`.
- `Clock.__init__` — `Clock span (stop - start)` not a multiple of dt.

Fix: pick a driver dt that divides every component timestep (their GCD —
what `couple()` does by default), put each component's `RunAction` in the
slot matching its exact timestep, and make `stop - start` a whole number of
dt steps. Related `ValueError`s: `Clock dt must be positive`, `Clock stop ...
must be after start`, and — from
`as_timedelta` — `Bare number 6 is ambiguous as a timedelta (hours? steps?) —
pass a string like '6h' or '2D', or a np.timedelta64`.

## AmbiguousCouplingError

Only `couple()` raises this (`api._synthesize_mediators`), in two spots: an
imported field is exported by more than one component, or a derived field's
*base* has multiple exporters.

```text
Import 'sea_surface_temperature' of component 'atmos' is exported by
multiple components: ocean, ocean2. Auto-wiring cannot choose — build the
Driver explicitly with Connector(src, dst, fields=[...]) or a run-sequence
DSL.
```

Fix: exactly what it says — auto-wiring refuses to guess, so hand-build the
Driver with explicit connectors and a DSL when several components export the
same field.

## SequenceError

The run sequence references unknown names or is malformed. Raise sites:

`RunSequence.validate()` (fires inside `Driver.initialize`):

- `Run sequence references unknown component 'atmso'. Did you mean:
  'atmos'? Known components: [...]` — the DSL name does not match a key of
  the `components` dict (also raised for connector endpoints and mediators).
- `Components never run by the sequence: ['ocean'] — add a RunAction (bare
  component name) to a slot matching their timestep` — every component,
  mediators included, must run somewhere.

`parse_run_sequence()`:

- `Line 1: action 'atmos' outside any @interval slot` — actions must follow
  an `@6h`-style header.
- `Line 2: cannot parse 'atmos ->' — expected 'name', 'src -> dst', or
  'mediator.compute'`.
- `Line N: Cannot parse timedelta ...` — a bad `@interval` header.
- `Run sequence is empty` — no slots at all (comments only, or a stray
  string).

## CouplingError (direct raises)

The base class is also raised directly for lifecycle and shape problems.
The complete set, grouped:

**Driver lifecycle** (`driver.py`):

- `Component 'atmos' needs an initial condition — pass ics={'atmos': (x,
  coords)}` — every `requires_ic` component (Prognostic, Callable) needs an
  ics entry; mediators, DataComponents, and DiagnosticComponents do not.
- `Driver.initialize(ics) must be called before running` — also after
  `reset()`, which deliberately invalidates initialization.
- `Driver clock exhausted: already at stop time ... Call driver.reset() and
  driver.initialize(ics) to run again` — `run()`/`steps()`/`rollout()` on a
  finished driver.
- `rollout(5) ran past the clock stop time ...: only 4 steps remained. Use
  n_steps <= clock.n_steps (4) ...` — mid-rollout exhaustion.
- `io= keys ['atmso'] are not component names; known components: [...]`.
- `Component 'x' export '...' carries a 'time' dimension of size 2; the
  driver records one snapshot per ring and can only absorb a size-1 'time'
  dim — publish a single time-slice per run` — recording/IO cannot absorb
  multi-time exports.

**Connector ordering** (`connector.py`):

- `Connector med->ocean: 'med' has not produced
  'geopotential_at_1000hpa_48h_mean' yet — check the run sequence ordering`
  — a connect scheduled before its source ever ran/computed (classic:
  `med -> ocean` placed before `med.compute`).

**Windowed connectors** (`connector.py`):

- `Connector atmos->ocean: window= and reduce= must be set together — a
  windowed reduction needs both the window length and the reduction method`
  — one of the two was passed alone.
- `... unsupported reduce='median'; choose 'mean', 'sum', 'max' or 'min'`.
- `Connector ocean->atmos: window='2D'/reduce='mean' is set but 'atmos'
  imports no derived field for [...] — register a
  FieldEntry(cell_method=CellMethod(base, 'mean', window='2D')) in the
  destination's dictionary and add its standard name to 'atmos''s imports`
  — the destination has no dictionary entry deriving from the source export
  with that exact method and window (a window mismatch, e.g. 24h vs the
  entry's 48h, fails the same way); the coupler never invents derived names.

**Import adapters** (`component.py`):

- `VariableOverwriteAdapter requires a 'variable' dim in the model state
  coords; use ConditioningKwargAdapter or ExtraTensorAdapter ...`.
- `Imported field 'air_temperature_2m' (model name '...') is not a state
  variable of the model (variables: [...]). If the model takes forcing as a
  conditioning kwarg or an extra tensor, pass the matching ImportAdapter.`
- `ExtraTensorAdapter: 2 imported fields but no field_order= given — the
  model's channel order cannot be inferred. Pass field_order=[...]` — see
  [troubleshooting](#forgotten-field_order) below.
- `... field_order names [...] are not in the import state (present: ...)`.
- `Imported field with shape ... has more dims than the model state slice`.

**Component phases** (`component.py`):

- `Component 'atmos' not initialized` — `run()` before `initialize()`;
  `Component 'atmos' not realized` — `should_run()` before `realize()` (the
  Driver sequences these for you; you only hit them driving components by
  hand).
- `Component 'c' advertises export 'sea_surface_temperature' but its output
  variables are [...]` — the model output has no variable resolving to an
  advertised export; check exports/aliases against the model's actual output
  coords.
- `Component 'prog': model outputs 1 lead times but takes 2 as input —
  supply a next_input hook to manage the sliding input window`.
- `DataComponent 'ocean' cannot fetch initial data before realize(clock)`.
- `DiagnosticComponent 'diag' is missing imports [...] at ... — check the
  run sequence connects its source before this component runs`.

**Mediators** (`mediator.py`):

- `AccumulationMediator 'med': 'sea_surface_temperature' has no cell_method
  in the field dictionary — register a
  FieldEntry(cell_method=CellMethod(base, method, window)) ...`.
- `... fields have differing windows [...]; split them across mediators or
  pass window= explicitly`.
- `TrailingAverageMediator 'med': fields [...] are not mean reductions — use
  AccumulationMediator`.
- `Mediator 'med': no samples of 'geopotential_at_1000hpa' accumulated
  before compute at ... — is a connector feeding this mediator in a faster
  slot?` — the `atmos -> med` connect must live in a slot faster than
  `med.compute`.

**Field/State invariants** (`field.py`): a `Field` must not carry a
`variable` dim (`use State.from_tensor to split`), data dims must match its
coords, `State` keys must equal the field's standard name, and
`State.as_tensor` with no fields raises. A missing key raises `KeyError:
State '...' has no field '...'; present: [...]`; `Driver.probe` with an
unknown name raises `KeyError: No connector '...'; have [...]`.

**YAML** (`config.py`): `to_yaml` raises `Component 'atmos'
(CallableComponent) is not serializable ... Set a yaml_spec attribute on the
component — a dict {'class': ..., 'kwargs': {...}} ...`; `from_yaml` raises
for missing top-level keys (`YAML config is missing required keys`), bad
import paths (`Cannot import module ...`, `... has no attribute ...`),
malformed component specs, connector endpoints that are not configured
components, and wraps any constructor failure as
`Component 'x': <class>(**kwargs) failed: ...`.

**DLESyM split** (`dlesym_split.py`): layout/window checks on the real-model
split — `expects the DLESyM input layout [...]`, `initial condition
lead_time window ... does not match DLESyM full_input_times`, `atmos output
times (N) do not chunk evenly into M ocean windows`, and the ocean run
needing its imports first (`schedule the atmos component and the atmos ->
ocean connector earlier in the same slot`). An honest caveat: the DLESyM
real-weights equivalence gate (`test_dlesym_weights_equivalence.py`) is
skipped unless `NVCOUPLER_DLESYM_WEIGHTS=1` is set with physicsnemo and the
checkpoints available, and has not been run here — the split is verified
against mock-model tests only.

## Troubleshooting: non-error failure modes

Things that used to fail silently and are now loud, plus the ones that remain
warnings or intentional behavior.

### Forgotten field_order

Multi-import `ConditioningKwargAdapter`/`ExtraTensorAdapter` **now raise**
instead of stacking alphabetically: silently permuted channels run fine and
predict garbage, which is the worst failure mode in ML coupling. With one
import no order is needed; with more, pass
`field_order=["field_a", "field_b", ...]` matching the model's channel order.

### Stale-IC forcing

Forgetting a connect line used to mean the destination ran the whole
simulation on its t0 import — plausible-looking, subtly wrong output. This
**now raises `UnmatchedImportError` at initialize** (see above). If a
free-running component is what you want, pass
`Driver(..., allow_unfed_imports=True)` and you get the explicit warning
`... it will run on stale initial-condition forcing` instead.

### Linear time policy that looks constant

`time_policy="linear"` extrapolates from the two most recent *distinct*
exports. A historical bug where repeated executes between source updates
collapsed the (prev, latest) baseline — degrading linear to constant after
the first step — is fixed: the history rotates only when a genuinely new
export (different `valid_time`) arrives, so the slope holds across every
intermediate step (see
`test_connector.py::test_time_policy_linear_holds_slope_across_repeated_executes`).
Two semantics still surprise people:

- The **first** transfer has one export in history and falls back to
  constant — extrapolation needs two points.
- Fields carrying a `lead_time` or `window` dimension have no single valid
  time, so linear is undefined for them; the connector warns once (grep
  `time_policy='linear' is undefined for field`) and holds constant.

### NaN in outputs

Two unrelated causes, both intentional:

1. **Unwritten IO rows.** Streamed backend arrays are NaN-initialized so
   never-written rows cannot masquerade as physical values (zarr's default
   fill reads back as 0.0). A mediator's t0 row is always NaN (it exports
   nothing before its first compute), as is everything after the point where
   a run crashed. NaN at exact ring boundaries of a slow component = normal;
   NaN spreading through a field mid-run = look upstream.
2. **Mask fill not set.** If a source's data is NaN over masked-out points
   (real SST products are NaN over land) and the connector has the default
   `fill="none"`, those NaNs bleed through the bilinear regrid into coastal
   destination cells. Set `fill="nearest"` (or `"zero"`) on every connector
   leaving a masked source.

### Exhausted clock

A Driver runs its clock exactly once. A second `run()`, `steps()`, or
`rollout()` raises `Driver clock exhausted` rather than silently yielding
nothing (the old generator behavior). The rerun recipe is `driver.reset()`
then `driver.initialize(ics)` — reset clears records, connector history, and
IO-ready state, and deliberately invalidates initialization so you cannot
resume from stale component state by accident.

### Duplicate mediator deliveries

Wiring the same field into a mediator twice per step (two connectors, or a
re-executed slot) does **not** double-count: `AccumulationMediator` ignores
arrivals whose `valid_time` it has already accumulated for that derived
field. This is silent by design. Corollary: if your source republishes with
an *unchanged* `valid_time`, the new values are ignored too — publish with
the current ring time.

### Warnings worth not ignoring

- `Component 'x' exports [...] but no connector consumes them` — harmless if
  the export is only for IO/records; a wiring bug otherwise.
- `In-memory collection will hold ~N GB of export fields; pass collect=False
  and a real IO backend (e.g. ZarrBackend) for runs of this size` — the
  memory guard at initialize; see [IO and outputs](user_guide.md#io-and-outputs).
- HEALPix/curvilinear sources: auto-regrid does not support them and raises
  (`pass a custom regridder=`) — this is a v1 limitation, not a bug; see
  [design and roadmap](design_and_roadmap.md).
