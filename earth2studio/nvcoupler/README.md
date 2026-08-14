# nvcoupler

A NUOPC/ESMF-inspired, Python-native coupling framework for AI Earth-system
inference — and, by design, for future coupled fine-tuning.

```python
import earth2studio.nvcoupler as nvc
```

## Why

AI Earth-system models are coupled today in three ad-hoc ways: baked into a
datapipe (PhysicsNeMo `ConstantCoupler` / `TrailingAverageCoupler`),
hard-coded inside one model's forward pass (DLESyM's atmos→ocean exchange),
or left entirely to the caller (StormScope `call_with_conditioning`).
Changing how DLESyM couples means forking `dlesym.py`; swapping its ocean for
observed SST means a rewrite.

nvcoupler factors coupling out of the models. Independent **Components**
exchange **Fields** by standard name through **Connectors** (regridding,
masking, time policies) and **Mediators** (windowed reductions), scheduled by
a **Driver** executing a NUOPC-style **run sequence** on a shared **Clock**.
Swapping a modeled ocean for prescribed observations becomes a one-line
change; a coupling-order experiment becomes a one-line DSL edit.

## Concepts (NUOPC → nvcoupler)

| NUOPC / ESMF        | nvcoupler                | Module          |
|---------------------|--------------------------|-----------------|
| ESMF_Field / State  | `Field`, `State`         | `field.py`      |
| Field dictionary    | `FieldDictionary`        | `dictionary.py` |
| ESMF Clock / Alarm  | `Clock`, `Alarm`         | `clock.py`      |
| NUOPC_Model         | `Component` subclasses   | `component.py`  |
| NUOPC_Connector     | `Connector`              | `connector.py`  |
| NUOPC_Mediator      | `Mediator` subclasses    | `mediator.py`   |
| NUOPC_Driver, runSeq| `Driver`, `RunSequence`  | `driver.py`, `sequence.py` |

### Field & State

A `Field` is one physical quantity: a **torch tensor** plus its earth2studio
`CoordSystem`, a canonical identity (`standard_name`, `units`), a
`valid_time`, and optional `mask` (True = valid, e.g. ocean points for SST)
and `vertical` metadata. A `State` is a bag of Fields keyed by standard name;
every component owns an import State and an export State.

Field data never round-trips through numpy inside the framework, so autograd
graphs survive every exchange.

### FieldDictionary

Connectors match fields by **standard name**, never raw model variable
strings. Aliases map model vocabularies onto standard names
(`z1000 → geopotential_at_1000hpa`); units are checked (not converted) on
every match. Derived fields carry a machine-readable `CellMethod`:

```python
nvc.FieldEntry(
    "total_precipitation_48h_sum", "kg m-2",
    cell_method=nvc.CellMethod("total_precipitation_6h", "sum", np.timedelta64(48, "h")),
)
```

which is how mediators (and future auto-wiring) know that a 48 h precip sum
derives from the 6 h precip field — no suffix string-parsing.

### Components

All components implement the NUOPC phases
`advertise → realize(clock) → initialize(x, coords) → run(time)* → finalize`
and declare their imports/exports as standard names.

- **`PrognosticComponent`** wraps any earth2studio `PrognosticModel`.
  Timestep and exports are inferred from `input_coords()`/`output_coords()`.
- **`CallableComponent`** wraps a plain `fn(x, coords) -> (x, coords)` — the
  entry point for synthetic components **and non-ML models** (a process-based
  hydrology or crop model can join the coupled system through it).

**ImportAdapters** — the critical seam. Real models receive coupled fields in
different call shapes, so the adapter owns the model invocation:

| Adapter | Call shape | Real-world pattern |
|---|---|---|
| `VariableOverwriteAdapter` (default) | overwrite state-variable slices, `model(x, coords)` | prescribed forcing as a state channel |
| `ConditioningKwargAdapter` | `model.call_with_conditioning(x, coords, conditioning=...)` | StormScope |
| `ExtraTensorAdapter` | `model(x, coords, coupling)` | DLESyM / PhysicsNeMo 4-tensor |

Models that manage a sliding input window internally take a
`next_input(prev_x, prev_coords, out, out_coords)` hook; the default handles
single-window models.

### Connector

`Connector(src, dst)` transfers the intersection of src's exports and dst's
imports (or an explicit `fields=[...]`). Each transfer runs a pipeline:

1. **Time policy** — `"constant"` holds the latest export (the PhysicsNeMo
   `ConstantCoupler` behavior); `"linear"` extrapolates from the two most
   recent exports toward the current time.
2. **Vertical** — when the destination declares `import_vertical` and it
   differs from the field's coordinate, hybrid→pressure interpolation runs
   (see below).
3. **Mask fill** — `"zero"`, or `"nearest"` (each invalid point takes its
   nearest valid neighbor via a cached KDTree — the principled version of
   DLESyM's SST NaN-interpolation hack). Always applied *before* regridding
   so invalid values can't bleed into the interpolation.
4. **Spatial regrid** — lazily built and cached per grid pair: identity when
   grids match, else bilinear via `earth2studio.utils.interp.
   latlon_interpolation_regular` (regular 1D lat/lon sources). HEALPix
   (`face` dim) and curvilinear sources need a user `regridder=` callable
   (build one with `earth2grid` as `models/px/dlesym.py` does).

### Mediator

`AccumulationMediator("med", ["geopotential_at_1000hpa_48h_mean"])` reads the
CellMethod off each derived field: it imports the base field, accumulates a
**running** reduction (mean/sum/max/min — O(1) memory in window length) on
every connector delivery, and exports the reduced field when its alarm rings.
`TrailingAverageMediator` is the mean-only restriction matching DLESyM's
ocean forcing exactly. Duplicate deliveries (same `valid_time`) are ignored.

### Vertical coordinates

For components with an explicit `level` dimension (chiefly chemistry
emulators on hybrid sigma-pressure levels):

```python
hybrid   = nvc.HybridLevels(a=(30000., 20000., 0.), b=(0., 0.5, 1.0))  # p = a + b·ps
pressure = nvc.PressureLevels((500., 850.))                            # hPa

met  = CallableComponent(..., export_vertical={"ozone_mixing_ratio": hybrid})
chem = CallableComponent(..., import_vertical={"ozone_mixing_ratio": pressure})
```

The connector interpolates linearly in log-pressure (differentiable), pulling
`surface_pressure` from the source's exports automatically and raising
`VerticalMismatchError` with a concrete fix when it can't. Models that encode
levels in variable names (`z500`, `t850`) never touch this machinery.

### Run sequence & Driver

The DSL mirrors NUOPC's runSeq. **Coupling semantics are pure ordering**: a
connect placed before the destination's run in the same slot is lagged
(NUOPC-explicit) coupling; placed after, sequential.

```
@6h
  atmos -> med          # accumulate atmos fields into the mediator
  ocean -> atmos        # lagged: atmos sees the ocean's previous state
  atmos
@48h
  med.compute
  med -> ocean          # 48h-averaged forcing
  ocean
@
```

```python
driver = nvc.Driver(
    {"atmos": atmos, "ocean": ocean, "med": med},
    sequence=dsl_text,                    # or a RunSequence object
    clock=nvc.Clock("2024-01-01", "2024-03-01", "6h"),
)
driver.initialize({"atmos": (x_a, coords_a), "ocean": (x_o, coords_o)})

datasets = driver.run()                   # dict[str, xr.Dataset]
for time, states in driver.steps():       # or notebook-style iteration
    ...
driver.probe("ocean->atmos")              # last exchanged fields
```

Validation is front-loaded: unknown names (with did-you-mean suggestions),
cadence misalignment, unmatched imports, unit mismatches, and unconsumed
exports are all reported at `initialize`, not mid-rollout.

## Training / coupled fine-tuning

The exchange path is autograd-clean end to end (regrid gathers, mediator
reductions, functional import injection). `driver.rollout(n_steps)` keeps
the graph when gradients are enabled:

```python
with torch.enable_grad():
    states = driver.rollout(16)
loss = criterion(states["atmos"]["geopotential_at_1000hpa"].data, target)
loss.backward()        # gradients reach parameters of BOTH components
```

This enables jointly fine-tuning coupled emulators so each learns to
tolerate the other's imperfect output — the standard remedy for coupled
drift. Optimizer loops, truncated BPTT, and per-component GPU placement are
intentionally out of scope for v1. `run()`/`steps()` execute under
`torch.inference_mode()`.

## Errors

All configuration errors derive from `CouplingError` and name the components,
the field, and the concrete fix: `UnknownFieldError`, `UnmatchedImportError`,
`UnitsMismatchError`, `IncompatibleFieldError`, `VerticalMismatchError`,
`CadenceError`, `SequenceError`, `AmbiguousCouplingError`.

## Current limitations (v1)

- HEALPix / curvilinear source grids need a user-supplied `regridder=`.
- Units are checked, not converted (no pint); convert in a Mediator.
- No checkpoint/restart, coupled ensembles, or concurrent slot execution.
- `Driver` IO is in-memory xarray (`collect=True`); direct `IOBackend`
  streaming is planned alongside the `couple()` auto-wiring layer.

## Examples

See `examples/09_nvcoupler/`:

1. `01_coupled_toy_workflow.py` — the full atmos⇄ocean loop on synthetic components
2. `02_lagged_vs_sequential.py` — coupling order as a one-line experiment
3. `03_impact_chain.py` — precip-sum / t2m-max mediators feeding an impact index
4. `04_vertical_chemistry.py` — hybrid→pressure coupling with auto ps dependency
5. `05_coupled_finetuning.py` — gradients across the exchange + a training step

Tests (`test/nvcoupler/`) double as executable specification, including a
fully hand-computed 96 h coupled run in `test_driver.py`.
