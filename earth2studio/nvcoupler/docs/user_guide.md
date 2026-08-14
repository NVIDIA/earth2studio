# User guide

Task-oriented recipes for building coupled systems with nvcoupler. Every
snippet on this page runs as-is against the toy components in
`earth2studio.nvcoupler.testing` — no weights, no network. For the mental
model behind Fields, Connectors, and the run sequence see
[concepts](concepts.md); for the full DSL and YAML grammar see the
[DSL and YAML reference](dsl_and_yaml_reference.md); for symbol-by-symbol
docs see the [API reference](api_reference.md). When something fails, start
at [errors and troubleshooting](errors_and_troubleshooting.md).

## Quickstart: couple two components in ten lines

`couple()` auto-wires components by standard name, synthesizes any mediator a
derived import needs, and returns a ready-to-initialize `Driver`:

```python
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

driver = nvc.couple(
    fake_atmos(), fake_ocean(), start="2024-01-01", stop="2024-01-05"
)
print(driver.describe())               # the coupling plan, before anything runs
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
datasets = driver.run()                # dict[str, xarray.Dataset]
print(datasets["atmos"]["geopotential_at_1000hpa"].shape)   # (17, 32, 64)
```

`describe()` prints a terraform-plan-style summary: one table row per
component (type, cadence, imports, exports), one per connector (fields, time
policy, fill, lagged/sequential mode, slot), then the generated run sequence.
Here the ocean imports `geopotential_at_1000hpa_48h_mean` — a derived field
nobody exports — so `couple()` synthesized an `AccumulationMediator`
(`med_z1000-48H`) between the 6 h and 48 h cadences. `run()` executes to the
clock's stop time under `torch.inference_mode()` and returns one
`xarray.Dataset` per component; the time axis is each component's own ring
times including t0, so atmos has 17 rows and the 48 h ocean has 3.

**Gotcha:** `couple()` uses lagged (NUOPC-explicit) coupling — every connect
precedes the runs in its slot, so each destination sees the source's
*previous* export. That is the reproducible default, not the only choice; see
the next section. The full walkthrough is
[example 01](../../../examples/09_nvcoupler/01_coupled_toy_workflow.py).

## Hand-built systems: Driver + DSL for ordering control

When the coupling *order* is the experiment, skip `couple()` and write the
run sequence yourself. Coupling semantics are pure ordering: a `src -> dst`
line before `dst`'s run in the same slot is lagged; after it, sequential.

```python
import numpy as np
from earth2studio.nvcoupler import Clock, Driver, TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

LAGGED = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  ocean -> atmos
  med -> ocean
  ocean
@
"""

SEQUENTIAL = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  med -> ocean
  ocean
  ocean -> atmos
@
"""

def build(dsl):
    components = {
        "atmos": fake_atmos(),
        "ocean": fake_ocean(),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }
    driver = Driver(components, dsl, Clock("2024-01-01", "2024-01-05", "6h"))
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    return driver

z_lag = build(LAGGED).run()["atmos"]["geopotential_at_1000hpa"].values[-1]
z_seq = build(SEQUENTIAL).run()["atmos"]["geopotential_at_1000hpa"].values[-1]
assert np.allclose(z_lag, 19.2, atol=1e-4)      # forced by the previous SST
assert np.allclose(z_seq, 19.2336, atol=1e-4)   # forced by the fresh SST
```

The only difference between the two DSLs is where `ocean -> atmos` sits
relative to `ocean`'s run — a one-line experiment
([example 02](../../../examples/09_nvcoupler/02_lagged_vs_sequential.py)).

**Gotcha:** a mediator's exports only exist after `med.compute`, so
`med -> ocean` must come after the `MediateAction` in the slot, or the
connector raises `CouplingError: ... has not produced ... yet`. Validation of
names, cadences, and unfed imports all happens at `initialize()`, not
mid-rollout.

## Wrapping your model as a PrognosticComponent

`PrognosticComponent` wraps any earth2studio `PrognosticModel`
(`models/px/base.py` interface). The contract it reads off the model:

- `input_coords()` / `output_coords(input_coords)` must return earth2studio
  `CoordSystem`s with a `"variable"` coordinate.
- **Timestep inference:** `timestep` defaults to
  `output_coords["lead_time"][-1] - input_coords["lead_time"][-1]`. Pass
  `timestep=` explicitly if your model has no `lead_time` axis.
- **Export inference:** with `exports=None`, every output variable whose raw
  name resolves through the field dictionary (or through your
  `variable_aliases`) becomes an export; unknown variables are silently
  skipped, so add aliases for anything you want exchanged.

```python
from collections import OrderedDict

import numpy as np

from earth2studio.nvcoupler import PrognosticComponent
from earth2studio.nvcoupler.testing import grid_coords


class MyModel:
    """Stands in for any earth2studio PrognosticModel."""

    def input_coords(self):
        return OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([np.datetime64("2024-01-01")]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["z1000", "sst"]),
                **grid_coords(8, 16),
            }
        )

    def output_coords(self, input_coords):
        out = OrderedDict({k: v.copy() for k, v in input_coords.items()})
        out["lead_time"] = input_coords["lead_time"] + np.timedelta64(6, "h")
        return out

    def __call__(self, x, coords):
        return x + 1.0, self.output_coords(coords)


atmos = PrognosticComponent(
    "atmos",
    MyModel(),
    imports=["sea_surface_temperature"],   # sst is also a state channel
)
# timestep inferred as 6h; exports inferred as both dictionary-known variables
imports, exports = atmos.advertise()
assert exports == ["geopotential_at_1000hpa", "sea_surface_temperature"]
```

`z1000` and `sst` resolve because they are registered aliases in the default
dictionary. For model vocabularies the dictionary does not know, map raw
names to standard names with `variable_aliases={"raw_name": "standard_name"}`
— the alias is also registered so exports resolve.

Published exports are *exchange-shaped*: size-1 `batch`/`time`/`lead_time`
dims are squeezed off the exported Fields (the internal model state keeps
them), so a `(1, 1, 1, var, lat, lon)` model couples cleanly to a plain
`(var, lat, lon)` one.

### Choosing an ImportAdapter

The adapter owns the model call, because real models disagree on how coupled
forcing arrives:

| Your model receives forcing as... | Adapter | Call shape |
|---|---|---|
| a state variable you overwrite before stepping | `VariableOverwriteAdapter` (default) | `model(x, coords)` after injecting import slices |
| a conditioning kwarg (StormScope) | `ConditioningKwargAdapter` | `model.call_with_conditioning(x, coords, conditioning=..., conditioning_coords=...)` |
| an extra positional/keyword tensor (DLESyM, PhysicsNeMo 4-tensor) | `ExtraTensorAdapter` | `model(x, coords, coupling)` or `model(x, coords, **{kwarg: coupling})` |

The default only works when every imported field is *also* a variable of the
model state (`"variable"` must be in the state coords); otherwise it raises
with a pointer to the other two adapters. **Gotcha:** with more than one
import, `ConditioningKwargAdapter` and `ExtraTensorAdapter` require
`field_order=[...]` — channel order cannot be inferred, and stacking
alphabetically would run fine and predict garbage, so the framework refuses
to guess.

### Multi-window models: the next_input hook

The default next-step input reuses the model output with the input
`lead_time` coordinates — correct only when the model outputs as many lead
times as it takes in. A model that consumes a sliding window (2 inputs, 1
output) raises
`CouplingError: model outputs 1 lead times but takes 2 as input — supply a
next_input hook`. Provide
`next_input=lambda prev_x, prev_coords, out, out_coords: (next_x, next_coords)`
to roll the window yourself; `earth2studio/nvcoupler/dlesym_split.py` is the
worked real-model example.

### requires_ic

`PrognosticComponent` and `CallableComponent` have `requires_ic = True`:
`Driver.initialize(ics)` demands an `(x, coords)` entry for each, and fails
with the exact `ics={...}` line to add. Mediators, `DataComponent`, and
`DiagnosticComponent` set `requires_ic = False` and can be initialized with
no arguments.

## Prescribed forcing: DataComponent

Swapping a modeled ocean for observed SST is the classic AMIP-style
experiment. A `DataComponent` fetches from any earth2studio `DataSource` at
its own cadence and publishes the results as exports — the atmos, connectors,
and sequence are untouched:

```python
import numpy as np
import xarray as xr

from earth2studio.nvcoupler import Clock, DataComponent, Driver
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, grid_coords


class ConstantSST:
    """Minimal DataSource: __call__(time, variable) -> xr.DataArray with
    dims (time, variable, lat, lon)."""

    grid = grid_coords(16, 32)

    def __call__(self, time, variable) -> xr.DataArray:
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = np.atleast_1d(np.asarray(variable))
        lat, lon = self.grid["lat"], self.grid["lon"]
        data = np.full((len(time), len(variable), len(lat), len(lon)), 3.0)
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": variable, "lat": lat, "lon": lon},
        )


ocean = DataComponent(
    "ocean",
    source=ConstantSST(),
    exports=["sea_surface_temperature"],
    timestep="24h",
)
dsl = """
@6h
  ocean -> atmos
  atmos
@24h
  ocean
@
"""
driver = Driver(
    {"atmos": fake_atmos(), "ocean": ocean},
    dsl,
    Clock("2024-01-01", "2024-01-03", "6h"),
)
driver.initialize({"atmos": atmos_ic()})   # no ocean entry: requires_ic=False
ds = driver.run()
```

The mock above is the whole DataSource contract for testing: a callable
`(time, variable) -> xr.DataArray` with dims `(time, variable, lat, lon)`.
`DataComponent` fetches through `earth2studio.data.utils.fetch_data`, so any
real source (WB2, GFS, an OISST archive) drops in. Standard names are mapped
to source vocabulary through `variable_map={"standard_name": "raw_name"}`
when the source's names are not dictionary aliases. On `initialize()` with no
IC it fetches at `clock.start`, so lagged coupling has t0 data; the connector
regrids the source grid onto each destination.

**Gotcha:** the DataComponent must be *realized* before a no-arg
`initialize()` (the Driver does this for you); standalone use raises
`CouplingError: ... cannot fetch initial data before realize(clock)`.

## Derived fields and impact chains

Derived (time-reduced) fields are first-class dictionary entries carrying a
`CellMethod` — machine-readable "I am `method` of `base` over `window`".
Two consumers turn that declaration into a running windowed reduction: the
**windowed connector** (the short form, preferred for one source feeding one
destination) and the **AccumulationMediator** (the general form for multiple
sources or custom reductions). Both share the same accumulator core and
produce identical numbers.

### The short form: a windowed connector

Set `window=` and `reduce=` on the connector itself; it accumulates the
source export every step and delivers the *derived* field (matched through
the destination's `CellMethod` entry) on each window boundary — no mediator,
no `med.compute` action, no extra slot lines:

```python
from earth2studio.nvcoupler import Clock, Connector, Driver
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

atmos, ocean = fake_atmos(), fake_ocean()   # ocean imports the 48h mean
conn = Connector(atmos, ocean, window="48h", reduce="mean")

DSL_WINDOWED = """
@6h
  atmos -> ocean       # windowed: accumulates every step, delivers each 48h
  ocean -> atmos
  atmos
@48h
  ocean
@
"""
driver = Driver(
    {"atmos": atmos, "ocean": ocean},
    DSL_WINDOWED,
    Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=[conn],
)
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
driver.run()
# probes carry the DERIVED name, delivered on the ocean grid
z48 = conn.last_transfer["geopotential_at_1000hpa_48h_mean"]
assert z48.data.shape == (16, 32)
```

The destination must import a dictionary entry whose `CellMethod` is
`(base=the source export, method=reduce, window=window)`; a plain import
raises `CouplingError: ... register a FieldEntry(...)` — the coupler never
invents names. Mid-window the destination's previous import stands; the
first delivery lands one window after t0 (the window origin is the
`valid_time` of the first execute's source field, i.e. the clock start under
lagged coupling). `time_policy` does not apply on the windowed path.

### The general form: an AccumulationMediator

Register the entry, and an `AccumulationMediator` knows what to import,
which reduction to run, and its cadence:

```python
import numpy as np

from earth2studio.nvcoupler import (
    AccumulationMediator,
    CellMethod,
    DEFAULT_DICTIONARY,
    FieldDictionary,
    FieldEntry,
)

dictionary = FieldDictionary(DEFAULT_DICTIONARY)
dictionary.register(
    FieldEntry(
        "geopotential_at_1000hpa_24h_max",
        "m2 s-2",
        "24 h max of z1000",
        cell_method=CellMethod(
            "geopotential_at_1000hpa", "max", np.timedelta64(24, "h")
        ),
    )
)
med = AccumulationMediator(
    "med", ["geopotential_at_1000hpa_24h_max"], dictionary=dictionary
)
assert med.import_names == ["geopotential_at_1000hpa"]
```

Wire it like any component: `atmos -> med` in the fast slot accumulates every
delivery (running mean/sum/max/min, O(1) memory in window length), and
`med.compute` in the slow slot exports the reduced field. Reach for the
mediator rather than a windowed connector when several sources feed one
reduction, when one reduced field fans out to several destinations, or when
the reduction needs custom code (subclass `Mediator` — also the v1 home for
unit conversions). Several derived fields can reduce the same base import in
one mediator, but with mixed windows you must pass `window=` explicitly or
split them across mediators. Deliveries carrying an already-seen `valid_time`
are ignored, never double-counted — in mediators and windowed connectors
alike.

The impact-chain pattern stacks a `DiagnosticComponent` downstream: mediators
turn fast fields into a 48 h precip sum and a 24 h t2m max, a diagnostic
model consumes both and exports an impact index. `DiagnosticComponent` is a
stateless single-step transform — imports/exports default to the model's own
`input_coords()`/`output_coords()` variables resolved through the dictionary,
so registered diagnostics wire up with just a name, the model, and a cadence.
See [example 03](../../../examples/09_nvcoupler/03_impact_chain.py) for the
full chain.

**Gotcha:** if the mediator's `med.compute` runs before any sample arrived
you get `CouplingError: no samples of ... accumulated before compute` — the
connector feeding the mediator must sit in a *faster* slot than the compute.

## Vertical coupling

Only components with an explicit `level` dimension (chiefly chemistry
emulators on hybrid sigma-pressure levels) touch this machinery. Models that
encode levels in variable names (`z500`, `t850`) never do — those are just
distinct fields.

Declare what you publish and what you expect; the connector interpolates
(linearly in log-pressure, differentiably) when they differ:

```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.nvcoupler import (
    CallableComponent,
    Clock,
    Connector,
    DEFAULT_DICTIONARY,
    FieldDictionary,
    FieldEntry,
    HybridLevels,
    PressureLevels,
)
from earth2studio.nvcoupler.field import Field
from earth2studio.nvcoupler.testing import grid_coords

d = FieldDictionary(DEFAULT_DICTIONARY)
d.register(FieldEntry("ozone_mixing_ratio", "kg kg-1", aliases=frozenset({"o3"})))

hybrid = HybridLevels(a=(30000.0, 20000.0, 0.0), b=(0.0, 0.5, 1.0))  # p = a + b*ps
pressure = PressureLevels((500.0, 850.0))                            # hPa

identity = lambda x, coords: (x, coords)
met = CallableComponent(
    "met", identity, "6h",
    exports=["ozone_mixing_ratio"],
    dictionary=d,
    export_vertical={"ozone_mixing_ratio": hybrid},
)
chem = CallableComponent(
    "chem", identity, "6h",
    imports=["ozone_mixing_ratio"],
    dictionary=d,
    import_vertical={"ozone_mixing_ratio": pressure},
)
clock = Clock("2024-01-01", "2024-01-02", "6h")
met.realize(clock)
chem.realize(clock)

grid = grid_coords(4, 8)
o3 = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 4, 8).clone()
met.initialize(
    o3, OrderedDict({"variable": np.array(["o3"]), "level": np.arange(3.0), **grid})
)
# surface pressure has no level dim, so it cannot ride in the same state
# tensor as o3; add it to the export state directly
met.export_state.add(
    Field(
        torch.full((4, 8), 100000.0), OrderedDict(grid),
        "surface_pressure", "Pa",
        valid_time=np.datetime64("2024-01-01"), source="met",
    )
)
chem.initialize(
    torch.zeros(1, 2, 4, 8),
    OrderedDict(
        {"variable": np.array(["o3"]), "level": np.array([500.0, 850.0]), **grid}
    ),
)
Connector(met, chem, fields=["ozone_mixing_ratio"]).execute(np.datetime64("2024-01-01"))
assert chem.import_state["ozone_mixing_ratio"].vertical == pressure
```

Hybrid sources depend on surface pressure: the connector pulls
`surface_pressure` (or whatever `HybridLevels.ps_field` names) from the
*source's* export state automatically, and raises `VerticalMismatchError`
with the fix (`add it to the source's export list`) when it is absent.
Interpolation clamps to the source column ends — no extrapolation beyond the
top/bottom levels — and v1 only interpolates *onto* `PressureLevels`.
[Example 04](../../../examples/09_nvcoupler/04_vertical_chemistry.py) runs
the met/chem pair end to end in a Driver.

**Gotcha:** for `PressureLevels` sources, the data's `level` coordinate (hPa)
must equal the declared levels exactly, in top-to-bottom (increasing)
order — a mismatch raises rather than silently pairing slices with wrong
pressures.

## Masked fields

Components exporting fields that are only valid on part of the grid (SST on
ocean points) declare `export_masks={"standard_name": bool_tensor}` (True =
valid). The mask travels on the Field; the *connector's* `fill=` option
decides what invalid points become, always before regridding so garbage
cannot bleed into the interpolation:

```python
from earth2studio.nvcoupler import Clock, Connector, Driver, TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

DSL = """
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""
components = {
    "atmos": fake_atmos(),
    "ocean": fake_ocean(with_mask=True),   # northern half is land
    "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
}
driver = Driver(
    components,
    DSL,
    Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=[
        Connector(components["ocean"], components["atmos"], fill="nearest")
    ],
)
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
driver.run()
assert driver.probe("ocean->atmos")["sea_surface_temperature"].mask is None
```

- `fill="nearest"`: every invalid point takes its nearest valid neighbor
  (great-circle KDTree, differentiable gather) — the principled version of
  DLESyM's SST NaN-interpolation. The filler is cached per (grid, mask).
- `fill="zero"`: invalid points become 0.0.
- Both consume the mask (`field.mask is None` downstream).

**Without fill** (`fill="none"`, the default) nothing happens: invalid values
pass straight into the regrid, bleeding into neighboring destination cells,
and the mask stays attached but describes the *source* grid while the data is
now on the destination grid. Import adapters ignore masks entirely. If a
source declares `export_masks`, set `fill=` on every connector that leaves
it — this is also why `couple()`'s describe output shows the fill column.

## IO and outputs

Two independent output paths:

- **In-memory collection** (`collect=True`, the default): every ring's export
  fields are recorded and `run()` / `to_xarray()` returns one
  `xarray.Dataset` per component. Records are detached clones, off the
  exchange path.
- **Streaming IO** (`io={"name": backend}`): each component gets its own
  `IOBackend`; arrays are allocated with a leading `time` axis covering the
  component's ring times including t0, and a row is written after the IC seed
  and after every run.

```python
import numpy as np

from earth2studio.io import ZarrBackend
from earth2studio.nvcoupler import Clock, Driver, TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

io = {"atmos": ZarrBackend(), "med": ZarrBackend()}
components = {
    "atmos": fake_atmos(),
    "ocean": fake_ocean(),
    "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
}
driver = Driver(
    components, DSL, Clock("2024-01-01", "2024-01-05", "6h"),
    io=io,
    collect=False,           # nothing kept in memory
)
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
result = driver.run()        # {} when collect=False
z = io["atmos"]["geopotential_at_1000hpa"][:]     # (17, 32, 64)
zm = io["med"]["geopotential_at_1000hpa_48h_mean"][:]
assert np.all(np.isnan(zm[0]))                    # t0 row: NaN, by design
```

(`DSL` as in the previous section.)

**NaN semantics:** backend arrays are NaN-initialized deliberately. Zarr's
default fill reads back as 0.0, which would let never-written rows — a
mediator's t0 row (mediators export nothing until their first compute), or
the tail of a crashed run — masquerade as physical values. NaN rows in your
output mean "this ring was never written", not a numerical blow-up.

**Memory guard:** with `collect=True`, `initialize()` estimates the total
collected size and logs a warning above ~4 GB telling you to pass
`collect=False` plus a real IO backend. IO and collection are independent —
you can have both, either, or neither.

## Gradients and coupled fine-tuning

`run()` and `steps()` execute under `torch.inference_mode()` — fast, no
graphs. `rollout(n_steps)` is the training entry point: under
`torch.enable_grad()` the autograd graph survives the entire exchange path
(regrid gathers, mediator reductions, functional import injection):

```python
import torch

from earth2studio.nvcoupler import Clock, Driver, TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

gain_atmos = torch.tensor(1.0, requires_grad=True)
gain_ocean = torch.tensor(1.0, requires_grad=True)
components = {
    "atmos": fake_atmos(gain=gain_atmos),
    "ocean": fake_ocean(gain=gain_ocean),
    "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
}
driver = Driver(components, DSL, Clock("2024-01-01", "2024-01-05", "6h"))
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})

with torch.enable_grad():
    states = driver.rollout(16)          # keeps the graph
loss = states["atmos"]["geopotential_at_1000hpa"].data.sum()
loss.backward()                          # reaches BOTH components' parameters
assert gain_atmos.grad is not None and gain_ocean.grad is not None
```

Collected records and IO writes are detached — they only feed
`to_xarray()`/backends and never pin a step's graph, so collection stays on
during training without a memory explosion. Optimizer loops, truncated BPTT,
and per-component device placement are out of scope for v1 (see
[design and roadmap](design_and_roadmap.md));
[example 05](../../../examples/09_nvcoupler/05_coupled_finetuning.py) runs a
real training step.

**Gotcha:** `rollout(n)` with more steps than remain on the clock raises a
`CouplingError` naming `clock.n_steps` — size your rollout to the clock, or
`reset()` + `initialize()` between epochs.

## YAML round-trip

`to_yaml(driver)` / `from_yaml(text_or_path)` serialize the clock, the run
sequence verbatim, non-default dictionary entries and aliases, component
specs, and connector settings (`src`, `dst`, `fields`, `time_policy`,
`fill`). What round-trips:

- `AccumulationMediator` / `TrailingAverageMediator`: automatically (name,
  fields, window).
- Anything else: only if it carries a `yaml_spec` attribute — a
  `{"class": "<import.path>", "kwargs": {...}}` dict naming a module-level
  class or factory that rebuilds the component from kwargs alone.

```python
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler import Clock, Driver, TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

atmos = fake_atmos()
atmos.yaml_spec = {
    "class": "earth2studio.nvcoupler.testing.fake_atmos",
    "kwargs": {"gain": 1.0, "timestep": "6h"},
}
ocean = fake_ocean()
ocean.yaml_spec = {
    "class": "earth2studio.nvcoupler.testing.fake_ocean",
    "kwargs": {"gain": 1.0, "timestep": "48h"},
}
med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
driver = Driver(
    {"atmos": atmos, "ocean": ocean, "med": med},
    DSL,
    Clock("2024-01-01", "2024-01-05", "6h"),
)
text = nvc.to_yaml(driver)      # or to_yaml(driver, path="system.yaml")
rebuilt = nvc.from_yaml(text)   # uninitialized; call initialize(ics) as usual
rebuilt.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
```

What cannot round-trip: components wrapping closures or live model objects
without a `yaml_spec` (`to_yaml` raises with the fix), custom `regridder=`
callables and custom ImportAdapter instances (connector `fields`/policies
serialize; callables do not), windowed connectors — `window=`/`reduce=` are
silently dropped by `to_yaml`, so windowed systems must be built in Python —
and model checkpoints referenced by load paths (out of scope for v1). Initial conditions are never serialized — a config
describes the system, not its state. Full schema in the
[DSL and YAML reference](dsl_and_yaml_reference.md).

## Inspection and debugging

- `driver.describe()` (or `nvc.describe(driver)`) — the plan: components,
  connectors with time policy/fill/mode/slot, and the run sequence. Works
  *before* `initialize()`; in Jupyter, a bare `driver` renders the HTML
  version.
- `driver.probe("src->dst")` — the last Fields exchanged on a connector
  (post-pipeline: time policy, vertical, fill, regrid already applied).
  Spaces are tolerated: `probe("ocean -> atmos")`.
- `driver.steps()` — iterate `(time, {component: export State})` per driver
  step, notebook-style, instead of one opaque `run()`.
- loguru levels: warnings (unconsumed exports, memory guard, linear-policy
  fallback) are on by default; every exchange also logs at DEBUG:

```python
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")   # show per-exchange lines

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

driver = nvc.couple(fake_atmos(), fake_ocean(), start="2024-01-01", stop="2024-01-03")
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
for time, states in driver.steps():
    z = states["atmos"]["geopotential_at_1000hpa"]
    print(time, float(z.data.mean()))
print(driver.probe("ocean -> atmos"))
```

Each DEBUG line reads
`exchange ocean->atmos: sea_surface_temperature (valid 2024-01-01T00:00...)`
— the `valid` timestamp is the fastest way to see lagged coupling in action
(the delivered field is older than the current step). When something raises
instead, every message names the components, the field, and the fix; the
complete catalog is in
[errors and troubleshooting](errors_and_troubleshooting.md).
