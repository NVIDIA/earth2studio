Hey @Peter Harrington, nothing fancy about it. HEre is the markdown: 

# Impact Modeling (Coupling + Applications API) -- Software Design

*High-level goals, requirements, proposed API, sequencing.*


The goal is being able to mix and match models (in a plug and play way) with remote sensing obs to make impact or decision models. 

---

## 1. High-level goals

**The problem.** Impact applications or workflows (hydrology, wildfire, energy, agriculture, coastal, air quality) consume forecast output at a different grid, cadence, and state model than the forecast, and ingest observations mid-run to produce a secondary index, application, or output (such as decision).

Today every team & SAs hand-writes the same plumbing: regridding, temporal alignment, state management, observation ingestion, I/O. Every operational forecast-to-impact chain will share the same engineering. 


Two main pieces: 
a. Coupler
b. Application API


**Major goals of this effort:**

1. **One component contract for everything downstream of (or beside) a forecast.** Forecast models, domain models, and data sources all declare imports/exports by standard name, keep private state, and run at their own cadence. A wind-farm power curve and DLESyM's ocean are configurations of the same contract, differing only along five axes: 
**state** (stateless/stateful),
**grid** (shared / own grid / points / mesh),
**cadence** (slower/same/faster than forcing), 
**coupling depth** (one-way / observation-corrected / two-way), 
**forcing count** (1 or N).

2. **Transforms live outside the models.** A minimum operation set covers all target applications; a component never knows its neighbor's grid or cadence.

3. **Observations are components, not a side channel.** Satellite/in-situ streams deliver into a running component's import state at their own cadence — the operational pattern (SMAP nudging, VIIRS perimeter re-initialization) becomes data streams, similiar to prognostic models....


4. **GPU-native, differentiable exchange.** Torch tensors end-to-end in physical units; autograd survives the exchange (`rollout(n)`), enabling coupled fine-tuning. No file-based handoffs, no numpy round-trips.

5. **Fail before compute.** Eager validation (names, units, cadences, unfed imports, cycles) with teach-the-fix errors, and a `describe()` plan preview before any tensor moves.

6. **An application-shaped view for the 80% case.** A declarative `Application(...)` constructor over the general `couple()` entry point, so a domain scientist assembles pipelines without touching Driver machinery — the on-ramp to an application catalog.


7. **Ensembles end-to-end.** Members as a batch dimension through stateful impact models, collapsed to statistics only at the product boundary — probabilistic impact products (flood exceedance, P10/P90 generation bands, loss exceedance curves) are what decision-makers actually buy.

**Non-goals:** owning any application's domain science/impact models (infrastructure for chaining these by Earth2Studio, science by the domain owner); GIS/catalog services; training pipelines.

---
## 2. Components the system needs

Seven pieces. Everything in the requirements and API sections maps onto one of these.

```
                 ┌──────────────────────────────────────────────┐
                 │                    Driver                     │
                 │  clock · run sequence · validation · preview  │
                 └───────┬───────────────┬──────────────┬────────┘
                         │ runs          │ runs         │ runs
                 ┌───────▼──────┐ ┌──────▼───────┐ ┌────▼─────────┐
                 │  Component   │ │  Component   │ │ DataComponent│
                 │ forecast @6h │ │ impact @24h  │ │ obs @72h     │
                 └───────┬──────┘ └──────▲───────┘ │ (satellite,  │
                         │  exports      │ imports │  tower, ...) │
                         │   ┌───────────┴──────┐  └────┬─────────┘
                         └──►│    Connector      │◄─────┘
                             │ regrid · temporal │
                             │ align · mask ·    │
                             │ subsample         │
                             └─────────▲─────────┘
                                       │ names · units · recipes
                             ┌─────────┴─────────┐
                             │  Field Dictionary │
                             └───────────────────┘
```

1. **Component** — the one contract every participant implements: declare imports/exports by standard name, keep private state, advance one timestep when the driver says so. Forecast models, impact models, and data sources are all Components; "application" vs "Earth-system model" is about where it sits in the graph, not its code. For example, diagnostic and prognostic models are components....

2. **Field** — the exchange unit: one variable, one grid, one valid time, physical units, torch tensor payload (batch axis = ensemble members), validity mask, **provenance metadata**.

3. **Connector** — everything *between* two components: 
- spatial regridding, 
- temporal alignment (aggregation and cadence bridging),
- masking, subsampling,
- vertical interpolation. 

A component never knows its neighbor's grid or cadence. A **Mediator** is the multi-source variant (several exporters feeding one import, e.g. blending nowcast and NWP).

4. **DataComponent** — a data source lifted into the coupled world at its own cadence. Covers four patterns: static rasters (DEM, land cover, fuel maps), satellite observation streams (SMAP, Sentinel-1, VIIRS — possibly irregular overpass cadence), **in-situ tower/station streams** (flux towers, met towers, gauges — point observations at sub-hourly cadence, entering through geometry sampling in reverse: sparse points corrected against or nudging gridded state), and versioned embedding rasters (foundation-model context).

5. **Driver** — the executor: one clock, a run sequence derived from the coupling graph (with a readable override), eager validation of the whole graph before any compute, a plan preview, and gradient-preserving rollout.

6. **Field Dictionary** — the semantic layer: standard names, canonical units, and machine-readable recipes for derived fields ("24 h precipitation sum") so composition can be auto-wired and checked. Grown one entry at a time from real applications.

7. **Application** — the declarative view for domain scientists: `forcing / context / transforms / model / diagnostics / io` kwargs that translate onto the driver with no new execution machinery. The unit that a catalog eventually lists.

---

## 3. Requirements

### 3.1 Functional requirements

| # | Requirement | Driven by |
|---|---|---|
| FR1 | **Component contract** — participants declare imports/exports by dictionary standard name; keep private state; run at their own timestep; accept mid-run observation correction (nudge *and* overwrite); initialization incl. warm-start | all applications; wildfire needs overwrite, soil moisture nudge |
| FR2 | **Field exchange** — single-variable fields in physical units (checked, **not** converted), torch tensors end-to-end (autograd survives), validity masks, valid-time + provenance metadata, leading batch axis | coupled fine-tuning; energy/downscaling ensembles |
| FR3 | **Transforms** — the full minimum operation set of §3.2, outside the models, declarable per field route, auto-synthesizable from dictionary recipes | every application |
| FR4 | **Context data plane** — four data-component patterns (static raster; satellite observation stream at its own, possibly irregular, cadence; in-situ tower/station stream at point locations; versioned embedding raster) + normalization contracts (per-channel stats persisted with checkpoints) | flood SAR, wildfire fuel maps, agriculture dual streams, tower verification/nudging, downscaling embeddings |
| FR5 | **Multi-rate stateful execution** — run sequence derived from the coupling graph with a human-readable override; lagged and concurrent exchange; subcycling; cycle detection; checkpoint/restart | DLESyM (both exchange semantics), seasonal agriculture |
| FR6 | **Application layer** — `Application(...)` with named roles, `derive=` pre-model diagnostics, solution nesting, warm-starting (`ics="warm"`) | GloFAS runs a dedicated daily chain just to warm-start river states |
| FR7 | **Ensembles** — perturbation at initialize, per-member state through stateful components, per-member or reduced IO, μ/σ + exceedance dictionary conventions, collapse only at the product boundary | flood exceedance, energy P10/P90, agriculture yield ranges |
| FR8 | **Multi-component forcing** — an application imports from ≥2 upstream components | coastal surge (atmos + ocean) |
| FR9 | **Decision-product stage** — report / alert / visualization components as first-class pipeline members | accurately forecast hazards have failed as warnings |
| FR10 | **Products layer** *(design-for, don't build)* — provides/requires registration enabling goal-directed resolution, continuous materialization, what-if substitution | longer-term direction |

### 3.2 Minimum transform (Connector) operation set

Nothing domain-specific should require an operation outside this list; anything absent blocks a named application.

| Category | Operation | Declaration | Needed by |
|---|---|---|---|
| **Spatial regrid** | identity | (grids match) | shared-grid coupling (DLESyM) |
| | bilinear / nearest | `regrid="bilinear"` | coarse↔fine fields |
| | conservative | `regrid="conservative"` | flood/wildfire feedback — any flux across grids |
| | user-supplied | `regridder=fn` | escape hatch; unstructured coastal meshes |
| **Temporal — aggregation** | sum / mean / max / min | `window="24h", reduce="sum"` | 6 h precip → daily totals; daily peak gust |
| **Temporal — cadence bridging** | constant hold | `time="constant"` | consumers faster than forcing (fire @1 h under 6 h atmos) |
| | linear interpolation | `time="linear"` | smooth in-between values |
| **Masking** | zero-fill / nearest-fill | `fill="zero"/"nearest"` | SST over land, swath-masked SAR |
| | valid-mask passthrough | field-level mask | partially valid fields carried, not silently dropped |
| **Subsampling** | spatial decimation | stride / `subsample="nearest"` | dense embeddings (10 m) → application grid |
| | temporal decimation | `subsample="Nh"` | sparse consumers / validation |
| **Geometry sampling** | points / lines / sites | `sample="points"` | energy assets, agriculture sites, downscaling stations — demanded independently 3× |
| **Vertical** | hybrid → pressure | — | level-coordinate mismatches |
| **Learned transforms** | trained model as a stage | — | downscaling; plain regridding leaves ~10 % skill on the table |
| **Multi-source** | blend / windowed reduce over N exporters | mediator | nowcast/NWP blend by lead time (energy) |
| **Reprojection** | CRS-aware (UTM tiles ↔ lat/lon) | — | satellite imagery (shared with GeoAI-data pillar) |

### 3.3 Non-functional requirements

| # | Requirement | Rationale |
|---|---|---|
| NF1 | **Fail before compute** — eager validation with teach-the-fix errors; a plan preview before anything runs | Misconfiguration must fail before GPU time is spent |
| NF2 | **GPU-native** — no numpy round-trips in the exchange; any pull-style coupling documented as inference-only | Autograd through the exchange is a hard requirement for coupled fine-tuning |
| NF3 | **Stability discipline** — conservative remapping gates any two-way flux; default time policies documented as *modeling assumptions*, never silent framework choices | Interface-oscillation results in the coupling literature; a 24 h hold changes flood peak timing — a science decision, not plumbing |
| NF4 | **Scale** — O(1) window accumulators, regridder caching; a stated memory target at 10 m DEM resolution | Flood on real DEMs hits it first |
| NF5 | **Reproducibility & lineage** — versioned weights, forcing cycles, embedding snapshots; graph-derived provenance on outputs | Cheap when the graph is explicit; decisive after a missed warning or in a partner marketplace |
| NF6 | **Extensibility without forks** — partner/customer models plug in via adapters only; no new hard dependencies | The division of labor that makes a catalog possible |

### 3.4 Verification requirements

- Binding rule: every claim of a working capability carries an executed test; everything else is labeled *design target*.
- **Equivalence gate:** a split coupled model (DLESyM atmosphere/ocean) must reproduce the fused original with real weights to tight tolerance before the coupler is considered validated.
- Conservation tests (flux integrals preserved across grids) required before any two-way edge ships.
- Ensemble N=1 batched run must be bit-identical to the unbatched path.
- At least one end-to-end reference application executed with a real forecast model, not mocks.

---

## 4. Summary of proposed API

### 4.1 Component — the core contract

```python
class Component(ABC):
    """One contract for forecast models, impact models, and data sources."""

    # --- Identity & declaration (set at construction) ---
    name: str                      # unique node name in the coupling graph
    timestep: TimeDelta            # the component's own cadence ("6h", "24h", ...)
    imports: list[str]             # dictionary standard names it consumes
    exports: list[str]             # dictionary standard names it produces
    dictionary: FieldDictionary    # shared vocabulary (names, units, recipes)
    requires_ic: bool = True       # False for data sources (nothing to initialize)

    # --- State (owned, private) ---
    import_state: dict[str, Field] # connectors deposit Fields here (push delivery)
    # internal model state is the subclass's business — never touched by the framework

    # --- Lifecycle phases (NUOPC-style, called by the Driver in order) ---
    def advertise(self) -> tuple[list[str], list[str]]:
        """Report declared imports/exports. No memory allocated, no data moved.
        Lets the Driver validate the whole graph before any compute."""

    def realize(self, clock: Clock) -> None:
        """Validate this component's timestep against the driver clock.
        Fail here — before any tensor moves — if cadences are incompatible."""

    def initialize(self, x: Tensor, coords: CoordSystem) -> None:
        """Seed internal state and publish exports at t=0,
        so lagged coupling has data on the first exchange."""

    @abstractmethod
    def run(self, time: datetime) -> None:
        """Advance exactly one timestep:
        1. read import_state   (fields already regridded/aggregated by connectors)
        2. step the model      (the subclass's science)
        3. self.publish(...)   (export fields, stamped with valid_time)"""

    # --- Provided by the base class ---
    def publish(self, x: Tensor, coords: CoordSystem, valid_time: datetime) -> None:
        """Place exports where connectors can pick them up, with units checked
        against the dictionary and provenance attached."""
```

Load-bearing properties:

1. **Declaration before allocation** — `advertise()` is metadata-only, so the Driver validates names, units, cadences, and unfed imports across the whole graph before any GPU memory or data movement (NF1).
2. **Push delivery** — connectors write into `import_state`; a component never fetches, never knows its neighbor's grid or cadence.
3. **Publish-at-init** — `initialize()` publishes t=0 exports so *lagged* exchanges (e.g. DLESyM's SST) have data on the first coupled step.
4. **State is private** — the framework only ever sees `import_state` and published exports; everything else belongs to the subclass. (Known cost: checkpoint/restart and per-member ensemble state require the subclass to expose serialization.)
5. **`run()` is the only abstract method** — a minimal component is one class with a constructor and `run()`; everything else is inherited.

Concrete wrappers, so existing objects join without modification:

```python
CallableComponent(name, fn, timestep=..., exports=[...])   # any fn(x, coords)
PrognosticComponent(name, model=<PrognosticModel>, timestep=...)
DiagnosticComponent(...)
DataComponent(name, source=<DataSource>, timestep=...)     # obs/context; requires_ic=False
# ImportAdapters inject imports into wrapped models without modifying them
```

**Component vs DataComponent.** `Component` is the abstract contract; its model-like subclasses *compute* (step FCN, integrate a bucket model). `DataComponent` is the one concrete subclass whose `run(time)` doesn't compute — it *fetches*: it wraps an existing `DataSource` (SMAP, Sentinel-1, a tower network, a DEM) and publishes whatever the source has for that time.

| | Component (model-like) | DataComponent |
|---|---|---|
| `run(time)` does | step a model using `import_state` | query the wrapped `DataSource`, publish the result |
| imports | usually yes | none — nothing feeds it |
| `requires_ic` | `True` — needs an initial state | `False` — nothing to initialize |
| state | carries model state between steps | stateless (the archive is the state) |
| role in the graph | producer *and* consumer | pure producer (source node) |

Making it a subclass rather than a separate concept is what makes "observations are components, not a side channel" literal: the Driver schedules, connects, and validates both identically, so swapping a modeled SST for an observed one is replacing a `PrognosticComponent` with a `DataComponent` — one line, and nothing downstream notices.

**Design decision — why wrap models rather than extend them.** The alternative (adding coupling methods to `PrognosticModel` itself) was considered and rejected: (1) the model contract is the wrong shape — tensor-in/tensor-out with no named imports/exports, units, cadence, or mid-run observation slot; (2) not every participant is a model — data sources, stateless functions, and diagnostics need the same contract, and a neutral `Component` covers all of them symmetrically; (3) no forks (NF6) — adapters let SFNO, FCN, DLESyM, and partner models join untouched, with no base-class change rippling through the model zoo. The honest cost: the wrapper holds mutable state, breaking the state-in-tensor convention — which is exactly why checkpoint/restart and per-member ensemble state are the two expensive gaps.

### 4.2 Driver — the core executor

```python
class Driver:
    """Executes a coupling graph: one clock, derived run sequence, eager validation."""

    # --- Construction (usually via couple()) ---
    components: dict[str, Component]
    connectors: dict[tuple[str, str, frozenset[str]], Connector]  # keyed src, dst, fields
    clock: Clock                   # dt = GCD of component timesteps
    sequence: RunSequence          # derived from the graph, or user-supplied override
    io: dict[str, IOBackend]       # optional per-component output backends

    # --- Graph analysis (at construction, before any compute) ---
    def _derive_sequence(self) -> RunSequence:
        """One slot per distinct cadence; lagged exchanges first, then a
        topological sort over concurrent edges. A cycle raises with the fix:
        'mark one edge lagged'."""

    def _validate(self) -> None:
        """Eager, teach-the-fix errors: unknown standard names, unit mismatches,
        incompatible cadences, imports no exporter feeds, duplicate connectors."""

    # --- User surface ---
    def describe(self) -> str:
        """Plan preview before anything runs: which slots fire at which cadence,
        which exchanges in what order, lagged vs concurrent."""

    def initialize(self, ics: dict[str, tuple[Tensor, CoordSystem]],
                   perturbation=None) -> None:
        """Seed every requires_ic component; optional perturbation expands
        member 1 -> N at the single place state enters the system."""

    def run(self) -> dict[str, xr.Dataset]:
        """Advance the clock start -> stop. At each tick, fire the slots whose
        cadence divides the time: run connectors (deposit into import_state),
        then components, in sequence order. Returns per-component datasets,
        each on its own time axis."""

    def rollout(self, n: int) -> Tensor:
        """n coupled steps with autograd kept through every exchange —
        the entry point for coupled fine-tuning."""

    def probe(self, name: str) -> ComponentReport:
        """Introspection: a component's current state, imports, staleness."""
```

Load-bearing properties:

1. **The sequence is derived but never hidden** — computed from the coupling graph by default, overridable with a human-readable sequence (DLESyM's 4-line sequence is the canonical override), and always inspectable via `describe()`.
2. **Lagged vs concurrent is positional** — a connector placed *before* its source's slot delivers the previous step's export; *after*, this step's. No mode flag; the sequence text is the single source of truth.
3. **All validation at construction** — a broken graph never reaches `run()`.
4. **One clock, many cadences** — `dt = GCD` of timesteps; each component fires only when its cadence divides the current time; connectors bridge the gaps per their time policy.
5. **Ensembles enter in exactly one place** — `initialize(perturbation=)`; after that, members are just a batch axis every Field already carries.

Composition entry point:

```python
driver = couple(*components, start=..., stop=..., dt=None, io=None)
```

Connectors auto-wire from field-dictionary `CellMethod` recipes (registering `total_precipitation_24h_sum` lets `couple()` synthesize the windowed connector); explicit `Connector(src, dst, regrid=, window=, reduce=, time=, fill=)` otherwise. A YAML config layer round-trips driver configurations.


### 4.3 Proposed surface, by priority

| Change | Surface | Priority |
|---|---|---|
| `Component` contract, concrete wrappers, `couple()` driver with derived sequencing and eager validation | core | P0 |
| `Application(...)` class | new class | P0 |
| Conservative regridding — gates any two-way edge | connector | P0 |
| Per-component IO backends exposed through `couple(io=)` | kwarg | P0 |
| `sample="points"` — geometry sampling; `asset_id` replaces lat/lon as output dim | transform | P1 |
| `subsample=` — spatial + temporal decimation | transform | P1 |
| `Driver.initialize(ics, perturbation=...)` + per-member/reduced IO — the ensemble API | driver | P1 |
| `StateStore` / `ics="warm"` — operational cycling as a framework primitive | application/driver | P1 |
| `EmbeddingRaster` component + CRS-aware tiled coords, COG windowed reads, "latest available at t" fetch | data plane | P1 |
| `derive=` kwarg · solution nesting with graph flattening · μ/σ field conventions · normalization contracts | application/dictionary | P2 |
| Unit *conversion* (baseline: checked, mismatches error) | dictionary | P2 |
| provides/requires registration (products layer) | dictionary | P3, design-for only |

**Cross-pillar seams:** the field dictionary must reconcile with `earth2studio.lexicon` (one vocabulary, not two); `DataComponent` is where the GeoAI data-layer pillar plugs in (any new `DataSource` is automatically a coupled participant); `Application` + catalog metadata is the hand-off to the applications-framework pillar.

---

## 5. Sequencing (this pillar only)

Strict order; each item unblocks the next.

| # | Item | Why this position |
|---|---|---|
| 1 | **Core contract + coupling validation** — implement the component/connector/driver core and pass the DLESyM real-weights equivalence gate | Everything downstream of an unvalidated coupler is speculation; the gate is the cheapest decisive test |
| 2 | **Conservative regridding** | Literature is unambiguous that non-conservative flux remap destabilizes two-way coupling; unblocks the flood/wildfire feedback class |
| 3 | **Coupled ensembles** | Must land before reference apps ship, or they ship deterministic |
| 4 | **First reference application: Food/Agriculture** | Named-customer validation (NSF Engine workflow end-to-end); forces geometry sampling, dual obs streams, normalization contracts |
| 5 | **Second application: Energy** | Cheapest case — the generality proof that the abstractions carry a second domain with near-zero new infrastructure, pressure-testing the API before harder domains lock it in |
| 6 | **Checkpoint/restart** | The bill for stateful components (serialize model windows, mediator accumulators, connector history); prerequisite for seasonal agriculture; design jointly with ensembles — N members = N state copies, one state-externalization fix serves both |
| 7 | **`Application` API hardening + catalog metadata** | Built last, from evidence: introduce the interface only after real workflows demonstrate the requirements |

**Dependencies on other pillars:** item 4 wants the GeoAI data layer's first sources (SMAP/Sentinel-class) but can start on existing `DataSource`s; item 7 hands off to the applications-framework pillar; CRS/reprojection work is shared scope with the data pillar and should be co-planned.

**Top risk for cross-pillar planning:** stability conditions for two-way feedback into an autoregressive ML atmosphere are unestablished — such models tolerate only in-distribution forcing, so slightly-off fed-back state can drift them with no numerical warning. Proposed mitigations: nudge don't overwrite, tunable feedback gain, conservative remap as a hard gate, coupled fine-tuning as the probable remedy. Characterizing these conditions is a publishable contribution Earth-2 is uniquely positioned to make.


# 6. Worked example — one instance of each component, FourCastNet-forced

A soil-moisture application driven by FourCastNet, corrected by satellite and tower observations. Every one of the seven components from §2 appears exactly once.

| § | Component | Instance in this example |
|---|---|---|
| 2.1 | Component | `SoilMoistureComponent` — a bucket model @24 h: soil moisture state, precip in minus evapotranspiration out |
| 2.2 | Field | `total_precipitation_24h_sum` — kg m⁻², FCN's 0.25° grid, valid-time stamped, physical units |
| 2.3 | Connector / Mediator | FCN's 6 h precip summed to 24 h totals; SMAP's coarse grid regridded (bilinear) to the soil grid; tower points sampled against the nearest grid cell |
| 2.4 | DataComponent | `smap` @72 h (satellite raster stream, nudges the state) and `towers` @1 h (in-situ point stream, verification/nudging) |
| 2.5 | Driver | sequences the three cadences (6 h / 24 h / 72 h), validates the graph, prints the plan, runs |
| 2.6 | Field Dictionary | entries for `total_precipitation_24h_sum` and `air_temperature_2m_24h_mean` — recipes the framework auto-wires into windowed connectors |
| 2.7 | Application | the declarative wrapper tying it all together |

```python
from earth2studio.models.px import FCN

# 2.6 Field Dictionary — derived-forcing recipes (auto-wired by couple())
DICT.register(FieldEntry("total_precipitation_24h_sum", "kg m-2",
              CellMethod("total_precipitation_6h", "sum", "24h")))
DICT.register(FieldEntry("air_temperature_2m_24h_mean", "K",
              CellMethod("air_temperature_2m", "mean", "24h")))

# 2.1 Component — the domain science (~40 lines total)
class SoilMoistureComponent(Component):
    def __init__(self):
        super().__init__("soil", timestep="24h",
            imports=["total_precipitation_24h_sum",       # from FCN, aggregated
                     "air_temperature_2m_24h_mean",       # from FCN, averaged
                     "observed_soil_moisture",            # from SMAP
                     "tower_soil_moisture"],              # from towers
            exports=["volumetric_soil_moisture_layer1"],
            dictionary=DICT)

    def run(self, time):
        f = self.import_state                             # 2.2 Fields, physical units
        pet = 0.4 * (f["air_temperature_2m_24h_mean"].data - 273.15).clamp(min=0)
        sm = (self._x + (f["total_precipitation_24h_sum"].data - pet)
              / self.depth_mm).clamp(0.0, 0.5)
        for obs_name, gain in [("observed_soil_moisture", 0.3),   # satellite nudge
                               ("tower_soil_moisture", 0.5)]:     # tower nudge (points)
            obs = f.get(obs_name)
            if obs is not None and self.is_fresh(obs):
                sm = sm + gain * obs.mask * (obs.data - sm)
        self._x = sm
        self.publish(self._x, self._coords, valid_time=time)

# 2.4 DataComponents — observations as first-class participants
smap   = DataComponent("smap",   source=SMAP(),      timestep="72h")
towers = DataComponent("towers", source=TowerObs(),  timestep="1h")   # sparse points

# 2.7 Application — 2.3 Connectors and 2.5 Driver are synthesized underneath
app = Application(
    forcing=PrognosticComponent("atmos",
        model=FCN.load_model(FCN.load_default_package()), timestep="6h"),
    context=[smap, towers],
    transforms=[
        dict(field="observed_soil_moisture", to="soil", regrid="bilinear"),
        dict(field="tower_soil_moisture",    to="soil", sample="points",
             fill="nearest"),
        # precip sum + t2m mean need no entry: auto-wired from the dictionary recipes
    ],
    model=SoilMoistureComponent(),
    io=ZarrBackend("soil_moisture.zarr"),
)
app.describe("2024-06-01", "2024-06-15")   # 2.5 Driver plan preview, then:
datasets = app.run("2024-06-01", "2024-06-15", ics={"soil": (x0, coords)})
```

What the framework owns here: 6 h→24 h summing and averaging, the satellite regrid, tower point sampling, four clocks (1 h / 6 h / 24 h / 72 h) sequenced correctly, freshness/latency semantics for both observation streams, graph validation before compute, and Zarr output on the soil component's own time axis. What the author writes: the bucket model and two nudging gains.
