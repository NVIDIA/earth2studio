# Earth2Studio Coupling (draft)

Goal: advance several models on one clock and serve each model's declared
forcing from the others' outputs, with no abstractions beyond the model,
source, regridder and execution contracts. `earth2studio.coupling` is to
coupled runs what `PrognosticRunner` is to single-model runs.

## Shape

- A `Coupler` holds models and connectors and executes one work item at a
  time. It is a `Runner` ([EXECUTION_CONTRACT_SPEC.md](EXECUTION_CONTRACT_SPEC.md)),
  so `Pipeline` owns distribution, resume, output and progress.
- Models are ordinary `PrognosticModel` and `DiagnosticModel` instances
  ([MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)). There is no component
  base class, protocol or mixin.
- A `Connector` is a frozen declaration of which destination labels a source
  serves. All runtime state lives in the coupler.

```python
coupler = Coupler(
    {"atmos": atmos, "ocean": ocean},
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={"atmos": (GFS(), None), "ocean": (ORAS5(), None)},
)
Pipeline(coupler, output={"atmos": store_a, "ocean": store_o}).run(items)
```

| Module | Contents |
| --- | --- |
| `coupling/coupler.py` | `Coupler`, per-item run state |
| `coupling/connector.py` | `Connector` |
| `coupling/sequence.py` | `RunSequence`, `Slot`, `derive_sequence` |
| `coupling/vertical.py` | `PressureLevels`, `HybridLevels`, `to_pressure_levels` |
| `coupling/errors.py` | `CouplingError` |

Public: `Coupler`, `Connector`, `CouplingError`. Semi-public: `RunSequence`,
`Slot`, `derive_sequence`, and the vertical helpers. Everything else is
private.

## What the coupler reads from a model

| Needed | Read from |
| --- | --- |
| imports | `forcing_coords()` of a prognostic; `input_coords()` of a diagnostic |
| exports | `output_coords()` |
| cadence | prognostic: final output lead minus final input lead; diagnostic: see below |
| grid | `infer_grid(signature)` per slot |
| initial conditions, unserved forcing | `default_sources()`, or `sources=` per slot |

A prognostic's cadence cannot be overridden. A diagnostic runs at the cadence
of the models feeding it (their GCD); one fed by no coupled model needs
`cadence=`. `sources=` lists one source per input-then-forcing slot.

## Interface

```python
class Coupler:  # satisfies Runner
    supports_member_batching: bool = False
    def __init__(
        self,
        models: Mapping[str, PrognosticModel | DiagnosticModel],
        connectors: Sequence[Connector] = (),
        sources: Mapping[str, Source | tuple[Source | None, ...] | None] | None = None,
        cadence: Mapping[str, str | np.timedelta64] | None = None,
        sequence: RunSequence | None = None,
        dt: str | np.timedelta64 | None = None,
        seed: int = 0,
    ) -> None: ...
    def to(self, device) -> Coupler: ...
    def output_coords(self, final_lead_time) -> Mapping[str, CoordinateSystem]: ...
    def data_requests(self, item) -> tuple[DataRequest, ...] | None: ...
    def run_item(self, item, ics=None) -> Iterator[Mapping[str, xr.DataArray]]: ...
    def describe(self) -> str: ...

@dataclass(frozen=True)
class Connector:
    src: str
    dst: str
    variables: tuple[str, ...] | None = None
    regridder: Regridder | None = None
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None
```

`run_item(item, ics=...)` accepts in-memory initial conditions, one array per
input slot, for callers already holding them; `data_requests` returns `None`
when an input slot has no source and so needs `ics`.

## Exchanges

**Matching.** Labels are Earth2Studio lexicon names. A destination import
`z1000` is served by a source exporting `z1000`. A qualified import such as
`z1000:mean:48h` ([TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md)) is served
by reducing the source's `z1000` frames over that window with
`utils.time_statistics`; the window must be a multiple of the source cadence.
A connector without `variables` serves every destination import its source
can. If there are none, it overwrites the matching **state** variables
(outputs that are also inputs) of a prognostic destination before each step,
for models that carry a coupled field in their own state.

**Leads.** A prognostic step needs its forcing at the slot's final declared
lead shifted by the lead of the `y` being advanced (model contract). The
coupler keeps a buffer of each model's frames per lead and serves exactly
those leads. A source slower than the destination holds its last frame; a
missing frame from a source of equal or faster cadence is an error naming the
run order. Diagnostics take the newest frame at or before the current lead;
a diagnostic over a window skips steps until the window exists. Where a
window reaches before the initial condition, or the source has not produced
a frame yet, the destination's bound source fills in.

**Order.** `derive_sequence` lays out one slot per cadence, fastest first,
so a faster model always advances before a slower one at a shared step.
Within a slot, a destination that needs the *same* step's frame (forcing lead
beyond its input time, or a plain diagnostic input) runs after its source;
otherwise declaration order, so it sees the previous frame (lagged). An
explicit `sequence=` is validated against these dependencies.

**Grids.** Differing grids resolve a built-in regridder (`bilinear` for
structured sources, else `nearest`) through `resolve_regridder`, or use the
connector's `regridder`, whose `source_grid`/`target_grid` must match the
models ([REGRID_SPEC.md](REGRID_SPEC.md)). Reductions happen on the source
grid, before regridding. A `NearestRegridder(..., source_mask=)` fills
masked (land) points from the nearest valid source point. State overwrites
require an exact coordinate match. `transform` runs after regridding and may
change only the `level` coordinate; `to_pressure_levels` builds one that
interpolates linearly in log pressure.

**Devices.** `Coupler.to(device)` moves every model and regridder. Fields
delivered to a model are staged onto that model's device, so CPU-only models
coexist with CUDA models; a model's device is read from its parameters, or
overridden with `devices=`.

**Streams.** Each step of `run_item` yields the outputs of the models that
advanced, keyed by model name (`name.1` for a second output slot), with
dimensions `time` then `lead_time` and the field dimensions. The first step
holds the prognostic initial conditions and diagnostics computable from
them. `output_coords(final_lead_time)` gives one schema per stream covering
the initial condition and every step; multi-lead steps publish all leads.

## Rules

| Rule | Requirement |
| --- | --- |
| `C1` | Models are accepted as `PrognosticModel`/`DiagnosticModel`; no coupler base class |
| `C2` | Imports, exports, cadence and grid come from the model declarations |
| `C3` | `Connector` is immutable and holds no runtime state |
| `C4` | Two connectors on one pair are distinct; one label from two connectors raises |
| `C5` | Forcing is served at the leads the model contract requires, never silently shifted |
| `C6` | A frame missing from an equal- or faster-cadence source raises, naming the run order |
| `C7` | Derived sequences satisfy every same-step dependency; explicit ones are validated |
| `C8` | Regridders implement `grids.Regridder` and match both models' grids |
| `C9` | State overwrites require exact coordinate equality |
| `C10` | Streams satisfy `X1`-`X4`; one stream per output slot |
| `C11` | `CouplingError` subclasses `ValueError` |

Tests live in `test/coupling/`; the hand-computable toy models are in
`test/coupling/coupling_toys.py`, and `dev/examples/10_coupling.py` and
`11_coupled_pipeline.py` demonstrate the concepts.

## Out of scope

Training rollouts, YAML or text configuration, model-owned data fetching.
Member batching is not implemented (`supports_member_batching = False`); a
leading `batch` dimension on initial conditions passes through unchanged.

## Open decisions

1. **Non-gridded payloads.** DataFrame observations and tabular impact
   outputs do not map onto DataArray streams; coupling an `AssimilationModel`
   needs a frame-aware slot kind.
2. **Irregular cadence.** "Latest available at *t*" observation sources are
   better composed as sources (`RegriddedSource`-style) than modelled in the
   clock.
3. **Ensembles.** Member batching for coupled runs, and whether perturbations
   are a coupler or runner argument.
4. **Time-varying surface pressure** for hybrid-level interpolation is a
   model-specific transform today; a general utility could pull `sp` from a
   declared slot.
