# nvcoupler

Couple Earth2Studio models on one clock.

A `Coupler` advances ordinary `PrognosticModel` and `DiagnosticModel` instances
together and serves each model's declared forcing from the others' outputs.
It is a `Runner`, so `Pipeline` drives coupled runs like any other. The design
follows NUOPC/ESMF: models advance in a run sequence derived from their
cadences, and exchanges are matched by variable label, regridded between grids
and reduced over time windows when a model asks for one.

```python
from earth2studio.nvcoupler import Connector, Coupler
from earth2studio.run import Pipeline, work_items

coupler = Coupler(
    {"atmos": atmos, "ocean": ocean},
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={"atmos": (GFS(), None), "ocean": (ORAS5(), None)},
)
print(coupler.describe())
Pipeline(coupler, output={"atmos": atmos_store, "ocean": ocean_store}).run(
    work_items(times, final_lead_time=np.timedelta64(240, "h"))
)
```

## Public API

- `Coupler`: the coupled system and its runner. Holds all runtime state.
- `Connector`: a frozen declaration of which destination labels a source
  serves, with an optional `Regridder` and `transform`.
- `CouplingError`: the one error type; subclasses `ValueError`.
- `RunSequence`, `Slot`, `derive_sequence`: semi-public, the order models
  advance, for overrides and inspection.

There is no component base class. Everything the coupler needs comes from the
model contract (`dev/spec/MODEL_CONTRACT_SPEC.md`):

- imports: `forcing_coords()` of a prognostic, `input_coords()` of a diagnostic;
- exports: `output_coords()`;
- cadence: final output lead minus final input lead for prognostics; the
  feeding models' cadence, or `cadence=`, for diagnostics;
- grid: the signature, via `earth2studio.grids.infer_grid`;
- initial conditions and unserved forcing: `default_sources()`, or `sources=`.

## How exchanges work

**Matching.** Labels are Earth2Studio lexicon names. A destination import
`"z1000"` is served by a source exporting `"z1000"`; a qualified import such as
`"z1000:mean:48h"` is served by reducing the source's `"z1000"` frames with
`earth2studio.utils.time_statistics` over that window (half-open, ending at
the required lead; the window must be a multiple of the source's cadence). A
connector without `variables` serves every destination import its source can;
if there are none, it overwrites matching **state** variables (outputs that are
also inputs) of a prognostic destination before each step, for models that
carry a coupled field in their own state rather than a forcing slot.

**When.** A prognostic step needs its forcing at the slot's final declared
lead shifted by the lead of the `y` being advanced, as in the model contract.
The coupler keeps a short buffer of each model's frames and serves exactly
those leads. A source slower than the destination holds its last frame
(zero-order hold); a missing frame from a source of equal or faster cadence
is an error naming the run order. Diagnostics take the newest frame at or
before the current lead. Where a window reaches before the initial condition,
or the source has not produced a frame yet, the destination's bound data
source fills in.

**Order.** `derive_sequence` puts one slot per cadence, fastest first, so a
faster model always advances before a slower one at a shared step. Within a
slot, a destination whose declaration needs a frame from the *same* step (a
forcing lead beyond its input time, or any diagnostic) runs after its source;
otherwise models keep declaration order and see their partner's previous
frame (lagged coupling). Pass `sequence=` to override; it is validated
against these dependencies.

**Grids.** When source and destination grids differ, the coupler resolves a
built-in regridder (`bilinear` for structured sources, else `nearest`) through
`earth2studio.grids.resolve_regridder`, or uses the connector's `regridder`,
whose `source_grid` and `target_grid` must match the two models. Reductions
happen on the source grid, before regridding. State overwrites require an
exact coordinate match after regridding.

**Streams.** Each step of `run_item` yields the outputs of the models that
advanced, keyed by model name (`name.1` for a second output slot), with
dimensions `time` then `lead_time` and the field dimensions. The first step
publishes the prognostic initial conditions and any diagnostics computable
from them. `output_coords(final_lead_time)` gives each stream's schema and
`data_requests(item)` every fetch, or `None` when a prognostic input has no
source and must come from `ics=`.

## Examples

See `examples/09_nvcoupler/` and the toy models in
`earth2studio.nvcoupler.testing`.

## Scope

In-memory coupled inference. Not included, by design: training rollouts,
YAML or text configuration, vertical-coordinate interpolation (a general
utility, when a model needs it), and model-owned data fetching, which the
model contract forbids. Ensemble member batching is not implemented yet.
