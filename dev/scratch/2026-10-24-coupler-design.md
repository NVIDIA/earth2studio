# Coupler design report (1.0.0-rc)

Status: implemented in `earth2studio/coupler/` (one file, ~390 lines, 22 tests).
Reference only the `1.0.0-rc` branch. Related: PR #1229 (nvcoupler, draft).

## Goal

Connect prognostic models that feed each other (StormScope GOES -> MRMS, an
atmosphere and an ocean at different cadences) and produce an ordinary
`earth2studio.run.Runner`, so `Pipeline(coupled_runner, output=...)` works with
no changes to `earth2studio.run`. It must stay small: no DAG, connector or node
concepts, only a clock and an entry point for the exchange logic.

## What the 1.0 contracts already give us

| Coupler need | Existing mechanism |
| --- | --- |
| Native timestep | `output_coords lead_time[-1] - input_coords lead_time[-1]` |
| Grid a forcing slot expects | `forcing_coords()` signature |
| Feed new data between steps | `create_iterator(...).send(forcing)` |
| Initial conditions / external forcing | `recommended_sources`, `fetch_data` |
| Output streams, resume, IO, distribution | `Runner` protocol + `Pipeline` |

The component *is* the `PrognosticModel`; the runner protocol *is* the driver.

## Implementation

`earth2studio/coupler/runner.py` exports `CoupledRunner` (plus the `Coupling` and
`Outputs` type aliases). Nothing else.

```python
runner = CoupledRunner(
    {"atmos": atmos, "ocean": ocean},         # name -> model; dict order = run order
    sources={"atmos": a_src, "ocean": o_src}, # optional: recommended sources
    couplings={                               # optional: model -> function
        "atmos": lambda out, leads: out["ocean"].sel(lead_time=leads, method="pad"),
        "ocean": lambda out, leads: out["atmos"].sel(
            lead_time=slice(leads[0] + H6, leads[0] + H48)
        ).mean("lead_time", keepdims=True),
    },
)
Pipeline(runner, output={"atmos": zarr_a, "ocean": zarr_b}).run(items)
```

- **Coupling = a function** `(outputs, leads) -> DataArray`. `outputs[name]` is
  that model's outputs from the previous window's last frame to now, along
  `lead_time`; `leads` are the lead times the forcing must be valid at. It
  supplies the model's *first* forcing slot (time-varying). Hold, window means,
  regridding, unit conversion and masking are ordinary xarray / `Regridder` calls
  inside it; the runner ships no reducers.
- **Clock.** Window = slowest model step (override `window=`); each step must
  divide it. One runner step = one window. Within a window models run in dict
  order, so a model sees earlier models' *new* outputs and later models' outputs
  as of the end of the previous window. Order is the lagged/sequential switch.
- **Checks.** Probe at construction (single in/out slot, one output lead,
  positive step, window divisibility, initial source present, uncoupled forcing
  slots have a source, couplings name real models with a time-varying slot).
  On every call a coupling's result must have one entry per requested lead and
  the forcing slot's spatial coordinates, else a clear error says to regrid.
- **Everything else** (initial conditions, other forcing slots, static fields)
  comes from sources exactly as in `PrognosticRunner`, via its private helpers.
  Stochastic models are seeded with `seed + crc32(name)` per member/time.
- **Streams** are named after the models; IC first, then per window.
  `supports_member_batching=False`.

Removed in the final simplification (all superseded by plain functions):
`Link`, auto-wiring, provider builders (`latest`/`reduced`/`then`), `NotReady`,
`window_*` reducers, scale/offset, units attr check, diagnostics inside the
coupled loop (apply a diagnostic to a stream afterwards, or inside a coupling),
`check_finite`, cycle/deadlock scheduling, `describe()`.

Not in scope: ensembles/member batching, mid-item resume, multi-slot or
multi-lead prognostics, conservative regridding.

## Lighthouse cases

1. **StormScope GOES + MRMS** (equal cadence, same grid):
   `couplings={"mrms": lambda out, leads: out["goes"].sel(lead_time=leads, method="pad")}`.
   GOES runs first in each window, so MRMS at lead `L` sees GOES at `L`. Replaces
   the hand-written loops in `examples/04_nowcasting/03_stormscope_goes_example.py`
   and `recipes/eval/src/pipelines/stormscope.py`. Needs real-weights verification.
2. **Atmosphere + ocean** (6 h atmosphere, 48 h ocean, DLESyM / SamudrACE-style):
   the example above. Atmosphere runs 8 steps per window with SST held; the ocean
   steps once with the 48 h mean atmosphere. Needs the fused DLESyM split into two
   models first.

## Mapping to nvcoupler (PR #1229)

| nvcoupler | Here |
| --- | --- |
| Components | The model itself |
| `Connector(time_policy, window, reduce, regridder)` | A user function in `couplings` |
| `Mediator` | The same function (xarray reduction) |
| `FieldDictionary` | Lexicon variable names (inside the function) |
| Run sequence lag vs sequential | Model order in the dict |
| `Driver` + `Clock` | `CoupledRunner` + window |
| `couple()` auto-wiring, YAML DSL, `collect` | Dropped |

## Survey of existing coupling frameworks

(Subagent reads of docs/code; items marked unverified were not read.)

- **legoESM** (JAX monolithic ESM): coupling is hard-coded sequential/lagged in
  drivers; the declared `Interface(form, transfer, regrid, coupling)` bricks are
  unused scaffolding. Useful: dt-weighted `FluxAccumulator` with residual
  carry, identity-remap fast path, fail-loud on missing/non-finite fields (silent
  zero defaults produced perpetual polar night), integer coupling-interval
  checks. Lesson: do not over-build interfaces before use.
- **ESMF/NUOPC**: Driver/Model/Mediator/Connector, advertise -> match by standard
  name -> realize. Lag vs sequential is purely connector position in the run
  sequence; child must advance exactly one parent step (integer multiple);
  no built-in averaging (mediators do it). Borrowed: name matching, probe before
  run, integer-multiple check. Skipped: phases, IPD, PET lists, namespaces.
- **OASIS3-MCT / YAC**: declarative couple record (src/dst, period,
  time_reduction instant/average/accumulate/min/max, lag, scale/offset,
  interpolation fallback stack); models just put/get with their own date;
  YAC requires coupling period to be an integer multiple of both timesteps. Borrowed: reduction
  vocabulary, period validation, lag as previous value, transform per link.
  Skipped: MPI, weight files, restart files.
- **ClimaCoupler.jl / MUSCLE3**: ClimaCoupler sub-cycles to a coupling dt and uses
  a flux accumulator for slow surface models; MUSCLE3 uses ports/conduits with
  strict-causality bridges, no many-to-one. Borrowed: sub-cycling by cadence
  ratio, accumulate-then-push for slow components, graph validation at
  construction (one producer per input).
- **preCICE / CMEPS / SamudrACE / DLESyM paper**: preCICE serial-explicit =
  sequential, parallel-explicit = lagged, window loop with subcycling, only the last
  sample is exchanged by default; CMEPS averages fast-period fluxes before
  mapping (linear ops, average on source grid, map once); SamudrACE averages 20
  six-hourly atmosphere flux steps into one 5-day ocean step, atmosphere forced by
  SST held for the window; DLESyM (96 h cycle) feeds time-averaged atmosphere
  fields over 48 h windows to the ocean. These define the `mean` and held-`last`
  semantics above. Alignment of averaging windows with ocean timestamps matters
  (SamudrACE Text S1): the window here is `(t, t + step]`.

## Open questions

1. Ensembles need member batching in the runner (contract lists it as open).
2. Fused multi-lead models (DLESyM) must be split model-side.
3. Automatic/conservative regridding: today the caller applies a bound `Regridder`
   (`grids.resolve_regridder` needs the `utils` extra).
4. Real-weights verification of StormScope equivalence to the `eval` pipeline.
5. Should the `Coupling` call signature gain the step/time, or a small helper
   module (hold, window_mean) once real use shows which patterns repeat?
