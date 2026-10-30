---
name: earth2studio-couple-models
description: >
  Use when running two or more Earth2Studio models together so one feeds
  another (e.g. StormScope GOES to MRMS, atmosphere and ocean at different
  cadences, a weather model driving an impact model): wires components and
  connectors into a CoupledRunner, checks the schedule, and runs it under
  Pipeline. Not for single-model forecasts, new model wrappers, or data sources.
argument-hint: The models to couple, and which outputs feed which slots (optional)
---

# Couple Models

`earth2studio.coupling.CoupledRunner` runs named components, each at its own
cadence and on its own grid. `Connector` edges pass fields between them. The
schedule is derived from what each model declares, and a coupled run is a
`Runner`, so `Pipeline` writes one store per stream. Components are ordinary
models. Do not wrap or subclass them, and do not add pair-specific code to the
runner: express the exchange with connectors.

Use `PrognosticRunner` instead when only one model runs and its forcing comes
from data sources.

## Workflow

1. **Read the declarations, not the docs.** For each model, print
   `input_coords()`, `forcing_coords()` and `output_coords(input_coords())`, as
   `inspect` in the [template](references/coupling-template.py) does.
   Record for each model:
   - its kind: stateful (`initialize`/`step`) or diagnostic,
   - its cadence: last output lead minus last input lead,
   - each slot's variables, including qualified labels such as `tp:sum:24h`,
   - its lead times and grids.

   Then list, slot by slot, which producer variable feeds which consumer slot.
   Anything left over comes from a source.
2. **Choose a connector for each fed slot** using the
   [patterns](references/connector-patterns.md):
   - `exact` for the same valid time or a reduction, `hold` for a lag or to
     break a feedback loop;
   - `variables` for renames;
   - a regridder whenever grids differ;
   - `state=True` for fields the model carries in its own state;
   - a diagnostic component between models for any conversion: derived
     variables (`DerivedWS`), units, or vertical levels
     (`DerivedPressureLevels`). Connectors never convert.

   State the timing you chose for each edge and why.
3. **Build the runner** from the [template](references/coupling-template.py):
   components in a dict, connectors, `sources` in input-then-forcing slot order
   (`None` keeps a recommendation), and optionally `devices` and `seed`.
   Construction validates everything; fix each `CouplingError` using the table
   below.
4. **Check the schedule** with `print(runner.describe(final_lead_time))` over a
   few cadences of the slowest component, before running anything. Each row is
   a call: when it is due, what it outputs, and where each connected slot reads
   from. `[forcing[0] <- land at +6h from +0h]` means the slot needed +6h and
   got the +0h frame (a `hold` lag); `from +0h..+18h` is a reduction window.
   Confirm every lag and window is the intended one.
5. **Run one item**, `runner.run_item(WorkItem(time, final_lead_time))`, and
   sanity-check a few values per stream. Then run
   `Pipeline(runner, output={stream: backend, ...}, device=...)`.
6. **Test** with hand-computable toys that mirror the real cadences, labels and
   grids. Follow the [testing template](references/testing-template.py): assert
   exact published values, the lead times each component was forced at, the
   `describe` rows, and that wrong wiring raises `CouplingError`. Tests with real
   weights are marked `@pytest.mark.package`.
7. **Verify** with the commands below, and report anything you did not run.

```bash
uv run python skills/earth2studio-couple-models/references/coupling-template.py
uv run pytest <your tests> -q
make format && make lint
```

## Errors

`CouplingError` is a `ValueError`. Its message names the rule it broke.

| Rule | Typical cause | Fix |
| --- | --- | --- |
| `C1` | Outputs not gridded, or output slots at different lead times | Split the model into components |
| `C2` | Diagnostic with no `exact` connector | Feed it with at least one `exact` edge |
| `C3` | Two connectors on one slot, or a slot that is not time-varying gridded forcing | One connector per slot; check `slot=` |
| `C4` | Label not published by the producer, or published by two output slots | Use `variables=` to rename, or pick the right producer |
| `C5` | Grids differ without a regridder, or the regridder's grids are wrong | `BilinearRegridder(src_grid, dst_grid)`, built from `infer_grid` |
| `C7` | `hold` on a qualified label | Use `exact`, or declare the slot at an earlier lead |
| `C8` | Reduction window not a multiple of the producer's spacing | Fix the window or the producer's cadence |
| `C11` | Cycle of `exact` edges, or a frame that never arrives | Make one edge `hold`; check the cadences and the final lead |
| `C12` | Two output slots resolve to the same stream name | Rename a component |

The messages suggest close names for misspelled components and variables.

## Common Mistakes

- **Expecting list order to sequence the models.** Order follows the edges; dict
  order only breaks ties between components ready at the same tick.
- **Using `hold` where `exact` was meant.** This silently lags by a step.
  Check the `describe` row.
- **Reductions use the window `[T - w, T)`.** The frame at `T` is excluded.
- **Expecting an initial condition in every stream.** Streams whose outputs
  differ from the inputs start at the first forecast.
- **Regridding the wrong way round.** The source grid is the producer's output
  grid and the target is the consumer's slot grid.
- **Converting inside a connector.** Connectors only select, reduce and
  regrid; put conversions in a diagnostic component.
- **Coupling a fused multi-cadence model.** Split it into components first.

## Reference Map

| Reference | Read when |
| --- | --- |
| [Coupling template](references/coupling-template.py) | Always; runnable CPU starting point |
| [Connector patterns](references/connector-patterns.md) | Choosing timing, variables, grids, units, state, sources, devices |
| [Testing template](references/testing-template.py) | Writing tests for a coupling |
| [Coupling package](../../earth2studio/coupling/) | Docstrings for `Connector` and `CoupledRunner` |
