**Point your agent here:** “Read `migration.md` on `1.0.0-rc`. Migrate my workflows
and models. Apply relevant changes only. Verify outputs.”

# Earth2Studio 1.0 migration checklist

Compared: `main` (`f8af4030`) → `1.0.0-rc` (`fa15ee13`). Trust RC code over old examples.

## Workflows

- **Arrays:** `(tensor, coords)` → `xr.DataArray`. Use `x = fetch_data(...)`,
  `y = model(x)`. Remove `legacy=...`. Multiple slots: `model(x1, x2)`; outputs: tuple.
- **Coordinates:** use `.dims`, `.sizes`, `.coords[dim].values`; align with `.sel()` /
  `.transpose()`. Fetch full history. `time`: datetime array; `lead_time`: timedelta array.
- **Grids:** `interp_to` / `interp_method` → `target_grid` / explicit `regridder`.
  Get target with `earth2studio.grids.infer_grid(signature)`. Preserve axes, lat/lon, CRS.
- **Variables:** copy declared labels, including `tp:sum:6h`. Temporal fetching needs
  `delta_t` or source `time_step`; windows include left endpoint, exclude right.
- **Torch:** `tensor, _ = x.e2s.to_torch()`; wrap with
  `earth2studio.utils.cupy.from_torch(tensor, signature)`. Default: NumPy CPU / CuPy CUDA.
  Use `backend="torch"` to retain gradients. Match model/data devices.
- **Rollout:** `y, state = model.initialize(x)`; then
  `y, state = model.step(y, state=state)`. Multiple outputs: `step(*y, state=state)`.
  Save **both** for restart. Repeated `model(x)` starts fresh.
- **Forcing:** initialize with inputs, then full forcing windows. Step with outputs,
  then new forcing. Required lead = forcing's final declared lead + current output lead.
  Static slots (no `lead_time`): initialization only. Inspect `forcing_coords()`;
  `default_sources()` recommends input-then-forcing providers. Fetch outside models.
- **Iterators:** first yield = first forecast. One yield = one core computation,
  possibly several lead times. Custom loops publish initial fields separately.
  Send new forcing via `.send(...)`; use `next()` only when none required.
- **Hooks:** accept/return DataArrays or output tuples; iterator-only.
  Rear runs after initialization/steps; front before subsequent steps.
- **Other calls:** perturbation/statistic → `op(x)`; metric → `metric(x, y)`.
  IO → `add_array(template)`, `write(x)`, `flush()` / `close()`.
  Template covers all output labels; creation writes no values. Keep native spatial axes.
- **Options:** regional model → `model = model.with_region(region)`; re-query signatures.
  Stochastic model → `model.set_rng(seed, reset=True)`.

## Workflow runners

- Keep `run.deterministic` / `diagnostic` / `ensemble` for single-input/output workflows
  without external forcing. They retain `nsteps`, dictionary `output_coords`, and initial output.
- For managed forcing/distribution/resume: import `Pipeline, work_items` from
  `earth2studio.run`; build `items = work_items(times, final_lead_time=duration)`;
  call `Pipeline.from_model(model, source, output={"forecast": io}).run(items)`.
- Add `forcing=(...)`, `diagnostics={"name": model}`; filter/regrid per output stream.
  Ensembles: `members=...`, `member_batch=1`. Built-in runner seeds stochastic models;
  no initial-condition perturbation. Multiple input/output slots, perturbations, or
  batching need custom runners/loops. Standalone diagnostics: `DiagnosticRunner`.
- `progress=ProgressStore(path)` skips finished items; interrupted items restart fully.
  See [Pipeline example](dev/examples/09_pipeline.py).

## Custom model wrappers

- **Signatures:** `CoordSystem` → `CoordinateSystem` (`earth2studio.utils.type`). Build
  with `coord_array` / `coord_array_like` (`earth2studio.utils.coords`). No field allocation.
  Wildcards: leading `dynamic=(...)`; resolve before execution. Geometry: `grid=...`.
  `output_coords(...)`: one argument per input slot; preserve concrete leading dimensions.
- **Contract:** fixed named slot parameters. Validate with `handshake_dataarray` /
  `handshake_time`. Match planned output metadata. Never mutate inputs or prior state.
  Add `default_sources()` and `stochastic: bool`; stochastic RNG must be model-local.
- **PX:** implement `initialize`, `step`, `forcing_coords`, `__call__`, `create_iterator`.
  Call = initialization forecast. State holds history/statics/RNG; serializable, replayable.
  Declare `state` positional-or-keyword after arrays. Mixin supplies private helpers only.
- **DX:** DataArray inputs/outputs; no forcing slots.
- **DA:** `init_coords()` → initialization slots in `input_coords()`.
  `create_generator()` → independent calls or `StatefulAssimilationModel` execution.
  Recurring conditioning → `forcing_coords()`. No priming yields. Keep `FrameSchema`,
  DataFrames, optional observation `None` slots. Deterministic DA: no-op `set_rng`.

References: [PX protocol](earth2studio/models/px/base.py),
[DX protocol](earth2studio/models/dx/base.py), [DA protocol](earth2studio/models/da/base.py),
and [model contract](dev/spec/MODEL_CONTRACT_SPEC.md).

## Verify

- Compare two advances: values, variables, dimensions, grids, units, lead times.
  Check forcing, hooks, IO readback, restart, and exactly one initial output where expected.
- Run model tests; PX/DX: `check_prognostic_contract` / `check_diagnostic_contract`
  from `earth2studio.models.conformance`. DA: [test patterns](test/models/da/).
