**Point your coding agent here:** “Read `migration.md` on `1.0.0-rc`. Convert coordinates
first, then migrate each affected component to its new protocol. Verify outputs.”

# Earth2Studio 1.0 migration checklist

Compared: `main` → `1.0.0-rc`. Trust RC code over old examples.

## New concepts: coordinates first

Update custom components to the RC protocols linked below. Shared changes:

- **Coordinate object:** `CoordSystem` (ordered dictionary) → `CoordinateSystem`
  (`xr.DataArray` signature, from `earth2studio.utils.type`). Build with `coord_array`;
  derive with `coord_array_like`. Signatures describe shape/dtype/coordinates, hold no field
  values, and allocate no field storage. Do not read their `.values`.
- **Grid object:** use `grid=` whenever possible. Pass a registered grid name or
  `GridDefinition` from `earth2studio.grids`; it supplies spatial axes, geometry, and CRS.
  Keep native projected/curvilinear axes and auxiliary lat/lon. A CRS alone is not a grid.
- **Field object:** `(torch.Tensor, coords)` → one `xr.DataArray` containing values and
  metadata. Coordinate signatures and concrete fields are different uses of DataArray.
- **Slots:** one input per positional argument; one output directly, multiple as a tuple.
  Implement fixed named parameters. `output_coords(...)` takes one signature per input slot.
  Pack compatible fields together along `variable`, as in model inputs. Separate arguments
  mainly for different grids; avoid unnecessary slots.

Example: replace a `variable, lat, lon` coordinate dictionary with a grid-backed signature:

```python
from earth2studio.grids import LatLonGrid
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch

signature = coord_array(
    dims=("variable", "lat", "lon"),
    coords={"variable": old_coords["variable"]},
    grid=LatLonGrid(old_coords["lat"], old_coords["lon"]),
)
x = from_torch(tensor, signature)  # Tensor shape/order must match signature.
signature = coord_array_like(x)   # Describe an existing field without copying values.
```

- Dictionary key order → `.dims`; lengths → `.sizes[dim]`; labels → `.coords[dim].values`.
  Select/reorder fields with `.sel()` / `.transpose()`. Preserve metadata when wrapping outputs.
- Empty wildcard axes → explicit leading `dynamic=(...)` in `coord_array`. Resolve them
  before execution or storage. Change spatial geometry with a new `coord_array(grid=...)`;
  use `coord_array_like(..., coords=...)` for non-spatial changes.
- Torch internals: `tensor, _ = x.e2s.to_torch()`; return `from_torch(result, output_signature)`.
  Default backend: NumPy on CPU, CuPy on CUDA. `backend="torch"` preserves autograd when
  wrapping tensors. Match model/data devices; use `.e2s.as_numpy()` for explicit CPU export.
- Preserve variable labels, including temporal qualifiers such as `tp:sum:6h`.
  Use datetime arrays for `time`, timedelta arrays for `lead_time`.

## Diagnostic models

Target: [DiagnosticModel](earth2studio/models/dx/base.py).

- `y, out_coords = model(tensor, coords)` → `y = model(x)`.
  Multiple inputs: `model(x1, x2)`; multiple outputs: `y1, y2 = model(...)`.
- Convert `input_coords()` / `output_coords(...)` dictionaries to coordinate signatures.
  Preserve concrete leading dimensions; validate fields with `handshake_dataarray` /
  `handshake_time`. Returned metadata must match `output_coords(...)`.
- Add `default_sources()`: one source, a tuple with one source or `None` per input, or `None`.
  Fetch outside the model. Diagnostics have no forcing slots. Keep `.to(device)`.

## Data assimilation models

Targets: [AssimilationModel / StatefulAssimilationModel](earth2studio/models/da/base.py).

- Gridded `CoordSystem` declarations → coordinate signatures. Keep `FrameSchema` for
  tabular slots and pandas/cuDF payloads. Pass signatures separately to `output_coords(...)`.
- Always-tuple outputs → one analysis directly, multiple analyses as a tuple.
  Preserve documented optional observation `None` slots and model-specific request metadata.
- `init_coords()` → initialization slots in `input_coords()`; recurring conditioning →
  `forcing_coords()` on stateful models. Respect each model's observation window.
- `create_generator()` → independent `model(...)` calls, or stateful `initialize`, `step`,
  `create_iterator`. Initialize with inputs then full forcing windows; returns `(analysis, state)`.
  Step takes previous outputs, new dynamic forcing, and `state=state`; omit static forcing.
  Declare `state` positional-or-keyword after slots. Iterators receive forcing via `.send(...)`;
  remove priming `None` / initial-condition yields. Execution hooks, if offered, are iterator-only.
- Add `default_sources()` in input-then-forcing order and `stochastic: bool`.
  Implement model-local `set_rng(seed, reset=True)`; deterministic DA uses a no-op.
  State must be serializable/replayable. Save analysis and state; mutate neither inputs nor state.

## Prognostic models

Target: [PrognosticModel](earth2studio/models/px/base.py).

- Tensor/dictionary calls → DataArray calls and coordinate signatures, as for diagnostics.
  Input history leads are increasing, relative to initialization, and end at zero.
- Hidden rollout state → `y, state = model.initialize(x)`, then
  `y, state = model.step(y, state=state)`. Multiple outputs: `step(*y, state=state)`.
  `model(...)` returns initialization's forecast and starts fresh each call.
- Implement `forcing_coords()` and `default_sources()` (input-then-forcing order).
  Initialize with inputs then full forcing windows; step with outputs then new dynamic forcing.
  Step forcing lead = final declared forcing lead + current output lead. Static forcing
  slots (no `lead_time`) are initialization-only. Fetch forcing outside the model.
- Iterator first yield is now the first **forecast**, not the initial condition.
  One yield is one core computation, possibly several leads. Supply new forcing with `.send(...)`.
  Hooks use DataArrays/tuples and run only in iterators: rear after initialization/steps,
  front before subsequent steps. Hook results feed the next step.
- State holds continuation history/statics/RNG; save both `y` and state for restart.
  Never mutate inputs or prior state. Declare `state` positional-or-keyword after array slots.
  `PrognosticMixin` supplies private helpers; implement fixed-signature public methods.
- Declare `stochastic`; stochastic models use model-local `set_rng(seed, reset=True)`.
  For `DomainAdaptable` models, assign `model = model.with_region(region)` and re-query signatures.

## Data sources and fetching

Targets: [source protocols](earth2studio/data/base.py), [fetch_data](earth2studio/data/utils.py).

- Sources already returned DataArrays/DataFrames; keep their source call signatures.
  Follow the `device` convention: NumPy/pandas on CPU, CuPy/cuDF on CUDA.
- `tensor, coords = fetch_data(...)` → `x = fetch_data(...)`; remove `legacy=...`.
  Gridded fetch output order is `time, lead_time, variable, ...`.
- `interp_to` / `interp_method` → `target_grid` / explicit `regridder`.
  Use `infer_grid(model.input_coords())` for a single gridded slot. A target alone handles
  matching/cropping/nearby selection; provide a `Regridder` for the intended interpolation.
- Fetch all declared variables/history. Temporal qualifiers need `delta_t` or source
  `time_step`; windows include the left endpoint, exclude the right. Do not aggregate
  already aggregated source fields again.
- Planned: declare supported statistics in lexicon `TIME_STATISTICS`; handle them per source.

## IO objects

Target: [IOBackend](earth2studio/io/base.py).

- `add_array(coords, names)` → `add_array(template)`; `write(tensor, coords, names)` → `write(x)`.
  Names now come from `variable` labels, or `x.name` when there is no variable dimension.
- Build the full output extent with `earth2studio.io.utils.output_template(...)` from an
  output signature. Resolve all dynamic axes; include every time/member/lead to be written.
  Creation reads metadata only, never field values; stores do not grow automatically.
- Writes select positions by coordinate label. Keep dimension order and native spatial axes;
  preserve auxiliary coordinates/CRS in the template. Pass device-backed fields directly.
- Custom backends must add `flush()` and `close()`. Flush before readback; close to finalize
  and release resources, including committing transactional stores.

## Perturbations

Target: [Perturbation](earth2studio/perturbation/base.py).

- `tensor, coords = perturbation(tensor, coords)` → `x = perturbation(x)` for every method.
  Custom perturbations return a DataArray with unchanged shape, dimensions, coordinates,
  and attributes. Never modify the input in place; use the Torch bridge for tensor math.
- `BredVector` / `HemisphericCentredBredVector`: migrate supplied model and seed-perturbation
  callbacks to `DataArray → DataArray`. Tensor/coordinate callbacks no longer work.
- Keep method-specific geometry: Brown/spherical methods require trailing `lat, lon`;
  spherical grids require N:2N or N+1:2N. `LaggedEnsemble` requires leading
  `ensemble, time, lead_time`, with one member per lag. Source-backed methods fetch on input device.
- Preserve noise amplitudes, seeds, and checkpoint behavior. Perturbation RNG is distinct from
  the model `set_rng` contract; Gaussian retains its own checkpointable Torch generator.

## Statistics and metrics

Targets: [Statistic / Metric](earth2studio/statistics/base.py).

- `stat(tensor, coords)` → `stat(x)`; `metric(tensor, coords, truth, truth_coords)` →
  `metric(x, truth)`. Each returns one DataArray, not a tensor/coordinate tuple.
- Convert `output_coords(...)` to coordinate signatures; keep `reduction_dimensions`.
  Reduced dimensions disappear. Align prediction/observation coordinate values explicitly.

## Workflows after component migration

Keep `run.deterministic` / `diagnostic` / `ensemble` for single-input/output models without
external forcing. They retain `nsteps`, dictionary `output_coords`, and initial output;
custom iterator loops must publish initial fields separately. Update custom callbacks and
IO using the sections above. For new orchestration, see [Pipeline](dev/examples/09_pipeline.py).

## Verify

- Compare two advances: values, variables, dimensions, grids, units, lead times.
  Check forcing, hooks, IO readback, restart, and exactly one initial output where expected.
- Run model tests; PX/DX: `check_prognostic_contract` / `check_diagnostic_contract`
  from `earth2studio.models.conformance`. DA: [test patterns](test/models/da/).
