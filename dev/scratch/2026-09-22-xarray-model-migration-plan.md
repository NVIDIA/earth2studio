# Remaining Model Xarray Migration Implementation Plan

> **For agentic workers:** Use the executing-plans skill to implement this plan
> inline, task by task. Keep detailed family-level steps in this scratch directory
> as each family is inspected. Do not delegate without user authorization.

**Goal:** Migrate all exported prognostic and diagnostic models and their callers
to native DataArray execution, reusing matching built-in grids.

**Architecture:** Coordinate signatures describe geometry without field allocation.
DataArrays carry labels and metadata through public calls, hooks, and iterators;
existing numerical kernels retain their native tensor or array representations.
Model-owned state and grid changes have explicit conversion boundaries.

**Tech Stack:** Python, Xarray, NumPy, PyTorch, CuPy, existing model dependencies,
pytest, pre-commit, Ruff, mypy, GitHub CLI.

## Execution and verification conventions

Work in `earth2studio-projected-grid-followup` on
`ngeneva/projected-grid-followup`. The base is release commit `33df61a1`.
Read the model, testing, and API documentation rules in `.agents/rules/` before
implementation. Read the repository model-creation skills for their validation
guidance; the newer `dev/spec/MODEL_CONTRACT_SPEC.md` governs the public API.

Run Python through `uv run` or an existing project virtualenv. The existing
`../earth2studio-pr-1154/.venv/bin/python` environment can run lightweight tests
with `PYTHONPATH=$PWD:/tmp/opencode/fetch-data-tools` without changing dependencies.
Confirm availability before using it.

For each model family, first convert an existing meaningful fixture/test to the
new contract, observe its failure against the old implementation, then migrate
the model and run that test plus the family suite. Retain numerical expectations;
do not replace execution assertions with signature-only checks. Commit each
verified family with explicit file staging and inspection of the staged diff.

## Task 1: Establish grid mapping and acceptance coverage

**Files:** `earth2studio/grids/__init__.py`,
`test/models/test_coordinate_signatures.py`,
`test/models/test_xarray_execution.py`, and the family files listed below.

- [ ] Inventory classes exported by both model package initializers, including
  subclasses sharing implementations and diagnostics omitted from `dx.__all__`.
- [ ] For each input and output domain, compare coordinates against registered
  grids. Use `grid="latlon-0.25deg"` for the exact 721-by-1440 geometry and
  `grid="latlon-0.25deg-south-pole-excluded"` for the exact 720-by-1440 geometry.
- [ ] Derive regional crops from `resolve_grid("hrrr-conus-3km")` when compatible,
  constructing `ProjectedGrid` with selected axes and the parent's CRS. Crops
  must not retain the full-domain registry ID.
- [ ] Check HEALPix ordering and layout explicitly. DLESyM's XY face layout cannot
  use `healpix-l6-nested`, even when the pixel count matches.
- [ ] For unmatched domains use existing `LatLonGrid`, `ProjectedGrid`,
  `CurvilinearGrid`, `PointGrid`, or `HEALPixGrid` definitions with their actual
  geometry. Preserve configurable grids and nongeographic dimensions.
- [ ] Add allocation-free signature and grid-metadata assertions to model tests.
  A fixed global signature should follow this construction:

```python
signature = coord_array(
    ("batch", "lead_time", "variable", "lat", "lon"),
    {"lead_time": np.array([0], dtype="timedelta64[h]"), "variable": variables},
    dynamic=("batch",),
    grid="latlon-0.25deg",
)
assert signature.data.nbytes == 0
assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
```

## Task 2: Lightweight models and conformance infrastructure

**Files:** `earth2studio/models/dx/identity.py`,
`earth2studio/models/px/persistence.py`, `earth2studio/models/px/datareplay.py`,
`earth2studio/models/conformance.py`, `earth2studio/models/px/utils.py`,
`test/models/dx/test_identity.py`, `test/models/px/test_persistence.py`,
`test/models/px/test_datareplay.py`, `test/models/test_conformance.py`,
`test/models/test_model_conformance.py`.

- [ ] Migrate Identity first. Its signature has a dynamic batch axis; output
  coordinates use `coord_array_like(input_coords)` and its call returns a shallow
  DataArray copy. Check field and coordinate identity with
  `xr.testing.assert_identical`, including name and auxiliary coordinates.
- [ ] Migrate Persistence's history selection, relative lead-time validation,
  DataArray hooks, and checkpoint state using FCN's metadata/state pattern.
  The initial yield selects `x.isel(lead_time=slice(-1, None))`; history updates
  use labelled concatenation along `lead_time`, preserving leading dimensions.
- [ ] Migrate DataReplay's fetch boundary to consume the DataArray returned by
  the current fetch API, preserving configured variables, domain, and lead times.
- [ ] Update conformance probes to concretize dynamic axes with
  `coord_array_like`, then create real fields using `from_torch`. Never allocate
  through a signature's `.values` or `.data`.
- [ ] Preserve every P1–P16 and D1–D10 check, aggregate violations, and retain
  explicit skip reasons. Test DataArray hook calls, input immutability, retained
  yields, shifted leads, and global RNG isolation with intentionally broken mocks.
- [ ] Run the three lightweight model suites and both conformance suites.

## Task 3: Regular-grid deterministic prognostics

**Files under `earth2studio/models/px/`:** `sfno.py`, `fengwu.py`, `fuxi.py`,
`pangu.py`, `dlwp.py`, `aifs.py`, `aifs2.py`, `aifsens.py`, `aurora.py`,
`graphcast_small.py`, `graphcast_operational.py`, `ucast.py`.
**Tests:** corresponding files under `test/models/px/`, including
`test_graphcast.py` and `test_pangu.py` for multiple exported classes.

- [ ] Inspect each wrapper's input history, output stride, normalization, and
  iterator-specific multistep schedule before editing it.
- [ ] Replace coordinate dictionaries on public signatures with allocation-free
  signatures and `handshake_dataarray` validation, preserving leading dimensions directly.
- [ ] Validate timedelta dtype, dimensionality, NaT, and relative history before
  computing output offsets from the final input lead time.
- [ ] Use a DataArray `_step` with `batch_func`, convert to the core representation,
  execute the unchanged numerical kernel, and reconstruct with `from_torch` or
  the equivalent native-array constructor. Preserve name and encoding.
- [ ] Keep hooks outside batch compression so they receive original leading axes.
  Preserve distinct Pangu schedules and FuXi model switching during rollouts.
- [ ] Convert the family tests and run mock inference, invalid-coordinate, and
  iterator tests on available CPU/CUDA devices after each wrapper.

## Task 4: Stateful, stochastic, and coupled prognostics

**Files under `earth2studio/models/px/`:** `fcn3.py`, `aifs2ens.py`, `atlas.py`,
`atlas_crps.py`, `aurora1p5.py`, `gencast_mini.py`, `weathernext2_cyclones.py`,
`ace2.py`, `samudrace.py`, `cbottle_video.py`.
**Tests:** corresponding files under `test/models/px/`, with
`test_weathernext2.py` covering WeatherNext2 variants.

- [ ] Identify recurrent buffers, forcing inputs, time coordinates, RNG objects,
  checkpoint bindings, and inherited overrides for every exported class.
- [ ] Migrate public signatures and execution using the native DataArray pattern;
  keep model-specific core packing explicit and reversible.
- [ ] Preserve per-model seeding behavior and existing conformance deviations.
  Test that migration itself introduces no new RNG interference or aliasing.
- [ ] Keep atmospheric/ocean component grids and internal coupling intervals
  distinct. Test a full public step rather than only an internal substep.
- [ ] Verify first yield, at least two forecast yields, hook placement, and
  checkpoint continuation wherever currently supported.
- [ ] Run each family's existing mock-weight suite and conformance checks.

## Task 5: HEALPix and regional prognostics

**Files under `earth2studio/models/px/`:** `dlesym.py`,
`dlesym_v0_isccp_era5.py`, `stormcast.py`, `stormcastconus.py`, `stormscope.py`,
`stormscope_meteosat.py`.
**Tests:** matching files under `test/models/px/`.

- [ ] Declare HEALPix level, ordering, layout, origin, and winding using the
  existing grid class; preserve face/height/width core ordering.
- [ ] Give latitude/longitude variants their own public signatures and construct
  output signatures from the actual output grid at each regridding boundary.
- [ ] Replace regional tensor-plus-CoordinateSystem public APIs with DataArrays.
  Derive projected axes from built-in parent grids when available; retain
  package-provided geometry where it differs.
- [ ] Migrate embedded conditioning fetches to the current DataArray API.
- [ ] Verify auxiliary latitude/longitude, CRS, cropped axes, multistep history,
  and retained-yield ownership using regional fixtures.

## Task 6: Diagnostics and derived fields

**Files under `earth2studio/models/dx/`:** `derived.py`, `climatenet.py`,
`precipitation_afno_v2.py`, `solarradiation_afno.py`, `wind_gust.py`,
`orbit2_precip.py`, `dlesym_v0_isccp_era5_precip.py`, `corrdiff.py`,
`corrdiff_cmip6.py`, `corrdiff_cosmo_era5.py`, `cbottle_sr.py`,
`cbottle_infill.py`, `cbottle_tc.py`, `stormscope_dx_nsrdb.py`, `tc_tracking.py`.
**Tests:** matching diagnostic suites, including `test_precip_afno_v2.py`,
`test_corrdiff_taiwan.py`, and every derived diagnostic in `test_derived.py`.

- [ ] Migrate signatures and single-field calls, preserving flexible-domain
  configuration and all exported subclasses.
- [ ] Use qualified output variable labels for existing time statistics.
  Recompute affected variable-dependent metadata through `coord_array_like`.
- [ ] Construct new signatures for superresolution and regional outputs, removing
  input spatial metadata that no longer describes the result.
- [ ] Keep tracking outputs on their actual track/feature dimensions; do not
  assign an image-grid identity to track values.
- [ ] Run numerical derived-field tests, mock neural diagnostics, spatial-output
  checks, stochastic checks, and diagnostic conformance.

## Task 7: Composition and inference callers

**Files:** `earth2studio/models/px/dxwrapper.py`,
`earth2studio/models/px/interpmodafno.py`, `earth2studio/run.py`,
`earth2studio/perturbation/hcbv.py`, affected model consumers in
`earth2studio/models/da/`, `recipes/`, and `examples/`.
**Tests:** `test/models/px/test_dxwrapper.py`,
`test/models/px/test_interpmodafno.py`, `test/perturbation/test_hcbv.py`,
affected runner and assimilation integration suites.

- [ ] Compose prognostic and diagnostic fields directly and validate intermediate
  variable/grid requirements using the new signatures.
- [ ] Update runner fetch/model boundaries to DataArrays; explicitly convert at
  still-legacy IO and perturbation boundaries rather than changing their APIs.
- [ ] Remove obsolete `cast(Callable, ...)` model-call bridges where migrated.
- [ ] Update recipe and example model calls, iterator unpacking, hook signatures,
  and signature-coordinate access. Preserve their observable inference behavior.
- [ ] Test deterministic and ensemble runs, diagnostic composition, HCBV, and
  assimilation components that call migrated models.

## Task 8: Full verification and release PR

- [ ] Audit all exported classes against the inventory, including inherited
  methods; search for remaining public tensor/coordinate-pair calls.
- [ ] Audit every model's grid choice against built-ins and record reasons for
  custom definitions in the implementation or existing model documentation.
- [ ] Run `test/models`, affected runner/perturbation tests, and coordinate/grid
  utility tests through the selected project environment. Investigate failures;
  distinguish optional dependency and weight-download skips from passing tests.
- [ ] Run `make lint` and `uv tool run pre-commit run --all-files`; rerun any hooks
  that modify files and recheck affected tests after behavioral edits.
- [ ] Review `git status`, `git diff`, recent commits, and the complete release-base
  diff; commit only intended changes, including scratch document relocation.
- [ ] Follow `skills/developer-open-refactor-pr/SKILL.md`, push upstream, and create
  the approved PR titled `[E2S 1.0] Migrate remaining models to xarray execution`,
  targeting `1.0.0-rc` with label `1.0.0`.
- [ ] Confirm remote head and CI status; update the shared coordination file and
  return the PR URL with actual verification results. Do not post PR comments.
