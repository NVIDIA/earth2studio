# Remaining model xarray migration

## Goal and scope

Migrate every remaining public prognostic and diagnostic model to the xarray
execution contract in `dev/spec/MODEL_CONTRACT_SPEC.md`. Follow the native
implementations introduced by PR #1168 in FCN, FuXi-S2S, and PrecipitationAFNO.
Include affected inference callers, composition wrappers, tests, and API examples
needed to exercise these models through their new interface.

The user approved this scope and PR title on 2026-09-22. The working branch is
`ngeneva/projected-grid-followup`, based on `upstream/1.0.0-rc` at `33df61a1`.
That release base includes the merged coordinate, execution, fetch, and projected
grid changes. Assimilation models remain outside this contract; their use of
migrated prognostic or diagnostic components must still work. Preserve the
existing Random and Random_FX data-source APIs.

## Architecture

### Public model interfaces

- `input_coords()` returns an allocation-free `CoordinateSystem` DataArray.
- `output_coords(input_coords)` validates and returns an allocation-free output
  signature without allocating a model field or running inference.
- `__call__(x: xr.DataArray) -> xr.DataArray` owns its output and borrows its input.
- Prognostic `create_iterator(x)` yields DataArrays, starting with the final input
  history entry, followed by complete forecast steps.
- Prognostic iterator hooks accept and return a single DataArray and operate on
  the caller's leading dimensions. Single-step calls do not invoke these hooks.

Migrate implementations natively. Do not install class-level compatibility
wrappers that expose DataArrays while retaining legacy public pair interfaces.
Internal numerical kernels may continue using tensors, NumPy, JAX, or external
model-specific data structures. Use the existing labelled-array conversion
utilities at those boundaries, including `.e2s.to_torch()` and `from_torch()`.

### Coordinate signatures and grids

Use `coord_array` and `coord_array_like`, registered grids, and concrete
`GridDefinition` objects. Read dimensions through `.dims`/`.sizes` and labels
through `.coords`; never materialize signature field storage.

Every model must first check for a matching built-in registered grid and reuse
it when available. Match coordinate values, axis order, pole inclusion, CRS,
and, for HEALPix, ordering, layout, origin, and winding. Prefer existing grid
classes for geometries without a registered match. Do not duplicate built-in
axis definitions in model wrappers or attach a full-domain grid ID to a crop.
Current registered choices include `latlon-0.25deg`,
`latlon-0.25deg-south-pole-excluded`, `hrrr-conus-3km`, and
`healpix-l6-nested`. A custom or cropped domain uses an appropriate concrete
grid definition derived from its actual geometry.

Dynamic axes form a leading prefix. Preserve arbitrary leading dimensions,
coordinate ordering, supported auxiliary coordinates, CRS, grid identifiers,
field name, and user metadata across model execution. Spatial changes construct
a new signature with the output grid rather than replacing axes on an existing
signature. Time-statistic declarations derive from qualified variable labels.

Validate lead-time histories as nonempty one-dimensional timedelta coordinates
without NaT, then compare relative offsets. Forecast output offsets are added to
the final input lead time, including nonzero starts and resumed rollouts.

### Model families and composition

Inventory exported classes, inherited implementations, and specialized entry
points before changing each family. Cover regular latitude/longitude,
unstructured, HEALPix, projected regional, and multiresolution grids. Preserve
constructor configuration, core inference algorithms, variable order, and
model-specific conditioning or output semantics.

Update composition wrappers and affected runners, perturbations, recipes, and
examples to pass labelled fields end to end. When a separate subsystem still
requires tensors and coordinate dictionaries, perform the conversion explicitly
at that subsystem's boundary. Remove obsolete legacy-call typing casts at
migrated model call sites.

### Iteration and state

Use DataArray-aware batching and prognostic hooks while preserving recurrent
history, coupled state, device placement, and checkpoint continuation. Persist
field storage separately from coordinate metadata where checkpoint machinery
requires it. Restored iteration advances to the next forecast step.

Neither a call nor an iterator may mutate caller data or coordinates. Advancing
an iterator must not alter earlier yields. Retain existing seeding controls and
stochastic declarations; report existing conformance deviations explicitly
rather than silently broadening exemptions.

## Validation and error handling

Update the conformance checker to exercise DataArray signatures and execution,
retaining validation, relative-time, batching, hook, ownership, and RNG checks.
Track exported models through the existing coverage registry. Invalid dimensions,
variables, lead-time histories, and grids fail before core inference.

Use mock-weight and lightweight model fixtures for executable regression tests.
Cover single-step and iterator results, arbitrary leading dimensions, coordinate
metadata, regional and nonrectilinear grids, checkpoint continuation, and
composition. Run CPU tests and available CUDA tests. Optional dependency skips
and unavailable weight-based integration tests must be reported accurately.

Finish with the applicable model and caller tests, `make lint`, and
`uv tool run pre-commit run --all-files`. Full-repository hooks are required
because PR #1168 exposed legacy callers outside a changed-files-only run.

## Delivery

Commit the verified migration and push to NVIDIA/earth2studio upstream. Create
the PR using `skills/developer-open-refactor-pr/SKILL.md` with:

- Title: `[E2S 1.0] Migrate remaining models to xarray execution`
- Base: `1.0.0-rc`
- Label: `1.0.0`

The PR body summarizes the committed changes and actual verification results.
Update the shared coordination notes with the branch, PR, results, and remaining
environmental limitations. Do not post PR comments.
