# Pipeline wave 1, track B: runner follow-up notes

Scope: `earth2studio/run/runner.py` and `earth2studio/run/_fields.py` --
DataArray stream schemas from `Runner.output_coords`, per-member seeding of
stochastic models, and removal of `_yields_initial_condition`. Spec:
`dev/spec/EXECUTION_CONTRACT_SPEC.md` ("Runner", "Stream schemas", "Yielded
outputs", "Seeding", "Built-in runners", rules X1-X5).

## What the contract now is, for the output/pipeline tracks

`Runner.output_coords(horizon) -> Mapping[str, CoordinateSystem]`. Each value is
an allocation-free signature DataArray (`data.nbytes == 0`), one per stream:

- Dimensions are `(*dynamic_prefix, "lead_time", *field_dims)`. The dynamic
  prefix is whatever the model declared (`("batch",)`, `("batch", "time")`,
  ...), each with size zero; `time` and `ensemble` are never fixed (X1).
- `lead_time` covers the whole rollout, from `_lead_times`: lead 0 plus every
  step's declared leads, so models publishing several leads per step (DLESyM)
  get all of them (X4).
- Grid metadata (`earth2studio_grid_id`, `earth2studio_crs`, `dims`, HEALPix
  `nside`/`ordering`/...), auxiliary coordinates (`lat`/`lon` on `(y, x)` for
  projected grids), dtype, name and other attrs carry over from the model
  signature (X2). HEALPix grids contribute index coordinates only -- that is
  `coord_array`'s behavior for `topology == "healpix"`, not a runner choice.

Consumers concretize with `earth2studio.io.utils.output_template(signature,
leading={"ensemble": ..., "time": ...}, coords=...)`; the result is accepted by
`IOBackend.add_array` unchanged, and every `run_item` output can be written to
it after adding the leading labels. `test_output_coords_round_trip_to_io_templates`
pins exactly this path; `run/output.py` should not need any schema fix-ups.

`seed` is a constructor argument on both built-in runners (default 0) and is
read as `runner.seed`. Supervision does not seed: it only chooses
`member_ids`.

## Decisions and judgment calls

1. **One signature per stream, prefix preserved.** The old `_stream_coords`
   returned an `OrderedDict` that dropped `time` and every zero-sized dynamic
   dimension. The new `_signature_for_stream` keeps the dynamic prefix instead of
   stripping it, because `output_template` is the component that replaces that
   prefix, and it needs `E2S_DYNAMIC_DIMS` to do so. A fixed (non-dynamic)
   `time` or `ensemble` in a model signature raises `ValueError` (X1).
2. **`lead_time` always leads the fixed dimensions**, whatever position the
   model put it in, and is added when the model's signature has none (several
   diagnostics drop it). This matches `_io_template`'s existing behavior for the
   workflows and X3's "`time`, then the fixed dims".
3. **Seeding uses the `stochastic` flag.** `_seed_model` seeds only models whose
   `stochastic` is truthy *and* which expose `set_rng`; a model with `set_rng`
   but `stochastic = False` is deliberately left alone, per the model contract.
   DLESyM's `stochastic` is a property, which `getattr` handles.
4. **Deterministic items are member 0.** `WorkItem.member_ids == ()` seeds with
   `member_seed(seed, time, 0)`, so a one-member run and member 0 of an ensemble
   reproduce each other.
5. **Diagnostics are seeded too**, in both runners, since CorrDiff-style
   diagnostics are stochastic. They share the prognostic's member seed; each
   model keeps its own stream, so sharing a seed does not correlate them beyond
   what the model contract already implies.
6. **Seeding happens once per item**, before fetching, not per step: `set_rng`
   with `reset=True` starts the member's stream, and the rollout state carries
   it forward (`initialize`/`step` thread RNG state explicitly). Never touches
   global RNG (X5); `test_seeding_leaves_models_without_randomness_and_global_rng_alone`
   asserts `torch.random.get_rng_state()` is unchanged.
7. **`_rollout` keeps its `publish_initial_condition` flag.** Only the
   unmigrated-wrapper branch went away; `run/__init__.py`'s restart path still
   needs the flag.

## Issues worth another pair of eyes

1. **`member_seed` lives in `run/work.py`, which imports `WorkItem` from
   `run/runner.py`.** Importing it at module scope in `runner.py` is a circular
   import, so `_seed_model` imports it inside the function. If the module layout
   is still open, `member_seed` (and maybe `WorkItem`) would be better in a leaf
   module both can import.
2. **Seeds are 63-bit** (`member_seed` returns `digest % 2**63`). Fine for
   `torch.Generator.manual_seed` and `np.random.default_rng`, but a model that
   seeds with legacy `np.random.seed` (needs < 2**32) would raise. No built-in
   model does today; worth a line in the model contract if we care.
3. **The published initial condition can be a label subset of its stream.**
   `PrognosticRunner` publishes `initial_condition(x)`, which carries the
   model's *input* variables. DLESyM's forecast stream adds diagnostic output
   variables (`atmos/ocean_diagnostic_variables`), so step 0 writes a subset --
   IO accepts that, and the missing labels stay fill-valued
   (`test_fused_dlesym_initial_condition_omits_diagnostic_variables`). Note two
   consequences for whoever replaces the eval pipelines: (a) the eval `dlesym`
   pipeline *skips* the IC entirely (`known_missing_leads() == {0}`), so moving
   to `PrognosticRunner` adds a lead-0 row with only the prognostic variables;
   (b) **`DLESyMLatLon` is worse than a subset** -- its `input_coords` swaps
   `ws10m`/`tau300-700` for `u10m`, `v10m`, `z300`, `z700`, which are *not* in
   its `output_coords`, so publishing the IC into the forecast store would write
   unknown labels. Either the IC publication needs the model's input signature
   mapped onto the stream (drop unknown labels), or `PrognosticRunner` needs a
   way to skip the IC for such models. Out of scope here (it predates this
   change and touches `models/**`), but it blocks integration check 4 for the
   lat/lon variant.
4. **`dev/examples/06_runner.py` is now misleading** (line ~96): it prints
   `{name: list(coords) ...}`, which iterated an `OrderedDict`'s keys and now
   iterates a DataArray's leading (size-zero) dimension, so it prints
   `{'forecast': [], 'copy': []}`. It should print `coords.dims` or
   `list(coords.sizes)`. Its `StopAtThreshold.output_coords` annotation also
   still says `Mapping[str, CoordSystem]`. Left untouched: outside this track's
   file list.
5. **Spec wording.** X3/X4 read as "other fixed labels equal the stream's"; the
   IC-subset case above means "a subset of" is the accurate rule for `variable`,
   which is also what the IO contract allows.

## Verification

- `uv run pytest test/run -q`: 228 passed, 196 failed. Identical failures to the
  branch point (`cd26aac3b`): 195 `cuda:0` parametrizations (no GPU on this
  machine) plus the pre-existing
  `test_deterministic.py::test_native_mapping_preserves_storage_and_rejects_wrong_geometry`.
  Baseline before the change was 218 passed / 196 failed; the 10 extra passes
  are the new tests.
- `uv run black`, `uv run ruff check`, `uv run mypy` clean on
  `earth2studio/run/runner.py`, `earth2studio/run/_fields.py`,
  `test/run/test_runner_models.py`. `mypy test/run/test_runner.py` still reports
  one pre-existing `timedelta64.__rtruediv__` complaint on an untouched line.
- `uv run python dev/examples/06_runner.py` still runs (see issue 4).
