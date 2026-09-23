# Earth2Studio Model Contract

## Goal

Define prognostic and diagnostic iterator, coordinate, hook, ownership, and RNG
semantics for model-independent execution, codifying existing correct behavior.

The native DataArray contract is enforced by `earth2studio.models.conformance`,
which reports the rule identifiers used below. Existing model tests exercise
execution, batching, metadata, hooks, ownership and checkpoint continuation.

`AssimilationModel` is out of scope; unresolved differences appear in Open Questions.

## Coordinate Systems

The new `CoordinateSystem` is an `xarray.DataArray` created by `coord_array()`.
Its backing array stores only shape and dtype: it allocates no field values,
including when every dimension has a concrete size. Read labels through `.coords`,
order through `.dims`, and lengths through `.sizes`. Accessing field `.values` is
unsupported. Dimension and auxiliary coordinate arrays themselves occupy memory.

`input_coords()` is a *declaration*. Dynamic dimensions form an explicitly marked,
zero-sized leading prefix (`earth2studio_dynamic_dims`). Any model can declare
dynamic dimensions, including `time` or other model-specific axes; this is not a
regional/global distinction and names are not restricted to `batch` and `time`.
The remaining dimensions, labels, auxiliary
coordinates, and declared grid/CRS/statistics metadata must match. Concrete inputs
may use any leading batch dimensions or none; fixed trailing dimension order is
authoritative. A zero-sized fixed dimension is not implicitly a wildcard.
Resolve multiple dynamic dimensions together or from right to left: concretizing
`time` in `(batch, time, ...)` retains a dynamic `batch`, but concretizing only
`batch` would leave a non-leading wildcard and is rejected. Dimensions are not
silently reordered or converted from wildcards to fixed zero-length axes.

Flexible input shapes and variable sets are configured on the model **instance**
before querying `input_coords()`. For example, a model supporting arbitrary crops
is configured with the chosen domain/grid, and an observation model accepting
optional channels is configured with the available variable set. Its declaration
then gives concrete spatial coordinates and variable labels for that configuration,
so fetching and output planning are unambiguous. Reconfigure before planning a
different domain or channel set. Keep that configuration fixed throughout a run;
changing it requires new signatures and a new input-fetch/output plan. Constructors
or model-specific setters currently provide configuration. A shared configuration
interface (for example, `set_domain` and variable selection) remains an open
contract question rather than an established method that callers can rely on.
An unresolved wildcard is not a request to fetch an unspecified region or variable
set. Model-specific validation enforces constraints such as patch-size multiples
or supported channel combinations in addition to coordinate handshakes.

`output_coords(input_coords)` validates and resolves output coordinates without
touching field data, enabling rollout planning before allocation.
Use `coord_array_like(input, replacements)` to preserve arbitrary
leading dimensions, dtype, name, metadata, and unaffected coordinates. Replacing
`lead_time` or `variable` removes dependent auxiliaries (for example `valid_time`
or variable-specific units), which must be recomputed explicitly if needed.
Temporal statistics are declared only in qualified variable labels such as
`tp:sum:6h`. `coord_array()` and `coord_array_like()` derive the
`earth2studio_statistics` attribute from those labels; it is not an independent
declaration. Replacing variables recomputes the metadata from the output labels.
Spatial replacements require `coord_array(grid=...)` with a new grid definition;
`coord_array_like()` rejects them to prevent stale grid metadata.

### Grid-backed signatures

Pass a registered name or `GridDefinition` to `coord_array(grid=...)` rather than
repeating grid axes in each model. The helper attaches grid metadata and
`earth2studio_crs` when the definition has a CRS; a registered name also attaches
`earth2studio_grid_id`. Projected, curvilinear, and point signatures include
geographic coordinates; HEALPix signatures use index coordinates. Handshakes
validate geographic coordinates as well as axes.

- `StormScopeGOES`, `StormScopeMRMS`: checkpoint `CurvilinearGrid` on `y, x`;
  output offsets are added to the final input lead time.
- `StormCastCONUS`: cropped `ProjectedGrid` with registered HRRR CRS on `y, x`;
  output advances the final input lead time by one hour.
- `PrecipitationAFNO`: registered `latlon-0.25deg-south-pole-excluded` grid
  (720 × 1440) on `lat, lon`;
  output is `tp:sum:6h` with derived statistics metadata.

Models declaring relative lead-time history subtract the final input lead time before checking the
declared history window. Before subtraction, `lead_time` must be an explicit,
nonempty one-dimensional timedelta coordinate with no `NaT` entries; datetime
and numeric labels are rejected. Output planning accepts both dynamic declarations and
concrete DataArrays without mutating either.

Coordinate validation uses the public handshakes in `earth2studio.utils.coords`:
`handshake_dim` reads DataArray dimension order (including unlabelled axes),
`handshake_size` reads dimension sizes, and `handshake_coords` compares coordinate
dimensions and label values without Xarray alignment or attached auxiliary coordinates.
These three APIs also accept legacy coordinate dictionaries. A tuple passed to
`handshake_dim` checks the complete ordered dimensions; `handshake_coords(...,
subset=True)` checks labels before an explicit model-specific selection.

`handshake_dataarray(input, signature, relative_lead_time=True)` composes those
checks with temporal and grid-metadata validation. `handshake_time` checks explicit,
nonempty finite datetime/timedelta labels and optionally interval alignment or a
minimum offset. Auxiliary/scalar validity times use `dimension=False`.
`handshake_metadata` compares explicitly named attributes; HEALPix signatures
require matching level, ordering, layout, origin and winding, including absent
CRS/registry attributes. None of these checks materializes field values.

Use `handshake_dataarray(input, runtime=True)` at execution boundaries to reject
unresolved zero-sized axes. Concrete field arrays are always checked as runtime
inputs, even when passed to `output_coords`; only explicitly declared dynamic
zero-sized signature axes are accepted for planning. Generic leading axes need
no labels; temporal labels are checked when present or required by the model.
Model methods retain their history/output transformations and invoke standard
handshakes for validation.

### Execution boundary

The public `PrognosticModel` and `DiagnosticModel` protocols use DataArrays:

```python
def __call__(self, x: xr.DataArray) -> xr.DataArray: ...
def input_coords(self) -> CoordinateSystem: ...
def output_coords(self, x: CoordinateSystem) -> CoordinateSystem: ...
# Prognostic models additionally expose:
def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]: ...
```

**All exported prognostic and diagnostic wrappers implement this execution API.**
Fields are NumPy-backed on CPU or CuPy-backed on CUDA. Wrappers document whether
they require matching input/model devices or move fields at the core boundary. Conversion at
the Torch boundary uses `.e2s.to_torch()` and `from_torch(tensor, signature)`; the
latter preserves all coordinates and output metadata without materializing the
signature. Real outputs omit signature kind/schema/dynamic attributes and preserve
user metadata, grid ID/CRS, and applicable statistics. Precipitation changes the
variable to `tp:sum:6h`, which declares its statistics. FCN requires finite timedelta-valued
lead times and advances by six hours, including from nonzero starting offsets.

`batch_func` dispatches DataArrays to `.e2s.batch()` / `.e2s.unbatch()`. It packs
arbitrary leading dimensions, handles an existing `batch` dimension, and inserts
a singleton for inputs without leading dimensions. Batch labels and batch-only
auxiliaries survive even when the core drops attrs; spatial auxiliaries survive,
and output variable counts may change. Mixed batch/fixed auxiliary coordinates
are unsupported. The core must retain the packed batch dimension, size, and labels
in their original order; reordering is rejected to prevent mislabeled output.

`DataArrayPrognosticMixin` hooks take and return one DataArray in the original
leading dimensions, and run only during iteration. `clear_hooks()` restores identity
hooks. Level-two checkpoints store the field tensor separately from dimensions,
coordinate values/attrs, name, attrs and encoding. Restarts yield the next forecast
step after the saved state, rather than repeating that state.

FuXi-S2S declares two consecutive daily means with start-of-day timestamps:
ordinary channels use `mean:0h:24h`, while hourly interval-ending `tp` and `ttr`
use `mean:1h:25h`. Qualified variable labels and matching statistics metadata are
required. Its iterator first yields the latest input day, then daily predictions;
front hooks see the two-day history and rear hooks see one prediction. Hook-modified
predictions feed the next rolling state. Hooks use original leading dimensions.
Field values move to the model device at the Torch/ONNX boundary. Input preparation
and units follow `TIME_STATISTICS_SPEC.md`'s calendar-day means section.

The legacy `CoordSystem` remains `OrderedDict[str, np.ndarray]` for tensor-based
IO, statistics, perturbations and private numerical helpers. Convert explicitly
at these boundaries with `.e2s.to_torch()` and `from_torch()`. `fetch_data()` returns
one field DataArray; unpacking it as a tensor/coordinate pair is invalid.
Random/Random_FX retain their existing data-source API. Assimilation protocols
are outside this migration. Regional prognostic wrappers also use native field
DataArrays. StormCastCONUS derives geographic auxiliaries through its cropped
projected grid declaration.
Its stored `ProjectedGrid` is the geometry source of truth; native axes come from
`input_coords()["y"]` and `input_coords()["x"]`. Read-only `hrrr_y` and `hrrr_x`
properties derive their values from those coordinates.
Runtime protocol membership checks method presence, not call signatures, so it is
not an execution-API detector. `models.conformance` dispatches to native DataArray
checks for DataArray signatures and retains legacy checks for dictionary signatures.
The rule identifiers below apply to both paths; native signatures are allocation-free
coordinate DataArrays rather than ordered dictionaries, and fields carry their coordinates.
See `dev/examples/03_coordinate_signatures.py` for signature planning and
`dev/examples/04_xarray_model_execution.py` for runnable DataArray execution.

## Rules

### Prognostic

These rules apply to native DataArray execution. The checker retains a separate
legacy path for dictionary-signature test fixtures.

| Rule | Requirement |
| --- | --- |
| `P1` | Structurally satisfies `PrognosticModel` |
| `P2` | Allocation-free DataArray declarations with explicit dynamic leading dimensions |
| `P3` | `input_coords()["lead_time"]` is relative, strictly increasing, and ends at zero |
| `P4` | `output_coords()` treats its argument as read-only |
| `P5` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `P6` | Shifting input `lead_time` by an offset shifts output `lead_time` by the same offset |
| `P7` | `create_iterator()` yields the initial condition as its 0th step |
| `P8` | The 1st yield matches the coordinate system `output_coords()` declared |
| `P9` | Every forecast matches its planned coordinates and structural metadata |
| `P10` | `create_iterator()` applies both hooks; `__call__` applies neither |
| `P11` | The model declares a boolean `stochastic` attribute |
| `P12` | A stochastic model implements `set_rng(seed, reset=True)` |
| `P13` | Seeding determines a rollout, and different seeds give different rollouts |
| `P14` | After `set_rng()`, seeding and stepping leave global RNG state unperturbed |
| `P15` | Stepping the model does not modify its input tensor or coordinate system |
| `P16` | A yielded tensor does not change once a later step is produced |

### Diagnostic

| Rule | Requirement |
| --- | --- |
| `D1` | Structurally satisfies `DiagnosticModel` |
| `D2` | Allocation-free DataArray declarations with explicit dynamic leading dimensions |
| `D3` | `output_coords()` treats its argument as read-only |
| `D4` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `D5` | `__call__` matches declared coordinates and structural metadata |
| `D6` | `__call__` does not modify its input tensor or coordinate system |
| `D7` | The model declares a boolean `stochastic` attribute |
| `D8` | A stochastic model implements `set_rng(seed, reset=True)` |
| `D9` | Seeding determines the output, and different seeds give different output |
| `D10` | After `set_rng()`, seeding and calling leave global RNG state unperturbed |

## Lead Time

Declared input `lead_time` is relative to analysis time and ends at zero (e.g.,
`[-6h, 0h]`). Output offsets are added to the *final input* lead time, never a
constant, allowing nonzero starts and resumed rollouts. Shifting every input lead
time must shift output equally (`P6`).

## Iteration

`create_iterator()` first yields the initial condition with `lead_time` and data
reduced to the final input entry. Complete forecast steps follow, advancing by the
model's step; `nsteps` forecasts require `nsteps + 1` yields. The model must not
consume its input before the 0th yield or emit partial steps.

## Hooks

**Hooks belong to the iterator (`P10`).** Every core advance applies `front_hook`
immediately before advancing; every forecast output applies `rear_hook` before it is
yielded. `__call__` applies neither. Native models declare `front_hook_interval`, the
positive number of forecast outputs per core advance (default 1). Multi-output cores
preserve their numerical cadence: DLWP declares 2 because one twelve-hour core call
produces two six-hour forecasts. Its hook order is front, rear, rear, then repeats;
the second output is already computed and does not invoke another front hook.
Conformance checks two complete declared cycles without model-name exceptions.
`PrognosticMixin` hooks transform `(tensor, coords)` pairs;
`DataArrayPrognosticMixin` hooks transform a single `xr.DataArray`.
The front hook reaches recurrent state otherwise inaccessible between
steps; see `examples/02_medium_range/02_model_perturbation_hook.py`.

Each hook is a single callable slot. Callers compose transformations explicitly;
there is no registration chain. `clear_hooks()` restores identity hooks.

Single-step callers can transform inputs and outputs directly. Iterator hooks
add access to internal history buffers and coupled recurrent state. Explicit
composition keeps ordering at the assignment site and avoids drivers silently
interleaving transformations with caller-configured hooks.

The migrated GraphCast, GenCast, and WeatherNext wrappers apply both hooks to
public DataArrays while preserving native recurrence when numerical inputs are
unchanged. Aurora1p5 declares six hourly outputs per core advance; its front hook
runs once per six-hour cycle and its rear hook runs on every hourly output.

## Ownership of Tensors

A model borrows its input and owns its output. Neither `__call__` nor
`create_iterator()` may modify caller input tensors or coordinates (`P15`, `D6`).
Earlier yields must remain unchanged after later steps (`P16`); views are allowed
only if their backing buffers will not be overwritten. This protects asynchronous
IO, resume buffers, and accumulators without caller-side defensive copies.

The motivating failure was `stormcast` overwriting the caller's initial condition
([issue #1133](https://github.com/NVIDIA/earth2studio/issues/1133), fixed in PR #1134).
`AsyncZarrBackend` currently copies every non-blocking write defensively; model-level
ownership guarantees address the underlying buffer-lifetime problem.

Legacy `batch_func` rebuilds coordinates but does not protect tensor inputs:
`_compress_batch` uses `unsqueeze`/`flatten` views, so internal writes reach callers.

## Stochasticity

`P11`–`P14` and `D7`–`D10` require a readable boolean `stochastic` declaration and,
when true, `set_rng(seed: int, reset: bool = True) -> None`. Absent declarations
default to `False`: `PrognosticMixin` supplies this default; diagnostics need no
shared base class. The declaration lets drivers plan ensembles before execution.

The integer seed is the first positional argument, supporting seed-only core APIs.
`reset=True` replaces the generator; `reset=False` initializes it only if absent,
otherwise ignoring the seed and preserving the current trajectory. Ensemble drivers
reseed each member with `reset=True`; per-step hooks may use
`set_rng(fallback_seed, reset=False)` without clobbering driver seeding. Resetting
to the same seed each step would repeat identical noise. Never-seeded models fall
back to the global RNG rather than failing.

The same seed must reproduce every rollout step (`P13`) or diagnostic output (`D9`)
exactly; different seeds must differ. A model declaring `stochastic=False` but
returning different results for the same input also fails these rules.

### RNG isolation

After `set_rng()`, neither seeding nor stepping/calling may leave global RNG state
perturbed (`P14`, `D10`). Use a local `torch.Generator` for every draw, a functional
PRNG key, or `torch.random.fork_rng()` around global seeding and execution to restore
state afterward. Forking supports external packages without generator injection
(e.g., `aifs2ens`/`anemoi`); pass `devices` explicitly because the default forks all
visible CUDA devices and warns. Unseeded global-RNG draws remain allowed.

For a core that accepts only global seeding, store the seed without drawing and
isolate each step:

```python
def set_rng(self, seed: int, reset: bool = True) -> None:
    if reset or self._seed is None:
        self._seed = seed

# Seeded step:
with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
    torch.manual_seed(self._seed + step)
    out = self.core_model(...)
```

Isolation protects other models, perturbations, and dataloaders from having their
streams reset. Reproducibility checks alone cannot catch this interference: a model
may reproduce perfectly in isolation while destroying independence in a cascade.
The rule constrains the observable effect, not the isolation mechanism.

**Known deviations:** `dlesym` uses a local generator and conforms; `fcn3` fails
`P14` because core noise-state refresh draws globally; `aurora1p5` fails through
bare `torch.manual_seed(seed)`. `Aurora1p5Ensemble` is exempt in
`test/models/test_model_conformance.py` pending a follow-up wrapper fix.

### Seeding is the only entry point

A constructor `seed` must not override later `set_rng()` calls.
`Aurora1p5Ensemble` violates this by reapplying `self.set_rng(self.seed)` in
`create_iterator()`; its exemption also covers this pending fix. Removing constructor
seeds remains open, including the `load_model(seed=...)` APIs of `corrdiff`,
`cbottle_sr`, and `stormscope_dx_nsrdb`.

### Current wrappers

The DataArray protocol migration preserves existing model randomness. The RNG
rules above remain targets for separate follow-up work, not changes in this PR.
`fcn3` and `dlesym` retain `(seed, reset)` controls; `Aurora1p5Ensemble` retains
`set_rng(seed: int | None)` and global seeding. WeatherNext retains its existing
functional-key `set_rng`, including updating `seed` with `reset=False` while
leaving the key unchanged. GenCast retains per-time functional keys and its seed
attribute. Composition wrappers do not add seeding dispatch.

CorrDiff and cBottle retain their existing constructor/attribute seed controls.
CBottleSR retains its pre-existing fork and per-sample seed progression; infill
retains its backend's lack of seed support. StormScopeDxNSRDB still globally seeds
each sample. StormCast, StormCastCONUS, StormScope and UCast retain their global
diffusion/dropout draws. Existing undeclared-randomness exceptions remain pinned
in `test/models/test_model_conformance.py` and the corresponding model tests.

## Conformance

`earth2studio.models.conformance.check_prognostic_contract(model)` evaluates every
applicable rule before failing, reporting all violations and returning unevaluated
rules with reasons. `rollout=False` runs only `P1`–`P6`, `P11`, `P12`, and the
seeding half of `P14`. `check_diagnostic_contract(model, forward=False)` similarly
runs `D1`–`D4`, `D7`, `D8`, and the seeding half of `D10`. RNG seeding and stepping
are checked separately so seeding violations need no forward pass.

Native probes concretize dynamic dimensions and use the signature's `.shape`,
including its auxiliary coordinates and grid/statistics metadata. Legacy probe
shapes follow `convert_multidim_to_singledim`: a 1-D coordinate contributes
its length; an n-D entry requires n−1 following partners of identical shape, with
the group contributing that shape once. Invalid groupings prevent probe creation
and skip `P7`–`P10` and `P13`–`P16`. Pseudo-random probes expose input mutation;
models rejecting unphysical data need realistic initial-condition fixtures.

Every forecast in the probed rollout is checked against independently planned
coordinates. The checker rebases the original declared input history at the
previous planned output's final lead before planning the next forecast, so it
supports multi-frame histories and multi-output steps without trusting corrupted
yielded coordinates. Calls and forecast yields must preserve declared grid, CRS,
and statistics metadata and must omit signature kind/schema/dynamic attributes.
User attributes remain free to change through hooks.

### Enforcement

Model creation skills produce `test_<model>_conformance` using existing mock-weight
fixtures, without real weights or network access. `test/models/test_conformance.py`
tests the checker; `test/models/test_model_conformance.py` introspects
`earth2studio.models.px`/`dx` and requires every class to be conformant or explicitly
exempt with a pinned reason. Backend-dependent execution requires its optional
dependencies; a dependency skip is not evidence that a model passes conformance.

## Open Questions

- Should `P13`'s `torch.allclose` tolerance be configurable for nondeterministic GPU
  kernels that make deterministic models appear stochastic?
- An opaque, unseedable dependency would require a separate `seedable` declaration
  and skipping `P13`, while retaining `P14`. No current wrapper requires this split.
- `P14`/`D10` currently cover only models implementing `set_rng`; widen them if a
  deterministic model that reseeds globally appears.
- `AssimilationModel` needs decisions on four divergences before a contract:
  1. Output resolution: `InterpEquirectangular` requires positional `request_time`,
     `HealDA` accepts it optionally, and `StormCastSDA`/`CorrDiffCosmoEra5SDA` reject
     it, preventing generic planning.
  2. Priming: `StormCastSDA` yields initial state; `HealDA`, `InterpEquirectangular`,
     and `CorrDiffCosmoEra5SDA` yield `None`.
  3. Stochasticity: `CorrDiffCosmoEra5SDA` uses `self.seed + i` without `set_rng`;
     `P11`–`P14` could transfer unchanged.
  4. Ownership: mutable DataFrame/DataArray inputs need `P15`/`D6` guarantees.

  `FrameSchema` (ordered column-to-array mappings) supports probe DataFrames, but
  mid-stream `send(None)`, single-call/step equivalence, and generator closure
  ownership remain undecided.
- Forcing/conditioning declarations require agreement with coupling and
  labelled-array proposals before inclusion here.
- Keep constructor `seed=` as a construction-time `set_rng` convenience or remove
  it? Account for loading APIs, Aurora's override bug, and preserving diagnostics'
  fresh-seed default (or communicating a breaking change).
