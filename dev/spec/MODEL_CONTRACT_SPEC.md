# Earth2Studio Model Contract

## Goal

Define prognostic and diagnostic iterator, coordinate, hook, ownership, and RNG
semantics for model-independent execution, codifying existing correct behavior.

The legacy tensor contract is enforced by `earth2studio.models.conformance`, which
reports the rule identifiers used below. This branch updates coordinate signatures;
DataArray execution migration is a separate effort.

`AssimilationModel` is out of scope; unresolved differences appear in Open Questions.

## Coordinate Systems

The new `CoordinateSystem` is an `xarray.DataArray` created by `coord_array()`.
Its backing array stores only shape and dtype: it allocates no field values,
including when every dimension has a concrete size. Read labels through `.coords`,
order through `.dims`, and lengths through `.sizes`. Accessing field `.values` is
unsupported. Dimension and auxiliary coordinate arrays themselves occupy memory.

`input_coords()` is a *declaration*. Dynamic dimensions form an explicitly marked,
zero-sized leading prefix (`earth2studio_dynamic_dims`), conventionally `batch`
and, for regional models, `time`. The remaining dimensions, labels, auxiliary
coordinates, and declared grid/CRS/statistics metadata must match. Concrete inputs
may use any leading batch dimensions or none; fixed trailing dimension order is
authoritative. A zero-sized fixed dimension is not implicitly a wildcard.
Resolve multiple dynamic dimensions together or from right to left: concretizing
`time` in `(batch, time, ...)` retains a dynamic `batch`, but concretizing only
`batch` would leave a non-leading wildcard and is rejected. Dimensions are not
silently reordered or converted from wildcards to fixed zero-length axes.

`output_coords(input_coords)` validates and resolves output coordinates without
touching field data, enabling rollout planning before allocation.
Use `coord_array_like(input, replacements)` to preserve arbitrary
leading dimensions, dtype, name, metadata, and unaffected coordinates. Replacing
`lead_time` or `variable` removes dependent auxiliaries (for example `valid_time`
or variable-specific units), which must be recomputed explicitly if needed.
Statistics are retained only for surviving variables unless explicitly supplied.
Spatial replacements require `coord_array(grid=...)` with a new grid definition;
`coord_array_like()` rejects them to prevent stale grid metadata.

### Grid-backed signatures

Pass a registered name or `GridDefinition` to `coord_array(grid=...)` rather than
repeating grid axes in each model. The helper attaches grid metadata and
`earth2studio_crs` when the definition has a CRS; a registered name also attaches
`earth2studio_grid_id`. Projected, curvilinear, and point signatures include
geographic coordinates; HEALPix signatures use index coordinates. Handshakes
validate geographic coordinates as well as axes.

| Model | Grid declaration | Spatial dimensions | Output |
| --- | --- | --- | --- |
| `StormScopeGOES`, `StormScopeMRMS` | `CurvilinearGrid` from checkpoint geometry | `y, x` | Configured output offsets added to final input lead time |
| `StormCastCONUS` | Cropped `ProjectedGrid` using registered HRRR CRS | `y, x` | Final input lead time plus one hour |
| `PrecipitationAFNO` | Registered `fcn1` grid (720 × 1440) | `lat, lon` | `tp`, with `sum:6h` statistics |

Regional validation subtracts the final input lead time before checking the
declared history window. Before subtraction, `lead_time` must be an explicit,
nonempty one-dimensional timedelta coordinate with no `NaT` entries; datetime
and numeric labels are rejected. Output planning accepts both dynamic declarations and
concrete DataArrays without mutating either.

### Migration boundary

Migrated models expose allocation-free coordinate signatures:

```python
def input_coords(self) -> CoordinateSystem: ...
def output_coords(self, x: CoordinateSystem) -> CoordinateSystem: ...
```

FCN, PrecipitationAFNO, StormCastCONUS, StormScopeGOES, and StormScopeMRMS
use these signatures for both planning and tensor-pair execution:

```python
def __call__(self, x: torch.Tensor, coords: CoordinateSystem
             ) -> tuple[torch.Tensor, CoordinateSystem]: ...
def create_iterator(self, x: torch.Tensor, coords: CoordinateSystem
                    ) -> Iterator[tuple[torch.Tensor, CoordinateSystem]]: ...
```

Field values remain separate tensors. Coordinate shapes must match the tensors;
wildcard dimensions must be concretized before execution. Batching preserves
leading dimension labels, geographic auxiliaries and metadata, and outputs remain
allocation-free. Converted models reject dictionary coordinate arguments.
`CoordSystem` remains the dictionary alias for unmigrated models and data helpers.
StormCastCONUS derives geographic auxiliaries from its cropped projected axes
and registered HRRR CRS through `coord_array(grid=...)`.
Its stored `ProjectedGrid` is the geometry source of truth; consumers obtain
native axes from `input_coords()["y"]` and `input_coords()["x"]`.
Read-only `hrrr_y` and `hrrr_x` properties derive their values from those coordinates.

Single field-DataArray inputs belong to the separate execution migration.
Runtime protocol membership checks method
presence, not call signatures. The conformance checker still expects dictionary
signatures; migrated public signatures are covered by focused coordinate tests.
See `dev/examples/03_coordinate_signatures.py` for allocation-free planning.

## Rules

### Prognostic

These rule tables describe the legacy conformance checker. Coordinate-signature
migration and its coverage are specified above.

| Rule | Requirement |
| --- | --- |
| `P1` | Structurally satisfies `PrognosticModel` |
| `P2` | `input_coords()` and `output_coords()` return ordered dicts of arrays led by `batch` |
| `P3` | `input_coords()["lead_time"]` is relative, strictly increasing, and ends at zero |
| `P4` | `output_coords()` treats its argument as read-only |
| `P5` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `P6` | Shifting input `lead_time` by an offset shifts output `lead_time` by the same offset |
| `P7` | `create_iterator()` yields the initial condition as its 0th step |
| `P8` | The 1st yield matches the coordinate system `output_coords()` declared |
| `P9` | Every yielded tensor shape matches its coordinate system |
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
| `D2` | Coordinate systems are ordered dicts of arrays led by `batch` |
| `D3` | `output_coords()` treats its argument as read-only |
| `D4` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `D5` | `__call__` returns the coordinate system `output_coords()` declared |
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

**Hooks belong to the iterator (`P10`).** Every forecast step applies `front_hook`
immediately before advancing and `rear_hook` immediately after; `__call__` applies
neither. `PrognosticMixin` hooks transform `(tensor, coords)` pairs.
The front hook reaches recurrent state otherwise inaccessible between
steps; see `examples/02_medium_range/02_model_perturbation_hook.py`.

Each hook is a single callable slot. Callers compose transformations explicitly;
there is no registration chain. `clear_hooks()` restores identity hooks.

Single-step callers can transform inputs and outputs directly. Iterator hooks
add access to internal history buffers and coupled recurrent state. Explicit
composition keeps ordering at the assignment site and avoids drivers silently
interleaving transformations with caller-configured hooks.

**Known deviation:** `gencast_mini`, `graphcast_small`, `graphcast_operational`, and
`weathernext2_cyclones_mini` apply only `rear_hook`, failing `P10`.

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

### Migration

Current `set_rng` implementations differ: `fcn3`/`dlesym` take `(seed, reset)`,
`aurora1p5` takes `(seed)`, and callers use `hasattr`. All three declare `stochastic`;
`dlesym` uses a property conditional on `use_cln`. Remaining declarations and seeding
implementations belong to wrapper owners, with this migration work:

| Mechanism today | Wrappers | Work |
| --- | --- | --- |
| Functional PRNG key | `gencast_mini`, `weathernext2_cyclones_mini` | declare and wrap |
| Local `torch.Generator` | `corrdiff` | declare and wrap |
| Seed passed to core model | `cbottle_video` | declare and wrap |
| Global `torch.manual_seed` | `aifs2ens`, `cbottle_sr`, `stormscope_dx_nsrdb` | fork the RNG |
| None at all | `atlas_crps`, `stormscope` | add seeding, forked |

"Declare and wrap" adds `stochastic`/`set_rng` around already-isolated randomness;
"fork" confines existing global seeding to `torch.random.fork_rng()`. No upstream
package changes are needed. Diagnostics currently use `seed=None` for a fresh seed
per call. Migration represents this as "never called `set_rng`" and must preserve
unseeded nondeterministic defaults.

## Conformance

`earth2studio.models.conformance.check_prognostic_contract(model)` evaluates every
legacy rule before failing, reporting all violations and returning unevaluated
rules with reasons. `rollout=False` runs only `P1`–`P6`, `P11`, `P12`, and the
seeding half of `P14`. `check_diagnostic_contract(model, forward=False)` similarly
runs `D1`–`D4`, `D7`, `D8`, and the seeding half of `D10`. RNG seeding and stepping
are checked separately so seeding violations need no forward pass.

Probe shapes follow `convert_multidim_to_singledim`: a 1-D coordinate contributes
its length; an n-D entry requires n−1 following partners of identical shape, with
the group contributing that shape once. Invalid groupings prevent probe creation
and skip `P7`–`P10` and `P13`–`P16`. Pseudo-random probes expose input mutation;
models rejecting unphysical data need realistic initial-condition fixtures.

### Enforcement

Model creation skills produce `test_<model>_conformance` using existing mock-weight
fixtures, without real weights or network access. `test/models/test_conformance.py`
tests the checker; `test/models/test_model_conformance.py` introspects
`earth2studio.models.px`/`dx` and requires every class to be conformant or explicitly
exempt with a reason. All three run in every PR's CI. Pre-spec models are currently
exempt pending incremental test backfill.

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
- Should this effort migrate other seed-attribute/global-RNG wrappers too:
  `aifs2ens`, `gencast_mini`, `cbottle_video`, `weathernext2_cyclones_mini`,
  `atlas_crps`, `stormscope`, `corrdiff`, `corrdiff_cosmo_era5`, `cbottle_sr`, and
  `stormscope_dx_nsrdb`?
