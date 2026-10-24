# Coupler vs reference frameworks: comparison report

Branch `ngeneva/coupler_ideas`, `earth2studio/coupler/runner.py` (665 lines,
`CoupledRunner`, `Link`, `window_mean`). Companion to
`2026-10-24-coupler-design.md`.

Seven reviewers each compared the implementation with one framework (or
group): NUOPC/ESMF, OASIS3-MCT + YAC, ClimaCoupler + MUSCLE3, preCICE +
CMEPS + SamudrACE/DLESyM, legoESM, and nvcoupler (PR #1229). Each read the
coupler code and the framework's docs/code. Each section lists what is
similar, different, the advantages and disadvantages of ours, what the
reference does that we do not, and the reference's own known issues.
Items marked *(verified)* were reproduced with a probe or test; the rest
are from reading.

## 0. Summary

### Where all references agree with our design

| Idea | Seen in |
| --- | --- |
| Match producers to consumers by field name, error on ambiguity | NUOPC (StandardName, bondLevel), MUSCLE3 (no many-to-one), nvcoupler (`Connector.match`) |
| Coupling window = integer multiple of every component step, fast side sub-cycles | NUOPC `SetRunClock`, YAC `coupling_period`, ClimaCoupler `dt_cpl`, preCICE time window, legoESM `n_sub` |
| Reduce on the source side, then regrid once | OASIS `LOCTRANS` below `put`, CMEPS linearity argument |
| Lagged vs sequential coupling is a matter of ordering | NUOPC run sequence, OASIS `LAG`, preCICE serial/parallel-explicit |
| Fail loud at construction, dry-run the schedule | legoESM `require_*`, nvcoupler `describe`, YAC `missing_definition_is_fatal` |
| "Hold" = last sample; "window_mean" = averaged fast output | OASIS INSTANT/AVERAGE, preCICE last sample, ClimaCoupler `FluxAccumulator`, MUSCLE3 theta/`all_in_window`, SamudrACE/DLESyM |

### Priority gaps found (ordered by how much they matter)

| # | Gap | Raised by | Evidence |
| --- | --- | --- | --- |
| 1 | No check that a delivered frame's valid time matches the label; `_deliver` relabels with `assign_coords(lead_time=...)` so a scheduling bug is masked | NUOPC, nvcoupler | read |
| 2 | Regridder output never validated at runtime (wrong coordinates are nearest-snapped into the slot) | nvcoupler | *(verified)* probe: lon `[5,15]` for slot `[0,20]` accepted |
| 3 | An explicit forcing source for a slot is overridden by an auto-wired link | NUOPC | *(verified)* probe |
| 4 | No units check; matching is by name only | NUOPC, OASIS/YAC, legoESM, nvcoupler | read |
| 5 | No finite/NaN guard; `window_mean` uses `skipna=False`, so one NaN poisons the mean silently | legoESM, OASIS/YAC, ClimaCoupler | read |
| 6 | Same derived seed for every stochastic model: correlated noise | OASIS/YAC, preCICE, legoESM | read (`_seed_model` has no per-model salt) |
| 7 | One reducer per link; cannot mix hold and mean, or weight (CMEPS `f*F`), per variable | legoESM, preCICE, OASIS | read |
| 8 | Cannot express DLESyM/SamudrACE semantics: hold-at-window-start, window offset/alignment, state overwrite, multi-lead models | preCICE/SamudrACE/DLESyM | read |
| 9 | No explicit lag/sequential flag; equal-cadence mutual hold is sequential for one side | NUOPC, OASIS, preCICE | *(verified)* result is dict-order independent |
| 10 | Slot-granular matching: one slot cannot take variables from two producers | NUOPC, ClimaCoupler/MUSCLE3 | read |
| 11 | No ensembles, no mid-item checkpoint of iterators/history | all | read |
| 12 | History holds all window frames; `xr.concat` copies them again | preCICE/CMEPS | read |

Items that are **not** gaps (deliberately out of scope): MPI/concurrency,
manager processes, phases, hierarchies, implicit iteration, run-sequence
DSL/YAML, second field vocabulary, flux physics, conservation, weight and
restart files.

## 1. NUOPC / ESMF

**Similar**
- Name-based pairing, error on ambiguity; one producer may feed many
  consumers, several producers into one field is unsupported in both.
- Declare, connect, run shape (our construction-time probing is the
  analogue of advertise/realize).
- Integer-multiple clock rule with sub-cycling.
- Lag vs sequential decided by ordering.
- Regrid is a per-connection transform set up before the run.

**Different**

| Aspect | NUOPC | Ours |
| --- | --- | --- |
| Component contract | Fortran/C "cap", IPD phases, `Advance`, `CheckImport` | Existing `PrognosticModel` + `create_iterator().send()` |
| Field identity | Field Dictionary: StandardName, canonical units, synonyms | Lexicon variable name, no units |
| Handshake | Advertise, realize, grid transfer offers | Metadata read at construction, no negotiation |
| Time | `ESMF_Clock` per component, timestamps on fields | Integer lead times, no timestamps |
| Run sequence | Declarative, nested, multiple clocks | Fixed rule: furthest-behind model steps first |
| Averaging | Not built in (Mediator or `Advance`) | `Link.reduce` |
| Regridding | RouteHandle, bilinear/patch/nearest/conserve | Injected bound `Regridder` |
| Parallelism | MPI PET lists, concurrent or sequential | Single process, sequential |
| Hierarchy / ensembles | Nested drivers, interactive ensembles | None |

**Advantages of ours:** no caps or wrappers; near-zero config; plugs into
`Pipeline`; `describe()`; strict construction checks; ~660 lines; plain
callables for reduce/regrid.

**Disadvantages of ours**
- No leap-frog: a consumer can never receive the producer's value at
  `t+step` unreduced.
- No timestamps/`CheckImport` (gap 1).
- No unit check (gap 4).
- Slot-granular matching (gap 10).
- Auto-wired link beats explicit source (gap 3, *verified*).

**NUOPC does, we do not:** units + synonyms (worth a light version via
lexicon units); `CheckImport` (worth adding: assert delivered lead equals
expected); implicit/leap-frog sequences (a per-link `lag`/sequential flag is
worth adding; implicit is out of scope); Mediator (covered by reduce);
namespaces, hierarchy, PET concurrency, init data-dependency loop, phase
maps (out of scope).

**NUOPC's own issues we avoid:** heavy cap boilerplate and IPD complexity;
ambiguous pairing needs namespaces; units verified but not converted; no
built-in averaging so window alignment is inconsistent between caps; strict
clock rule needs a custom `SetRunClock` for mixed cadence; default
`CheckImport` requires all imports at `currTime`; initialize-time data
dependencies can deadlock; parallel regrid sums are not bit-reproducible;
overlapping PET lists serialise execution.

## 2. OASIS3-MCT and YAC

**Similar**
- Topology separate from component code (namcouple/YAML vs `Link` plus
  signature probing).
- Period is an integer multiple of component steps (YAC states this).
- Time reduction on the source side before interpolation.
- Fatal error at setup when a coupled field is undefined.
- `window_mean` ~ AVERAGE, hold ~ INSTANT; one producer may feed many
  consumers.

**Different**

| Aspect | OASIS/YAC | Ours |
| --- | --- | --- |
| Reductions | instant, average, accumulate, min, max | `None` (hold) or any callable; only `window_mean` ships |
| Lag | `LAG`/`src_lag`/`tgt_lag` + restart file | None; ordering by `_next`, cyclic reducing links rejected |
| Component interface | model calls `put`/`get` with own date | runner drives `create_iterator().send()` |
| Concurrency | MPI, distributed grids | Single process |
| Remap | weight files, SCRIPR/CONSERV, YAC interpolation stack with fallbacks | One injected regridder, no fallback |
| Scale/mask | `scale_factor`/`scale_summand`, masks | None |
| Restart | restart files hold lagged fields and partial accumulators | Item-level resume via `Pipeline` only |
| Config | text/YAML | Python, auto-wired |

**Advantages of ours:** no wrappers or new data types; no config file;
runs under `Pipeline`; in-memory, no MPI handshake; callables for custom
reductions; explicit `(t, t+step]` window.

**Disadvantages of ours:** no ensembles (gap 11); same seed for every
stochastic model (gap 6, read); single device and strictly serial; no units
(gap 4); no NaN/mask handling (gap 5); one producer or source per slot; no
accumulate/min/max/weighted helpers; newest-frame-only history; no explicit
lag; single-slot single-lead models only; mock tests only; no accumulator
checkpoint.

**OASIS/YAC do, we do not**
- Worth adding: `window_last/min/max` and weighted accumulate (one-liners as
  callables); `scale`/`offset` on `Link`; NaN guard and optional fill value
  (YAC `fixed` fallback); distinct seeds; explicit lag.
- Out of scope: MPI partitions, weight and restart files, grid-definition
  APIs, conservative stacks, fractional masking.

**Their own issues:** deadlock without lag (+ restart file); lag must not
exceed the coupling period; restart file must match namcouple exactly and a
missing file aborts; run must be an integer number of coupling periods (we
have the same rule via `final_lead_time`); interpolation not bit
reproducible across process counts/compilers; YAC pole edge cases (Gaussian
barycentric fixed in 3.13.1) and compiler-specific failures; name mismatch
is the most common YAC failure; CONSERV normalisation trades conservation
against unreasonable fluxes; MPI/NetCDF/Fortran/CMake build burden;
positional legacy namcouple with unused keywords.

## 3. ClimaCoupler and MUSCLE3

**Similar**
- Sequential sub-cycling to a coupling step (`step!` / our window loop).
- Slow-component averaging: `FluxAccumulator` (sum / `n_steps`) = `window_mean`.
- Held value: MUSCLE3 bridge theta mode / `strict_causality` ~ `reduce=None`.
- One producer per input (MUSCLE3); components own their stepping.

**Different**

| Aspect | ClimaCoupler | MUSCLE3 | Ours |
| --- | --- | --- | --- |
| Component contract | Julia subtype; `get_field`, `update_field!`, `step!`, `sim_dt`, `will_step` | libmuscle `Instance` with ports per operator | existing prognostic model |
| Wiring | hard-coded atmos + surfaces | explicit yMMSL conduits | auto by variable name, optional `Link` |
| Exchange | shared boundary space of ClimaCore fields; coupler computes turbulent fluxes | timestamped messages over the network | in-memory DataArrays |
| Time | fixed `dt_cpl` + callbacks | per-message timestamps, bridges | window = slowest step |
| Process model | single program, MPI/GPU | manager process + peers | single process |
| Fault tolerance | `Checkpointer` | snapshots | item-level resume only |

**Advantages of ours:** no new component API; everything validated at
construction; `Runner`/`Pipeline` unchanged; small; no serialisation; no
deadlock risk because the schedule is computed, not negotiated.

**Disadvantages of ours:** single-slot/single-lead models only (fused
DLESyM excluded); one producer per slot (gap 10); no interpolation, only
hold or unweighted mean; `skipna=False`; no instantaneous vs accumulated
distinction (ClimaCoupler issue #1972); name-only matching; device from
first model and one seed for all (gap 6); no ensembles/resume; history
tied to current link semantics; `_covers` is a heuristic; mock tests only.

**They do, we do not:** flux calculation and conservation checks, exchange
grid with area fractions (all out of scope: ML models emit state, not
fluxes); checkpoint/snapshots (worth adding lightly once models can expose
iterator state); MUSCLE3 bridge `theta` and linear interpolation of a held
link (worth adding); concurrency (out of scope); deadlock detection
(unnecessary, static schedule); per-field units/level/mask metadata (worth
considering units).

**Their issues:**
- ClimaCoupler: restarts fragile (same MPI rank count, not portable,
  restart from component state alone currently impossible); accumulator
  state must be checkpointed; a missing interface method silently falls back
  to a default (`update_field!` only warns); fixed atmos+surface topology;
  open issues #1994 (30+ day instability TripolarGrid +
  ConservativeRegridding), #1972, #1948, #2041 (ocean/sea-ice fluxes out of
  sync), #1917 (NaN tracking), #1847.
- MUSCLE3: unequal-step peers wired directly skew or deadlock (hence
  bridges); bidirectional bridges with variable stepping can deadlock;
  checkpointing needs every component to support it and topology changes on
  resume are unsupported; manager process, network transport and
  serialisation (GPU tensors unverified); issues #411, #356, #419.
  Window convention: `all_in_window` is `[t_cur, t_next)` versus our
  `(t, t+step]`, expect an off-by-one when comparing.

## 4. preCICE, CMEPS, SamudrACE, DLESyM

**Similar**
- preCICE: fixed window with subcycling; last-sample default = our hold;
  parallel-explicit = hold link; window must be a multiple of each step.
- CMEPS: average fast fields before the slow component, on the source grid,
  then map once (linear); we reduce then regrid.
- SamudrACE: atmosphere held SST at window start, ocean takes window-mean
  fluxes: same shape as ours.
- DLESyM: `A48bar = avg(A6..A48)`, `A96bar = avg(A54..A96)` match our
  right-closed window.

**Different**

| Aspect | preCICE | CMEPS | SamudrACE/DLESyM | Ours |
| --- | --- | --- | --- | --- |
| Components | processes + adapters | ESMF/NUOPC, MPI | hand-written loop | models in one process |
| Time scheme | serial/parallel, explicit/implicit | run sequence, lags | fixed explicit | explicit; hold is lagged |
| Sub-window data | constant or waveform | averaged/accumulated | mean atm->ocn, hold ocn->atm | hold or user reduce |
| Mapping | consistent/conservative, RBF | ESMF weights, fraction-weighted | pre-remapped | injected regridder; repo has bilinear and nearest only |
| Weights | n/a | fraction-weighted | equal | unweighted `mean(skipna=False)` |
| Ensembles | n/a | n/a | 5 members (SamudrACE) | rejected |

**Advantages of ours:** no adapters; `Runner` + `Pipeline`; `describe()`;
GPU in-memory exchange; reduce-then-regrid ordering; small and tested.
`timedelta64` leads avoid preCICE's tiny-last-sub-step float problem.

**Disadvantages of ours (what we cannot reproduce)**
- DLESyM as written: SST `S0*` is held for the whole 96 h cycle; our hold
  gives "latest output <= t", so a 48 h ocean would be visible to the
  atmosphere from t=48. No `hold` at window start. The 48 h example in the
  design note therefore differs from the paper.
- SamudrACE Text S1 aligned vs misaligned windows (2.5-day offset): `Link`
  has no offset or window shape; `(t, t+step]` is hard-coded; nothing records
  whether an output is instantaneous, an interval mean ending at the lead,
  or centred.
- State overwrite: SamudrACE overwrites the atmosphere's surface-temperature
  state; we only pass forcing via `send`.
- One reduction per slot; no `f*F` weighting; no conservation; no unit
  checks.
- `history` memory: whole frames for the whole window plus an `xr.concat`
  copy; grows with window/step on 0.25 degree grids.
- Same seed for every stochastic model (read).
- Final lead must be a multiple of the window.
- Mean labelled at the lead of the window start (misleading).
- Tests do not exercise multi-lead forcing with reduce, GPU, or real
  models.

**They do, we do not:** preCICE implicit iteration with checkpoints
(out of scope; `PrognosticMixin.step` is functional so a rewind could be
added later); preCICE waveform interpolation (worth adding as optional
linear slow->fast interpolation); consistent vs conservative constraint
(worth adding as a flag once conservative regridders land); CMEPS
fraction-weighted accumulation (worth adding a `weight=` option for flux
links); CMEPS restart bookkeeping (later). Needed for DLESyM/SamudrACE:
hold-at-window-start, window offset/alignment, multi-lead outputs,
per-variable link semantics, state overwrite hook.

**Their issues:** preCICE tiny last sub-steps from accumulated float error
(not exposed), waveform unavailable in serial-explicit for the first
participant, restricted mappings for parallel participants; CMEPS must
accumulate `f*F`, never `Sum f * Sum F`, and fields are undefined where the
fraction is zero; SamudrACE: mismatch between the training and coupled
window alignment produced SST dipole/ENSO-like biases (they kept
"misaligned" for results while stating "aligned" is correct); DLESyM: T2m
deliberately not passed to the ocean, train and inference coupling must
match, `A48bar`/`A96bar` are non-overlapping.

## 5. legoESM

**Similar**
- Sub-cycle to a coupling window; fast mean to slow (`FluxAccumulator`);
  hold slow state for the fast side; fail-loud construction validation;
  lagged exchange as default; identity-remap fast path when grids match
  (`GridRemapper.identity` vs `same_geometry`).

**Different**

| Aspect | legoESM | Ours |
| --- | --- | --- |
| Components | hand-written JAX atm/ocean/land/ice; `SurfaceToAtm`, `TileResponse` NamedTuples | any prognostic model |
| Field identity | fixed-slot typed structs (21 fields) | lexicon variable names |
| Wiring | hard-coded in the driver; `Interface` bricks are exported but never constructed | auto + `Link` |
| Exchange physics | tile blending, bulk fluxes, implicit surface solve | none |
| Time reduction | dt-weighted accumulator in flux space with residual carry | unweighted mean |
| Regrid | conservative bidirectional `a2o`/`o2a` with masks | injected one-way regridder |
| Driver | wraps the atmosphere driver's segment callback | an E2S `Runner` |
| State | full checkpoint (v3 with lag buffers) | none |

**Advantages of ours:** small, no wrappers, usable by an agent from
signatures alone; `Pipeline` handles IO; strong construction checks; first
forcing window reuses the producer's IC; legoESM's slab loop
(`sfno_slab/coupling.py`) is a monolith that ours makes declarative.

**Disadvantages of ours**
- No conservation concept; unweighted mean.
- Reducer sees only one stacked array: not `dt`, variable or units, so it
  cannot treat instantaneous vs accumulated fields differently, or mix them
  in a slot.
- No NaN/inf guard (`grep isfinite` finds nothing); legoESM raises
  explicitly (`sfno_slab/coupling.py:124-144`) and refuses to carry the
  previous SST forward.
- No units/semantics check.
- Ambiguity all-or-nothing; no partial variable coverage.
- Reducing-link `_covers` returns `leads.size == 1` without checking the
  producer's IC: unverified quirk.
- History pruned to newest frame: no lookback longer than one window.
- Lead-time only; cannot couple models with different init times.
- ICs are fetched independently per model; consistency unchecked beyond
  `item.time`.
- No per-link transform hook (legoESM glue does unit conversion and NaN
  fill inline).
- Equal-lead ties resolve by dict order. The mutual-hold probe showed
  results are identical for both dict orders *(verified)*, but this is not
  asserted by a test.

**legoESM does, we do not:** dt-weighted accumulator with residual carry
(only needed for non-integer ratios; a weighted `window_mean` is worth
adding if steps vary); flux-space averaging of nonlinear terms (docstring
caveat only); fail-loud NaN/missing checks (**worth adding**, opt-in
`check_finite`; also gate on config, not key presence, to avoid false
errors); tile blending, conservative masks, ocean CFL sub-stepping, MPI,
budget tripwires (out of scope); checkpoint of lag buffer and accumulator
(only when mid-item resume lands).

**Their issues:** silent zero defaults (perpetual polar night, runaway
salinification); `Interface` bricks declared but never instantiated (our
"no components" decision matches); fixed-struct ABI accretes fields and
touches accumulator, blender and checkpoint each time; window-flush
constraint (mean of means double-weights; we reduce on raw frames so we are
safe); nonlinear averaging (`T_rad` vs mean T^4; same trap for any
`window_mean` over derived fields: document it); lag buffers must be
checkpointed; implicit unit/sign conventions; ad-hoc ML glue; hardcoded
driver ordering with two divergent drivers.

## 6. nvcoupler (PR #1229)

**Similar:** single process, in-memory, DataArray-native, inference only;
no MPI, checkpoint or ensemble batching in either; name-based wiring with
loud construction errors; `describe()`; integer-multiple cadences; lagged vs
sequential via ordering; windowed reductions for fast->slow.

**Different**

| Aspect | nvcoupler | Ours |
| --- | --- | --- |
| Component | `Component` ABC with Prognostic/Diagnostic/Callable/Data/Mediator adapters | the model itself |
| Field identity | own `FieldDictionary` (CF names, aliases, units) + `Field` + `State` | lexicon name, bare DataArray |
| Clock | fixed global `dt`, components run when aligned | window = slowest step, furthest-behind first |
| Schedule | `RunSequence`, derived or DSL/YAML | implicit via `_next` |
| Reductions | windowed `Connector`, `AccumulationMediator`, `TrailingAverageMediator`, `CellMethod` | one `Link.reduce` callable |
| Forcing delivery | adapters: overwrite state, conditioning kwarg, `PullAdapter` | `send()` into declared forcing slots |
| Output | own `to_xarray()` Datasets | `Runner`, so `Pipeline` handles IO |
| Spatial | built-in bilinear, point sampling, mask fill, vertical interpolation | injected `grids.Regridder` |

**Advantages of ours:** 2 public classes + 1 helper vs ~50 exports; reuses
`Runner`, `Pipeline`, `_fetch`, `_map_field`, `_seed_model`,
`recommended_sources` (what reviewer pzharrington says nvcoupler
reimplements); cadence and grid read from signatures; uses the shared
`Regridder` protocol and `same_geometry` that he asked for; strict
construction validation; producer IC reused for the first wired window and
omitted from `data_requests`; `Link` is a frozen dataclass and per-item
state is local, iterators closed in `finally`.

**Disadvantages of ours**
- No units/semantic checks (gap 4).
- No valid-time check (gap 1).
- Regridder output unchecked (gap 2, *verified*).
- No NaN masks/weights.
- Held links are single-frame (design-note open question 4).
- Mutual-hold pair at equal cadence: one side sees the other sequentially;
  true lagged coupling is not expressible (gap 9).
- Narrow model support (single slot, single lead).
- No diagnostics in the loop, no non-gridded payloads, no ensembles.
- One reduce per link, no `CellMethod` metadata.
- Tests: mocks only, no real weights, no GPU/CuPy, no bad-regridder test.

**nvcoupler does, we do not**

| Capability | Verdict |
| --- | --- |
| FieldDictionary units check | add, built on lexicon units, not a second vocabulary |
| Valid-time / coordinate assertion on delivery | add |
| Regridder output validation | add (compare to slot signature after `link.regridder(x)`) |
| Windowed connectors / Mediators | covered by `Link.reduce`; add sum/max/min presets and per-variable reductions |
| Run-sequence DSL / YAML | out of scope (reviewer flagged as unwanted config systems) |
| Point sampling / `PullAdapter` | out of scope |
| Vertical interpolation | out for now |
| Mask fill | out unless a masked pair needs it |
| `DataComponent` | not needed; a source bound to a slot does it |
| In-memory Dataset collection | out; use `Pipeline` backend |
| `Driver.initialize(ics)` | worth adding: accept a dict of IC DataArrays |
| Linear extrapolation connectors | out |
| Diagnostics in loop | worth adding |

**nvcoupler's review blockers: do we share them?**

| Blocker | Ours |
| --- | --- |
| Longitude wrapping in bilinear interp | avoided: uses `grids.BilinearRegridder`, which wraps |
| Multi-valued `time` exports | avoided: no collection |
| Alias publishing / alias-keyed metadata | avoided: single vocabulary |
| Coordinate alignment on injection (`broadcast_like`) | partly shared: relabelling via `assign_coords` |
| Regridder output validation | **shared** *(verified)* |
| CuPy nearest fill | n/a; GPU path untested |
| Pressure degeneracy / hybrid levels | n/a |
| Mask aliases / mask left on source grid after regrid | n/a, but no mask handling |
| State reset / partial step | avoided: state local to `run_item` |
| Metadata dropped in Datasets | n/a; no units check either |
| Valid-time / cadence checks | window multiples validated; delivered valid time **not** checked |
| Misspelled `time_policy` silently falls back | avoided: `reduce` is `None` or callable |

## 7. Recommended follow-ups

Small and in the spirit of the current design (each is a few lines):

1. Assert delivered lead/valid time before relabelling (`_deliver`).
2. Validate regridder output geometry against the slot signature.
3. Explicit source for a slot wins over auto-wiring (or error on conflict).
4. Per-model salt in the seed so stochastic models do not share noise.
5. Ship `window_last`, `window_min`, `window_max`, `window_sum`; optional
   `check_finite` guard.
6. `Link(hold="window_start")` and a `window_offset` for DLESyM/SamudrACE.
7. Lexicon-based units check; `scale`/`offset` on `Link`.
8. Accept a dict of IC DataArrays; run diagnostics inside the loop.
9. Tests: bad regridder output, mutual-hold both dict orders, multi-frame
   forcing with reduce.

Deferred: ensembles, iterator/history checkpoint, multi-slot and multi-lead
models (needs the model-side DLESyM split), linear interpolation, automatic
regridder selection and conservative regridding, weighted reductions.

## 8. What was and was not verified

Verified by running code in this review: the 26 coupler tests pass; the
bad-regridder acceptance; the explicit-source override; the mutual-hold
pair gives the same result under both dict orders; `_seed_model` has no
per-model salt (read).

Not verified: anything using real StormScope or DLESyM weights, GPU/CuPy,
framework behaviour beyond the docs and issue titles the reviewers read,
and nvcoupler's blockers on its current head (taken from PR comments).
