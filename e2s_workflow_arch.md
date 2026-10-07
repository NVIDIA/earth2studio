# Earth2Studio Workflow & Execution Architecture

Working design memo: API changes/extensions for the execution model, plus a
sequencing plan.

Scope: distributed/restartable inference, the boundary between data sources and
models, and what should move from `recipes/eval/src/` into the core package.

Status: the five original design questions and the six follow-up TODOs are all
settled, with conclusions folded into the sections below. One item is
deliberately left open pending outside input — the name of the transform
protocol. The device-transfer mechanism at the source boundary is settled as
**DLPack, shipped in Phase 1 as an interim mechanism**; the parallel `cupy`
proposal may replace it later, and the coupling is confined to a single
function so the swap stays cheap.

The full plan — together with the `cupy` proposal and related enhancements —
targets the package's official **v1.0** bump, so breaking changes are
acceptable where flagged; see [Migration mechanics](#migration-mechanics).

## Contents

- [Motivating findings](#motivating-findings)
- [API changes and extensions](#api-changes-and-extensions)
  - [1. Data layer](#1-data-layer) — incl. [grid representation](#grid-representation)
  - [2. Model contract](#2-model-contract) — incl. [hooks](#why-promote-the-hooks),
    [`default_source()`](#what-default_source-should-return),
    [forcing seam](#the-forcing-seam), [DA/SDA](#da-and-sda-models)
  - [3. Execution substrate](#3-execution-substrate) — incl.
    [assumptions](#what-each-abstraction-assumes), [grain](#work-item-grain)
  - [4. Cross-cutting cleanups](#4-cross-cutting-cleanups)
- [Implementation plan](#implementation-plan) — incl.
  [testing/CI](#testing-and-ci), [migration](#migration-mechanics)
- [Serving layer (`earth2studio/serve`)](#serving-layer-earth2studioserve)
- [Deferred: level-2 rollout checkpointing](#deferred-level-2-rollout-checkpointing)
- [Known bug: resumed ensemble corruption](#known-bug-resumed-ensemble-corruption)

## Motivating findings

Evidence gathered from the current tree; these drive the proposals below.

**User need for ready-to-run workflows.** Feedback from SAs indicates many
users desire ready-to-use workflows that are easy to configure and get to
forecast results quickly. The current `run.deterministic` and related
interfaces are good but do not cover most complex use-cases; the bulk of
users work at a complexity somewhere between the notebook examples and the
full recipes.

**Core has no large-scale story.** No `torch.distributed` anywhere in
`earth2studio/` — no work sharding, no collectives, no barriers.
`run.deterministic` / `run.diagnostic` have no batching over initial
conditions, so the whole IC list must fit in device memory. `run.py` never
calls `close()`, though `AsyncZarrBackend` and `netcdf4` both define one.

**The same engine has been built four times.** `run_with_rank_ordered_execution`
exists in 2 copies with zero behavioral divergence; rank work-splitting in 4;
output-coordinate assembly in 4; ensemble batching in 3 (core `run.ensemble`,
HENS `get_batchid_from_ensid`, eval `EnsembleGroup`). `s2s` is a near-verbatim
fork of `hens` — 20 identically-named functions, seed derivation byte-identical
including a shared typo.

**Recipe-only code rots.** `eval` is actively maintained; `hens` and `s2s` were
last touched ~5 months ago, still pin `earth2studio>=0.9.0`, still `exit()` on
idle ranks, and `hens` has no resume at all (a killed 1000-member run restarts
from zero).

**Core's restart system is unused by recipes.** `earth2studio/utils/checkpoint.py`
(1484 lines, documented) has zero uses across `recipes/`. It solves *mid-rollout
state* resume with per-rank catalogs; every recipe actually needs
*work-item-identity* resume that survives a world-size change.

**Model wrappers own data sources.** Many wrappers hold a live `DataSource`
internally — `stormcast`, `stormcastconus`, `stormscope`, `ace2`, `datareplay`.
Archetype: `StormCast(conditioning_data_source=GFS_FX())`. This blocks engine
prefetch, breaks offline runs, and makes predownload undeclarable.

**Model contracts are conventions, not contracts.** The lead-time rebasing rule
(`output_coords["lead_time"] = input_coords["lead_time"][-1] + ...`) is written
longhand in `ucast.py:745-746` and `atlas.py:282-283` and nowhere else — which
is why the recent `Aurora.create_iterator` bug needed hand-alignment against
FuXi, DLWP and FengWu. `PrognosticMixin.front_hook`/`rear_hook`
(`models/px/utils.py:21-30`) exists in 25 of 29 wrappers but is absent from the
`PrognosticModel` protocol, has no coord declaration, and has one doc example
and zero recipe uses.

**The hide-or-expose question is already being answered ad hoc.** `aifs2`
(`preload_invariants=False`) and `ucast` (`preload_static_fields=False`)
independently grew the same toggle for whether static fields are wrapper-loaded
or caller-supplied.

## API changes and extensions

### 1. Data layer

- **Upstream source combinators** *(add)* — `CompositeSource`,
`CadenceRoundedSource`, `ValidTimeForecastAdapter`, `PredownloadedSource`,
`RegriddedSource` from `recipes/eval/src/data.py`.
- **Upstream `Regridder` ABC** *(add)* — from `recipes/eval/src/regrid.py`; NN +
bilinear. The numerics already exist as device-movable `nn.Module`s in
`utils/interp.py`, so only the ABC and wrapper move.
- **Device in the `DataSource` contract** *(extend protocol)* — today
`fetch_data` takes `device=` but sources don't, so a wrapper cannot know what
to emit and chains ping-pong host↔device.
- **Grid in the `DataSource` contract** *(extend protocol)* — sources advertise
their grid declaratively. See [Grid representation](#grid-representation).
- **No host round-trip at the source boundary** *(fix; DLPack interim)* —
`Regridder.apply_dataarray` currently forces numpy via `.values`
(`recipes/eval/src/regrid.py:147`). The *requirement* this plan depends on is
that source-boundary regrid must not round-trip through the host when the
data is already on device. **Mechanism: ship the DLPack path in Phase 1 as an
interim**, to be reconciled with — and possibly replaced by — the parallel
deeper-`cupy` proposal. Two facts make the later swap cheap: the `Regridder`
numerics are already device-agnostic torch `nn.Module`s, so nothing there
changes; and the coupling is confined to a single xarray→tensor boundary, so it
is one function to revisit, not a cross-cutting concern.
- **Retire `fetch_data(interp_to=)`** *(deprecate)* — already rejected on the
`legacy=False` path (`data/utils.py:152-155`); superseded by an explicit
`Regridder`.
- **Remove the StormCast hack** *(fix)* — `data/utils.py:286-291` hardcodes
dropping `hrrr_y`/`hrrr_x` for one model, TODO-flagged, in the shared fetch
path. The root cause is a naming inconsistency; see below.
- **Rename HRRR's projected dims** *(fix, small)* — HRRR is the only source that
prefixes its projected dims (`hrrr_x`/`hrrr_y`); goes, himawari\_ahi,
meteosat\_fci, jpss, opera, planetary\_computer and dynamical all use plain
`x`/`y`. Normalizing HRRR deletes the StormCast special case outright rather
than working around it.
- **Grid-equality predicate** *(add)* — so the no-op fast path in
`forecast.py:_align_to_grid` becomes a shared, tested function rather than an
ad-hoc check repeated per call site. Must define tolerance semantics explicitly
(`allclose` with a stated `atol`, not bitwise equality) — sources round
coordinates differently, and a bitwise check would make the fast path flap.

Keep **both** regrid application points. Source-boundary regrid is correct for
predownload/caching (materialize on the target grid once); tensor-space regrid
is correct for the hot loop. `RegriddedSource`'s docstring already documents
this split — it is a feature, not a defect.

#### Grid representation

**A registry, not a lexicon.** A lexicon solves *vocabulary translation* (model
says `t2m`, GFS says `TMP:2 m above ground`, ERA5 says `2t`) — bidirectional,
many-to-many, per source pair. Grids have no such disagreement. They need three
cheaper things:

1. **A value type — already exists.** A grid is the spatial subset of a
   `CoordSystem`: dimension names plus `_lat`/`_lon` coordinate *value* arrays.
   The underscore prefix already distinguishes value arrays from dimension names,
   and `prep_data_array` (`data/utils.py:264-299`) already handles the full
   regular↔curvilinear cross product with 1D or 2D arrays. Keep this; do not
   invent a parallel type. It is what `map_coords`, `handshake_coords` and
   `Regridder.target_coords()` already speak.
2. **A registry — exists in embryo, needs promoting.**
   `recipes/eval/src/grids.py` is already "Hydra-instantiable grid resolvers":
   `gfs_grid()`, `arco_grid()`, `goes_grid()`, `glm_grid()`, `mrms_grid()`, each
   returning `(lats, lons)`. Names buy config ergonomics and compact
   serialization only — embedding HRRR's curvilinear arrays by value is ~7.6M
   floats. BYO grids pass values directly and skip the registry.
3. **Advertisement — the real gap.** `mrms_grid()` discovers the grid by
   *probing a sample fetch*. If knowing a grid requires I/O, offline and
   predownload planning cannot work — precisely what the forcing spec exists to
   fix. Only **2 of ~50** data sources expose `.lat`/`.lon` today
   (`data/ace2.py`, `data/goes_glm.py`), yet ACE2 depends on that convention.

**The taxonomy.** A sweep of all ~50 sources says the descriptor is a small sum
type — `regular | curvilinear | mesh(id)` — plus one category that is explicitly
*not* a grid:

- `lat`/`lon` 1D — **regular**; the large majority (arco, gfs, gefs, cds, cfs,
  cmip6, wb2, mrms, ncar, nclimgrid, metop\_\*, …).
- `x`/`y` (and today `hrrr_x`/`hrrr_y`) — **curvilinear/projected**; goes,
  himawari\_ahi, meteosat\_fci, jpss, opera, planetary\_computer, dynamical.
- **mesh** — HEALPix dims `face` (12) `/height/width` (`dlesym.py:309`), regrid
  via `earth2grid`, identified by `nside`; and reduced/octahedral Gaussian, where
  AIFS carries flat native lat/lon buffers (`aifs.py:77-80`) and regrids via
  `earthkit.regrid` against a *remote* grid database (`aifs.py:394`), identified
  by a name like `N320`. The four duplicated `earth2grid` wrappers are evidence
  this case is real, not speculative.
- `station` — **point/irregular observations**; ghcn, iem, isd, ufs,
  utils\_ncep. *Not grids*, and the architecture already reflects that: they are
  served by the separate `DataFrameSource`/`ForecastFrameSource` protocols in
  `data/base.py:150,215`. The grid descriptor should not try to cover them.

**Not part of the grid:** `interp_method`. StormScope proves it is per-channel
physics (nearest for radar/satellite, bilinear for sparse GLM), so it belongs on
the `Regridder`/regrid intent, not the grid identity.

### 2. Model contract

- **Write down the iterator contract** *(spec + test)* — step-0 yield semantics,
history-window representation, lead ticks per step, axis names
(`batch`/`ensemble`/`sample`), and the `output_coords` rebasing rule. Add a
conformance test every px/dx wrapper must pass.
- **Promote `front_hook`/`rear_hook`** *(extend protocol)* — give it declared
`input_coords`/`output_coords`, put it in `PrognosticModel`, and make it a
composable chain rather than attribute monkeypatching. Currently absent from
`dxwrapper` and `interpmodafno`, and half-wired in `gencast_mini` and the two
`graphcast_*` wrappers (`rear_hook` only, no `front_hook`). See
[Why promote the hooks](#why-promote-the-hooks).
- **Declared forcing/conditioning** *(add; breaking for 5 models)* — models
*declare* exogenous needs (variables, grid, cadence, lead offsets) and are
*handed* data instead of owning a `DataSource`. Makes predownload derivable,
enables engine prefetch, and unblocks offline runs. See
[The forcing seam](#the-forcing-seam).
- **`default_source()`** *(add; name TBD, return shape TBD)* — mirrors the
existing `load_default_package()` convention. Returns a **raw source,
constructed lazily inside the method** — not a default-argument literal, and
not pre-composed with a regridder. **Open question**: whether this stays a
single-source method (name likely needs to disambiguate it as *forcing*, not
IC) or becomes a plural, keyed `default_sources() -> dict[str, DataSource]` to
cover models with more than one forcing need (e.g. StormScope's separate MRMS
and GLM sources). See
[What default_source() should return](#what-default_source-should-return).
- **Generalize the static-field toggle** *(add)* — subsume `preload_invariants`
and `preload_static_fields` into one declared mechanism.
- **Data-assimilation models strain several of the above** — see
[DA and SDA models](#da-and-sda-models). Stateless analysis models fit
`WorkItem` but not the iterator contract; cycling SDA models break work-item
independence outright.

**Naming.** The transform protocol gets a distinct name — *not* `DiagnosticModel`,
even though that is structurally identical (`__call__(x, coords)` +
`input_coords` + `output_coords` + `to`). Structural similarity is not intent: a
diagnostic produces a derived geophysical quantity as a *product*; a transform
adapts tensors to satisfy a contract. Conflating them would make both harder to
explain and would imply diagnostics are interchangeable with adapters in configs,
which they are not. Name TBD, but it should read as adaptation, not derivation.

**Explicitly out of scope: intrinsic forcing stays hidden.** Cosine zenith angle
(17 wrappers), packaged statics, and normalization constants are pure functions
of valid time, the model's own grid, and its own package assets. Hiding them is
correct, re-surfacing them would yield no benefit. DLESyM is the proof case: it
has **no** data source, computing insolation analytically from times
(`dlesym.py:605-634`) plus static buffers.

**Also not a problem: history.** ~19 model classes need >1 input timestep and
all declare it uniformly as negative `timedelta64` offsets in
`input_coords["lead_time"]`, which the generic fetch already honors.

#### Why promote the hooks

**What they enable.** The one documented use is *stochastic model perturbation*
(`examples/02_medium_range/02_model_perturbation_hook.py:159`) — perturbing state
each step to generate ensemble spread from model error rather than IC error, an
SPPT-like technique and a genuinely different source of spread from
`perturbation/`. Beyond that, several things currently done as bespoke code
become expressible as reusable hooks: DLESyM's ocean masking (today post-hoc in
`recipes/eval/src/pipelines/dlesym.py:204-259`), FuXi's `tp06` unit-convert and
clip (`fuxi.py:364-369`), StormScope's GLM state injection
(`stormscope.py:2276-2288`), and nudging/bias-correction toward analysis.

**Why the current form is inadequate** — five concrete defects:

1. **No coord contract**, so users hand-write model-specific tensor surgery. The
   shipped example says so in its own comment: *"`center.unsqueeze(-1)` is DLWP
   specific since it operates on a cubed sphere… To switch out the model,
   consider removing the `unsqueeze`."* A declared `input_coords`/`output_coords`
   is exactly what makes a hook portable across models.
2. **Not in the protocol** (`models/px/base.py` never mentions `PrognosticMixin`),
   so no generic workflow can rely on it; 2 wrappers apply neither hook and 3
   more apply only `rear_hook`.
3. **Applied only in `_default_generator`, never in `__call__`.** Verified across
   `aifs`, `ace2`, `aurora`, `atlas`, `fcn3`, `sfno`: every `__call__` invokes
   `_forward` directly. So a hook set by a user is **silently ignored** for
   single-step use — a quiet correctness trap.
4. **Placement is not uniform.** Aurora applies `rear_hook` to a *slice*,
   `x[:, :, 1:]` (`aurora.py:435`), rather than the full tensor.
5. **No composition.** `model.front_hook = lambda …` is plain instance-attribute
   assignment shadowing the class attribute; a second assignment silently
   replaces the first.

**Assessment: the case is moderate, not overwhelming.** This is cheap
(the mechanism exists in 25 wrappers) and fixes a real silent-ignore bug, but no
recipe uses hooks today, so the demand is latent. Worth doing in Phase 1 because
it is small and because defects 3 and 4 are bugs regardless of whether the
protocol promotion happens.

#### What `default_source()` should return

Correcting an earlier overstatement: **a live source is not inherently an
anti-pattern.** `GFS_FX.__init__` sets `self.store = None` and only builds the
object store on first fetch (`data/gfs.py:129-131`), so construction does no
network I/O. `load_default_package()` is a good precedent precisely because
`Package` is also a lazy handle — cheap to construct, describes *where*, fetches
later.

The real defect is narrower: **it is a mutable default argument.**
`conditioning_data_source: … = GFS_FX()` (`stormcast.py:284`,
`stormcastconus.py:807`) and `forcing_data_source: … = ACE2ERA5Data(...)`
(`ace2.py:167,333`) are evaluated **once at module import** and shared by every
caller who takes the default. `GFS` carries per-instance mutable state that
populates lazily — `self.store` and `self._tmp_cache_hash` (`data/gfs.py:129-131,
518-520`) — so two StormCast instances in one process silently share one
source's store and cache identity. The fix is the ordinary one: default to
`None` and construct inside.

**So the spec-vs-object framing was the wrong axis.** What predownload actually
needs is not "a spec instead of an object" but *a declaration of what will be
requested*, which is `forcing_spec` — a separate thing from the source handle. A
lazy source object is fine; an undeclared request is not.

**On the regridding gap** (correct concern): a bare `default_source()` returning
`GFS_FX()` does not put data on the model's grid. Two options —

- *(a)* return a pre-composed `RegriddedSource(GFS_FX(), regridder)`. Rejected:
  it bakes one interpolation method in, and StormScope proves method is
  per-channel physics; it also collapses the two regrid application points we
  deliberately keep, forcing source-boundary regrid even in the hot loop.
- *(b)* **return the raw source; put grid and method in `forcing_spec`; let the
  engine compose the regridder.** Consistent with the forcing seam (model
  declares, engine executes), and it preserves the predownload-vs-hot-loop
  choice.

Take *(b)*.

**Open question, not yet settled: naming, and what happens when a model needs
more than one forcing source.** Everything above was worked out against
single-need models — StormCast's one GFS conditioning source, ACE2's one ERA5
forcing source. Two problems surface once that assumption is dropped:

- **Naming is ambiguous even in the single-source case.** `default_source()`
  mirrors `load_default_package()`, which is singular because there really is
  one default weights package. But "source" doesn't say *which* data — IC data
  still flows through the caller-supplied source passed to the existing
  `fetch_data` path, unrelated to this method; `default_source()` only ever
  meant *forcing*. A caller reading `model.default_source()` cold has no way to
  know that. `default_forcing_source()` (or similar) removes the ambiguity for
  free, independent of the question below.
- **Multi-source models break the singular-return shape entirely.**
  StormScope needs two forcing sources with different grids and different
  regrid methods (nearest for MRMS radar, bilinear for the sparse GLM field) —
  not substitutable, and a caller may reasonably want to BYO just one leg
  (swap GLM, keep default MRMS). Two candidate shapes:
  1. *Model composites internally, still returns one source* — e.g. StormScope
     builds a `CompositeSource` dispatching by variable and returns that.
     Reuses the `CompositeSource` combinator this plan already promotes to
     core, and the call site stays a single line. But it quietly re-admits
     source *composition* into the model wrapper right as this plan is trying
     to remove data-sourcing decisions from models, and there's no clean way
     to override a single leg from outside the composite.
  2. **Plural, keyed return** — `default_sources() -> dict[str, DataSource]`,
     keyed to match the `name`/`target` fields each `forcing_spec` request
     already carries (`"mrms"`, `"glm"`, or `state`/`conditioning`). A
     single-source model is just the one-entry-dict degenerate case. This
     matches existing precedent better than either alternative:
     `StormScopeMRMS.__init__` already takes `conditioning_data_source` and
     `glm_data_source` as two separate kwargs rather than one merged source
     (`stormscope.py:1969,1975`), and the eval recipe's predownload stores are
     already named and cached separately (`data_goes.zarr`, `data_mrms.zarr`,
     `pipelines/base.py:122`) rather than as one composited store. It also
     makes partial override trivial (`{**model.default_sources(), "glm":
     MyGLM()}`) and keeps composition an explicit, caller-visible decision
     instead of one hidden inside the model.

  Leaning toward *(2)* on the strength of that existing-precedent argument, but
  this has not been decided — flagging as an open design question rather than
  folding it into the settled *(b)* above, since it changes the method's
  return type and therefore every call site, and is worth a second pair of
  eyes before locking in.

#### The forcing seam

Split it: **tensor argument on the model side, declarative spec consumed by the
engine.** The model-side entry point and the scheduling are separate decisions,
so the original callback-vs-push framing was a false choice.

*Enabling fact:* **no model's forcing fetch depends on its own output.** Every
`fetch_data` call site in `ace2`, `stormcast`, `stormcastconus`, `stormscope`
and `datareplay` is a pure function of `coords["time"]`, `coords["lead_time"]`,
and a fixed variable list. So `(init_time, nsteps)` is enough to enumerate every
request up front, and a declarative spec is possible.

- **Model side**: adopt StormScope's `call_with_conditioning(x, coords,
  conditioning, conditioning_coords)` (`stormscope.py:1367`) as the universal
  contract. The model keeps regrid, normalize, and channel layout — all of which
  depend on model internals; the caller owns only *when* and *what data*. This
  seam already exists and is already driven by the eval pipeline and a public
  example.
- **Declaration**: `forcing_spec(init_time, nsteps) -> list[(time, variables,
  grid)]` is a **model method** — only the model knows what it needs (ACE2's
  forcing variable list comes from its own checkpoint's
  `stepper._input_only_names`, `ace2.py:203-215`, not from user config).
- **Execution**: the engine batches, dedups, and prefetches from that spec, then
  pushes step-by-step into the same tensor argument.

*Why the declaration is not on `Pipeline`:* one declaration has **three**
consumers — the inference engine (prefetch + push per step), predownload (derive
what to cache, replacing StormScope's hand-maintained `predownload_stores`), and
offline-run validation (does this cached store cover what the model will
request?). Putting it on `Pipeline` hides it from predownload, which is today's
problem inverted.

*Why not a pull-callback:* it reproduces exactly the per-step fetch that ACE2
already had to build a cache around, and it cannot be batched across steps.
ACE2's `_fetch_forcing` cache (`ace2.py:471-546`) is a hand-rolled prefetch
engine solving three problems the declarative layer would absorb: bulk fetch
aligned to the source's file granularity (one netCDF *per year*), dedup of the
overlapping `[t, t+dt]` windows that consecutive steps both request, and a
bounded residency policy (it clears so at most one year is GPU-resident).

*The push path must survive alongside it.* Coupled StormScope feeds one model's
**prediction** in as the other's conditioning
(`recipes/eval/src/pipelines/stormscope.py:313-318`). That can never be
pre-enumerated — but it uses the identical tensor entry point, so one contract
covers both. `stormcastconus.create_generator`'s `obs = yield x, coords`
coroutine (`stormcastconus.py:674-681`) is an existing in-repo precedent for
pushing per-step data into a rollout.

**Acknowledged gap: orchestrating the coupled rollout itself is out of scope
here.** The push path above solves *how* one model's output reaches another as
a tensor. It does not solve *who drives two models concurrently, in lockstep,
wiring one's output into the other's input each step* — coupled StormScope
(GOES model forcing MRMS model) needs both running together as a single unit,
which is why it currently needs its own hand-rolled eval `Pipeline` subclass
rather than fitting the shared `run` loop. Nothing in `WorkItem` or `Pipeline`
as scoped here changes that: a `WorkItem` is `(time, member_ids)` driving one
model, with no notion of a work item that owns a pair (or graph) of coupled
models stepping together. Structurally this is the same shape as cycling SDA's
"not independently schedulable" problem (see [DA and SDA
models](#da-and-sda-models)) — a coupled rollout is not order-free and not
decomposable into independent `WorkItem`s either.

*Near-term mitigation, with a real cost:* a thin composite model wrapper (e.g.
`CoupledStormScope`) that owns both sub-models internally and presents as one
`PrognosticModel` to `Pipeline`/`WorkItem` — precedented by DLESyM's existing
internal coupling. This needs no execution-substrate changes, but it is in
tension with this plan's own goal of pulling composition logic *out* of model
wrappers (the forcing seam moves data-sourcing decisions out; a composite
wrapper moves model-coupling decisions further in), and it does not generalize
— every new coupled pair gets its own bespoke wrapper class, reproducing the
"same engine built N times" problem this plan otherwise targets.

*The real fix is out of scope for this plan* and belongs to the parallel
model/earth-system coupler effort, which would let "model B's input is model
A's output" be a declarable relationship the `Pipeline`/scheduler understands
directly, rather than something hand-built into a wrapper class or a one-off
`Pipeline` subclass. This plan should not foreclose that: the `WorkItem`
order-constrained/rank-pinned scheduling flag already proposed for cycling SDA
(see [Mitigation](#da-and-sda-models)) is a plausible shared primitive both
cases could sit on top of, so `WorkItem` and `Pipeline` should be kept general
enough to host it later rather than assuming every work item drives exactly
one independent model. **Sequencing implication:** if the coupler effort is
going to change how `Pipeline` accepts and steps models, it is worth landing
enough of that effort *before* Phase 3 (`Pipeline` + `run.py` convergence) to
shape `Pipeline`'s constructor and stepping contract for it up front, rather
than upstreaming `Pipeline` in Phase 3 and having to break it again shortly
after for coupling. This is a call for the two efforts' owners to make
together, not a decision made here — but Phase 3 should not be scheduled
assuming the coupler effort is irrelevant to `Pipeline`'s shape.

*Spec must express, at minimum:* lookahead/lookbehind (ACE2 needs `[t, t+dt]`;
StormCastCONUS needs `t+1h`; StormScope needs a past window of N frames), target
grid, and a `target: state | conditioning` distinction (StormScope's GLM writes
into *state* channels, `stormscope.py:2276-2288`).

*Two implementation gotchas:* `call_with_conditioning` is not `@batch_func`
decorated and hard-codes literal `"batch"`/`"time"` keys
(`stormscope.py:1394-1404`); and ACE2 builds its regridder from the source
object's `.lat`/`.lon` attributes at construction (`ace2.py:227-240`), so grid
must come from the spec rather than a live source handle — hence the
`DataSource` grid-advertisement change in the Data layer.

#### DA and SDA models

These strain the plan in specific, enumerable ways. There are **three shapes**,
not one:

| Shape | Models | Fits `WorkItem`? | Fits iterator contract? |
| --- | --- | --- | --- |
| Stateless single-shot | `HealDA`, `InterpEquirectangular` | yes | no — no state, no lead time |
| Background + obs, single-shot | `CorrDiffCosmoEra5SDA` | mostly | no — primes with `yield None` |
| **Cycling, stateful** | `StormCastSDA`, `StormCastCONUS` | **no** | yes |

**The protocol is already different.** `AssimilationModel`
(`models/da/base.py:34-70`) is variadic over `pd.DataFrame | xr.DataArray`, not
`(torch.Tensor, CoordSystem)`, and `input_coords()` returns
`tuple[FrameSchema | CoordSystem, ...]` — a union gridded code paths do not
handle. `HealDA.__call__(conv_obs, sat_obs)` takes **no state argument at all**
and smuggles valid time in via `df.attrs["request_time"]`
(`healda.py:445,483-490`).

**What breaks, concretely:**

- **`WorkItem` independence.** `StormCastSDA`'s unit of work is a *cycle chain*,
  not a time: `create_generator` autoregresses, feeding each analysis into the
  next (`sda_stormcast.py:852-920`). Such items are not order-free, not
  independently schedulable, and cannot be retried per-`attempt` without
  replaying the chain. The eval recipe already concedes this —
  `StatelessAssimilationRunner` **hard-rejects** any model with non-`None`
  `init_coords()` (`recipes/eval/src/assimilation.py:363-371`), and the
  `AssimilationRunner` ABC docstring reserves a future cycling runner that
  "constrains work distribution (analysis times must be processed in order on one
  rank)" (`:309-329`). So StormCastSDA is *currently undrivable* by eval.
- **The iterator contract is meaningless for single-shot models.** HealDA has no
  lead time; the recipe fabricates `lead_time=[0ns]` purely to fit the output
  schema (`assimilation.py:466-486`). `CorrDiffCosmoEra5SDA` deliberately primes
  with `yield None` (`:703`), violating "step-0 yields initial state" by design.
- **Observations do not fit `forcing_spec`.** An obs request is
  `(window=[t+lo, t+hi], variable, *fields*, tabular)` — no grid, and `fields`
  (satellite metadata: `type/elev/pres/sensor_index/satellite/scan_angle/…`,
  `healda.py:247-274`) has no slot in a `(time, variables, grid)` triple. The
  codebase already needed a **parallel** declaration type for this,
  `PredownloadFrameStore` (`assimilation.py:258-266`) — good evidence one gridded
  spec will not absorb obs.
- **Observations are not conditioning.** They enter as a likelihood/guidance term
  (DPS score correction) with a mask and noise model, not as a dense network
  input — which is why these models deliberately omit `torch.inference_mode`
  ("DPS guidance requires gradient computation through the denoiser",
  `sda_stormcast.py:804-806`). Forcing obs through a single `conditioning` tensor
  loses the mask and the noise parameters.
- **`member_ids` mis-models CorrDiff-SDA.** Its ensemble axis is `sample` and is
  produced *inside one call* (`sda_corrdiff_cosmo_era5.py:357`), so batching by
  member would double-count.

**Landmine worth fixing regardless:** `StormCastCONUS` ships two entry points —
`create_generator` (send-based, receives obs) and `create_iterator`
(`stormcastconus.py:685-691`), which is `yield from create_generator` — so the
send channel technically survives, but the method is *typed and documented* as a
plain `Iterator` "without observation input". Any consumer driving it with
`for`/`next()` implicitly sends `None` each step and gets an **unconditioned
rollout with no warning**; the interface gives callers no way to know a send
channel exists. Any generic runner defaulting to `create_iterator` would quietly
produce an un-assimilated forecast.

**Mitigation.** Preserve an explicit *runner* seam that owns stepping semantics
and may declare scheduling constraints — the `AssimilationRunner` ABC is the
existing acknowledgement of this. If the new design has no equivalent, the
cycling case becomes unrepresentable. Concretely: a work item should be able to
declare itself **order-constrained and rank-pinned**, which is a scheduling
attribute rather than a new grain.

Also worth noting: `StormCastSDA` (`models/da/`) and `StormCastCONUS`
(`models/px/`) have near-identical semantics on opposite sides of the package
boundary. Reconciling them is out of scope here but is the kind of thing the
contract spec should surface.

### 3. Execution substrate

- **`WorkItem` + `distribute_work`** *(add)* — de-Hydra'd. Returns `[]` for idle
ranks rather than calling `exit()`. See [Work-item grain](#work-item-grain).
- **Progress/resume store** *(add)* — work-identity keyed so it survives a
world-size change, replacing eval's four near-duplicate marker families. This is
what Phase 2 builds on; `utils/checkpoint.py` level 2 is a different concern and
stays as-is (see [Deferred](#deferred-level-2-rollout-checkpointing)).
- **Ensemble group planning** *(add)* — `EnsembleGroup`, `plan_ensemble_groups`;
collective groups for online CRPS.
- **`OutputManager`** *(add)* — also the natural home for enforcing
`AsyncZarrBackend`'s shard-ownership rule, where violation is currently
*silent* data loss.
- **`IOBackend`: `close`/`flush`/`exists`** *(extend protocol)* — today it is
two methods, so a workflow cannot ask what has already been written.
- **`run_on_rank0_first`, `configure_logging`** *(add)* — 3 and 2 copies
respectively, zero divergence.
- **`Pipeline` ABC** *(add, last)* — `earth2studio.run` functions become thin
wrappers so there are not two competing execution APIs.
Generality and future-proofing of these abstractions is analyzed in
[What each abstraction assumes](#what-each-abstraction-assumes).

**De-Hydra'ing is the main refactor cost.** Only 4 of 19 `recipes/eval/src/`
modules avoid `omegaconf` (`distributed`, `grids`, `metrics`, `regrid` — and
notably those are exactly the obviously-general ones). `work.py` alone
references `cfg` 63 times, and 14 of 21 eval test files build `DictConfig`
fixtures. Pattern: library takes plain dataclasses/primitives; the recipe keeps
a thin `cfg → spec` adapter and retains `hydra.utils.instantiate`.

**`Pipeline` itself is the concrete instance of this cost, not just `work.py`.**
Today's `Pipeline.setup(cfg: DictConfig, device)` (`pipelines/base.py:345`) and
`Pipeline.run(work_items, data_source, output_mgr, output_variables, device,
cfg, scorer, member_batch)` (`:784-794`) take a Hydra `DictConfig` and a long,
loosely-typed argument list — `cfg` is read for `resume`, and subclasses read
whatever else they need out of it during `setup`. The upstreamed `Pipeline`
takes typed, explicit arguments (model, IC source, forcing declaration/source,
output manager, resume flag, …) with no `DictConfig` anywhere in the core
signature. This is a real, user-visible break in how a `Pipeline` is
constructed and driven — every eval pipeline subclass's `setup` body has to be
rewritten to pull its config out of typed constructor arguments instead of
`cfg.pipeline.*` lookups — and should be scoped and estimated as part of
Phase 3 rather than discovered while writing it.

#### Work-item grain

Unify across recipes, but change the **grain** — not just add a field. Adding
`package_id` is trivial and was the wrong thing to worry about. The real finding
is that **eval is the outlier**: HENS's true inference unit is `(package, IC,
batch_id) -> members[...]` (`hens_utilities.py:311`, `hens_ensemble.py:348-365`)
and tc_tracking's is `(ic, mems, seed)`
(`generate_tc_hunt_ensembles.py:294-336`) — *both batch-grained* — while eval
emits one item per member (`work.py:292-300`).

- **`member_ids: tuple[int, ...]` replaces scalar `ensemble_id`.** The one
  irreversible decision; everything else is additive. Eval's current behavior is
  the degenerate `len == 1` case, and eval already has the concept as
  `members_per_rank` (K), just as a *rank* property rather than an *item*
  property.
- **`package_id: str | None`** plus an explicit package-major ordering contract
  and a `Pipeline.ensure_package(package_id)` hook. Today HENS's O(1) weight
  loading is a *load-bearing accident* of `for pkg: for ic:` plus contiguous
  slicing — there is no sort key saying so. Any round-robin distribution silently
  turns it into O(n_items) full checkpoint reloads.
- **`seed` supplied, not derived in the constructor.** HENS derives from a
  *string* domain (`"{base}_{alnum(pkg)}_{ic}"` → sha256, per *batch*), eval from
  three int64s via FNV-1a (per *member*). Not reconcilable in one function, so
  derivation becomes a pluggable strategy and `WorkItem` carries the result.
- **`attempt: int = 0`, folded into seed derivation.** tc_tracking's retry does
  `seed + 1` (`generate_tc_hunt_ensembles.py:245-256`), which escapes eval's
  derivable space and breaks its reproducibility invariant.
- **`batch_id: int | None`** as the replay handle, recorded into output metadata.
  This is what makes HENS's batch-ID replay work: batch IDs are global across
  packages and the seed depends only on `(pkg, ic, batch_id)` — never on rank or
  world size — so an arbitrary subset can be regenerated bit-for-bit on a
  different GPU count.

*Good news:* ensemble-group collectives do **not** require identical weights
across a group — same architecture and same `nsteps` satisfies step-sync, which
is exactly what a HENS ensemble is. The real constraints are that `G * K ==
ensemble_size` (`work.py:196`) must become *per package*, weight swaps must be
group-collective rather than per-item, and the completion-marker key
(`work.py:412-414`) must include `package_id` or resume silently skips real work.

#### What each abstraction assumes

Each abstraction is generalized from eval's patterns. Below: the hidden
assumption, what breaks it, and whether to hedge now or later. The rule applied
is *hedge now only where the fix is expensive later* — i.e. where it changes a
data shape or a contract rather than an implementation.

**`WorkItem`** — assumes items are independent and order-free. Broken by cycling
SDA (above) and by tc_tracking's dynamic re-queue. *Hedge now*: the `member_ids`
grain (already decided) plus an order-constrained/rank-pinned scheduling flag.
Both change the shape, so they are expensive later.

**`distribute_work`** — assumes work is *homogeneous* (contiguous ceil-slicing
balances only by count). Breaks for mixed-cost campaigns: multi-checkpoint runs
where some packages are larger, or mixed `nsteps` per IC. *Hedge later*: an
optional cost weight is additive, and static partitioning is correct for the
common case.

**Progress store** — assumes marker count stays modest.
`filter_completed_items` does `d.iterdir()` over the whole directory
(`work.py:451`) and **every rank calls it at startup**. At HENS scale (100 ICs ×
1000 members) that is 100k files in one directory, listed simultaneously by every
rank — a known metadata-contention pathology on Lustre/GPFS. *Hedge now, cheaply*:
shard the marker directory (e.g. by IC) or use one compact per-rank manifest
instead of one file per item. This is a storage-layout decision, so it is
awkward to change once runs exist on disk.

**`OutputManager`** — assumes **one store per run with `total_coords` known up
front**. Breaks for: unbounded/streaming IC lists (operational cycling), runs
whose extent is discovered rather than declared, and **HENS's existing pattern of
one store per (IC, package)** (`hens_run.py:70`). *Hedge now*: do not bake
"exactly one store" into the type; let a run own a *collection* keyed by
something. Retrofitting a store axis later means rewriting the schema-validation
path.

**`IOBackend.exists()`** — the granularity question is unresolved and matters.
Chunk-level is what resume wants; `AsyncZarrBackend` can only answer at *shard*
level (`_shard_exists`, `async_zarr.py:1003`), and conservatively assumes
existence on probe failure. *Decide now*: specify `exists()` as a coarse,
advisory predicate and keep authoritative completion in the progress store —
otherwise callers will assume precision the backend cannot deliver.

**Ensemble groups** — assume `G * K == ensemble_size` exactly, rejecting ragged
groups (`work.py:196-202`). Fine today; already flagged as needing to become
per-package for HENS.

**`Pipeline`** — assumes one model per process for the whole run
(`setup` is "called exactly once", `pipelines/base.py:345-347`). HENS needs
per-item weight swaps. *Hedge now*: the `ensure_package` hook, since it changes
the call contract.

**Cross-cutting: nothing here contemplates a second parallelism axis.** Every
recipe parallelizes over (IC × member). Region tiling, variable sharding, and
model parallelism for large models would all add an axis. Not worth building,
but worth *not precluding* — which argues for `distribute_work` operating on an
opaque item list rather than anything IC-shaped.

### 4. Cross-cutting cleanups

- **Cosine zenith angle**: hand-reimplemented from scratch (declination,
equation of time) in 4 AIFS variants, while 5 other models import
`physicsnemo.utils.zenith_angle`. Duplication of *science*, not plumbing —
weight accordingly.
- **HEALPix regridders**: `earth2grid` wrappers duplicated across `dlesym`,
`dlesym_v0_isccp_era5`, `cbottle_video`, `cbottle_infill`.
- **Preserve on migration**: HENS's multi-checkpoint fan-out (ranks load
*different* weights; parallelism unit is checkpoint×IC×batch) and its
batch-ID replay (reproduce an arbitrary subset of a huge ensemble from a base
seed string). Neither has an eval analogue. If the upstreamed `WorkItem`
cannot express both, HENS will fork again.

## Implementation plan

Sizing is relative (S/M/L), not calendar-committed. Ordering and dependencies
matter more than the estimates.

### Phase 0 — Contract spec (S, do first)

Write down the iterator/coord contract and add the conformance test. Cheap,
non-breaking, and **codifies existing behavior rather than designing new
semantics** — the rebasing rule already exists in `ucast`/`atlas`. Permanently
closes the `Aurora`-class bug family.

Prerequisite for everything else: you cannot define a transform boundary until
both sides' contracts are pinned.

### Phase 1 — Data layer + hooks (M) — starts now

Upstream the combinators and `Regridder`. Add device and grid to the source
contract, the grid registry and equality predicate, and the DLPack path. Promote
`front_hook`/`rear_hook` to a declared, composable protocol member.

Independent of distribution, testable single-process, and immediately widens
what `run.deterministic` can drive. Ship the DLPack fix here — it unblocks GPU
regridding **without** waiting on `cupy-xarray` maturity.

### Phase 2 — Execution substrate (L) — after eval's output-side work

`WorkItem`, `distribute_work`, the progress store, ensemble groups,
`OutputManager`, and the `IOBackend` protocol extension.

Deferred behind the eval recipe's near-term extensions, which are on the
*output* side rather than the execution core: region-broken-down scoring,
temporal aggregation (e.g. monthly means), and broader S2S/S2D metrics. Those
land first, which bounds the online-scoring timing risk.

Land the forcing declaration here, not earlier: prefetch, caching, and
predownload derivation are engine concerns, and the declaration that tells
predownload what to cache is the *same* declaration that tells the engine what
to feed per step.

*Scope control:* a minimal `WorkItem` can ship first, deferring `package_id` and
`batch_id` — but make the **grain** decision (`member_ids` vs `ensemble_id`) up
front, because it is the expensive one to change later.

**Suggested first PR**: Phase 1 plus a single Phase 2 slice — `WorkItem` +
`distribute_work` + progress store, de-Hydra'd, with tests.

**Then port HENS onto it** *(P1, after the refactor lands)*. `hens` and `s2s`
are both lower priority than `eval`, but each gets a planned port once the
substrate is in — enough to keep them from rotting outright. HENS goes first
and is the honest test: it has no resume today, so the port is a visible
capability win rather than a lateral refactor, and it forces the
multi-checkpoint question early rather than after the API hardens. `s2s`
follows, folded into the HENS port where possible — it is a near-verbatim fork
of HENS, and the port is the natural moment to de-fork it.

### Phase 3 — `Pipeline` + `run.py` convergence (M, last)

Upstream `Pipeline`; make `earth2studio.run` functions thin wrappers over it.

Deliberately last: `run_item_batched`, `supports_online_scoring`, and the group
synchronization invariant are all recent and still moving, and
`supports_member_batching` has a known soundness hole (a subclass overriding
`run_item` while inheriting `run_item_batched` diverges silently).

**Phases 0–2 shrink this phase rather than growing it.** If input preparation
and forcing declaration land, DLESyM and StormScope may not need bespoke
`Pipeline` subclasses at all — which both de-risks the upstreaming and
retroactively validates the plan.

**This phase also carries the `Pipeline` construction API change**, not just
the upstreaming: today's eval `Pipeline.setup(cfg: DictConfig, device)` /
`run(work_items, data_source, output_mgr, output_variables, device, cfg,
scorer, member_batch)` (`pipelines/base.py:345,784-794`) take a Hydra config
blob and a long, loosely-typed argument list. The upstreamed `Pipeline` takes
typed, explicit constructor arguments instead (model, forcing source(s),
output manager, resume flag, …), with no `DictConfig` in the core signature —
every existing eval pipeline subclass's `setup` has to be rewritten against
this. Budget for it as the bulk of this phase's work regardless of how many
subclasses survive the Phases 0–2 consolidation above.

**Known gap this phase does not close: multi-model coupled rollouts** (e.g.
StormScope's GOES-forces-MRMS execution) — see the [orchestration gap
note](#the-forcing-seam) under the forcing seam. Not addressed by `WorkItem`
or `Pipeline` as scoped in Phases 0–2; near-term mitigation is a composite
model wrapper, real fix belongs to the parallel model/earth-system coupler
effort. If that effort is going to change how `Pipeline` accepts and steps
models, landing enough of its design *before* this phase — rather than after —
avoids upstreaming `Pipeline` here and breaking its shape again shortly after
for coupling. A sequencing call for the two efforts' owners to make together,
not decided here.

### Testing and CI

Phase 0's conformance test covers the model contract; the execution substrate
needs its own story, and it implies CI-side changes. All of the below runs on
CPU — no GPU CI dependency.

- **Multi-process tests via the `gloo` backend** for `distribute_work`,
  ensemble groups, and step-sync collectives — `torch.multiprocessing.spawn`
  (or `torchrun`-launched pytest) at world sizes 1/2/4 with toy models. This is
  the main CI addition: a new pytest marker or tox environment on the existing
  CPU matrix, kept fast (target: a couple of minutes).
- **Progress-store tests**: concurrent marker writes from N processes; resume
  after a simulated kill; **resume across a world-size change** (the property
  that distinguishes it from `utils/checkpoint.py`); and marker-directory
  layout behavior at synthetic scale (~10k items on tmpfs) to pin the
  sharding/manifest decision.
- **`world_size=1` degradation as an explicit test**, not an accident — this is
  also the serve-layer requirement (constraint 1 in the serving section).
- **Seed/replay tests**: the same `(package, IC, batch_id)` reproduces
  bit-identical members at different world sizes; a bumped `attempt` provably
  changes the stream.
- **The known-bug regression test** (resumed ensemble, `batch_size <
  nensemble`, non-identity model, asserting *values*, not labels) lands with
  the bug fix, ahead of everything above.

### On breaking changes

Smaller blast radius than it first appears:

- **Do not break** intrinsic forcing (zenith, statics, normalization).
- **Preserve the "forecast in four lines of code" path** via `default_source()`.
The ethos is *"the user doesn't have to write the assembly"*, not *"no assembly
exists"* — and an explicit, inspectable, swappable assembly is strictly better
than a hidden one because it is discoverable and diffable.
- **Accept breakage** where hiding is silently *lossy*. StormScope deliberately
uses nearest-neighbor for radar/satellite but bilinear for the sparse GLM
lightning field because the physics differs per channel. Buried, that choice
is unauditable. For a package whose credibility rests on verification, making
it explicit is the same transparency argument as the scorecard work.

#### Migration mechanics

Deliberately slim: everything here lands as part of the **v1.0** bump, so
major changes are in-bounds. The rule is one release of warnings where a shim
is cheap, and a clean break where a shim would cost more than it saves.

- **Forcing seam (5 models)** — one deprecation release. The
  `conditioning_data_source=`/`forcing_data_source=` kwargs keep working with a
  `DeprecationWarning` (internally wrapping the given source into the new
  engine-side path), then are removed at v1.0. Cheap because the new tensor
  entry point coexists with the old constructor plumbing.
- **HRRR dim rename** — rip the bandaid at v1.0, no dual-emission shim. Ship a
  changelog entry with a one-line `xr.Dataset.rename` snippet for existing
  saved datasets, and delete the StormCast hack in the same PR so the breakage
  and its payoff land together.
- **`fetch_data(interp_to=)`** — already rejected on the non-legacy path; emit
  a `DeprecationWarning` on the legacy path immediately, delete at v1.0.
- **Hooks** — additive, no break: plain attribute assignment keeps working
  (it becomes a chain of one).
- **`run.*` functions** — remain as thin wrappers post-Phase 3 with unchanged
  signatures; no user action.
- **Level-2 checkpoint gate** — non-breaking for the two models that support
  it; for the other 30 the prior behavior was silent truncation, so the
  "break" is an error message where a wrong answer used to be.

## Serving layer (`earth2studio/serve`)

Lower priority, and partly bound for physicsnemo-serve. Assessment of how the
REST `Workflow` abstraction relates to `Pipeline`.

**Verdict: the "thin REST wrapper" read is right, with one refinement — the seam
is `Earth2Workflow.__call__`, not `Workflow` itself.** `Workflow` and `Pipeline`
decompose along *orthogonal* axes, which is why they compose rather than compete:
`Pipeline` factors the inference loop into base machinery plus narrow subclass
hooks; `Workflow` does the opposite, leaving compute as one opaque `run()` and
factoring out the *serving* machinery. The design docs say this outright —
`README_workflows.md:20` ("Transform Python scripts into REST APIs") and
`README_earth2workflows.md:5`.

**The abstract surface is tiny**: two methods, `validate_parameters` and
`run(parameters, execution_id) -> dict` (`serve/server/workflow.py:321,358`).
`Earth2Workflow` (`e2workflow.py:114`) narrows it further to a single
`__call__(io: IOBackend)`, auto-deriving the pydantic parameter model from the
signature via an `AutoParameters` metaclass (`e2workflow.py:87`).

**Two tiers exist today, and only one duplicates `Pipeline`:**

- Direct `Workflow` subclasses (`deterministic_workflow`, `deterministic_fcn`,
  `ensemble_workflow`, `diagnostic_workflow`) — thin adapters delegating to
  `earth2studio.run.*`. Nothing to reclaim.
- `Earth2Workflow` subclasses — mostly also delegate, but the two `foundry_*`
  ones **hand-roll the rollout** (`foundry_fcn3.py:238-272`): `create_iterator`,
  manual `map_coords`, ad-hoc ensemble `unsqueeze`, `io.write(*split_coords(...))`.
  No regrid, no resume, no seeding contract. **This is the concrete win** — they
  are reimplementing `Pipeline.run` badly, for the same reason the recipes did:
  multi-model assembly `run.deterministic` cannot express.

**What would NOT transfer**, and is genuinely serve's own (~4000 lines with no
`Pipeline` analogue): execution IDs, RQ enqueue and admission control, the status
state machine, Redis TTL state, the four-stage post-processing chain
(zip → object storage → GeoCatalog → finalize), streaming download, and the
exposure allowlist. Plus the one real impedance mismatch: `AutoParameters`
derives an OpenAPI schema from a `__call__` signature, and `Pipeline` has no
per-run signature to introspect — its input is a Hydra `DictConfig`. Serve would
keep hand-written parameter models per pipeline.

**Progress already has an elegant answer, and it transfers.** `BackendProgress`
(`e2workflow.py:263`, installed at `:193`) decorates the `IOBackend` and infers
`current_step`/`total_steps` from the *write stream* — which is why delegating
workflows need no progress awareness inside `run.deterministic`. Because
`Pipeline` funnels every write through `output_mgr.write` (`base.py:916`), the
same decorator works unchanged by wrapping `OutputManager`'s backend. The
residual gap is narrow: `BackendProgress` tracks a single `progress_dim`
(default `lead_time`), so `foundry_fcn3`'s sample×step message
(`foundry_fcn3.py:243-252`) cannot be expressed by it. That one case wants an
optional `progress_cb` on `Pipeline.run` — the same shape as the existing
`scorer` parameter (`base.py:788`), which already proves the ABC tolerates an
injected observer.

**Two constraints this places on the plan:**

1. **`Pipeline` must be usable with no distributed init.** Serve is RQ/Redis with
   one blocking worker per request — no `torchrun`, no `init_process_group`,
   nothing distributed anywhere in `earth2studio/serve/` (`worker.num_workers: 1`,
   with a config comment admitting the scheduler does not exist yet). Running at
   `world_size=1` is workable since `get_rank()` already degrades gracefully, but
   this should be an explicit requirement rather than an accident.
2. **Promoting `Pipeline` into core is a prerequisite, not a consequence.**
   `pipelines/base.py:55-59` imports `src.data`, `src.distributed`, `src.output`,
   `src.regrid`, `src.work` — all recipe-local and unimportable from
   `earth2studio/serve/`.

**Recommendation: do not make `Workflow` a subclass or wrapper of `Pipeline`.**
After Phase 3 promotes `Pipeline`, add a `PipelineWorkflow(Earth2Workflow)`
adapter whose `__call__` builds config from validated params, calls
`pipeline.setup()` once in `__init__` (matching the documented per-worker model
caching), and runs the pipeline against the request's `IOBackend`. Each
abstraction stays at the layer it was designed for, and the two foundry workflows
delete their hand-rolled rollouts.

## Deferred: level-2 rollout checkpointing

`utils/checkpoint.py` level 2 (mid-rollout state resume) **stays** — it was
recently added and has users. It is simply not what Phase 2 builds on, and
improving it is a later, separately-scoped effort. The findings below scope that
effort; they are not an argument for removal.

*It is unsound for almost every model.* `run.py:119-134` gates resume on the
requested checkpoint *level* only — it never checks whether the model actually
implements `_restore_checkpoint_state`. Only **2 of 32** prognostic models bind
checkpoint state at all (`fcn.py:117`, `persistence.py:94`; plus
`perturbation/gaussian.py:66`). For the other 30, a level-2 resume re-writes the
t=0 field and then terminates `restart_step` steps early — a silently truncated
forecast, no error, no warning.

*It is structurally unachievable for the models people run long.* Rollout state
lives in **generator frame locals**, not on the model object: U-CAST's two-step
history roll (`ucast.py:1027`), StormScope's `next_input` sliding window
(`stormscope.py:1441`), DLESyM's multi-frame coupled history plus recurrent
hidden warmup (`dlesym.py:893`), cbottle's atomic 12-frame chunks
(`cbottle_video.py:540-573`). Only the last frame is *yielded*, so the true
state cannot be reconstructed from the checkpoint or from IO. FCN3 is worse than
unachievable — `_reset_internal_state` (`fcn3.py:338-357`) draws **fresh noise**
on every generator entry, so a restart provably diverges. And the codec captures
no global/CUDA RNG state at all, which several stochastic samplers depend on.

*No in-repo workload needs it, which is why Phase 2 does not build on it.* The
longest rollout anywhere in the repo is `nsteps: 176`
(`recipes/s2s/cfg/pnw_sfno.yaml:11`); the median is 2–56. The expensive axis here
is **ensembles × initial times**, not rollout depth — which is exactly what
work-item resume covers. No recipe references it, though external users do. The
forward-looking case is real: `ace2.py:471-505` caches forcing a full calendar
year at a time, clearly built for multi-year integration.

*First fix when this is picked up*: invert the default with an explicit
`PrognosticModel.supports_rollout_checkpoint` that `run.py` **enforces** before
setting `restart_step`, falling back to level-1 behavior otherwise. Today the
runtime assumes support and the docs push verification onto the user
(`docs/userguide/advanced/checkpointing.md:16-21`). This is non-breaking for the
two models that do support it and closes the silent truncation for the other 30.

## Known bug: resumed ensemble corruption

Separable from this plan — fix independently.

On a *resumed* `run.ensemble` at level 2 with `batch_size < nensemble`, every
batch after the first is silently corrupted. `_state_loaded` is set once at
session bind (`checkpoint.py:822`, `loaded_state is not None`) and never reset
per batch, while `_save_checkpoint_state` overwrites `checkpoint.x` every step.
So for `batch_index >= 1` the workflow sets `restart_step = None` (`run.py:480`)
but `FCN._default_generator` still restores unconditionally (`fcn.py:295`),
handing batch 1 **batch 0's final state** as its IC and skipping its own
initial-condition yield (`fcn.py:300-302`). `batch.py:_decompress_batch` then
relabels that tensor with the current member's `ensemble`/`time` coords, so the
data is member 0's and the label says member 1.

The test suite cannot catch it: both ensemble checkpoint tests
(`test/utils/test_checkpoint.py:590,834`) use `batch_size=1` with
`Persistence`, an identity operator, and assert labels, write counts and lead
times — but never tensor-value continuity across a resume, which is the thing
that breaks. Verified by reading the call sites.

**Fix**: make restore *per-rollout* rather than per-session — reset
`_state_loaded`, or gate `_restore_checkpoint_state` on the workflow's
`restart_step`, at each `create_iterator` call. Add an ensemble test with
`batch_size < nensemble` on a non-identity model asserting values and lead times,
not just labels.
