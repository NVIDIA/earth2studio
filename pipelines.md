# Earth2Studio Workflow & Execution Architecture — Proposed Changes

## 1. Goals

- **Close the mid-complexity gap for users.** SA feedback: most users sit between the notebook  
examples and the full recipes in complexity, and the existing `run.deterministic`-style interfaces  
don't reach that middle ground for many more advanced models. A shared, tested execution  
substrate is what lets core offer more than "toy example" or "bespoke recipe."
- **Sharpen model contracts to be more explicit.** Lead-time rebasing, the forcing/conditioning  
data injection, and pre/post hooks are currently either hand-copied per wrapper or half-wired  
(present in some models, silently ignored in others).
- **Decouple models from data sourcing.** Several model wrappers (`stormcast`, `stormcastconus`,  
 `stormscope`, `ace2`, `datareplay`) own a live `DataSource` internally. This blocks the calling  
`Pipeline` from prefetching or caching on the model's behalf, breaks fully offline runs, and makes  
it impossible to declare in advance what data a predownload step needs to cache. It can also force  
data preprocessing code into model wrappers when it would be a better fit elsewhere (data-side  
preprocessing steps), blurring the scope/purpose of model wrappers. Declaring the need is separate  
from deciding how to satisfy it — a `Pipeline` can still fetch eagerly, per-step, exactly as models  
do today; the declaration just makes prefetch/caching *possible* where it wasn't before, it doesn't  
mandate it.
- **Minimize the need for bespoke `Pipeline` subclasses.** Today, models with nuanced execution
patterns (DLESyM, StormScope) each required a hand-written `Pipeline` subclass in `recipes/eval/`
even though `Pipeline` itself is fairly general — the nuance leaked out of the model/data contracts
and into one-off execution code. If the refined contracts above (forcing declaration, hooks,
grid/device awareness) are sufficient, most models should run under one shared `Pipeline` implementation  
with zero custom subclassing.
- **Give core a large-scale story.** Today nothing in `earth2studio/` uses
`torch.distributed` — no sharding, no collectives, no batching over initial conditions. Every
team that needs distributed, restartable inference (HENS, s2s, eval, tc_tracking) has built
its own engine in `recipes/`, independently.
- **Stop paying for the same engine N times.** Work-splitting, output-coordinate assembly,  
and ensemble batching each exist in 3-4 near-identical copies across recipes. `s2s` is a  
near-verbatim fork of `hens`, seed-derivation typo included. Duplication across our recipes  
implies duplication/hand-rolling in user codebases as well

## 2. Proposed API changes

All changes target the package's **v1.0** bump, so breaking changes can be acceptable provided we  
establish a migration plan.

**Data layer**

- Upstream the source combinators (`CompositeSource`, `RegriddedSource`, etc.) and the
`Regridder` ABC from `recipes/eval/` into core — these are already general-purpose, just stranded in a recipe.  
Will help absorb some data-preprocessing needs into the data source side rather than model wrappers.
- Extend the `DataSource` protocol with **device** and **grid** awareness, so sources can hand
data over on the target device/grid instead of forcing a host round-trip. Ships behind DLPack
as an interim device-transfer mechanism (Phase 1), with a possible later swap to a `cupy`-based
mechanism — the coupling is confined to one function, so that swap stays cheap either way.
- Add a grid **registry + advertisement** convention (regular / curvilinear / mesh) so grids can
be known without probing a live fetch — a prerequisite for offline/predownload planning.
- Deprecate `fetch_data(interp_to=)` in favor of an explicit `Regridder`; remove the
StormCast-specific `hrrr_x`/`hrrr_y` special case by normalizing HRRR's dimension names like
every other projected source.

**Model contract**

- **Breaking for 5 models**: models stop owning a `DataSource` and instead *declare* their exogenous data needs (variables, grid, cadence, lead offsets) via a `forcing_spec` method, and are *handed* a tensor by the calling `Pipeline` at each step instead of fetching it themselves. How that data actually arrives — fetched live per step, prefetched in bulk, or predownloaded ahead of the run — is entirely the `Pipeline`'s call; the model's contract is just "here's what I need," not "here's when to get it." One deprecation release keeps `conditioning_data_source=`/`forcing_data_source=` working via an internal shim before removal at v1.0.
- Add `default_source()` (name TBD), mirroring the existing `load_default_package()` pattern — returns a raw, lazily-constructed source (not a mutable default argument, not pre-composed with a regridder), preserving the "forecast in four lines" path.
- General sharpening:
  - Write down the iterator/coordinate contract (step-0-yield semantics, lead-time rebasing, batch/sample axis naming)  
  as a spec plus a conformance test every prognostic/diagnostic wrapper must pass. This is non-breaking — it codifies behavior that already exists in the correct wrappers. Also come up with a standardized way for models to expose/specify/report stochasticity (`set_rng` or other).
  - Generalize the `preload_invariants`/`preload_static_fields` duplication into one static-field
  toggle mechanism.

**Execution substrate**

- Add `WorkItem` + `distribute_work` (de-Hydra'd) as the shared unit of parallel work, replacing four divergent implementations across recipes. Idle ranks get an empty list instead of `exit()`.
- Add a work-identity-keyed progress/resume store that survives a world-size change
- Add `EnsembleGroup`/`plan_ensemble_groups` for collective ensemble scheduling, an
`OutputManager` that enforces shard-ownership (currently silent data loss on violation), and
extend the `IOBackend` protocol with `close`/`flush`/`exists`.
- Add a `Pipeline` ABC last, once the above has stabilized, so `earth2studio.run.`* functions
become thin wrappers over one execution API instead of a second, competing one.
- **Breaking-ish (changing recipe code)**: today's eval `Pipeline.setup(cfg: DictConfig, device)` and `Pipeline.run(work_items, data_source, output_mgr, output_variables, device, cfg, scorer, member_batch)` (`recipes/eval/src/pipelines/base.py`) take a Hydra config blob and a long,
loosely-typed argument list — subclasses read whatever they need out of `cfg` inside `setup`. The
upstreamed `Pipeline` takes typed, explicit constructor arguments instead (model, IC/forcing
sources, output manager, resume flag, …), with no `DictConfig` in the core signature. Every eval
pipeline subclass's `setup` has to be rewritten against this, so it's real migration work, not a
drop-in — see [Sequencing](#3-sequencing).

**Serving layer** — lower priority, mostly independent. The REST `Workflow` abstraction and
`Pipeline` are orthogonal (one factors serving machinery out, the other factors inference-loop
machinery out) and should compose via a thin `PipelineWorkflow` adapter rather than one wrapping
the other. The two `foundry`_* workflows that currently hand-roll their own rollout loop are the
concrete beneficiaries — they'd delete that code once `Pipeline` exists.

## 3. Sequencing

Ordering is driven by dependency, not calendar estimate.

1. **Phase 0 — Contract spec (do first, small).** Write down the iterator/coordinate contract
  and its conformance test. Cheap, non-breaking, and a prerequisite for everything else: you
   can't define the data↔model boundary in Phase 1 until both sides of the contract are pinned.
2. **Phase 1 — Data layer + hooks (starts now, independent of distribution).** Source
  combinators, `Regridder`, device/grid contract extensions, DLPack path, hook promotion. Fully
   testable single-process; immediately widens what today's `run.deterministic` can drive without
   needing any of the distributed machinery.
3. **Phase 2 — Execution substrate (large, after eval's near-term output-side work).**
  `WorkItem`, `distribute_work`, progress store, ensemble groups, `OutputManager`. Deferred
   behind eval's in-flight scoring/aggregation work so that work lands first and doesn't get
   entangled with the substrate rewrite. The forcing declaration (`forcing_spec`) lands here, not
   in Phase 1, because deciding *when* and *how* to fetch — live, batched, or predownloaded — is a
   `Pipeline`/scheduling concern, not a data-layer one.
4. **Phase 3 — `Pipeline` + `run.py` convergence (last).** Deliberately last because the pieces
  it would formalize (`run_item_batched`, member-batching support) are still moving. Phases 0-2  
   shrink this phase: if forcing declaration and input-prep land first, DLESyM and StormScope may  
   not need bespoke `Pipeline` subclasses at all — the real measure of whether Phases 0-2 succeeded  
   is how few (ideally zero) new subclasses Phase 3 has to introduce to cover them. This phase also
   carries the `Pipeline.setup`/`run` signature change called out above — every existing eval
   pipeline subclass gets migrated off `cfg: DictConfig` here, which is the bulk of this phase's
   work regardless of how many subclasses survive.
  1. Note this plan currently does not include automatic `Pipeline` support for coupled model rollouts like
    `StormScopeGOES`+`StormScopeMRMS`. Partly intentional -- if parallel work on a general model coupler 
    interface proceeds, a possible merge/integration point with that effort could happen between Phases 2-3
    which could enable the `Pipeline`s to eventually cover coupled rollout orchestration too.

## What success could look like

StormCast declares its forcing need (GFS conditioning) instead of owning a `DataSource`, and no
custom `Pipeline` subclass is needed. Data can be fetched live (the default, shown below) or predownloaded ahead of time from the same declaration — the model's contract doesn't change based on that choice. (Field/method names are illustrative — several are marked "name TBD" in the design doc). Multi-model coupling, e.g. StormScope's GOES-forces-MRMS rollout, is a known gap not addressed by this example.

```python
from earth2studio.models.px import StormCast
from earth2studio.run import WorkItem, distribute_work, Pipeline, OutputManager
from earth2studio.io import AsyncZarrBackend

# Model no longer owns its data source — it declares what it needs, and hands the
# calling Pipeline a raw source to fetch from.
package = StormCast.load_default_package()
model = StormCast.load_model(package)

# forcing_spec enumerates every request up front (variables, grid, cadence) —
# consumed by predownload tooling, or just informative for a live-fetch run.
for req in model.forcing_spec(init_time="2024-01-01T00", nsteps=8):
    print(req.variables, req.grid, req.cadence)

forcing = model.default_source() # name TBD, likely need to expand to multiple source support
# Predownloaded/offline alternative, same declaration, no code change below:
# forcing = PredownloadedSource.from_spec(model.forcing_spec, "./cache")

items = [WorkItem(time=t, member_ids=members) for t in init_times for members in member_batches]

# distribute_work is only needed for a distributed (world_size > 1) run — a single-
# process run just passes `items` straight to Pipeline.run.
work = distribute_work(items, world_size=world_size, rank=rank) if world_size > 1 else items

pipeline = Pipeline(
    model=model,
    forcing=forcing,  # Pipeline fetches per-step by default, batching/dedup'ing via forcing_spec
    output=OutputManager(AsyncZarrBackend("s3://bucket/run.zarr")),
    resume=True,  # keyed on work-item identity, survives a world-size change
)
pipeline.run(work)  # same call whether world_size is 1 (serving) or 512 (HENS-scale)
```

Compared to today: no `conditioning_data_source=` kwarg threaded through the model constructor,
no per-recipe reimplementation of work-splitting or resume, and the same `Pipeline.run` call scales from a single-process serve request to a HENS-scale distributed ensemble without a rewrite at either end. With comments and explanatory statements trimmed, we recover the ~8-liner run script that is attractive for push-button users.