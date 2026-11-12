# Model Evaluation (Eval) Recipe

> 🏗️ **Under construction** — This recipe is still being developed; behavior,
> defaults, and documentation may change without notice.

General-purpose recipe for model verification and validation (V&V) using
Earth2Studio. Run any prognostic model (with optional diagnostic models)
across a set of initial conditions, distributing work across GPUs, and save
forecast outputs to zarr.

Key features:

- Works with **any** Earth2Studio prognostic or diagnostic model
- **Data assimilation support** — evaluate `AssimilationModel`-class models
  (e.g. HealDA) directly against reanalysis, or use them to initialize a
  forecast (e.g. HealDA + FCN3)
- Custom inference loops through the `earth2studio.run.Runner` protocol
- Multi-GPU distributed inference via `torchrun` / SLURM / MPI
- Clean work-distribution with automatic load balancing across ranks
- Parallel, non-blocking zarr I/O via thread pool
- Ensemble support with configurable perturbation
- Hydra-based configuration with composable model configs

## Contents

- [Quick Start](#quick-start)
- [Pre-downloading Data](#pre-downloading-data)
  - [Zarr stores](#zarr-stores)
  - [Disable the per-source cache](#disable-the-per-source-cache)
  - [Resume after interruption](#resume-after-interruption)
- [Using Your Own Data](#using-your-own-data)
- [Data Assimilation Models](#data-assimilation-models)
- [Multi-GPU Execution](#multi-gpu-execution)
- [Resuming and Multi-Job Runs](#resuming-and-multi-job-runs)
  - [Resuming after a failure](#resuming-after-a-failure)
  - [Resuming across jobs](#resuming-across-jobs)
- [Configuration](#configuration)
  - [Campaign configs](#campaign-configs)
  - [Model selection](#model-selection)
  - [Ensemble runs](#ensemble-runs)
  - [Scoring](#scoring)
  - [Regional scoring](#regional-scoring)
  - [Event scoring](#event-scoring)
  - [Online scoring](#online-scoring)
  - [Report](#report)
- [Architecture](#architecture)
  - [Runner interface](#runner-interface)
  - [Custom runners](#custom-runners)
- [Extending the recipe](./docs/extending.md) for BYO data, custom models, custom runners
- [Testing](#testing)

## Quick Start

Install the required packages:

```bash
# With uv (runtime dependencies only)
uv sync

# Include pytest and related tooling to run tests (see Testing below)
uv sync --extra dev
```

**Step 1 — Pre-download data** (required before inference):

```bash
python predownload.py
```

**Step 2 — Run inference:**

```bash
python main.py
```

`main.py` will refuse to start if the pre-download step has not been completed.
To bypass this check (e.g. data is already cached from a prior run), pass
`require_predownload=false`.

## Pre-downloading Data

Run `predownload.py` before `main.py`. It fetches the data the configured
runner reads (and optionally verification data) and writes it into zarr
stores under `<output.path>/`. It accepts the same config overrides as
`main.py`, so both commands read one campaign definition.

The runner describes every fetch it makes through
`runner.data_requests(item)` (and `frame_requests(item)` for observation
DataFrames). The model signatures fix the initial-condition variables,
input lead times, and forecast valid times; the config never lists them.

```bash
# Single process (login node or interactive session)
python predownload.py

# With a campaign config
python predownload.py campaign=fcn3_2024_monthly

# Distributed — parallelise across CPU workers
torchrun --nproc_per_node=8 --standalone predownload.py

# Also pre-fetch ERA5 verification data for the full forecast window
# (variables taken from output.variables — only what will be scored)
python predownload.py predownload.verification.enabled=true
```

> **Note.** Verification always lives in its own `verification.zarr`,
> never in the IC store. A bring-your-own `verification_source` wins;
> otherwise `score.py` and online scoring read `verification.zarr` when
> it exists, and fall back to a live fetch from
> `predownload.verification.source` or `data_source`. Set
> `predownload.verification.enabled=true` in your campaign config, not
> just as a one-shot command-line override to `predownload.py`, so every
> entry point agrees on where verification comes from. The bundled campaigns
> (`dlwp_*`, `fcn3_*`, `dlesym_*`, `healda_*`) set it explicitly.

### Zarr stores

`predownload.py` groups the runner's requests by the source they target and
writes one store per source *role* (`runner.source_roles`) in `<output.path>/`:

<!-- markdownlint-disable MD013 -->
| Store | Dimensions | Contents |
| --- | --- | --- |
| `<role>.zarr` | one array per variable on `(time, <spatial...>)`, or `(time, lead_time, <spatial...>)` for a forecast source | Every valid time a role needs (every initial time and lead time for a forecast source): `data.zarr` for initial conditions, `forcing<k>.zarr` per forcing slot, `goes.zarr` / `mrms.zarr` for StormScope |
| `verification.zarr` | one array per variable on `(time, <spatial...>)` | Verification fields at every forecast valid time, on the `output.regrid` grid (reached with its method) when the campaign sets one; always a separate store (`predownload.verification.enabled=true`) |
| `<slot>.parquet/` | one parquet file per analysis time | DA runners only: observation DataFrames (see [Observation pre-download](#observation-predownload)) |
<!-- markdownlint-enable MD013 -->

At inference time each runner slot resolves through `src.data.resolve_source`:
a bring-your-own node wins, then `src.data.open_predownloaded` reads an
existing `<role>.zarr`, then the live source.
`require_predownload=true` (the default) only makes `main.py` insist on the
`.predownload.done` sentinel, and only for runners that fetch something.

### Disable the per-source cache

Earth2Studio's remote data sources (`ARCO_ERA5`, `GFS_FX`, `GOES`, `MRMS`, …)
default to `cache=True`, which keeps a copy of every byte they fetch in
`~/.cache/earth2studio` (or `$EARTH2STUDIO_CACHE`).  Predownload then
writes its own dedicated zarr under `<output.path>/`, so with the cache
left on **the same data lands on disk twice** — once in the shared
cache, once in `data.zarr`.  For the campaign-scale fetches this recipe
targets, that doubles the disk footprint of every run.

The bundled configs all set `cache: false` on their data sources for
this reason.  When you add a new source — at the top level via
`data_source:` or per-component (e.g. StormScope's `ic_source` /
`conditioning_source` blocks), pass it through as well:

```yaml
data_source:
    _target_: earth2studio.data.ARCO_ERA5
    cache: false
```

Leave the cache on only when you have a separate reason to populate the
shared cache (e.g. another recipe will reuse the same data and you don't
want to refetch it).

### Resume after interruption

A `ProgressStore` under `<output.path>/.progress_predownload_<role>` tracks
progress per time. If a SLURM time limit or a crash ends a pre-download job
early, re-running with the same config skips already-completed times:

```bash
# Just re-run — resume is automatic
python predownload.py campaign=fcn3_2024_monthly
```

To recreate stores from scratch, set `predownload.overwrite=true`. The zarr
stores have a fixed layout, so changing the campaign's times, variables,
sources, output grid or averaging window also requires it; the progress
manifest records all five, and a mismatch fails with that hint.

## Using Your Own Data

If you already have initial-condition or verification data on disk (for
example, a zarr store you ETL'd for training), you can skip predownload
entirely and point the recipe directly at it.  This avoids doubling disk
usage by re-caching data that already exists locally.

The two escape hatches are top-level config keys:

| Key | What it overrides | Consumer |
| --- | --- | --- |
| `ic_source` | The initial-condition source used during inference | `main.py` |
| `verification_source` | The verification source used during scoring | `score.py` |

Either key accepts any `_target_` pointing at a class that implements
Earth2Studio's `DataSource` protocol (`__call__(time, variable) -> xr.DataArray`).
For a new store layout, write a small wrapper — the
[`create-data-source`](../../skills/earth2studio-create-datasource/) skill
scaffolds one from a reference script.

### Configuration matrix

<!-- markdownlint-disable MD013 -->
| ICs | Verification | Config |
| --- | --- | --- |
| package | package | defaults — run `predownload.py`, then `main.py`, then `score.py` |
| **user** | package | set `ic_source`; run `predownload.py predownload.verification.enabled=true` to cache verif |
| package | **user** | set `verification_source`; run `predownload.py` (caches IC only) |
| **user** | **user** | set both; skip `predownload.py` entirely — `main.py` drops the sentinel check |
<!-- markdownlint-enable MD013 -->

### Example

A minimal user `DataSource` for a `(time, variable, lat, lon)` zarr:

```python
# my_pkg/my_data.py
import xarray as xr

class MyZarrSource:
    def __init__(self, store_path: str) -> None:
        self._da = (
            xr.open_zarr(store_path)
            .to_array("variable")
            .transpose("time", "variable", ...)
        )

    def __call__(self, time, variable) -> xr.DataArray:
        if not isinstance(time, (list, tuple)):
            time = [time]
        if not isinstance(variable, (list, tuple)):
            variable = [variable]
        return self._da.sel(time=time, variable=variable)
```

Wire it in via Hydra override or a campaign config:

```bash
python main.py \
    ic_source._target_=my_pkg.my_data.MyZarrSource \
    ic_source.store_path=/data/my_training_set.zarr \
    require_predownload=false
```

> **Note on resume:** output-side completion markers are indexed by IC time,
> not by data source.  If you change `ic_source` or `verification_source`
> between runs that share the same `output.path`, either start clean with
> `resume=false` / `output.overwrite=true` or move to a new `run_id` —
> otherwise stale markers may reference data that no longer matches.

### Coupled multi-source runners

The default StormScope checkpoint is the **`3km_10min`** CONUS nowcasting
variant (3 km grid, 10-minute step). The `stormscope` builder
(`src/runners/stormscope.py`) couples two models with the library's
`earth2studio.coupling.CoupledRunner`:

- **`StormScopeGOES`** — forecasts the eight ABI satellite channels.  The
  `3km_10min` variant is *pure obs*: it takes **no external conditioning**.
- **`StormScopeMRMS`** — forecasts radar reflectivity (`refc`, `refc_base`)
  **plus a gridded GLM lightning channel** (`glm_density`), conditioned on
  GOES (observations at the IC time, then the GOES model's own forecast
  frames during rollout).

Both models run on the shared HRRR projected grid and land in `forecast.zarr`
as the streams `goes` and `mrms`. One `Connector("goes", "mrms")` feeds the
GOES model's frames into the radar model's conditioning slot: the observed
history it holds at initialization, then its forecast at every step. Members
batch on a leading `ensemble` axis. The runner seeds each model once per
work item, from the item's first member, under the model's own offset of
`random_seed` (execution contract rule X5). With `member_batch: 1` every
member reproduces exactly. In ensemble runs, `model.<name>.perturbation`
perturbs that model's initial condition per member. The builder binds one source per
role rather than using the top-level `data_source` / `ic_source`:

<!-- markdownlint-disable MD013 -->
| Role | Config node | Default |
| --- | --- | --- |
| `goes` | `model.goes.ic_source` | the GOES model's `default_sources()` |
| `mrms` | `model.mrms.ic_source` | the MRMS model's `default_sources()` (radar composed with GOES GLM lightning) |
| `mrms_conditioning` | `model.mrms.conditioning_source` | the `goes` source; the GOES model feeds the slot, so a source only serves history frames the GOES model does not hold (a radar history longer than the GOES one) |
| `goes_forcing` | `model.goes.forcing_source` | the GOES model's recommendation; conditioned variants only (`6km_1hr`) |
<!-- markdownlint-enable MD013 -->

The models interpolate onto their own grid. Their default
sources wrap raw GOES and MRMS in nearest-neighbour `earth2studio.grids`
mappings bounded by the checkpoint's `max_dist_km`. GLM lightning
(`earth2studio.data.GOESGLMGrid`, a 0.1° lat/lon grid) goes through bilinear
interpolation, matching the model's own GLM interpolator. A bring-your-own
source must already deliver fields on the model `y`/`x` grid: a pre-downloaded
store, or `earth2studio.data.RegriddedSource` around a raw source:

```yaml
model:
    goes:
        ic_source:
            _target_: my_pkg.MyGoesZarrSource
            path: /data/my_goes.zarr
    mrms:
        ic_source:
            _target_: my_pkg.MyMrmsGlmZarrSource   # refc, refc_base, glm_density
            path: /data/my_mrms.zarr
        conditioning_source: null                  # the GOES model conditions MRMS
```

An optional `model.region` box (`{kind: box, params: [west, south, east,
north]}`) restricts both models to the same sub-domain through
`with_region`, so their grids keep matching.

#### Predownload with HRRR-aligned storage

Running `predownload.py` for a StormScope campaign writes one zarr store per
role, each already on the model grid:

```bash
python predownload.py campaign=stormscope_2023_convection
# → <output.path>/goes.zarr   (time, y, x)  GOES ABI channels
# → <output.path>/mrms.zarr   (time, y, x)  refc, refc_base, glm_density
```

Each store holds every valid time the runner requests for the IC history.
At inference time `resolve_source` prefers these stores for the `goes` and
`mrms` roles, so `main.py` needs no network access to the GOES and MRMS
archives, and the stores hold only the model grid rather than the raw
GOES and GLM footprints. After the first step,
`glm_density` comes from the model's own predictions.

Verification for the GOES, radar and lightning channels comes from the same
observation sources at every forecast valid time. The campaign leaves
`data_source` unset, so the runner composes its live GOES and MRMS sources
and `predownload.py` writes them into `verification.zarr` for `score.py`. A
bring-your-own `verification_source` replaces that.

Because the `3km_10min` GOES model reads observations only, there is no forcing source
to pre-download. The GOES model conditions the radar model and the `goes`
source serves any earlier history frame, so the planner writes no
`mrms_conditioning.zarr` either.

## Data Assimilation Models

Models implementing the `AssimilationModel` protocol
(`earth2studio.models.da`) produce gridded analyses from sparse
observations rather than stepping a gridded state forward.  The recipe
supports two evaluation modes, both driven by the same predownload /
inference / scoring / report entry points:

<!-- markdownlint-disable MD013 -->
| Mode | Runner | Campaign example |
| --- | --- | --- |
| Analysis vs. reanalysis | `runner: assimilation` (`src.runners.assimilation.AssimilationRunner`) | `healda_2024_analysis` |
| DA-initialized forecast | `runner: assimilation_forecast` (the forecast runner reading initial conditions from `AnalysisSource`) | `healda_ucast_2024_monthly` |
<!-- markdownlint-enable MD013 -->

```bash
# Analysis mode — score HealDA analyses directly against ERA5
python predownload.py campaign=healda_2024_analysis   # obs + verification
python main.py campaign=healda_2024_analysis
python score.py campaign=healda_2024_analysis
python report.py campaign=healda_2024_analysis

# Forecast mode — HealDA analyses initialize a prognostic rollout.
# healda_ucast_2024_monthly uses UCast (1.5-degree grid, 12-hour step,
# slim dependencies, HuggingFace checkpoint).
python predownload.py campaign=healda_ucast_2024_monthly
python main.py campaign=healda_ucast_2024_monthly
```

### Observation sources

The default DA runner consumes observation **DataFrames** only; it does not
supply background fields or gridded forcing, so it has no gridded DA IC store.
The general DA interface also supports DataArray inputs. Observations here come from the sources
declared under `model.da.obs_sources`:

```yaml
da:
    architecture: earth2studio.models.da.HealDA
    load_args:
        lat_lon: true
        output_resolution: [721, 1440]   # match the verification grid
    obs_sources:
        conv:
            _target_: earth2studio.data.UFSObsConv
            time_tolerance_hours: [-21, 3]
        sat:
            _target_: earth2studio.data.UFSObsSat
            time_tolerance_hours: [-21, 3]
```

Entries are matched **positionally, in declaration order** against the
model's `input_coords()` slots. Two convenience keys are handled by the
recipe rather than passed to the source constructor: `enabled: false`
disables a slot (the model receives `None` — e.g. a conv-only ablation),
and `time_tolerance_hours: [lo, hi]` sets the observation window as
hours around each analysis time.

### Observation predownload

`predownload.py` caches observations alongside the gridded stores: one
parquet file per analysis time under `<output.path>/<slot>.parquet/`
(e.g. `conv.parquet/`, `sat.parquet/`), each holding the full
`time_tolerance` window for that analysis time. The runner declares them
through `frame_requests(item)`, the DataFrame counterpart of
`data_requests`. The same per-time progress store and multi-rank
distribution as the zarr stores apply, so
an interrupted download continues where it left off and
`torchrun --nproc_per_node=N predownload.py` parallelizes the fetch.

At inference time the runners prefer a pre-downloaded
`<slot>.parquet` store over the live source when it exists, the
observation analogue of the `<role>.zarr` substitution. `main.py` then
needs no network access to the obs archives.  Requesting an analysis
time that was not predownloaded fails loudly rather than silently
falling back to a live fetch.

In forecast mode, observations are cached at every analysis time IC
assembly touches (`IC time + each prognostic input lead offset`).  To
skip obs caching, for example when the source's own `~/.cache/earth2studio`
cache is already warm, set `predownload.observations.enabled=false`.
Inference reads an existing `<slot>.parquet` whenever one is present, so
remove it to fetch live again.

### Missing observations and archive coverage

GSI observation archives have gaps — an individual platform or obs type
can be absent for some assimilation cycles (a decommissioned satellite,
or a GNSS radio-occultation `gps` cycle with no contributing RO
satellite).  The UFS obs sources **tolerate** this: a missing diag file
is logged and skipped rather than aborting the fetch, and the analysis
runs on whatever observation subset is present.  This is intended — DA
models are built to assimilate the obs actually available at a given
time — so scattered `File ... not found` warnings during predownload are
benign.

Two degrees of missingness matter for interpreting results:

- **Partial gaps** (some files present) — normal.  The analysis is built
  from a reduced obs set; predownload logs a per-store summary of how
  many frames were affected.
- **Total gaps** (every file missing for a request) — the obs source
  returns an empty frame and the DA model produces an **all-NaN
  analysis**, which propagates to NaN scores.  This almost always means
  the campaign's `start_times` fall outside the observation archive's
  coverage window.  Because the obs sources might not raise on missing
  files, `predownload.py` guards this instead: if *every* fetched frame
  for a store comes back empty it logs a prominent `ERROR` pointing at
  the date range.  **If your scores come back all-NaN, check that
  message first** — verify the IC dates are within the UFS replay
  archive's range before anything else.

Verification and fill data come from gridded reanalysis (ARCO_ERA5 /
WB2ERA5), which is effectively gap-free *within its range* but has a
trailing edge a few days behind real time — the same constraint as any
forecast campaign.  Predownload of `verification.zarr` / `data.zarr`
fails loudly (not silently) on an out-of-range valid time, so those
gaps surface immediately.

### Analysis mode

Each work-item time produces one analysis, yielded as the `analysis`
stream with a singleton `lead_time=[0]` axis, the same layout as the
library's `DiagnosticRunner`. The standard scoring then aligns
verification at the analysis time itself, and the report reads as analysis
error per cycle time. The runner requires `nsteps: 0`. The DA model must
emit its analysis on the verification grid; for HealDA, `lat_lon: true` with
`output_resolution: [721, 1440]` matches ARCO_ERA5 0.25°. `predownload.py`
writes the `<slot>.parquet` stores plus a `verification.zarr` (set
`predownload.verification.enabled=true` in the campaign); there is no
gridded initial-condition store in this mode.

### Forecast mode (HealDA initializing `UCast`)

The `assimilation_forecast` builder constructs the ordinary forecast runner
on `model.forecast` and replaces only its initial-condition source with an
`AnalysisSource`. That `DataSource` runs the DA model at every requested
valid time, so a history model gets one analysis per prognostic input lead
offset. Rollout, diagnostics, ensemble/perturbation, output, scoring, and
report are the standard forecast path. Diffing the `scores_summary.csv` of
a HealDA-initialized `UCast` campaign against a plain `UCast` campaign with
reanalysis initial conditions compares the two directly.

A `CompositeSource` fills prognostic input variables the DA analysis
does not provide from the standard initial-condition path: `ic_source` BYO,
then pre-downloaded `data.zarr`, then `data_source`, then the model's
recommended source.
The runner's `data` role then carries only the fill variables, so
`predownload.py` narrows `data.zarr` to exactly that set and creates no
store when the DA model covers all inputs. The builder logs the fill set
prominently. Set `model.fill_missing_variables=false` to error on any gap
instead (pure-DA initial conditions only).

### Stateless assimilation models only

The runners support independent (stateless) DA models: each analysis is
one `model(*obs)` call, so work items distribute freely across ranks.
Observations arrive as DataFrames, one `DataFrameSource` per model input
slot; the runners reject gridded DA inputs (a background field, as CorrDiff
needs). Cycling `StatefulAssimilationModel` runs are out of scope, and a
`model.da.runner` entry in the config is an error.

## Multi-GPU Execution

Use `torchrun` to distribute forecasts across GPUs:

```bash
torchrun --nproc_per_node=$NGPU --standalone main.py
```

Each work item covers one initial time and a group of `member_batch`
ensemble members. `earth2studio.run.work.distribute_work` splits the items
into contiguous, balanced slices; the first `len(items) % world_size` ranks
take one extra item, so exact divisibility is not required.

## Resuming and Multi-Job Runs

Set `resume=true` to skip already-completed work items and append to the
existing zarr store.  This is useful in two scenarios:

### Resuming after a failure

If a job is killed or times out partway through, re-submit with the same
config plus `resume=true`. An `earth2studio.run.ProgressStore` in
`<output.path>/.progress/` records completed work items, which the rerun
skips; an item counts as done only after its outputs flush, so an
interrupted item reruns in full:

```bash
torchrun --nproc_per_node=$NGPU --standalone main.py resume=true
```

### Resuming across jobs

Jobs that share an `output.path` run one after another. Resubmit the same
command with `resume=true` after a failure or an expired allocation, and
the next job skips the items the progress store records as done:

```bash
# After the previous job ended, the same command again
torchrun --nproc_per_node=$NGPU --standalone main.py resume=true
```

Two jobs running at once do not split the work. Each lists the remaining
items when it starts, so both run them all. The second job's progress
setup also compacts the logs the first job still writes to, which loses
the first job's later records. To run a campaign on more than one node at
once, launch one `torchrun` job spanning them (`--nnodes`), or give each
job its own `start_times` and `run_id`.

When `resume=true`, the `output.overwrite` setting is ignored — existing
data is never deleted.  When all items are complete, subsequent runs exit
immediately with a success message.

## IO backend and sharding

By default the recipe writes through `AsyncZarrBackend`, which runs
`output.thread_writers` writer threads of its own and supports Zarr v3
sharding.  Setting `output.io_backend: zarr` falls back to `ZarrBackend`,
which writes synchronously (`thread_writers` is ignored there).

```yaml
output:
    io_backend: async_zarr
    shard_coords: {lead_time: 8}
    max_inflight_shards: 4
```

Note that larger *chunks* are not the alternative: `time`, `lead_time` and
`ensemble` are written one slice at a time, so a chunk spanning several of them
would let concurrent writes read-modify-write the same object and lose one of
them. `build_backend` in `src/output.py` rejects a chunk size other than 1
on those axes.

Only `lead_time` is safe to shard on a multi-rank run.  `time` and `ensemble`
are the distributed axes, and shard buffering is per-process, so a shard
spanning two ranks is written in full by both and the later write silently
discards the other's data. The library `Pipeline` rejects stores sharded on
those axes whenever the world size is greater than one. Sharding also
interacts with resume. `flush()` runs after every work item there, writing
out any incomplete shard, and re-entering that shard later falls back to
rewriting the whole object. Sharding on `lead_time` only avoids this.
Pre-downloaded stores hold a single `lead_time` and never use shards.

Reading a sharded store requires zarr ≥ 3 (and a correspondingly recent
xarray); readers on zarr 2.x fail with a codec error.

## Configuration

All configuration lives under `cfg/` and uses [Hydra](https://hydra.cc/docs/intro/).
The config is organized into four layers:

| Layer | Location | Purpose |
| --- | --- | --- |
| Base | `cfg/default.yaml` | Shared defaults for runner, data, output, pre-download and scoring |
| Model | `cfg/model/*.yaml` | Model architecture and checkpoint |
| Data source | `cfg/data_source/*.yaml` | Source of initial conditions and verification |
| Campaign | `cfg/campaign/*.yaml` | ICs, ensemble, variables, forecast length |

### Campaign configs

Campaign configs are the primary way to set up evaluation runs.  They
override only what differs from the base config — model, ICs, ensemble
size, and output variables.  Apply with `campaign=`:

```bash
# DLWP monthly deterministic
python main.py campaign=dlwp_2024_monthly

# FCN3 4-member ensemble, monthly ICs
python main.py campaign=fcn3_2024_monthly
```

Both `main.py` and `predownload.py` accept the same `campaign=` flag,
so the two scripts stay in sync automatically.

To add a new model benchmark, create one file in `cfg/campaign/`:

```yaml
# cfg/campaign/my_model_2024.yaml
# @package _global_
defaults:
    - override /model: my_model

run_id: my_model_2024
start_times:
    - "2024-01-01 00:00:00"
nsteps: 40
output:
    variables: [t2m, z500]
```

### Model selection

Models are selected via Hydra defaults. Each model config lives in
`cfg/model/` and specifies the architecture class. Campaign configs
override the model via `defaults: [override /model: ...]`, or you
can switch on the command line:

```bash
python main.py model=fcn3
```

### Data source selection

The data source is a Hydra group as well.  `cfg/data_source/arco_era5.yaml`,
the default, reads hourly ARCO ERA5. Models that need time statistics as
input, such as FuXi-S2S's UTC calendar-day means, declare them with qualified
variable labels in their input signature (`t2m:mean:0h:24h`,
`tp:mean:1h:25h`, and so on). `earth2studio.data.fetch_data` then computes
the statistic from the hourly source itself, so the recipe needs no daily
mean source wrapper. `predownload.py` caches the reduced fields under their
qualified labels, and the cached store answers `fetch_data` for them.
`PredownloadedSource` in `src/data.py` explains how. A campaign switches
sources with
`defaults: [override /data_source: ...]`. An inline `data_source:` block
merges over the selected source instead, which is how the custom examples
above work.

### Ensemble runs

Set `ensemble_size > 1` and provide a perturbation config:

```yaml
ensemble_size: 10
perturbation:
    _target_: earth2studio.perturbation.CorrelatedSphericalGaussian
    noise_amplitude: 0.05
```

The runner draws each member's perturbation under `member_seed(random_seed,
time, member)`, one member at a time. A member's initial condition does not
depend on the members it runs with. The runner seeds a stochastic
model (for example FCN3) once per work item, from its first member, so the
trajectories do depend on the item's members. The default `member_batch: 1`
reproduces every member exactly. [Members per rank](#members-per-rank-k--1)
covers batching.

### Scoring

Score inference outputs against verification data using any
`earth2studio.statistics` metric.

**Step 1 — Pre-download with verification data:**

```bash
python predownload.py predownload.verification.enabled=true
```

**Step 2 — Run inference:**

```bash
python main.py
```

**Step 3 — Run scoring:**

```bash
python score.py
```

**Distributed scoring** (parallelises across IC times):

```bash
torchrun --nproc_per_node=$NGPU --standalone score.py \
    campaign=fcn3_2024_monthly
```

Scoring writes a `scores.zarr` store in the output directory.  Each
metric × variable combination is a separate zarr array named
`{metric}__{variable}` (e.g. `rmse__t2m`, `crps__z500`).

Configure metrics in the campaign config or via CLI overrides:

```yaml
scoring:
    metrics:
        rmse:
            _target_: earth2studio.statistics.rmse
            reduction_dimensions: [lat, lon]
        crps:
            _target_: earth2studio.statistics.crps
            reduction_dimensions: [lat, lon]
            ensemble_dimension: ensemble
    lat_weights: true           # cosine latitude weighting
    lead_time_chunk_size: 8     # memory control
```

Custom metrics work too — any class satisfying the `earth2studio.statistics.Metric`
protocol can be specified via `_target_`.

**RMSE note:** With `reduction_dimensions: [lat, lon]`, scoring produces
per-(time, lead_time) RMSE values.  To compute aggregate RMSE across
all IC times, use `sqrt(mean(rmse²))` in post-processing (equivalent
to `sqrt(mean(MSE))`).  The report step handles this automatically.

### Regional scoring

`scoring.regions` scores every configured region alongside the full
grid and adds a labeled `region` axis to the score stores. Both
pathways read the same block: the offline path masks each metric's
weights per region, and the online path folds the masks into its
accumulation weights.

```yaml
scoring:
    regions:
        global: null                                   # the whole grid
        europe: {lat: [35, 75], lon: [-12.5, 42.5]}    # one box
        extra_tropics:                                 # union of boxes
            - {lat: [20, 90]}
            - {lat: [-90, -20]}
```

A region is `null` for the whole grid, one box, or a list of boxes
scored as their union. A box maps spatial dimension names to
`[min, max]` coordinate ranges, so limited-area grids work the same way
with their own dimension names, for example HRRR `y`/`x` coordinates.
Dimensions left out of a box cover their full extent. Longitudes may
be negative, meaning degrees west, and wrap the dateline where min
exceeds max.

The offline path applies regions at `score.py` time against the retained
`forecast.zarr`, so re-scoring with new regions or metrics needs no new
inference. The online path bakes the masks into the accumulated sums,
so choose regions before the run. The offline path can only
region-mask a metric that reduces over all spatial dimensions and takes
a `weights` argument. Any other metric, such as a spectral one, scores
the whole grid only.

### Event scoring

`scoring.events` reports skill for one named event: a time window
paired with a region. An event's box joins `scoring.regions` under the
event's name, so both pathways score it like any other region. The
scorecard exporter applies the window when it reads the per-IC scores.
Time windows cost nothing after a run. Regions, as above, have to be in
the config before an online run.

```yaml
scoring:
    events:
        storm_eowyn:
            label: "Storm Éowyn"                    # optional display name
            start: "2025-01-23 12:00:00"            # inclusive, UTC
            end: "2025-01-25 00:00:00"
            region: {lat: [48, 62], lon: [-15, 5]}  # box, list of boxes, or region name
            window: valid                           # valid (default) | init
            ics: {step_hours: 12, lookback_hours: 336}
```

`window: valid` keeps every pair of initial condition and lead time whose
valid time falls inside the window. Each lead time then averages over a
different set of forecasts, and lead times that never reach the window
come out empty. `window: init` keeps whole forecasts by their initial
time. The optional `ics` block adds initial conditions to the campaign,
every `step_hours` from `start - lookback_hours` up to `end`. They join
`start_times` or the `ic_block_*` sweep, or stand alone when the config
has neither. A `valid` window must state `lookback_hours`. Set it to the
forecast horizon so every lead time has valid times inside the window.
Each event's `ics` block adds initial conditions, so a campaign grows with
every event. Trim it with fewer events, a larger `step_hours`, or a
smaller `lookback_hours`.

The score stores carry the event definitions in their `events`
attribute. The scorecard exporter turns them into
`eval_scores_<model>_events.json`. Refer to the
[scorecard README](scorecard/README.md#event-campaigns) for the export.

### Online scoring

The flow above is *store-then-score*: `main.py` writes every
(member, IC, lead, variable) field to `forecast.zarr` and `score.py`
re-reads it.  At campaign scale that raw store is the binding constraint —
a 50-member, 730-IC, 14-day global campaign is ~170 TB of raw fields to
produce ~20 MB of scores.

**Online scoring** reduces each forecast to sufficient statistics inside
the inference loop, so no raw store is needed:

```bash
torchrun --nproc_per_node=$NGPU --standalone main.py \
    campaign=fcn3_2024_monthly \
    scoring.mode=online output.retain=none
```

This writes `stats.zarr` (~400 MB for the campaign above) and derives a
`scores.zarr` from it that `report.py` reads unchanged.

**How work is distributed.**  Ranks are partitioned into *ensemble
groups*.  A group owns one IC at a time and its ranks carry disjoint
slices of the ensemble — rank `g` runs members `[g*K, (g+1)*K)` where
`G * K = ensemble_size` and `K` is `members_per_rank`.  ICs are
distributed across **groups**, not ranks.  With the default `K = 1` you
need `world_size >= ensemble_size`.  Online scoring runs the same
`run_item` as an offline run, so perturbed members match offline ones
whatever `K`.  The runner seeds a stochastic model once per item, from
its first member. Its trajectories match an offline run only when both
group members alike: `members_per_rank` equal to `member_batch`, or both 1.

<!-- markdownlint-disable MD013 -->
| Setting | Meaning |
| --- | --- |
| `scoring.mode` | `offline` (default) · `online` |
| `output.retain` | `all` (default) · `none` — drop `forecast.zarr` entirely; only legal with online scoring |
| `scoring.online.scores_store` | Where the finalize step writes; `null` → `scoring.output.store_name` |
| `scoring.online.ensemble_group_size` | Ranks per group `G`; `null` → `G = ensemble_size / K` |
| `scoring.online.members_per_rank` | Members per rank `K`; `> 1` batches `K` members per item (every runner but `assimilation`) |
| `scoring.online.stats_store` | The durable artifact (default `stats.zarr`) |
| `scoring.online.climatology` | A `DataSource` on the verification grid; required for ACC |
<!-- markdownlint-enable MD013 -->

#### Members per rank (`K > 1`)

`K = 1` is the simplest layout but spreads a group across `M` ranks —
usually several nodes — which puts the CRPS member exchange on the
inter-node fabric.  Running `K` members per rank shrinks the group to
`M/K` ranks; once a group fits inside one node the exchange moves onto
NVLink, roughly a 10× reduction in the dominant communication cost.  It
also lets you run an ensemble at all when `world_size < ensemble_size`.

```bash
# 56 members, 7 per rank -> 8-rank groups, each inside one node
torchrun --nproc_per_node=8 --standalone main.py \
    campaign=fcn3_2024_monthly scoring.mode=online output.retain=none \
    ensemble_size=56 scoring.online.members_per_rank=7
```

`G * K` must equal `ensemble_size` exactly — ragged groups are rejected
with a clear error rather than silently padded.

Every distributed process fetches through its own temporary data cache
(`src/logs.py`). Sources with `cache: false` such as `HRRR` delete their
download directory after each fetch, which would remove files another rank
is still reading. Set `EARTH2STUDIO_DATA_CACHE` yourself to keep one shared
root; model packages use the model cache either way.

A batched rollout stacks members on the model's batch axis: one work item
carries more than one `member_ids` entry. The library runners behind
`forecast`, `assimilation_forecast`, `window_mean`, `diagnostic` and
`stormscope` all batch members; only the `assimilation` runner runs one
member per item, and the pipeline rejects batched items for it. Offline
runs batch members with the top-level `member_batch` setting instead. The
runner draws each member's perturbation under that member's own seed, so
initial conditions still match a `K = 1` run exactly.

Stochastic models, such as FCN3, declare `stochastic = True`. The runner
seeds them once per work item, from its first member (execution contract
rule X5). The members of an item share that random stream, so single
trajectories differ from a `K = 1` run while the ensemble stays
statistically valid. `K = 1` reproduces every member exactly. The bundled
FCN3 and Atlas CRPS scorecard campaigns run two members per rank to keep
their groups node-local; the Aurora 1.5 ensemble campaign runs one member
per rank, 16 ranks over two nodes, because a two-member batch of that
model exceeds the GPU memory.

**Metrics.**  The online path does *not* use `scoring.metrics` — it
accumulates a fixed set of mergeable sums and derives the metric set from
them at finalize time:

<!-- markdownlint-disable MD013 -->
| Array in `scores.zarr` | Definition |
| --- | --- |
| `mse__{var}` | per-member MSE (with an `ensemble` axis when `M > 1`) |
| `ensemble_mean_mse__{var}` | MSE of the ensemble mean |
| `ensemble_variance__{var}` | Bessel-corrected ensemble variance (spread²) |
| `crps__{var}` | fair (unbiased) CRPS |
| `rank_reliability__{var}` | `sum_k \|p_k - 1/(M+1)\|` of the rank histogram |
| `bias__{var}`, `corr__{var}` | mean error, centered spatial correlation |
| `acc__{var}` | anomaly correlation — only when a climatology is configured |
<!-- markdownlint-enable MD013 -->

Spread and spread-skill ratio come out of the report's existing
`ensemble_mean_mse` + `ensemble_variance` path.  The full rank histogram
(with its `rank_bin` axis) stays in `stats.zarr`.

#### The cost of CRPS

Every online statistic except CRPS reduces spatially *before*
communicating, so only scalars cross the wire.  CRPS's `E|X-X'|` term is
the exception: every member has to meet every other member, costing
`(G-1) * K * n_vars * field` bytes of egress per rank per lead step — for
a 50-member 0.25° campaign, ~4 GB per lead step, dominating everything
else by ~25×.  That single term is where an online run's communication
budget goes.

Four levers, roughly in order of leverage:

<!-- markdownlint-disable MD013 -->
| Setting | Effect |
| --- | --- |
| `defer_pairwise_one_step: true` (default) | Issues the exchange for step *n* asynchronously and finalizes it during step *n+1*'s model forward.  Nearly free; hides most of the cost. |
| `members_per_rank: K` | A smaller group fits inside a node → NVLink instead of IB (~10× cheaper).  See above. |
| `pairwise_comm_dtype: bfloat16` (default) | Halves the bytes.  Arithmetic upcasts to float64 on arrival, so the precision loss is far below ensemble sampling noise. |
| `pairwise_variables: [z500, t2m]` | Restrict CRPS to the variables you actually report on; the moment-based metrics stay on the full set. |
| `crps: false` | Drop it entirely — the run then needs no field-sized communication at all. |
<!-- markdownlint-enable MD013 -->

Members meet via a single all-gather per variable chunk, which is what
makes the exchange deferrable; `variable_chunk` bounds both the payload
and the float64 working set.

**Re-deriving scores.**  `stats.zarr` holds *sums*, not scores, so
re-deriving scores from the already-stored sums -- e.g. recomputing rank
reliability from the stored rank counts, or bootstrap CIs over ICs --
never requires re-running inference. Spatial weighting is baked into the
sum at run time, though: cosine-latitude weighting is on or off for the
whole grid for the entire run, and there is currently no per-region
weighting exposed. So while a *new* metric that is derivable from the
existing sums is free, evaluating a *different* spatial weighting/region,
a genuinely new metric that needs its own accumulator, or anything else
that needs the raw fields, all require re-running the inference portion --
which is the main downside of online scoring. To consolidate the
summation results in `stats.zarr` into the final scores, run:

```bash
python score.py campaign=fcn3_2024_monthly scoring.mode=online
```

In online mode `score.py` only re-runs the (single-process, cheap)
finalize step.

**Validating an online campaign.**  With `output.retain=all` (the
default) an online run keeps its raw fields, so the very same forecasts
can be scored the slow way and the two compared:

```bash
torchrun --nproc_per_node=$NGPU --standalone score.py \
    campaign=fcn3_2024_monthly scoring.mode=offline \
    scoring.output.store_name=scores_offline.zarr
```

The `store_name` override keeps the offline pass off `scores.zarr`, which
the online finalize already wrote; setting
`scoring.online.scores_store` moves the online store instead if you'd
rather that change. Both passes read `scoring.nan_policy`. `propagate` and
`zero_fill` mean the same on both. `mask`, the per-gridpoint masking for
sea-only fields, exists online only. The offline pass of such a campaign
needs `scoring.nan_policy=propagate` (or `zero_fill`) on the command line,
and only its unmasked variables are then comparable.

**Resume.**  Online runs record finished items in the same
`<output.path>/.progress` store as offline runs. An item counts as done only
after its round's group reduction, so a crash loses at most one IC per
group; re-submit with `resume=true`. A resumed online run needs whole
initial conditions: an IC with some member groups recorded and others not
(a job that stopped between the ranks' records) reruns whole. The recipe
drops its records from the progress store and the rerun rewrites the
statistics of that initial condition.

**Failure domain.**  A dead rank hangs its whole group rather than merely
forfeiting its own items.  `scoring.online.nccl_timeout_s` (default 30 min)
turns that into a clean job failure that `resume=true` picks up from.

**Caveats.**

- `verification.zarr` must exist (run `predownload.py` first); the setup
  checks that it covers every `(IC + lead)` valid time before the first
  work item runs, once the model has loaded.
- Online scoring prioritizes support for a fixed set of important metrics
  amenable to the partial summation approach that produces `stats.zarr`.
  Users with custom metrics will have to fall back to offline scoring,
  or open an issue with a request to add new online metrics.
- With `output.retain=none` the report's visualization sections have no
  raw fields to plot and degrade gracefully.

### Report

Generate a self-contained markdown report with summary tables, lead-time
skill curves, per-IC heatmaps, and field visualizations.

**Step 4 — Generate report** (no GPU required):

```bash
python report.py campaign=fcn3_2024_monthly
```

This produces a `report/` directory inside the output path:

```text
<output.path>/report/
├── report.md                    # Collapsible markdown report
├── figures/
│   ├── rmse_vs_leadtime_surface.png
│   ├── crps_vs_leadtime_upper_air.png
│   └── ...
└── tables/
    ├── scores_summary.csv       # (model, metric, variable, lead_time, value)
    ├── scores_summary_DJF.csv   # Per-group CSVs (when time_groups configured)
    └── ...
```

The markdown report uses `<details>` blocks for collapsible sections —
the summary table is always visible, while detailed plots expand on click.

#### Variable groups

Group variables that share physical scales onto the same plot.
Variables not listed in any group automatically get their own
individual plot:

```yaml
report:
    variable_groups:
        geopotential: [z500, z850]
        temperature: [t500, t850]
        u_wind: [u500, u850]
        # t2m, msl, tcwv are not listed → each gets its own plot
```

If `variable_groups` is omitted entirely, every variable gets its own plot.

The summary table also supports a `variables` key to feature only a
subset of variables:

```yaml
report:
    sections:
        - type: summary_table
          variables: [t2m, z500, msl]
          lead_times: ["1 days", "5 days", "10 days"]
```

#### Time groups (seasonal breakdown)

Break down lead-time curves by time period using date ranges:

```yaml
report:
    time_groups:
        DJF:
            - start: "2024-01-01"
              end: "2024-02-29"
            - start: "2024-12-01"
              end: "2024-12-31"
        MAM:
            - start: "2024-03-01"
              end: "2024-05-31"
        JJA:
            - start: "2024-06-01"
              end: "2024-08-31"
        SON:
            - start: "2024-09-01"
              end: "2024-11-30"
```

Each group is a list of date ranges (inclusive), allowing wrap-around
groups like DJF.  The same config works for any IC frequency — monthly,
12-hourly, or otherwise.

To include seasonal curves in the report, add a `lead_time_curves`
section with `time_groups: true`:

```yaml
report:
    sections:
        - type: lead_time_curves
          collapsed: true

        - type: lead_time_curves
          time_groups: true
          title_suffix: "Seasonal Breakdown"
          collapsed: true
```

#### Map projection

For global models, enable a map projection for visualization panels
(requires [cartopy](https://scitools.org.uk/cartopy/)):

```yaml
report:
    projection: robinson   # or mollweide, platecarree, orthographic
```

If omitted or `null`, a flat (equirectangular) plot is used.  If cartopy
is not installed, the projection setting is ignored with a warning.

#### Section types

| Type | Description | Key options |
| --- | --- | --- |
| `header_visualization` | Hero prediction vs truth map | `variable`, `lead_time`, `time` |
| `summary_table` | Scores at key lead times | `lead_times`, `variables` (subset) |
| `lead_time_curves` | Metric vs lead time plots | `metrics`, `time_groups`, `title_suffix` |
| `ic_heatmap` | Time × lead_time heatmap | `metrics`, `variables` |
| `visualization` | Pred vs truth for many vars/lead times | `variables`, `lead_times`, `time` |

The `header_visualization` and `visualization` sections require the raw
`forecast.zarr` and verification data to still be present.  If the data has
been cleaned up, these sections degrade gracefully with a note.

## Architecture

```bash
recipes/eval/
├── main.py              # Hydra entry point — distributed inference
├── predownload.py       # Hydra entry point — data pre-fetch
├── score.py             # Hydra entry point — distributed scoring
├── report.py            # Hydra entry point — report generation
├── cfg/
│   ├── default.yaml     # Base config (runner, sources, output, predownload, scoring, report)
│   ├── predownload.yaml # Thin overlay (hydra.run.dir only)
│   ├── data_source/
│   │   └── arco_era5.yaml
│   ├── model/           # One file per model (dlwp, fcn3, dlesym, fuxi_s2s, ...)
│   └── campaign/        # One file per evaluation campaign
├── src/
│   ├── campaign.py      # Initial times, members, work items, ensemble-group ordering
│   ├── runners/         # Runner builders: src.runners.build_runner and the built-ins
│   │   ├── __init__.py  #   BUILTIN_RUNNERS, build_runner, forecast, diagnostic builders
│   │   ├── assimilation.py # AssimilationRunner, AnalysisSource, DA builders
│   │   ├── stormscope.py #  stormscope builder: CoupledRunner over GOES + MRMS (goes/mrms)
│   │   └── seasonal.py  #   WindowMeanRunner (window means of a rollout)
│   ├── data.py          # resolve_source, Predownloaded(Frame)Source, FrameRequest
│   ├── models.py        # Model loading (prognostic, diagnostic, assimilation)
│   ├── output.py        # Store paths, zarr backends, output regridders, per-stream variables
│   ├── logs.py          # Logging and process-group setup
│   ├── scoring.py       # Offline scoring: metrics, data alignment, score loop
│   ├── online.py        # Online scoring — ensemble groups, accumulators, stats.zarr
│   ├── metrics.py       # Recipe metrics (mse, ensemble_variance) as library Statistics
│   ├── regions.py       # Regional masks for both scoring pathways
│   └── report/          # Report package: aggregation, plotting, sections
│       ├── aggregation.py #  Score loading, time/ensemble aggregation, spread-skill
│       ├── plotting.py  #   Matplotlib + cartopy primitives
│       ├── sections.py  #   Section renderers (summary, curves, heatmaps, visualization)
│       └── main.py      #   generate_report orchestration
├── scorecard/           # Scorecard campaigns, baselines, exporter
│   └── utils/
│       ├── baselines.py #   PersistenceBaseline, ClimatologyForecast
│       └── sources.py   #   StormCast grid, conditioning and window helpers
├── test/
└── pyproject.toml
```

The recipe builds on `earth2studio.run`: the library owns work distribution
(`WorkItem`, `work_items`, `distribute_work`), resume (`ProgressStore`), the
per-item execution contract (`Runner`, `PrognosticRunner`,
`DiagnosticRunner`) and the supervising loop (`Pipeline`). Interpolation
between grids comes from `earth2studio.grids`, fetching from
`earth2studio.data.fetch_data`, and metrics from `earth2studio.statistics`.
Each recipe module has a scoped responsibility:

<!-- markdownlint-disable MD013 -->
| Module | Responsibility |
| --- | --- |
| `campaign.py` | Parse initial conditions from config; `build_items`; order items into lockstep ensemble groups for online scoring |
| `runners/` | `build_runner` from `cfg.runner`; built-in runners and their `source_roles` |
| `data.py` | `resolve_source` (BYO, then pre-downloaded, then live); `PredownloadedSource` / `PredownloadedFrameSource` serving the pre-downloaded zarr / parquet stores |
| `models.py` | Load prognostic, diagnostic, and assimilation models from config nodes |
| `output.py` | Store paths; `build_backend` (ZarrBackend / AsyncZarrBackend); `build_regridders` from `output.regrid`; `stream_variables` |
| `scoring.py` | Metric instantiation, data loading/alignment, offline scoring loop |
| `online.py` | In-loop scoring — ensemble-group comms, sufficient-statistic accumulators, `stats.zarr` → `scores.zarr` finalize |
| `metrics.py`, `regions.py` | Recipe-specific statistics and regional masks |
| `report/` | Score aggregation, matplotlib plotting, section rendering, markdown report assembly |
<!-- markdownlint-enable MD013 -->

### Runner interface

A runner drives all inference logic: an object satisfying the
`earth2studio.run.Runner` protocol that executes one work item at a time
and yields outputs keyed by *stream*. `main.py` builds it, builds the work
items, and hands both to `earth2studio.run.Pipeline`. The pipeline owns
everything above a single item: splitting items over ranks, filtering
outputs to `output.variables`, interpolating onto the output grid, online
scoring, zarr writes, and resume.

| Member | Purpose |
| --- | --- |
| `supports_member_batching` | Whether one item may carry more than one member in `member_ids` |
| `to(device)` | Move the models to `device` |
| `output_coords(final_lead_time)` | Per-stream signature: `lead_time`, then the field dimensions |
| `data_requests(item)` | The item's gridded fetches (`DataRequest`), used by `predownload.py` |
| `run_item(item)` | Yield, per step, a mapping of stream name to `xr.DataArray` |

Every runner the recipe builds also carries `source_roles`, a mapping of
role name to bound source (`data`, `forcing0`, `goes`, and so on), which names
the stores `predownload.py` writes and `resolve_source` reads back.
Runners binding observation DataFrames add `frame_requests(item)`.

A stream is a named sequence of outputs sharing variables, grid and
lead-time pattern: `forecast` plus one stream per diagnostic for the
forecast runner, `analysis` for DA, `goes` and `mrms` for StormScope.
The recipe writes every stream into the one `<output.path>/forecast.zarr`
group and splits `output.variables` over the streams carrying each name.
It interpolates each stream onto the `output.regrid` grid when set, and
records finished items in a `ProgressStore` at `<output.path>/.progress`
for `resume=true`.

`cfg.runner` selects the builder. A name picks a built-in; any further keys
pass as keyword arguments to the builder:

```yaml
runner: forecast                 # a built-in name
runner: {name: window_mean, window: 30D}   # a built-in with options
runner: {_target_: my_pkg.my_runner, option: value}   # any importable builder
```

<!-- markdownlint-disable MD013 -->
| `runner:` | Implementation | What it does |
| --- | --- | --- |
| `forecast` (default) | `earth2studio.run.PrognosticRunner` over `model` | Prognostic rollout with optional diagnostics, perturbation, and member batching; the `forecast` stream plus one per diagnostic. Coupled models such as DLESyM and forced models such as StormCast run here too |
| `diagnostic` | `earth2studio.run.DiagnosticRunner` | Diagnostics applied to source data (for example CorrDiff); one stream per diagnostic at lead time zero |
| `assimilation` | `AssimilationRunner` | One DA analysis per item as the `analysis` stream; requires `nsteps: 0` |
| `assimilation_forecast` | `PrognosticRunner` reading from `AnalysisSource` | DA analyses initialize the prognostic rollout of `model.forecast` |
| `stormscope` | `earth2studio.coupling.CoupledRunner` over `model.goes` and `model.mrms` | The GOES model feeds the radar model through one connector; streams `goes` and `mrms`, members batched |
| `window_mean` | `WindowMeanRunner` over the forecast runner | Rollout stored as means over fixed windows (`window: 30D`); the `lead_time` axis holds the window ends. Verification needs a `verification_source` or `predownload.verification.source` of window means at those ends, and a saved `verification.zarr` must hold means over the run's own window. See `campaign=fuxi_s2s_2024` |
<!-- markdownlint-enable MD013 -->

### Custom runners

To add a custom inference loop, write a builder `(cfg, prefer_predownloaded=True,
**options) -> Runner` and point `runner` at it with a `_target_` block.
See [docs/extending.md](./docs/extending.md) for the full contract, the
worked example (`src/runners/seasonal.py`) and how `predownload.py`
derives its stores from the runner. Models that exchange fields need no
custom runner: `src/runners/stormscope.py` builds the library's
`earth2studio.coupling.CoupledRunner` instead.

```python
# my_runner.py
from earth2studio.run import Runner

class MyRunner:
    supports_member_batching = False

    def to(self, device): ...
    def output_coords(self, final_lead_time): ...
    def data_requests(self, item): ...
    def run_item(self, item):
        ...
        yield {"forecast": x}

def my_runner(cfg, prefer_predownloaded=True, **options) -> Runner:
    return MyRunner(...)
```

```yaml
# In your Hydra config override:
runner: {_target_: my_runner.my_runner}
```

`Pipeline` supplies the rest for a custom runner as for the built-ins, as
described under [Runner interface](#runner-interface).

## Testing

Install dev dependencies first (`uv sync --extra dev`). Unit and integration
tests live in `test/`. They use a lightweight `Persistence` model and
`Random` data source so no network access or model checkpoints are required.

```bash
# Run all tests (CPU only, no downloads)
pytest test/ -v

# Only the entry point: build_runner and main.run on the fixtures
pytest test/test_main.py -v

# Only predownload planning and the store writers
pytest test/test_predownload.py -v
```

`test/conftest.py` composes the real `cfg/default.yaml` over a temporary
output path (`compose_cfg`), so tests exercise the same config the entry
points read. `test/test_online.py` ends with a two-rank `gloo` test that
launches `torchrun` on itself to cover the real collective code path; set
`E2S_EVAL_SKIP_MULTIRANK` to skip it.
