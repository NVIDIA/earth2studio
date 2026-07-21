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
- **Extensible pipeline interface** — subclass `Pipeline` to add custom inference loops
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
  - [Splitting work across multiple SLURM jobs](#splitting-work-across-multiple-slurm-jobs)
- [Configuration](#configuration)
  - [Campaign configs](#campaign-configs)
  - [Model selection](#model-selection)
  - [Ensemble runs](#ensemble-runs)
  - [Scoring](#scoring)
  - [Report](#report)
- [Architecture](#architecture)
  - [Pipeline interface](#pipeline-interface)
  - [Custom pipelines](#custom-pipelines)
- [Extending the recipe](./docs/extending.md) — BYO data, custom models, custom pipelines
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

`predownload.py` must be run before `main.py`.  It fetches initial condition
data (and optionally verification data) and writes it into explicit zarr stores
under `<output.path>/`.  It accepts the **same config overrides** as `main.py`,
so the two commands stay in sync with no extra bookkeeping.

IC variables, lead times, and model step size are inferred automatically from
the model's `input_coords()` / `output_coords()`.

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

> **Note.** `predownload.verification.enabled` is also consulted by
> `score.py` to decide whether `data.zarr` holds verification data: when
> it's false (the default), the scorer refuses to treat an IC-only
> `data.zarr` as a verification store.  Set it in your **campaign
> config**, not just as a one-shot CLI override to `predownload.py` —
> otherwise `score.py` will read the default (`false`) and fail with
> "No verification data found".  The bundled campaigns (`dlwp_*`,
> `fcn3_*`, `dlesym_*`) set it explicitly.

### Zarr stores

Predownload creates the following stores in `<output.path>/`:

| Store | Dimensions | Contents |
|---|---|---|
| `data.zarr` | `(time, variable, <spatial...>)` | IC data (plus verification if same source) |
| `verification.zarr` | `(time, variable, <spatial...>)` | Only when verification source differs |
| `obs_<name>.parquet/` | one parquet file per analysis time | DA pipelines only — observation DataFrames (see [Observation predownload](#observation-predownload)) |

`main.py` automatically detects and reads from `data.zarr` when
`require_predownload=true` (the default).

### Disable the per-source cache

Earth2Studio's remote data sources (`ARCO`, `GFS_FX`, `GOES`, `MRMS`, …)
default to `cache=True`, which keeps a copy of every byte they fetch in
`~/.cache/earth2studio` (or `$EARTH2STUDIO_CACHE`).  Predownload then
writes its own dedicated zarr under `<output.path>/`, so with the cache
left on **the same data lands on disk twice** — once in the shared
cache, once in `data.zarr`.  For the campaign-scale fetches this recipe
targets, that doubles the disk footprint of every run.

The bundled configs all set `cache: false` on their data sources for
this reason.  When you add a new source — at the top level via
`data_source:` or per-component (e.g. StormScope's `ic_source` /
`conditioning_data_source` blocks) — pass it through as well:

```yaml
data_source:
    _target_: earth2studio.data.ARCO
    cache: false
```

Leave the cache on only when you have a separate reason to populate the
shared cache (e.g. another recipe will reuse the same data and you don't
want to refetch it).

### Resume after interruption

Progress is tracked per-timestamp via marker files.  If a predownload job is
killed (e.g. by a SLURM time limit), re-running with the same config skips
already-completed timestamps:

```bash
# Just re-run — resume is automatic
python predownload.py campaign=fcn3_2024_monthly
```

To recreate stores from scratch, set `predownload.overwrite=true`.

## Using Your Own Data

If you already have initial-condition or verification data on disk (for
example, a zarr store you ETL'd for training), you can skip predownload
entirely and point the recipe directly at it.  This avoids doubling disk
usage by re-caching data that already exists locally.

The two escape hatches are top-level config keys:

| Key | What it overrides | Consumer |
|---|---|---|
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
|---|---|---|
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

### Multi-source pipelines (StormScope)

The default StormScope checkpoint is the **`3km_10min`** CONUS nowcasting
variant (3 km grid, 10-minute step).  The pipeline runs two coupled models:

- **`StormScopeGOES`** — forecasts the eight ABI satellite channels.  The
  `3km_10min` variant is *pure obs*: it takes **no external conditioning**.
- **`StormScopeMRMS`** — forecasts radar reflectivity (`refc`, `refc_base`)
  **plus a gridded GLM lightning channel** (`glm_density`), conditioned on
  GOES (observations at the IC time, then the GOES model's own predictions
  during rollout via `call_with_conditioning`).

So the pipeline needs three IC-side sources — GOES satellite, MRMS radar, and
GLM lightning — living on three different native grids.  A single top-level
`ic_source` doesn't fit, so the pipeline declares `needs_data_source = False`
and resolves its own sources from per-component config blocks; `main.py` skips
top-level source resolution and the predownload sentinel check for these
pipelines.

The **GLM channel** (`glm_density`) is a *state* channel — both an input
(observation history) and a predicted output that evolves autoregressively.
It comes from `earth2studio.data.GOESGLMGrid` on a 0.1° lat/lon grid and is
regridded **bilinearly** onto the model grid (`src.regrid.BilinearRegridder`),
matching the model's own `build_glm_interpolator` — unlike the
nearest-neighbor path used for radar/satellite.

BYO is done by overriding the per-component entries in `cfg.model`:

```bash
python main.py campaign=stormscope_2023_convection \
    model.goes.ic_source._target_=my_pkg.MyGoesZarrSource \
    model.goes.ic_source.path=/data/my_goes.zarr \
    model.goes.ic_grid._target_=my_pkg.my_goes_grid \
    model.mrms.ic_source._target_=my_pkg.MyMrmsZarrSource \
    model.mrms.ic_source.path=/data/my_mrms.zarr \
    model.mrms.ic_grid._target_=my_pkg.my_mrms_grid \
    model.mrms.glm_data_source._target_=my_pkg.MyGlmSource \
    model.mrms.glm_grid._target_=my_pkg.my_glm_grid
```

`ic_grid` (and `glm_grid`) is a Hydra-instantiable callable that returns
`(lats, lons)` for the source's native grid — the pipeline uses it to build
StormScope's internal interpolator.  If your BYO store is already on the HRRR
grid, point `ic_grid` at a resolver that returns the model's own `y`/`x` —
StormScope's `prep_input` detects the match and skips interpolation.

#### Predownload with HRRR-aligned storage

Running `predownload.py` for a StormScope campaign writes **three** zarr
stores, each already resampled onto the model's HRRR sub-region — radar and
satellite via a nearest-neighbor regridder
(`src.regrid.NearestNeighborRegridder`), GLM via the bilinear regridder
(`src.regrid.BilinearRegridder`):

```bash
python predownload.py campaign=stormscope_2023_convection
# → <output.path>/data_goes.zarr   (time, y, x)  GOES ABI channels (nearest)
# → <output.path>/data_mrms.zarr   (time, y, x)  radar refc/refc_base (nearest)
# → <output.path>/data_glm.zarr    (time, y, x)  glm_density (bilinear)
```

Each store covers both the IC input window and every forecast valid time, so
`data_mrms.zarr` / `data_glm.zarr` double as verification for scoring the
radar and lightning channels.

At inference time, `main.py` auto-detects these stores under `<output.path>`.
The MRMS state spans two grids (radar + GLM), so its IC is reassembled from
`data_mrms.zarr` + `data_glm.zarr` via a `CompositeSource` that dispatches
each variable to the store providing it; `data_goes.zarr` is wired in with
`PredownloadedSource` directly.  Because every store is already on the model
`y`/`x`, StormScope's `prep_input` skips its live interpolators — so the
predownload is both a disk-size win (raw GOES is ~10× the regridded footprint;
raw GLM is far larger still) and an inference-speed win.  After the first
step, `glm_density` flows autoregressively from the model's own predictions.

Per-model BYO overrides of `ic_source` (or `glm_data_source`) bypass
predownload for that source.  To fully skip predownload, either set
`model.goes.ic_byo=true model.mrms.ic_byo=true` or simply don't run
`predownload.py` — the pipeline falls back to the live sources, wrapping each
in the appropriate regridder (nearest for radar/satellite, bilinear for GLM)
so the assembled state still lands on the model grid.

Because the `3km_10min` GOES model is pure-obs, there is **no** conditioning
source to predownload (no `cond_goes.zarr`).  MRMS conditioning is supplied
externally in the coupling loop (the GOES state via `call_with_conditioning`),
so its internal conditioning source is bypassed too.

## Data Assimilation Models

Models implementing the `AssimilationModel` protocol
(`earth2studio.models.da`) produce gridded analyses from sparse
observations rather than stepping a gridded state forward.  The recipe
supports two evaluation modes, both driven by the same predownload /
inference / scoring / report entry points:

| Mode | Pipeline | Campaign example |
|---|---|---|
| Analysis vs. reanalysis | `src.pipelines.assimilation.AssimilationPipeline` | `healda_2024_analysis` |
| DA-initialized forecast | `src.pipelines.assimilation.AssimilationForecastPipeline` | `healda_dlwp_2024_monthly`, `healda_fcn3_2024_monthly` |

```bash
# Analysis mode — score HealDA analyses directly against ERA5
python predownload.py campaign=healda_2024_analysis   # obs + verification
python main.py campaign=healda_2024_analysis
python score.py campaign=healda_2024_analysis
python report.py campaign=healda_2024_analysis

# Forecast mode — HealDA analyses initialize a prognostic rollout.
# healda_dlwp_2024_monthly is the lightweight example (DLWP, slim deps,
# NGC checkpoint); healda_fcn3_2024_monthly is the FCN3 variant.
python predownload.py campaign=healda_dlwp_2024_monthly
python main.py campaign=healda_dlwp_2024_monthly
```

### Observation sources

DA models consume observation **DataFrames**, not gridded fields, so
there is no gridded IC store.  Observations come from the sources
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
model's `input_coords()` tuple.  Two convenience keys are handled by the
recipe rather than passed to the source constructor: `enabled: false`
disables a slot (the model receives `None` — e.g. a conv-only ablation),
and `time_tolerance_hours: [lo, hi]` sets the observation window as
hours around each analysis time.

### Observation predownload

`predownload.py` caches observations alongside the gridded stores: one
parquet file per analysis time under `<output.path>/obs_<name>.parquet/`
(e.g. `obs_conv.parquet/`, `obs_sat.parquet/`), each holding the full
`time_tolerance` window for that analysis time.  The same per-timestamp
resume markers and multi-rank distribution as the zarr stores apply, so
an interrupted download continues where it left off and
`torchrun --nproc_per_node=N predownload.py` parallelizes the fetch.

At inference time the pipelines **automatically prefer** a predownloaded
`obs_<name>.parquet` store over the live source when it exists — the
observation analogue of the `data.zarr` substitution — so `main.py`
needs no network access to the obs archives.  Requesting an analysis
time that was not predownloaded fails loudly rather than silently
falling back to a live fetch.

In forecast mode, observations are cached at every analysis time IC
assembly touches (`IC time + each prognostic input lead offset`).  To
skip obs caching and always fetch live (e.g. when the source's own
`~/.cache/earth2studio` cache is already warm), set
`predownload.observations.enabled=false`.

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

* **Partial gaps** (some files present) — normal.  The analysis is built
  from a reduced obs set; predownload logs a per-store summary of how
  many frames were affected.
* **Total gaps** (every file missing for a request) — the obs source
  returns an empty frame and the DA model produces an **all-NaN
  analysis**, which propagates to NaN scores.  This almost always means
  the campaign's `start_times` fall outside the observation archive's
  coverage window.  Because the obs sources might not raise on missing
  files, `predownload.py` guards this instead: if *every* fetched frame
  for a store comes back empty it logs a prominent `ERROR` pointing at
  the date range.  **If your scores come back all-NaN, check that
  message first** — verify the IC dates are within the UFS replay
  archive's range before anything else.

Verification and fill data come from gridded reanalysis (ARCO / WB2
ERA5), which is effectively gap-free *within its range* but has a
trailing edge a few days behind real time — the same constraint as any
forecast campaign.  Predownload of `verification.zarr` / `data.zarr`
fails loudly (not silently) on an out-of-range valid time, so those
gaps surface immediately.

### Analysis mode

Each work-item time produces one analysis, written with a singleton
`lead_time=[0]` axis (the same layout as `DiagnosticPipeline`), so the
standard scoring aligns verification at the analysis time itself and the
report reads "analysis error per cycle time".  The DA model must emit
its analysis on the verification grid — for HealDA, `lat_lon: true` with
`output_resolution: [721, 1440]` matches ARCO/ERA5 0.25°.  Predownload
declares the `obs_*.parquet` stores plus a `verification.zarr` (set
`predownload.verification.enabled=true` in the campaign).

### Forecast mode (e.g. HealDA + FCN3)

`AssimilationForecastPipeline` subclasses `ForecastPipeline` and replaces
only initial-condition acquisition: one analysis per prognostic input
lead offset (history models get one per offset) instead of a
`data_source` fetch.  Rollout, diagnostics, ensemble/perturbation,
output, scoring, and report are the standard forecast path — a HealDA+FCN3
campaign is directly comparable against a plain FCN3 campaign with
reanalysis ICs by diffing their `scores_summary.csv`.

Prognostic input variables the DA analysis does not provide are **filled**
from the standard IC path (`ic_source` BYO → predownloaded `data.zarr` →
`data_source`); predownload narrows `data.zarr` to exactly that fill set
(no store is created when the DA model covers all inputs).  The fill set
is logged prominently at setup.  Set `model.fill_missing_variables=false`
to error on any gap instead (pure-DA ICs only).

### Stateful DA models

The pipelines drive DA models through an `AssimilationRunner` seam
(`src/assimilation.py`).  Stateless models (where `init_coords()` is
`None`, e.g. in HealDA) use the default `StatelessAssimilationRunner`, which calls the
model independently per analysis time so work items distribute freely
across ranks.  Stateful/cycling models (e.g. StormCastSDA, which carries
a background state between cycles) need a cycling runner that steps
`model.create_generator()` sequentially — wire one in via the optional
`model.da.runner` Hydra block; the stateless runner refuses stateful
models with a clear error.

## Multi-GPU Execution

Use `torchrun` to distribute forecasts across GPUs:

```bash
torchrun --nproc_per_node=$NGPU --standalone main.py
```

Work items (one per initial-time / ensemble-member pair) are partitioned
automatically and evenly across ranks. Remainder items are absorbed by the
first rank rather than requiring exact divisibility.

## Resuming and Multi-Job Runs

Set `resume=true` to skip already-completed work items and append to the
existing zarr store.  This is useful in two scenarios:

### Resuming after a failure

If a job is killed or times out partway through, re-submit with the same
config plus `resume=true`.  Completed work items are detected via marker
files in `<output.path>/.progress/` and automatically skipped:

```bash
torchrun --nproc_per_node=$NGPU --standalone main.py resume=true
```

### Splitting work across multiple SLURM jobs

Submit N identical jobs with `resume=true`.  The first job to start creates
the zarr store; subsequent jobs validate the schema and append.  Each job
skips items that have already been completed by earlier jobs:

```bash
# Submit the same command multiple times (or as a SLURM array)
torchrun --nproc_per_node=$NGPU --standalone main.py resume=true
```

Because zarr chunks are non-overlapping per `(time, lead_time)` slice,
concurrent writes from different jobs to different ICs are safe.

When `resume=true`, the `output.overwrite` setting is ignored — existing
data is never deleted.  When all items are complete, subsequent runs exit
immediately with a success message.

## Configuration

All configuration lives under `cfg/` and uses [Hydra](https://hydra.cc/docs/intro/).
The config is organized into three layers:

| Layer | Location | Purpose |
|---|---|---|
| Base | `cfg/default.yaml` | Shared defaults (pipeline, data source, output, predownload) |
| Model | `cfg/model/*.yaml` | Model architecture and checkpoint |
| Campaign | `cfg/campaign/*.yaml` | ICs, ensemble, variables, forecast length |

### Campaign configs

Campaign configs are the primary way to set up evaluation runs.  They
override only what differs from the base config — model, ICs, ensemble
size, and output variables.  Apply with `campaign=`:

```bash
# DLWP monthly deterministic
python main.py campaign=dlwp_2024_monthly

# FCN3 full 56-member ensemble
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

### Ensemble runs

Set `ensemble_size > 1` and provide a perturbation config:

```yaml
ensemble_size: 10
perturbation:
    _target_: earth2studio.perturbation.CorrelatedSphericalGaussian
    noise_amplitude: 0.05
```

For stochastic models (e.g. FCN3), the pipeline also calls
`model.set_rng(seed=...)` per ensemble member when available.

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
|---|---|---|
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
│   ├── default.yaml     # Base config (shared defaults + predownload + scoring + report)
│   ├── predownload.yaml # Thin overlay (hydra.run.dir only)
│   ├── model/
│   │   ├── dlwp.yaml
│   │   └── fcn3.yaml
│   └── campaign/        # One file per evaluation campaign
│       ├── dlwp_2024_monthly.yaml
│       └── fcn3_2024_monthly.yaml
├── src/
│   ├── pipelines/       # Pipeline package — ABC + built-in pipelines
│   │   ├── base.py      #   Pipeline ABC, Predownload(Frame)Store, shared run loop
│   │   ├── forecast.py  #   ForecastPipeline + DiagnosticPipeline
│   │   ├── assimilation.py # AssimilationPipeline + AssimilationForecastPipeline
│   │   ├── dlesym.py    #   DLESyMPipeline (HEALPix forecast variant)
│   │   └── stormscope.py #  StormScopePipeline (coupled GOES+MRMS nowcasting)
│   ├── report/          # Report package — aggregation, plotting, sections
│   │   ├── aggregation.py #  Score loading, time/ensemble aggregation, spread-skill
│   │   ├── plotting.py  #   Matplotlib + cartopy primitives
│   │   ├── sections.py  #   Section renderers (summary, curves, heatmaps, visualization)
│   │   └── main.py      #   generate_report orchestration
│   ├── assimilation.py  # DA plumbing — model loader, obs sources, runners
│   ├── scoring.py       # Scoring logic — metrics, data alignment, score loop
│   ├── work.py          # WorkItem, distribution, resume markers
│   ├── distributed.py   # Rank-ordered execution, logging setup
│   ├── models.py        # Model loading (prognostic + diagnostic)
│   ├── output.py        # OutputManager (zarr lifecycle)
│   ├── grids.py         # Grid-resolver helpers (goes, mrms, glm, gfs, arco)
│   └── data.py          # Predownloaded(Frame)Source (zarr/parquet → source)
└── pyproject.toml
```

Each source module has a specific scoped responsibilities:

<!-- markdownlint-disable MD013 -->
| Module | Responsibility |
|---|---|
| `pipelines/` | `Pipeline` ABC and built-in implementations (Forecast, Diagnostic, Assimilation, DLESyM, StormScope) |
| `assimilation.py` | DA plumbing — `load_assimilation`, `ObsSourceSet`, `AssimilationRunner`, analysis→tensor conversion |
| `report/` | Score aggregation, matplotlib plotting, section rendering, markdown report assembly |
| `scoring.py` | Metric instantiation, data loading/alignment, scoring loop |
| `work.py` | Define work units; parse ICs from config; distribute across ranks |
| `distributed.py` | Rank-ordered execution primitive; logging setup |
| `models.py` | Load prognostic/diagnostic models from config |
| `output.py` | Zarr store creation, validation, threaded writes, consolidation |
| `grids.py` | Hydra-instantiable grid resolvers (`goes_grid`, `mrms_grid`, `glm_grid`, `gfs_grid`, `arco_grid`) |
| `data.py` | `PredownloadedSource` / `PredownloadedFrameSource` — wrappers serving predownloaded zarr / parquet stores |
<!-- markdownlint-enable MD013 -->

### Pipeline interface

All inference logic is driven by a **Pipeline** — an abstract base class
(`src/pipelines/base.py`) that separates per-work-item inference from the
shared scaffolding (work iteration, output filtering, ensemble injection,
zarr writes).  Subclasses implement three methods:

| Method | Purpose |
|---|---|
| `setup(cfg, device)` | Load models, move to device, cache coordinate metadata |
| `build_total_coords(times, ensemble_size)` | Define the full zarr output coordinate system |
| `run_item(item, data_source, device)` | Yield `(tensor, coords)` pairs for one work item |

The base class `Pipeline.run()` handles everything else: iterating work
items, building the output variable filter, injecting the ensemble dimension,
and writing to the `OutputManager`.

Built-in pipelines (pass the fully qualified class path via `cfg.pipeline`):

- **`ForecastPipeline`** (`src.pipelines.forecast.ForecastPipeline`, the
  default) — prognostic rollout with optional diagnostic models.  Yields
  one output per lead-time step.
- **`DiagnosticPipeline`** (`src.pipelines.forecast.DiagnosticPipeline`) —
  diagnostic-only (no prognostic model).  Yields a single output per work
  item.
- **`AssimilationPipeline`** (`src.pipelines.assimilation.AssimilationPipeline`)
  — data-assimilation analysis scored directly against reanalysis.  Yields
  a single `lead_time=0` output per work item.
- **`AssimilationForecastPipeline`**
  (`src.pipelines.assimilation.AssimilationForecastPipeline`) — prognostic
  rollout initialized from a data-assimilation analysis (e.g. HealDA+FCN3).
- **`DLESyMPipeline`** (`src.pipelines.dlesym.DLESyMPipeline`) — coupled
  Earth-system forecast (atmos + ocean on different cadences).
- **`StormScopePipeline`** (`src.pipelines.stormscope.StormScopePipeline`) —
  coupled GOES/MRMS nowcasting.

### Custom pipelines

To add a custom inference loop, subclass `Pipeline` and set `pipeline` in
your Hydra config to the fully-qualified class name:

```python
# my_pipeline.py
from src.pipelines import Pipeline

class MyPipeline(Pipeline):
    def setup(self, cfg, device):
        ...
    def build_total_coords(self, times, ensemble_size):
        ...
    def run_item(self, item, data_source, device):
        ...
        yield x, coords
```

```yaml
# In your Hydra config override:
pipeline: my_pipeline.MyPipeline
```

Custom pipelines inherit the full shared machinery — distributed output
management, ensemble dimension handling, threaded zarr writes — for free.

## Testing

Install dev dependencies first (`uv sync --extra dev`). Unit and integration
tests live in `test/`. They use a lightweight `Persistence` model and
`Random` data source so no network access or model checkpoints are required.

```bash
# Run all tests (multi-GPU tests auto-skip when GPUs are unavailable)
pytest test/ -v

# Only the pure-logic work-distribution tests (no GPU needed)
pytest test/test_work.py -v

# Only multi-GPU tests (requires 2+ CUDA GPUs)
pytest test/test_multigpu.py -v
```

Multi-GPU tests launch `torchrun` as a subprocess, so each test exercises
the real distributed code path. They are guarded by `skipif` markers based
on `torch.cuda.device_count()` and will be collected but skipped when the
required number of GPUs is not present.
