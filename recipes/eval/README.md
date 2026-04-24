# Model Evaluation (Eval) Recipe

> 🏗️ **Under construction** — This recipe is still being developed; behavior,
> defaults, and documentation may change without notice.

General-purpose recipe for model verification and validation (V&V) using
Earth2Studio. Run any prognostic model (with optional diagnostic models)
across a set of initial conditions, distributing work across GPUs, and save
forecast outputs to zarr.

Key features:

- Works with **any** Earth2Studio prognostic or diagnostic model
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
  - [Resume after interruption](#resume-after-interruption)
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

### Zarr stores

Predownload creates the following stores in `<output.path>/`:

| Store | Dimensions | Contents |
|---|---|---|
| `data.zarr` | `(time, variable, <spatial...>)` | IC data (plus verification if same source) |
| `verification.zarr` | `(time, variable, <spatial...>)` | Only when verification source differs |

`main.py` automatically detects and reads from `data.zarr` when
`require_predownload=true` (the default).

### Resume after interruption

Progress is tracked per-timestamp via marker files.  If a predownload job is
killed (e.g. by a SLURM time limit), re-running with the same config skips
already-completed timestamps:

```bash
# Just re-run — resume is automatic
python predownload.py campaign=fcn3_2024_monthly
```

To recreate stores from scratch, set `predownload.overwrite=true`.

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
│   ├── pipeline.py      # Pipeline ABC + built-in pipelines
│   ├── scoring.py       # Scoring logic — metrics, data alignment, score loop
│   ├── report.py        # Report generation — aggregation, plotting, markdown
│   ├── work.py          # WorkItem, distribution, resume markers
│   ├── distributed.py   # Rank-ordered execution, logging setup
│   ├── models.py        # Model loading (prognostic + diagnostic)
│   ├── output.py        # OutputManager (zarr lifecycle)
│   └── data.py          # PredownloadedSource (zarr → DataSource)
└── pyproject.toml
```

Each source module has a specific scoped responsibilities:

| Module | Responsibility |
|---|---|
| `pipeline.py` | `Pipeline` ABC and built-in implementations (Forecast, Diagnostic) |
| `scoring.py` | Metric instantiation, data loading/alignment, scoring loop |
| `report.py` | Score aggregation, matplotlib plotting, markdown report assembly |
| `work.py` | Define work units; parse ICs from config; distribute across ranks |
| `distributed.py` | Rank-ordered execution primitive; logging setup |
| `models.py` | Load prognostic/diagnostic models from config |
| `output.py` | Zarr store creation, validation, threaded writes, consolidation |
| `data.py` | `PredownloadedSource` — DataSource wrapper for predownloaded zarr stores |

### Pipeline interface

All inference logic is driven by a **Pipeline** — an abstract base class
(`src/pipeline.py`) that separates per-work-item inference from the shared
scaffolding (work iteration, output filtering, ensemble injection, zarr
writes).  Subclasses implement three methods:

| Method | Purpose |
|---|---|
| `setup(cfg, device)` | Load models, move to device, cache coordinate metadata |
| `build_total_coords(times, ensemble_size)` | Define the full zarr output coordinate system |
| `run_item(item, data_source, device)` | Yield `(tensor, coords)` pairs for one work item |

The base class `Pipeline.run()` handles everything else: iterating work
items, building the output variable filter, injecting the ensemble dimension,
and writing to the `OutputManager`.

Two built-in pipelines are provided:

- **`ForecastPipeline`** (`pipeline=forecast`) — prognostic rollout with
  optional diagnostic models.  Yields one output per lead-time step.
- **`DiagnosticPipeline`** (`pipeline=diagnostic`) — diagnostic-only (no
  prognostic model).  Yields a single output per work item.

### Custom pipelines

To add a custom inference loop, subclass `Pipeline` and set `pipeline` in
your Hydra config to the fully-qualified class name:

```python
# my_pipeline.py
from src.pipeline import Pipeline

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
