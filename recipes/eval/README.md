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

`predownload.py` must be run before `main.py`.  It fetches and caches all
initial condition data needed for inference, and optionally pre-fetches
reanalysis data for verification.  It accepts the **same config overrides** as
`main.py`, so the two commands stay in sync with no extra bookkeeping.

IC variables, lead times, and model step size are inferred automatically from
the model's `input_coords()` / `output_coords()`.

```bash
# Single process (login node or interactive session)
python predownload.py

# With a campaign config
python predownload.py +campaign=fcn3_2024_full

# Distributed — parallelise across CPU workers
torchrun --nproc_per_node=8 --standalone predownload.py

# Also pre-fetch ERA5 verification data for the full forecast window
# (variables taken from output.variables — only what will be scored)
python predownload.py predownload.verification.enabled=true
```

### Custom cache location

To redirect all I/O to a shared filesystem, set `predownload.cache_dir`.
Point the inference job at the same location via the `EARTH2STUDIO_DATA_CACHE`
environment variable that earth2studio already supports:

```bash
# Pre-download to shared path
python predownload.py predownload.cache_dir=/lustre/shared/e2s_cache

# Inference reads from the same location
EARTH2STUDIO_DATA_CACHE=/lustre/shared/e2s_cache torchrun \
    --nproc_per_node=$NGPU --standalone main.py
```

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
size, and output variables.  Apply with `+campaign=`:

```bash
# DLWP monthly deterministic
python main.py +campaign=dlwp_2024_monthly

# FCN3 full 56-member ensemble
python main.py +campaign=fcn3_2024_full
```

Both `main.py` and `predownload.py` accept the same `+campaign=` flag,
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

### Scoring (planned)

A scoring section is stubbed in the config for future implementation. The
scoring workflow will read forecast zarr outputs, compare against
verification data, and write skill metrics.

## Architecture

```bash
recipes/eval/
├── main.py              # Hydra entry point — distributed inference
├── predownload.py       # Hydra entry point — data pre-fetch
├── cfg/
│   ├── default.yaml     # Base config (shared defaults + predownload)
│   ├── predownload.yaml # Thin overlay (hydra.run.dir only)
│   ├── model/
│   │   ├── dlwp.yaml
│   │   └── fcn3.yaml
│   └── campaign/        # One file per evaluation campaign
│       ├── dlwp_2024_monthly.yaml
│       └── fcn3_2024_full.yaml
├── src/
│   ├── pipeline.py      # Pipeline ABC + built-in pipelines
│   ├── work.py          # WorkItem, distribution, resume markers
│   ├── distributed.py   # Rank-ordered execution, logging setup
│   ├── models.py        # Model loading (prognostic + diagnostic)
│   └── output.py        # OutputManager (zarr lifecycle)
└── pyproject.toml
```

Each source module has a specific scoped responsibilities:

| Module | Responsibility |
|---|---|
| `pipeline.py` | `Pipeline` ABC and built-in implementations (Forecast, Diagnostic) |
| `work.py` | Define work units; parse ICs from config; distribute across ranks |
| `distributed.py` | Rank-ordered execution primitive; logging setup |
| `models.py` | Load prognostic/diagnostic models from config |
| `output.py` | Zarr store creation, validation, threaded writes, consolidation |

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
