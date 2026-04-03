# Model Evaluation (Eval) Recipe

> 🏗️ **Under construction** — This recipe is still being developed; behavior,
> defaults, and documentation may change without notice.

General-purpose recipe for model verification and validation (V&V) using
Earth2Studio. Run any prognostic model (with optional diagnostic models)
across a set of initial conditions, distributing work across GPUs, and save
forecast outputs to zarr.

Key features:

- Works with **any** Earth2Studio prognostic or diagnostic model
- Multi-GPU distributed inference via `torchrun` / SLURM / MPI
- Clean work-distribution with automatic load balancing across ranks
- Parallel, non-blocking zarr I/O via thread pool
- Ensemble support with configurable perturbation
- Hydra-based configuration with composable model configs

## Quick Start

Install the required packages:

```bash
# With uv
uv sync
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

# Distributed — parallelise across CPU workers
torchrun --nproc_per_node=8 --standalone predownload.py

# Match IC range and model to your planned eval config
python predownload.py model=dlwp \
    ic_block_start="2024-01-01" ic_block_end="2024-03-31" ic_block_step=24

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

## Configuration

All configuration lives under `cfg/` and uses [Hydra](https://hydra.cc/docs/intro/).

### Project and initial conditions

```yaml
project: eval_run
run_id: dlwp_deterministic

# Explicit list of ICs
start_times:
    - "2024-01-01 00:00:00"
    - "2024-01-02 00:00:00"

# Or a range (remove start_times first)
# ic_block_start: "2024-01-01 00:00:00"
# ic_block_end: "2024-03-31 00:00:00"
# ic_block_step: 24   # hours
```

### Model selection

Models are selected via Hydra defaults. To switch models, either override
on the command line or create a new YAML under `cfg/model/`:

```bash
python main.py model=dlwp
```

### Ensemble runs

Set `ensemble_size > 1` and provide a perturbation config:

```yaml
ensemble_size: 10
perturbation:
    _target_: earth2studio.perturbation.CorrelatedSphericalGaussian
    noise_amplitude: 0.05
```

### Output

```yaml
output:
    path: outputs/${project}_${run_id}
    variables: [t2m, z500]
    overwrite: true
    thread_writers: 4
    chunks:
        time: 1
        lead_time: 1
```

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
│   ├── default.yaml     # Main config
│   ├── predownload.yaml # Pre-download config (inherits default.yaml)
│   └── model/
│       └── dlwp.yaml    # DLWP model config
├── src/
│   ├── work.py          # WorkItem, build_work_items, distribute_work
│   ├── distributed.py   # Rank-ordered execution, logging setup
│   ├── models.py        # Model loading (prognostic + diagnostic)
│   ├── output.py        # OutputManager (zarr lifecycle)
│   └── inference.py     # Core inference loop
└── pyproject.toml
```

Each source module has a specific scoped responsibilities:

| Module | Responsibility |
|---|---|
| `work.py` | Define work units; parse ICs from config; distribute across ranks |
| `distributed.py` | Rank-ordered execution primitive; logging setup |
| `models.py` | Load prognostic/diagnostic models from config |
| `output.py` | Zarr store creation, validation, threaded writes, consolidation |
| `inference.py` | Fetch ICs, perturb, run model iterator, apply diagnostics, write |

## Testing

Unit and integration tests live in `test/`. They use a lightweight
`Persistence` model and `Random` data source so no network access or model
checkpoints are required.

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
