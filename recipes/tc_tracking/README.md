# Tropical Cyclone Tracking

<!-- markdownlint-disable MD033 -->
<div align="center">

![Helene Prediction](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/main/recipes/tc_tracking/helene_pred.gif)

</div>
<!-- markdownlint-enable MD033 -->

> [!Note]
> This recipe serves as an example of how to integrate
> custom downstream models into a forecasting pipeline.
> While TempestExtremes is used here for tropical cyclone
> tracking, the patterns demonstrated---such as asynchronous
> CPU/GPU execution, in-memory file handling, and diagnostic
> model integration---can be adapted for other downstream
> analysis tools.

This pipeline produces ensemble prediction and tracks
tropical cyclones (TCs) within the prediction. Here, we
demonstrate how to integrate downstream analysis tools like
[TempestExtremes][te-repo]
into a prediction workflow while avoiding the need to store
large atmospheric field data to disk, which can grow
prohibitively large and significantly impact inference speed.

Although TempestExtremes is not GPU-accelerated, its
execution time on CPU is typically shorter than the time
required for the GPU to produce the next prediction. We
therefore implement an asynchronous mode where the CPU
executes TempestExtremes on completed time steps while the
GPU continues producing subsequent predictions, resulting
in virtually no computational overhead from the tracking
process.

## Table of Contents

1. [Setting up the Environment](#1-setting-up-the-environment)
   - [1.1 Container (Recommended)](#11-container-recommended)
   - [1.2 UV Environment](#12-uv-environment)
2. [Configure and Execute Runs](#2-configure-and-execute-runs)
   - [2.1 Generate Ensemble](#21-generate-ensemble)
   - [2.2 Reproduce Individual Ensemble Members](#22-reproduce-individual-ensemble-members)
   - [2.3 Extract Reference Tracks from ERA5](#23-extract-reference-tracks-from-era5)
3. [Visualisation](#3-visualisation)
4. [TempestExtremes Integration](#4-tempestextremes-integration)
5. [Example Workflow](#5-example-workflow)
   - [5.1 Extract Baseline](#51-extract-baseline-optional)
   - [5.2 Produce Ensemble Forecasts](#52-produce-ensemble-forecasts)
   - [5.3 Analyse Tracks](#53-analyse-tracks)
   - [5.4 Reproduce Interesting Members](#54-reproduce-interesting-members-to-extract-fields)

## 1. Setting up the Environment

There are two ways to set up the environment: either
directly using a uv environment, or using uv to install
dependencies inside a container. Since flash-attention,
torch-harmonics and TempestExtremes all require compilation,
**the container approach is strongly recommended**.

### 1.1 Container (Recommended)

The [Dockerfile](./Dockerfile) in this directory builds an image
on top of the PhysicsNeMo container, which already ships PyTorch,
flash-attention and NATTEN. It adds TempestExtremes and installs
the recipe's dependencies with uv, including all supported models
(FCN3, SFNO, AIFS-ENS, the hourly interpolation models and
WeatherNext 2). The recipe code itself is not part of the image,
it is mounted at runtime.

```bash
cd recipes/tc_tracking
docker build -t tc_tracking:1.0 .
```

The image installs Earth2Studio from the latest commit of the
`1.0.0-rc` branch; rebuild with `docker build --no-cache` to pick
up newer commits. Building takes a few minutes, most of it for
compiling torch-harmonics. The build can be adjusted with build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `AIFS_VERSION` | `1` | AIFS-ENS version, `1` or `2`; both cannot be installed together |
| `MAX_JOBS` | from free RAM | Source files compiled in parallel |
| `NVCC_THREADS` | `5` | GPU architectures compiled in parallel per file |

Every parallel compile process needs up to 3 GB of RAM. By
default, `MAX_JOBS` is therefore derived from the memory available
at build time (at most one job per core). If the build fails with
`Killed` or `signal 9` in the log, or the machine becomes
unresponsive while compiling, compiling ran out of memory. In that
case, reduce the parallelism, for example for an image with
AIFS-ENS v2:

```bash
docker build --build-arg AIFS_VERSION=2 \
    --build-arg MAX_JOBS=1 --build-arg NVCC_THREADS=1 \
    -t tc_tracking:1.0-aifs2 .
```

The image builds on x86 and ARM machines (e.g. DGX Spark); on ARM,
build it on the machine itself.

[launch_tc_tracking.sh](./launch_tc_tracking.sh) starts the
container with the repository mounted at `/workspace` and the
shell in the recipe directory:

```bash
./launch_tc_tracking.sh                                    # interactive shell
./launch_tc_tracking.sh python main.py --config-name=helene.yaml
GPUS=0,1 IMAGE=tc_tracking:1.0-aifs2 ./launch_tc_tracking.sh
```

The script

- runs the container as the calling user, so all outputs and
  cache files are owned by that user on the host,
- caches model weights and data on the host in
  `~/.cache/earth2studio` (`E2S_CACHE`) and `~/.cache/huggingface`
  (`HF_CACHE`), so they are downloaded only once,
- mounts `~/.cdsapirc` if present, which `CDS_ERA5` needs,
- uses the host's shared memory (`--ipc=host`), where
  TempestExtremes stages its input. Docker's default of 64 MB is
  too small for that.

With `LOCAL_E2S=1`, the container uses the Earth2Studio code of the
mounted repository instead of the one installed in the image, to
develop the recipe and Earth2Studio together.

### 1.2 UV Environment

<!-- markdownlint-disable MD013 -->

> [!Note]
> When installing through uv, note that compiling
> `flash-attn` (required for AIFS-ENS) can take several
> hours. See the [troubleshooting notes][flash-attn-help]
> for more information. Consider using the container
> instead, where `flash-attn` is pre-installed in the
> base image.
>
> Additionally, you will need to install TempestExtremes
> yourself and point the configuration files to the
> TempestExtremes executables.

<!-- markdownlint-enable MD013 -->

> [!Important]
> Both `torch-harmonics` and `flash-attn` are built without
> build isolation and need to discover the installed PyTorch
> version, CUDA version, and include paths at compile time,
> so PyTorch must be present in the environment before
> running `uv sync`.

The dependencies are defined in `pyproject.toml`, with one extra
per model. AIFS-ENS v1 (`aifs`) and v2 (`aifs-v2`) depend on
incompatible versions of `anemoi-models`, so only one of them can
be installed. `all` installs everything with AIFS-ENS v1,
`all-aifs-v2` everything with AIFS-ENS v2. First install PyTorch
for your system, then sync the remaining dependencies:

```bash
uv pip install torch
uv sync --extra all        # or: uv sync --extra all-aifs-v2
```

This creates a uv environment in the `.venv` directory.
Optionally, activate the virtual environment:

```bash
source .venv/bin/activate
```

Additionally, you need to [install TempestExtremes][te-install]
and either add the executables for `DetectNodes` and
`StitchNodes` to the `PATH` or point the associated commands
in the configs to these executables.

#### Potential Issues with torch-harmonics

uv builds torch-harmonics while resolving the dependencies,
which can fail if other dependencies are not yet installed.
Additionally, a Git LFS error occasionally appears. To
address these issues, split the installation into a two-step
process and skip Git LFS:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all --no-install-package torch-harmonics
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all
```

## 2. Configure and Execute Runs

Runs are configured through YAML files located in
`tc_tracking/cfg`. To execute the pipeline:

```bash
python main.py --config-name=config.yaml
```

The pipeline has three operational modes:

- **`generate_ensemble`**: Generate an ensemble prediction
  and extract tropical cyclones.
- **`reproduce_members`**: Reproduce
  individual ensemble members to store atmospheric fields
  of interesting tracks.
- **`extract_baseline`**: Extract tropical
  cyclone tracks from historical reanalysis data (e.g. ERA5)
  for validation purposes.

**Parallelism:**

The two GPU-bound modes (`generate_ensemble` and
`reproduce_members`) can be executed in distributed
settings using Slurm, MPI, or torchrun, e.g.

```bash
torchrun --nproc-per-node=2 main.py --config-name=config.yaml
```

The batches (one IC and `batch_size` members each) are split
into contiguous blocks of nearly equal size, one per rank, so
the batches of an IC stay on few ranks. A rank without
batches still runs the setup. A batch that becomes unstable
(see the stability check below) goes into a queue shared by
all ranks. Once a rank has finished its own batches, it takes
batches from this queue until no rank has work left, so ranks
help each other with retries instead of idling. If a rank
crashes, the others would wait for it: `torchrun` and
`srun --kill-on-bad-exit` stop all ranks in that case.

The CPU-bound `extract_baseline` mode does not use
  `torchrun` and does not require a GPU. Instead, individual
  storms are distributed across CPU worker processes via the
  ``num_workers`` configuration entry (see
  [Section 2.3](#23-extract-reference-tracks-from-era5)),
  so a plain ``python main.py ...`` invocation suffices.

In the following we will explain how to configure the yaml
files for those three modes. You can find example configs
in `./cfg`.

### 2.1 Generate Ensemble

Here we explain how to configure a yaml file to produce an
ensemble prediction with either FCN3 or AIFS-ENS and track
tropical cyclones within the predictions. A full example can
be found at `cfg/helene.yaml`.

> [!Note]
> The pipeline currently supports probabilistic models
> (FCN3, AIFS-ENS) that generate ensemble spread
> internally. Deterministic models can be added as well,
> but require an additional perturbation method to produce
> meaningful ensemble members. When adding a deterministic
> model, register it in `load_model` in
> `src/modes/generate_tc_hunt_ensembles.py` **and** perturb
> the initial condition `x` in `run_inference` before the
> rollout.

**Basics:**

```yaml
project: 'helene'             # project name
mode: 'generate_ensemble'     # choose mode
model: 'aifs'                 # 'fcn3', 'aifs' or 'aifs-v2'
# optional: bitwise reproducible runs, see 2.2 (default: False)
deterministic: False
# optional: random seed, from which the seeds of all batches
# are drawn (does NOT control model-internal random state).
# If not set, rank 0 draws one and logs it.
random_seed: 7777
# optional: continue an interrupted or partly failed run,
# see "Resuming a Run" below (default: False)
resume: False
# optional: path to model weights
# (uses default package if omitted)
model_package: '/path/to/weights'
```

<!-- markdownlint-disable MD033 -->

**<a name="config_ens"></a>Ensemble Configuration**

<!-- markdownlint-enable MD033 -->

```yaml
# list of initial conditions
ics: ["2024-09-24 00:00:00", "2024-09-25 18:00:00"]
n_steps: 28          # number of forecast steps
ensemble_size: 16    # members per initial condition
batch_size: 2        # batch size for inference
```

**GPU Memory Requirements**: For a batch size of 1,
AIFS-ENS v1 peaks at about 38GB of GPU memory, which fits
on a 48GB GPU, and scales nearly linearly with increasing
batch size. FCN3 requires over 60GB for batch size 1, but
memory usage increases only marginally with larger batch
sizes, making it more memory-efficient for batched
inference.

**IO Configuration:**

```yaml
# data source for initial conditions
data_source:
    # choose any online or on-prem source
    _target_: earth2studio.data.NCAR_ERA5

# 'none', 'netcdf', or 'zarr'
# 'none' produces no field output
store_type: "netcdf"
# directory for all outputs
store_dir: "./outputs_${project}"
# field variables to write to file
out_vars: ['msl', 'z300', 'z500', 'u10m', 'v10m']
# optional, Zarr only: one shard per member, IC and variable
# (default: True), see below
zarr_shard_per_member: True
```

The data source can be any of the Earth-2 Studio data
sources or a custom data source implementation.

**Output Files**: Every batch writes its own store, so that
any rank can run any batch, including retries. NetCDF output
is one file per batch,
`<project>_<ic>_mems<first>-<last>.nc`. Zarr output is
written into one store per batch under `<project>_batches/`,
which rank 0 merges into `<project>.zarr` at the end of the
run. The merge moves no data, it hard-links the batch's
chunk files into the combined store. If a run is
interrupted, `mode: 'merge_zarr'` with the same config merges
the batch stores that are complete. Both formats contain a
variable `random_seed` with the seed each member was run
with, `-1` for members without valid fields (unstable with
all seeds, or not run).

Zarr chunks hold one member, IC, lead time and variable.
With `zarr_shard_per_member: True`, all lead times of a
member, IC and variable form one shard, i.e. one file. Zarr
would rewrite a shard on every partial write, so the shard
is kept in host memory until the member is complete and
then written at once, in the background. This needs about
`batch_size x len(out_vars) x (n_steps + 1) x 4 MB` of host
memory per rank at 0.25 degree. For very long rollouts or
little host memory, set it to `False`: every step is then
written as its own files right away.

**Storage Considerations**: Exercise caution when storing
field variables, as storage requirements grow rapidly
(proportional to number of ICs x ensemble size x forecast
length). Additionally, writing large volumes of data to disk
can significantly slow down inference. For FCN3 workflows,
consider storing only track data initially and using the
[reproducibility feature](#22-reproduce-individual-ensemble-members)
to regenerate fields for ensemble members with interesting
tracks.

**Cyclone Tracking with TempestExtremes** (optional)

<!-- markdownlint-disable MD013 -->

```yaml
cyclone_tracking:
    # run TempestExtremes while GPU produces next step
    asynchronous: True
    # variables required by TempestExtremes
    vars: ['msl', 'u10m', 'v10m', 'z300', 'z500']
    # DetectNodes command
    detect_cmd: >-
      DetectNodes --verbosity 0 --mergedist 6
      --closedcontourcmd
      _DIFF(z300,z500),-58.8,6.5,0.;msl,200.,5.5,0
      --searchbymin msl --outputcmd
      msl,min,0;_VECMAG(u10m,v10m),max,5;height,min,0
    # StitchNodes command
    stitch_cmd: >-
      StitchNodes
      --in_fmt lon,lat,msl,wind_speed,height
      --range 8.0 --mintime 54h --maxgap 4
      --out_file_format csv --threshold
      wind_speed,>=,10.,10;lat,<=,50.,10;lat,>=,-50.,10;height,<=,150.,10
    # path to orography data; local NC file or URL
    # (only needed if height is used)
    orography_path: https://huggingface.co/nvidia/fourcastnet3/blob/main/orography.nc
    # keep raw field data used for tracking
    keep_raw_data: False
    # scratch directory for intermediate files (null = auto-detect /dev/shm, fall back to disk)
    scratch_dir: null
    task_timeout_seconds: 120   # timeout for tracking tasks
    print_te_output: False      # print TE output to terminal
    max_workers_per_rank: 8     # max concurrent TE subprocesses per rank
    # run DetectNodes every n steps (null = once per rollout)
    detect_chunk_steps: null
```

<!-- markdownlint-enable MD013 -->

For more information on [asynchronous mode](#async), see
the dedicated section below.

**Debugging TempestExtremes Commands**: To debug
`detect_cmd` and `stitch_cmd`, set `print_te_output` to
`True` to see TempestExtremes logs. However, iterating
through full inference runs for debugging can be cumbersome.
For more efficient debugging, set `keep_raw_data` to `True`
to write field snapshots as NetCDF files to
`${store_dir}/cyclone_tracks_te/raw_data`. You can optionally
set `scratch_dir` to a persistent path if you want intermediate
files to survive between runs. You can then debug the
TempestExtremes commands directly from the command line using
these files. Once the commands work
correctly, copy them back to the configuration file.
When running TempestExtremes from the command line, you
need to specify file paths explicitly (see
[TempestExtremes documentation](https://climate.ucdavis.edu/tempestextremes.php)):
use `--in_data_list` and `--out_file_list` for
`DetectNodes`, and `--in` and `--out` for `StitchNodes`.

**Long Rollouts**: By default, the tracking variables of all
steps are kept in host memory and written to `scratch_dir`
(in memory as well, if it is `/dev/shm`) before DetectNodes
runs. With `detect_chunk_steps`, DetectNodes runs on chunks
of that many steps while the model rolls out, and StitchNodes
on the nodes of all chunks at the end. DetectNodes works on
one time step at a time, so the tracks are the same, but
memory no longer grows with the rollout length. For rollouts
of months, also set `scratch_dir` to a directory on disk and,
for Zarr output, consider `zarr_shard_per_member: False`.

**Stability Check** (optional)

FCN3 can perform stable rollouts on subseasonal timescales
(weeks to months), but occasionally experiences numerical
instabilities on seasonal timescales. A simple stability
check has been implemented to detect and handle these cases:

**Key Points**:

- **Applicable to**: FCN3 for (sub)seasonal forecasts.
  The long-term behaviour of AIFS-ENS is still under
  investigation.
- **Detection mechanism**: Instabilities manifest in global
  averages of field variables, which typically build up over
  ~10 days before diverging rapidly.
- **Implementation**: Provide a list of variables to monitor
  and a threshold for each. Global averages are computed at
  initialisation and at each time step. If any variable's
  global average exceeds its threshold, the simulation of
  the batch is interrupted and queued for a retry with a new
  random seed, which any rank can take. A retried batch keeps
  its member numbers, and its output records the seed it was
  run with.
- **Retries**: At most `max_retries` (default 3) per batch.
  The retry seeds are drawn from the same generator as the
  original seeds, so all seeds of a run are unique and depend
  only on `random_seed`. A batch that stays unstable with all
  its seeds is logged as failed, writes no track files, and
  its fields are marked with `random_seed = -1`.
- **Limitation**: This is a basic implementation. More
  advanced detection methods that don't require 10 extra
  simulation days would be beneficial improvements.

```yaml
stability_check:
    variables: ['t1000', 'u250', 'v850']
    thresholds: [10, 10, 10]
    max_retries: 3  # then the batch is logged as failed
```

**Resuming a Run** (optional)

A run records its settings in `<project>_run.json` and every
finished batch in `<project>_status/` (both in `store_dir`).
Rerunning the same config with `resume: True` skips the
batches that are done (fields written and all track files
present), reruns the missing ones, and continues failed
batches with their next retry seed. For the latter, raise
`max_retries`; the seeds of all other batches stay the same.
The settings that define the batches and their seeds
(`random_seed`, `model`, ICs, `ensemble_size`, `batch_size`,
`n_steps`, `out_vars`, `store_type`) must not change; the
number of ranks may. Without `resume`, a run starts from
scratch and replaces the outputs of an earlier run of the
project.

To stop a run, send SIGTERM (`kill <pid>` of `torchrun` or
of the Python process, `scancel` on Slurm): the run exits
and removes its scratch files. Ctrl-C does not work, as a
dependency of Earth2Studio replaces Python's SIGINT handler.

**Hydra Configuration** (optional)

```yaml
hydra:
    job:
        name: ${project}
    run:
        dir: ${store_dir}/${project}_logs/${now:%Y-%m-%d_%H-%M-%S}
```

This propagates the project name and output directory to
Hydra's logging system. You will most likely never need to
modify this section.

### 2.2 Reproduce Individual Ensemble Members

A common workflow is to run ensemble forecasts without
storing atmospheric fields, since track data is approximately
five orders of magnitude smaller than the associated field
data. This enables sending results of hundreds of
hypothetical cyclone seasons as an email attachment. However,
for ensemble members with particularly interesting tracks,
examining the atmospheric fields can help understand the
storm's evolution.

FCN3 and AIFS-ENS draw their noise from a random state that
is seeded per batch, enabling exact reproduction of individual
ensemble members.

For AIFS-ENS, set `deterministic: True` in both the original
and the reproducing run. Its graph layers otherwise sum in a
varying order on the GPU, and the resulting round-off
differences grow to a few hPa within days. Deterministic GPU
kernels cost roughly 10% runtime; FCN3 is reproducible
without them.

> [!Warning]
> AIFS-ENS v2 is not bitwise reproducible, even with
> `deterministic: True`. Repeated runs with the same seed give
> one of two slightly different results (a few Pa in `msl`
> after 18 h, growing to a few hPa), so reproduced tracks may
> differ slightly from the original ones. The cause is not
> known yet.

Note that
[reproducibility is challenging](https://arxiv.org/abs/1605.04339)
and only works reliably when using the same environment on
the same system to produce the predictions.

To reproduce a run, you need:

- Configuration file from the original run (ideally)
- Ensemble member ID and random seed for all members to be
  reproduced (both provided in the track CSV filename, the
  seed also in the `random_seed` variable of the field
  output). For a member that was retried after an
  instability, this is the seed of the retry.
- Batch size (also provided in the track CSV filename)

**Configuration Changes:**

The following describes what to change in the configuration
file to reproduce ensemble members. You can compare
`cfg/helene.yaml` and `cfg/reproduce_helene.yaml` as
reference examples. In the original configuration, consider
changing the project name to avoid overwriting files, and
set the mode to:

```yaml
mode: 'reproduce_members'
```

In the [**ensemble configuration section**](#config_ens),
remove the `ics` variable and replace it with a list of
ensemble members to reproduce. For each ensemble member,
provide the initial condition, member ID, and random seed:

```yaml
reproduce_members: [
                    ['2024-09-24 00:00:00',  5, 2148531792],
                    ['2024-09-24 00:00:00',  8, 1342799488],
                    ['2024-09-24 12:00:00',  4, 2528887706],
                    ['2024-09-24 12:00:00',  5, 2528887706],
                    ['2024-09-24 12:00:00', 13, 1968734508],
                   ]
```

The example above would reproduce members 5 and 8 from the
ensemble starting at midnight, and members 4, 5, and 13 from
the ensemble starting at midday.

**Batch Reproduction Behaviour**: Note that the internal
random state can only be set for a complete batch. While
individual members within a batch will produce independent
predictions, they share the same random seed. This means a
full batch must always be reproduced. For example, with a
batch size of 2 in the example above, the run would reproduce
members 4, 5, 8, and 9 from the midnight ensemble, and
members 4, 5, 12, and 13 from the midday ensemble.
Use the same `batch_size` and `ensemble_size` as in the
original run: they define the batches, and a member's noise
depends on the batch it was generated in.

### 2.3 Extract Reference Tracks from ERA5

[IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
(International Best Track Archive for Climate Stewardship)
is considered the gold standard for tropical cyclone
observation data. However, it represents point observations
that a global model at quarter-degree resolution cannot
reproduce accurately. Therefore, a meaningful validation of
model predictions must be performed against ERA5. To do so,
we first have to extract tropical cyclone tracks directly
from reanalysis data (such as ERA5) and compare the
predictions against these reference tracks. This pipeline
demonstrates how to extract reference tracks for named
storms, using IBTrACS to identify the temporal windows and
approximate locations of storms of interest.

**Basics:**

```yaml
project: 'reference_tracks'
mode: 'extract_baseline'
```

**Cases:**

Specify which storms to extract by providing their name,
year, and basin for each case:

```yaml
cases:
    hato:
        year: 2017
        basin: west_pacific
    helene:
        year: 2024
        basin: north_atlantic
```

The script queries the IBTrACS database to determine when
each storm was active, retrieves the corresponding
atmospheric data from the specified data source, and applies
TempestExtremes to extract tropical cyclone tracks. Since
multiple storms may be active globally at any given time,
a second matching step compares all extracted tracks against
the IBTrACS ground truth to identify the correct track for
each named storm. A track is matched at the first IBTrACS
observation time for which the extracted storm position lies
within ``max_dist`` metres of the IBTrACS reference position.
The ``max_dist`` setting is optional and defaults to
``300000`` (300 km) if omitted:

```yaml
max_dist: 300000   # optional; max. distance (in metres) between
                   # extracted track and IBTrACS reference at the
                   # matching time step; defaults to 300000 (300 km)
```

**IO Configuration:**

**IBTrACS Data:**

The pipeline requires the IBTrACS dataset in CSV format.
Specify a local path in the configuration:

```yaml
ibtracs_source_data: "./aux_data/ibtracs.ALL.list.v04r01.csv"
```

If the file does not exist at the configured path, it is
downloaded automatically from NCEI (~300 MB). The download
only occurs once; subsequent runs use the cached file.

**Reanalysis Data:**

- For extracting multiple storms, we recommend using an
  on-premise dataset to avoid lengthy download times.
- The data source can be configured using the standard
  approach (as shown in the ensemble generation section).
- Alternatively, different data sources can be specified for
  different year ranges, which is particularly useful when
  datasets are partitioned into
  training/testing/out-of-sample periods.
- The example below shows how to configure different ERA5
  sources for different time periods:

```yaml
store_dir: "./outputs_${project}"
ibtracs_source_data: "./aux_data/ibtracs.ALL.list.v04r01.csv"

data_source:
    era5_train:
        years: "1980-2016"
        source:
            _target_: earth2studio.data.NCAR_ERA5

    era5_oos:
        years: "2017-2022"
        source:
            _target_: earth2studio.data.CDS_ERA5
```

**Parallel Extraction:**

`extract_baseline` is CPU-bound (no GPU required) and
distributes individual storms across worker processes via
a [`ProcessPoolExecutor`][ppe-docs]. Set the desired number
of workers in the configuration:

```yaml
num_workers: 2     # 1 = serial; capped at len(cases)
```

Each worker handles one storm at a time and writes its own
per-storm CSV file independently, so output is
deterministic and no inter-process communication is needed.
Serial runs (`num_workers: 1`) skip the pool entirely and use the
same code path inline. If `num_workers` exceeds the number
of configured cases, it is silently capped to `len(cases)`
and a warning is logged.

## 3. Visualisation

Two [JupyText](https://jupytext.readthedocs.io/) notebook scripts are
provided in the recipe root for analysing and visualising tropical cyclone
tracking results:

- **`plot_tracks_analysis.py`**: Ensemble track analysis
  including spaghetti plots (trajectory visualisation),
  absolute and relative intensity metrics (wind speed,
  MSLP), comparisons against ERA5 reference tracks and
  IBTrACS observations, extreme value statistics, and
  error moment analysis over lead time.

- **`plot_tracks_n_fields.py`**: Create animated
  visualisations of storm tracks overlaid on atmospheric
  field data.

Both scripts can be run as plain Python files or converted to Jupyter
notebooks via JupyText. From the recipe root:

```bash
jupytext --to notebook plot_tracks_analysis.py
jupytext --to notebook plot_tracks_n_fields.py
jupyter notebook plot_tracks_analysis.ipynb
```

The recipe root must be the working directory so that `src.plt.*` imports
resolve correctly. Helper modules (`analyse_n_plot.py`, `data_handling.py`,
`plotting_helpers.py`) live in `src/plt/`.

## 4. TempestExtremes Integration

This pipeline uses TempestExtremes to demonstrate how to
integrate custom downstream analysis tools into a prediction
workflow. TempestExtremes requires file-based input, does not
run on GPU, and cannot be installed as a Python
library---making it a useful reference case, as tools with
fewer constraints should be simpler to integrate. The current
implementation avoids writing atmospheric field data to disk
and reading it back, which would otherwise slow down
inference significantly and become prohibitive given the
large data volumes involved.

**General Integration Approach:**

The `TempestExtremes` class in `src/tempest_extremes.py`
collects the forecast steps of a batch in memory. Once the
batch is complete, it writes one input file per ensemble
member and runs the TempestExtremes command-line tools
DetectNodes and StitchNodes on it in a subprocess. The track
files are written by TempestExtremes directly to the output
directory.

Since TempestExtremes requires file-based input, we must
provide atmospheric data through files. To avoid the
overhead of writing and reading large volumes of data
to/from disk, we write temporary files to `/dev/shm`
(shared memory filesystem), effectively creating in-memory
files that TempestExtremes can access while avoiding
physical disk I/O. Each tracker uses its own subdirectory,
so several runs can share a node. In Docker, `/dev/shm`
is limited to 64MB unless the container is started with
`--ipc=host` or a larger `--shm-size`, as
`launch_tc_tracking.sh` does.

<!-- markdownlint-disable MD033 -->

**<a name="async"></a>Asynchronous Mode**

<!-- markdownlint-enable MD033 -->

On many platforms, executing TempestExtremes on the CPU is
faster than producing the prediction on the GPU. We
therefore implemented an asynchronous mode where a CPU
thread executes TempestExtremes on completed time steps
while the GPU continues producing the next prediction,
enabling tracking with virtually no overhead.

**Implementation Details**:

- In asynchronous mode, each ensemble member in a batch is
  tracked in its own background thread, allowing parallel
  tracking across the batch. A failure of TempestExtremes
  stops the run at the next batch or at the end of the
  run, with the output of TempestExtremes.
- If TempestExtremes execution becomes slower than GPU
  prediction (which might occur with AIFS-ENS on newer GPUs
  like B200), consider increasing the batch size. GPU
  prediction time scales approximately linearly with batch
  size, while the time for a single CPU thread to process
  one member remains constant. There is no backpressure:
  members waiting for tracking pile up in `/dev/shm`
  (about 0.7 GB per member for the Helene setup).
- **Concurrency control**: On multi-GPU nodes, multiple
  ranks share the same CPUs. Use `max_workers_per_rank` to
  limit the number of members tracked in parallel per
  rank. A good default is
  `cpu_count / gpus_per_node` (e.g. 8 on a 64-core node
  with 8 GPUs). If omitted, the value is derived
  automatically from `cpu_count / gpu_count`.
- If you encounter issues, first verify whether they also
  occur with asynchronous mode disabled before debugging
  further.

## 5. Example Workflow

This section walks through an example workflow demonstrating
the pipeline's core functionality: generating an ensemble
prediction, reproducing individual members, visualising
results, and comparing predicted tracks against reference
tracks extracted from ERA5 data.

### 5.1 Extract Baseline (Optional)

This step extracts reference tracks from ERA5 reanalysis for
comparison with forecast predictions. Note that downloading
the ERA5 field data for Hato and Helene is required and can
take some time. If it takes too long, you can skip this step
and use the pre-computed reference tracks provided in
`./aux_data` instead.

The IBTrACS data file is downloaded automatically on first
use. Extract the baseline tracks (no `torchrun` / GPU
needed; storms are spread across `num_workers` CPU worker
processes as configured in the YAML):

```bash
python main.py --config-name=extract_era5.yaml
```

This should produce a folder called
`outputs_reference_tracks/` containing two reference track
files:

- `reference_track_hato_2017_west_pacific.csv`
- `reference_track_helene_2024_north_atlantic.csv`

### 5.2 Produce Ensemble Forecasts

Run the forecast loop for Helene using FCN3 and for Hato
using AIFS-ENS:

```bash
python main.py --config-name=helene.yaml
python main.py --config-name=hato.yaml
```

This should produce two output folders: `outputs_helene` and
`outputs_hato`, each containing tracked tropical cyclone
trajectories.

### 5.3 Analyse Tracks

Visualise the results using the notebook
`plot_tracks_analysis.py` (or convert to `.ipynb` via jupytext).

**For Helene**, configure the first cell of the notebook
with:

```python
case = 'helene_2024_north_atlantic'
pred_track_dir = '/path/to/outputs_helene/cyclone_tracks_te'
tru_track_dir = '/path/to/outputs_reference_tracks'
# If you skipped baseline extraction, use:
# tru_track_dir = '/path/to/tc_tracking/aux_data'
```

**For Hato**, use:

```python
case = 'hato_2017_west_pacific'
pred_track_dir = '/path/to/outputs_hato/cyclone_tracks_te'
tru_track_dir = '/path/to/outputs_reference_tracks'
# If you skipped baseline extraction, use:
# tru_track_dir = '/path/to/tc_tracking/aux_data'
```

### 5.4 Reproduce Interesting Members to Extract Fields

Suppose that after conducting the above analysis you want
to take a closer look at ensemble members 0, 8, 9, and 12
of the Helene prediction. Since the forecast was produced
with FCN3, we can reproduce these members and store the
atmospheric fields for detailed analysis. For AIFS-ENS,
both runs need `deterministic: True`.

**Reproduce with Field Storage**:

1. In `cfg/reproduce_helene.yaml`, verify that the random
   seeds in the `reproduce_members` variable match the seeds
   from the track filenames in
   `outputs_helene/cyclone_tracks_te`. If they do not,
   update the seeds in the config accordingly.
2. Double-check that the batch size, ensemble size, and data
   source are identical to the original run.
3. Execute the pipeline:

   ```bash
   python main.py --config-name=reproduce_helene.yaml
   ```

4. This should produce a folder `outputs_reproduce_helene`
   containing both track files and one NetCDF file per
   reproduced batch with the atmospheric fields, e.g.
   `reproduce_helene_2024-09-24T00.00.00_mems0008-0009.nc`
   for members 8 and 9.

**Visualise Tracks and Fields**:

Use the notebook `plot_tracks_n_fields.py` to
create animated visualisations:

1. Configure the file paths:

   ```python
   field_data = (
       '/path/to/outputs_reproduce_helene'
       '/reproduce_helene_2024-09-24T00.00.00_mems0008-0009.nc'
   )
   track_dir = (
       '/path/to/outputs_reproduce_helene'
       '/cyclone_tracks_te'
   )
   ```

2. Select visualisation parameters:

   ```python
   variable = 'wind_speed'  # or 'msl', etc.
   ensemble_member = 8      # from reproduced subset
   region = 'gulf_of_mexico'
   ```

3. Run the notebook to generate an interactive animation
   where you can step through time.

**Verification**: To verify that reproducibility worked
correctly, point `track_dir` to the original Helene run
(`outputs_helene/cyclone_tracks_te`) and confirm that the
tracks are identical.

<!-- markdownlint-disable MD013 -->

[te-repo]: https://github.com/ClimateGlobalChange/tempestextremes
[te-install]: https://github.com/ClimateGlobalChange/tempestextremes?tab=readme-ov-file#installation-via-cmake-recommended
[flash-attn-help]: https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html#flash-attention-has-long-build-time-for-aifs-models
[ppe-docs]: https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor

<!-- markdownlint-enable MD013 -->
