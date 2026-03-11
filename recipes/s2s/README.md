# Subseasonal-to-Seasonal (S2S) Forecasting Recipe

Subseasonal-to-seasonal (S2S) forecasting bridges the gap between weather forecasts (up to 2
weeks) and seasonal forecasts (3-6 months). This recipe demonstrates how to run ensemble
forecasts for S2S timescales using Earth2Studio.
Reflecting the need for larger ensembles and longer forecast
timescales, the recipe supports multi-GPU distributed inference, along with parallel I/O to
efficiently save forecast data as it is generated, and permits only saving a subset of the
forecast outputs if storage space is a constraint. Among the models in Earth2studio, DLESyM
and HENS-SFNO are best-posed for S2S forecasting, so this recipe supports both models; it
can be extended to support other models as well.

Key features include:

- Multi-GPU inference
- Parallel, non-blocking I/O using zarr format
- Usage of one or more diagnostic models in the forecast pipeline
- Storage space reduction using regional output or coarsening
- Multi-GPU scoring of forecast outputs
- Capability to score using ECMWF AIWQ S2S metrics

<!-- markdownlint-disable MD033 MD013 -->
<div align="center">
<img src="https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/0.1.0/recipes/s2s/pnw_demo.png" width="90%" alt="Earth2Studio S2S Banner">
</div>
<!-- markdownlint-disable MD033 -->

## Prerequisites

### Software

To use the HENS component of this recipe, we need to download the pre-trained checkpoints,
kindly provided by the authors of the method. If desired, the following command can be used
to download the entire model registry from [NERSC][nersc-registry]:

<!-- markdownlint-disable MD028 -->
> [!WARNING]
> The entire model registry is 241 Gb.

> [!TIP]
> To download the individual checkpoints on an "as need" basis, just run the recipe with
> out executing this command.
> The checkpoints will get downloaded to the registry folder defined in the config from
> the HuggingFace mirror. (coming soon)
<!-- markdownlint-enable MD028 -->

```bash
wget --recursive --no-parent --no-host-directories \
--cut-dirs=4 --show-progress --reject="index.html*" \
--directory-prefix=s2s_model_registry \
https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
```

Using HENS in this recipe also requires the skill file for scaling the noise vector in the
breeding steps of the initial condition (IC) perturbation.
This can be downloaded with the following command:

```bash
wget --no-parent --no-host-directories --show-progress \
--directory-prefix=s2s_model_registry \
https://portal.nersc.gov/cfs/m4416/hens/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc
```

### Hardware

The HENS checkpoints have a larger memory footprint than other models supported in
Earth2Studio.
For the best experience the following hardware specifications are recommended:

- GPU: CUDA Compute Compatibility >8.0
- GPU Memory: >40Gb
- Storage: >512Gb NVMe SSD

## Quick Start

Start by installing the required packages with pip:

```bash
# If using uv package manager
uv sync
# Pip
pip install -r recipe-requirements.txt
```

To run an example ensemble forecast with DLESyM, run

```bash
# With uv
uv run python main.py --config-name=pnw_dlesym ncheckpoints=2
# With pip
python main.py --config-name=pnw_dlesym ncheckpoints=2
```

Here, we use a command-line config override to limit the number of model checkpoints in the
forecast to just two, for speed. More instructions on configuring the ensemble can be found
below in the [Configuration](#configuration) section.

## Documentation

The overall structure and usage of this recipe is similar to that of the HENS recipe, since
many of the features useful for HENS in medium-range forecasts are also useful for S2S
ensemble forecasting (whether with HENS or other models).

### Reference workflows

This recipe includes a few reference workflows which are configured
through the use of [Hydra YAML][hydra-docs] files.
These can get executed by directly invoking the `main.py` file.

#### 2021 Pacific Northwest Heat Wave

This example takes the heatwave that occurred in June 2021 over western North America as a
case-study for S2S forecasts. This event was extreme in both its duration and intensity,
setting several temperature records and causing severe wildfires and substantial damage.
Research has shown that S2S forecast models began indicating warm anomalies as far as about
3 weeks in advance, though none could capture the exact location and intensity of the
heatwave at that time ([Lin, Mo, & Vitart 2022][pnw-heatwave-s2s]).

Since S2S forecasts can potentially generate a lot of data with large ensembles and longer
rollouts, this example showcases how the output saved to disk can be limited to just a few
variables of interest subselected to a specific region ("cropbox") to cut down on storage
requirements.

Execute this example by running:

```bash
uv run main.py --config-name=pnw_dlesym
```

We also provide a corresponding config for the HENS-SFNO model to run for this case-study:

```bash
uv run main.py --config-name=pnw_sfno
```

The latter model is larger, and takes longer to complete a forecast. Both commands above
may be parallelized to accelerate execution with multiple GPUs, which is highly recommended
if possible. See the section on [Parallelism](#parallelism) for detailed instructions.

#### Global S2S scoring over extended periods

In developing and assessing S2S forecast models, it is important to evaluate their
performance across a range of conditions rather than just individual case-studies. Since
predictability of a given spatial scale rapidly decreases with lead time, it only makes
sense to evaluate S2S forecasts at coarser resolution if looking at global skill metrics; a
common evaluation resolution used by ECMWF and others is 1.5 degrees, which conveniently
also reduces the data volume generated by S2S ensembles.

This example generates S2S forecasts for the entire globe across a range of initial
conditions spanning a full year (2018 taken as a test year), so despite coarsening the data
before saving to disk it still generates a substantial amount of data (about 300 GB).
To make this run in a reasonable amount of time, multi-GPU execution is highly recommended.

<!-- markdownlint-disable MD028 -->
> [!WARNING]
> This example generates a lot of data, roughly 300 GB.
<!-- markdownlint-enable MD028 -->

On a system with `ngpu` GPUs available, run:

```bash
uv run torchrun --nproc_per_node=$ngpu --standalone main.py --config-name global_dlesym
```

More considerations on multi-GPU execution are discussed in the [Parallelism](#parallelism)
section.

#### Using diagnostic models with S2S forecast models

Similar to what is demonstrated in the HENS recipe, it is possible to use diagnostic models
"on top of" the core prognostic model used for S2S forecasting, provided all the input
variables of the diagnostic model are available in the prognostic model outputs.

For reference, we provide a config that diagnoses total precipitation using the outputs
of a HENS-SFNO ensemble, again subselected to the Pacific Northwest region for efficiency:

```bash
uv run python main.py --config-name pnw_sfno_precip.yaml
```

### Configuration

This recipe is highly customizable and extensible using the use of [Hydra][hydra-docs]. While
config items are documented with in-line comments in the `yaml` configuration files under
`cfg`, we describe a few key high-level configuration settings here:

#### Project Basics

The essential settings to identify a given forecast run are the `project` and `run_id`,
which are composed with the `output_path` to determine where all run outputs and logs
will be saved (`${output_path}/${project}_${run_id}`). Then, the core settings for
ensemble size and forecast duration are specified as:

```yaml
nsteps: 12         # number of steps to run the forecast
nperturbed: 4      # Number of perturbations applied to the initial condition for each model checkpoint
                    # Total ensemble size is nperturbed * ncheckpoints
ncheckpoints: 16 # number of model checkpoints to use
                # For DLESyM, ncheckpoints represents the total number of atmos/ocean model checkpoint pairs
batch_size: 4      # inference batch size
```

As mentioned in the in-line comments, this recipe creates ensembles by combining initial
condition perturbations with multiple model checkpoints to produce variability in the
ensemble. Each model checkpoint (set of trained weights) is paired with `nperturbed`
initial condition perturbations to produce a total of `ncheckpoints*nperturbed` ensemble
members. With DLESyM, there are 4 atmosphere and 4 ocean models available, leading to a
total of 16 possible checkpoints; HENS-SFNO has up to 29 model checkpoints available for
use in an ensemble.

To provide maximum flexibility and extensibility, we use a hydra defaults list to select
model and perturbation configs from the `cfg/forecast_model` and `cfg/perturbation`
folders. This allows one to add additional models/perturbations, if desired. To select a
specific combination, set the defaults in the top-level config file (see existing configs
for a reference):

```yaml
defaults:
    - forecast_model: dlesym
    - perturbation: gaussian
```

<!-- markdownlint-disable MD028 -->
> [!TIP]
> Not all perturbation methods in Earth2studio are compatible with
> the DLESyM model, currently. In particular, multi-timestep perturbation
> methods like (normal/hemispheric-centered) bred vectors will not work.
<!-- markdownlint-enable MD028 -->

Aside from these main config items, there are also options to choose a `data_source` and
configure the `file_output` and `scoring` behavior. Refer to the example config files
to see how these are set.

#### Output subselection

To reduce the storage requirements when running larger ensembles and longer rollouts, there
are several configurations available to subselect outputs if desired under the `file_output`
section of the config. First, one can only save variables of interest by specifying a list
in `file_output.output_vars`, which is a required config item. Also required is the output
resolution to save data at, which would be `latlon721x1440` for default full resolution or
`latlon121x240` for 1.5 degree resolution, commonly used in S2S scoring.

One can also choose to save outputs only over a specific region of the globe using the
optional `file_output.cropboxes` config setup. For each region, or "cropbox", you can
specify the name of the region and the min/max latitude/longitude values of it. Refer to
`pnw_dlesym.yaml` for a full cropbox example config.

#### Scoring forecast outputs

This recipe also provides mechanisms to score model outputs using a variety of metrics. The
scoring script is `score.py` and is run in much the same way `main.py` is used to run
a forecast. It is expected that the same config file is used when scoring a given forecast
as was used when running it.

The scoring script will read data from the output of the forecast run (`forecast.zarr` in)
the run's output directory and write scores to a new file `score.zarr`. There are two
general ways of scoring model outputs:

1. Using metrics defined in Earth2studio
2. Using metrics defined as part of the ECMWF AI Weather Quest competition

These can be configured under the `scoring` section of the config file:

```yaml
scoring:
    aiwq:
        variables: ["t2m"]
    variables: ["t2m", "z500"] # variables to score
    temporal_aggregation: "weekly" # temporal aggregation to apply before scoring
    metrics:
        crps:
            _target_: earth2studio.statistics.crps
            ensemble_dimension: "ensemble"
            reduction_dimensions: ["lat", "lon"]
            fair: True
    score_path: '${output_path}/${project}_${run_id}'
```

The `aiwq` subsection can be omitted if AI Weather Quest scoring is not desired. For the
more general `metrics` section, each metric can be instantiated using hydra, and they
will be computed for each variable specified in `scoring.variables` and saved to disk. As
weekly averaging is common in S2S scoring, we provide a mechanism to do so using the
`scoring.temporal_aggregation`. We can also take advantage of multi-GPU execution to speed
up the scoring process as before:

```bash
uv run torchrun --nproc_per_node=8 --standalone score.py --config-name global_dlesym
```

#### AI Weather Quest scoring

To use the [AI Weather Quest](https://aiweatherquest.ecmwf.int/) scoring routines, you must
have installed the [AI-WQ-package](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/index.html)
and have registered with AI Weather Quest. After you have registered, set your password in the
following environment variable before running any scoring routines:

```bash
export AIWQ_SUBMIT_PWD=<your_AI-WQ_password>
```

This will permit you to use ECMWF's official scoring metrics computed against the official
competition verification data. The AI Weather Quest focuses on RPSS scores of
weekly-averaged forecast outputs, computed against climatological quintiles.

There are a number of limitations associated with using `AI_WQ_package` for scoring:

- The initialization date of the forecast must be on a Thursday
- The initialization date must be on or after March 2025 to be able to download
  the corresponding verification data. This precludes using some ERA5 data sources.
- The scoring routines will download data (verification ERA5 data, weekly mean climatology,
  land-sea mask) when run.

The example config `global_dlesym_aiwq.yaml` demonstrates a config that can run and score
an ensemble appropriate for the AI Weather Quest surface temperature variable.

## Parallelism

This recipe uses the `physicsnemo.distributed.DistributedManager` to handle multi-GPU
execution. Example commands in this document make use of the `torchrun` launcher provided
by PyTorch as it is widely available and compatible. To turn a single-GPU run (for example, one
launched by `uv run python main.py --config-name pnw_dlesym.yaml`) into a multi-GPU run,
 add `torchrun` and the number of GPUs `$ngpu` you'd like to parallelize over:

```bash
uv run torchrun --nproc_per_node=$ngpu --standalone main.py --config-name global_dlesym
```

The `--standalone` argument indicate to `torchrun` that all GPUs are on the same
machine/node; if running over multiple nodes refer to the [`torchrun` guide](https://docs.pytorch.org/docs/stable/elastic/run.html)
for how to properly configure the process group.

`torchrun` is not required, and the `DistributedManager` will also work for
process groups initialized with SLURM (`srun`) or MPI (`mpirun`).

There are a number of considerations with multi-GPU execution worth highlighting here for
more advanced users and developers. Primarily, certain download and caching operations in
Earth2studio utilities are not thread-safe, and can cause uncontrolled behavior if not run
in a coordinated fashion:

- `load_default_package` of an Earth2studio model
- `fetch_data` of some Earth2studio data sources
- `__call__` of some perturbation methods, which implicitly call `fetch_data`
- Initialization of output files in an I/O backend

To handle these, a utility routine `run_with_rank_ordered_execution` is provided and used
in this recipe, which can wrap a function call and ensure that one rank (by default rank 0)
will run the function first, before the rest. This allows, for example, filesystem objects in a
cache or output directory to be created properly before other ranks access them.
Developers extending this recipe should use `run_with_rank_ordered_execution` for any
operations that might lead to race conditions; however it is important that when using it,
all ranks enter and leave the function the same number of times (that is, execution must be
load-balanced so all ranks enter and exit the barriers in the routine as expected).

For the parallel I/O capabilities in this recipe, it is important to consider the chunk
size used with the output zarr file. Execution is parallelized across GPUs over the ensemble
and time (initial condition) dimensions, so for safe concurrent writes we keep the chunk
size at 1 in those dimensions; I/O performance then depends on the chunking used for the
remaining dimensions. A good rule of thumb for zarr chunking is to aim for chunk size of
around 10MB, but other sizes may work well too in this case depending on the output grid
(resolution or cropbox size) data is being saved on.

## References

- [The 2021 Western North American Heatwave and Its Subseasonal Predictions][pnw-heatwave-s2s]
- [A Deep Learning Earth System Model for Efficient Simulation of the Observed Climate][dlesym-paper]
- [Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators][hens-paper]
- [Spherical Fourier Neural Operators](https://arxiv.org/abs/2306.03838)
- [NERSC Registry][nersc-registry]

<!--Common Links-->
[dlesym-paper]: https://arxiv.org/abs/2409.16247 "A Deep Learning Earth System Model for Efficient Simulation of the Observed Climate"
[hens-paper]: https://arxiv.org/abs/2408.03100 "Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators"
[nersc-registry]: https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
[hydra-docs]: https://hydra.cc/docs/intro/
[pnw-heatwave-s2s]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL097036 "The 2021 Western North American Heatwave and Its Subseasonal Predictions"
