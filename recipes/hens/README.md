# Earth2Studio Huge Ensembles (HENS) Recipe

This project implements a multi-checkpoint inference pipeline for large-scale
ensemble weather forecasting.
The pipeline enables parallel processing of multiple model checkpoints, providing a
flexible framework for uncertainty quantification in weather prediction systems.

A key application is the recovery of the HENS method, as described in
[Huge Ensembles Part I][hens-paper], which provides a calibrated ensemble forecasting
system based on Spherical Fourier Neural Operators (SFNO). HENS uses customized
bred vectors for initial condition
perturbations and multiple independently trained model checkpoints to represent
model uncertainty, specifically designed for better representation of extreme
weather events.

This approach enables unprecedented sampling of the tails of forecast
distributions for improved extreme event prediction.
It offers better uncertainty quantification and reduces the probability
of outlier events while requiring orders of magnitude fewer computational
resources than traditional physics-based models.
Key features include:

- Multi-GPU inference
- Reproducibility of individual inference batches
- Integration of one or more Diagnostic models in the pipeline
- Tropical cyclone tracking
- Regional output for efficient data storage

<!-- markdownlint-disable MD033 -->
<div align="center">

![Earth2Studio Banner](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/0.1.0/recipes/hens/hens_cyclone_tracks.gif)

</div>
<!-- markdownlint-enable MD033 -->

> [!TIP]
> For a simple example of running the HENS checkpoints, see the dedicated
> [example](https://nvidia.github.io/earth2studio/examples/11_huge_ensembles.html#sphx-glr-examples-11-huge-ensembles-py)
> for a basic introduction.

## Prerequisites

### Software

To run HENS, we need to download the pre-trained checkpoints, kindly provided
by the authors of the method.
The following command can be used to download the entire model registry from [NERSC][nersc-registry]:

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
--directory-prefix=hens_model_registry \
https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
```

This recipe also requires the skill file for scaling the noise vector in the breeding
steps of the initial condition (IC) perturbation.
This can be downloaded with the following command:

```bash
wget --no-parent --no-host-directories --show-progress \
--directory-prefix=hens_model_registry \
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

The provided JupyText python script `hens_notebook.py` is already set up to run
predictions for hurricane Helene track ensembles using two of the HENS
checkpoints.
This has comments through out the file documenting the steps used in this recipe.
`hens_notebook.py` will step users through different parts of configuring the workflow,
execution and also post processing.
Run ensemble inference using the following commands:

```bash
# uv package manager
uv run python hens_notebook.py
# system python
python hens_notebook.py
```

For notebook environment, convert the `hens_notebook.py` into a notebook with:

```bash
uv run jupytext --to notebook hens_notebook.py
```

Output results will be stored in the `outputs` folder.
For further configuration options, see the following section.

## Documentation

### Reference Workflows

This recipe includes a few reference workflows which are configured
through the use of [Hydra YAML][hydra-docs] files.
These can get executed by directly invoking the `main.py` file.

#### Hurricane Helene

[Hurricane Helene][helene-wiki] was a significant tropical cyclone that made landfall
in September 2024.
This workflow demonstrates ensemble inference for Helene, with the model
initialized approximately two and a half days before landfall.
In the configuration file `cfg/helene.yaml`.
Users are encouraged to look at this configuration file and adjust properties
accordingly.

The current configuration is set up for two checkpoints, two initial conditions, and
four ensemble members per checkpoint-IC pair, resulting in 16 ensemble members in
total.
Once the small configuration is verified, you can expand the number of checkpoints,
initial conditions and ensemble members per checkpoint-IC pair.

Execute the ensemble inference by running:

```bash
[mpirun -n XX] uv run main.py --config-name=helene.yaml
```

#### Storm Bernd

This example demonstrates the use of diagnostic models in ensemble forecasting,
using the case of Storm Bernd which caused [widespread flooding across central
Europe][bernd-wiki] in July 2021. The workflow showcases how multiple diagnostic
models can be chained together in a specific order to derive complex meteorological
variables.

In this particular case, the pipeline first calculates relative humidity using a
numerical model. This derived variable, along with other forecasted fields, serves as input
for the precipitation diagnostic model. The sequential processing is essential as each
diagnostic model's output can become an input for subsequent models in the chain.

Execute this example by running:

```bash
uv run main.py --config-name=storm_bernd.yaml
```

#### Batched Helene Ensemble

In large ensemble forecasting scenarios, storing all forecast fields may
be impractical due to storage constraints. However, you may want to reproduce
and store specific ensemble batches, particularly those containing interesting
cyclone tracks. This section explains how to reproduce specific batches of the
Helene ensemble.

##### Key Considerations

- Ensemble batches (including perturbations) are generated collectively,
  so individual ensemble members cannot be reproduced in isolation
- The entire batch must be reproduced together
- The configuration must match the original run, except for the file output section
- A fixed random seed is required to ensure identical initial condition perturbations
- Below, we list the steps to set up a config file for reproducing, but you can also
  use the `reproduce_helene_batches.yaml` config file.
- To demonstrate the benefits of reproducing individual batches, the reproduced field
  data will store more variables on a larger domain than the original run.

##### Steps to Reproduce a Batch

> [!Note]
> If you have deleted the outputs from the Helene run, re-run that recipe first to
generate data to compare against.

1. **Identify the Batch to Reproduce**
   - Determine the batch IDs you want to reproduce
   - Set these IDs in the `batch_ids_reproduce` parameter

2. **Configure the Reproduction Run**
   - Use the same configuration as `helene.yaml` for all parameters except file output
   - If the original run didn't specify a random seed, you can find it in:
     <!-- - The cyclone tracker file TODO: add to file-->
     - The field output files when using `KVBackend` or `XarrayBackend`
     - Other IObackends, including the netcdf4 backend do not provide the
       seed in the output file
   - If the original run didn't specify a random seed, different processes will
     generate different seeds

3. **Execute the Reproduction**

   ```bash
   python src/hens_run.py --config-name=reproduce_helene_batches.yaml
   ```

4. **Verify the Output**
   - Compare the output tracks with the selected batches of the original run
     using the scripts in the `test` folder:

     ```bash
     cd test
     uv run test_reprod.py
     ```

   - Running without any arguments will compare tracks in the `outputs/helene/cyclones`
     and `outputs_reprod/cyclones` directories.
     See the [test README](test/README.md) for more details on how to choose other directories
     or on how to compare individual files.

### Configuration

The pipeline follows Earth2Studio's modular design approach.
It is highly customizable using [instantiating through Hydra][hydra-docs].
This allows for flexible customization of components. Additional functionality
can be provided through custom functions and instantiated using Hydra, as
demonstrated in the configuration of the HENS perturbation.

#### Project Basics

Select a project name to uniquely identify your run. This prevents overwriting
files from previous runs and helps with organization. `nensemble`
refers to the number of ensemble members per (IC × number of checkpoints).
When used with the HENS perturbation, `nensemble` and `batch_size` have to be even
as the perturbation method is symmetric, that is the perturbation is added once and
subtracted once from the initial condition.

```yaml
project: 'project_name' # project name

start_times: ["2024-09-24", "2024-09-24 12:00:00"] # list of ICs
nsteps: 40     # number of forecasting steps
nensemble: 32  # ensemble size **per checkpoint, per IC**
batch_size: 2  # inference batch size
```

Alternatively, you can specify a block of equidistant initial conditions. In the
following example, individual initial conditions are spaced 48 hours apart:

```yaml
ic_block_start: "2022-02-08" # first IC
ic_block_end: "2022-02-09"   # upper boundary for ICs
ic_block_step: 48            # number of hours between individual ICs
```

#### Forecast Model

The forecast model configuration allows you to specify the model
architecture for forecasting.
It provides the path to the model checkpoints downloaded in [section 2](#2-prerequisites)
and sets the maximum number of checkpoints to be used (selected in alphabetical order).
Note that the choice of model architecture is not restricted to SFNO and
the default model package is used if `package` is not provided.

```yaml
forecast_model:
    architecture: earth2studio.models.px.SFNO   # forecast model class
    package: '/path/to/hens_model_registry'
    max_num_checkpoints: 1 # max number of checkpoints which will be used
```

#### Diagnostic Models

Diagnostic models can augment the generated data by deriving additional variables
from the forecast model output. Multiple diagnostic models can be integrated into
the pipeline, with their order of application being crucial for correct computation.

```yaml
diagnostic_models:
    rel_hum:
        _target_: earth2studio.enterprise.models.dx.DerivedVariables
        calculation_name: 'r_from_q_t_sp'
        levels: [850,500]
    precip:
        architecture: earth2studio.models.dx.PrecipitationAFNO
```

#### CorrDiff Downscaling Models

CorrDiff downscaling models can be integrated into the pipeline to generate high-resolution
ensemble outputs from the forecast model predictions. Each CorrDiff model is configured
in the `corrdiff_models` section. The output for each CorrDiff model is saved as a single
NetCDF file per batch, with each variable as a separate data variable in the file. The filename
includes the project name, CorrDiff model name, model package, initial condition, and batch
ID for traceability.

```yaml
corrdiff_models:
    wind:
        architecture: earth2studio.models.dx.CorrDiff
        package: /output/mini_package
        path: /output/
```

- `architecture`: The CorrDiff model class to use for downscaling.
- `package`: Path to the model weights or checkpoint.
- `path`: Output directory for the CorrDiff results.

The CorrDiff output NetCDF will have the same lead_time coordinate as the main HENS output, and
each variable will be a separate data variable in the resulting xarray Dataset.

#### IC Data Source

The IC data source configuration determines where the pipeline retrieves
its input data. Earth2studio supports various data sources, including
common online repositories and on-prem solutions. Users can also implement
custom data sources to meet specific requirements.

```yaml
data_source:
    _target_: earth2studio.data.NCAR_ERA5
```

#### IC Perturbation

The pipeline supports various perturbation methods for initial conditions.
While this example demonstrates the HENS perturbation configuration, any
perturbation method can be implemented. Some perturbation methods
require runtime information that is only available during pipeline execution.

```yaml
perturbation:
    skill_path: '/path/to/channel/specific/skill.nc' # downloaded in section 2
    noise_amplification: .35  # scaling amplitude of noise vector
    perturbed_var: 'z500'     # variable to perturb in seeding step
    integration_steps: 3      # vector breeding steps
```

#### Tropical Cyclone Tracking

Cyclone tracking can be triggered by providing the `cyclone_tracking` section in the
config.
The can be selected and configured in the config as shown below.
Tracking results are exported as netCDF files to the directory
specified under `path`.

```yaml
cyclone_tracking:
    path: 'outputs'
    tracker:
        _target_: earth2studio.models.dx.tc_tracking.TCTrackerWuDuan
        path_search_distance: 250
        path_search_window_size: 2
```

#### Writing Fields to Disk

The pipeline supports writing forecast fields to disk through the `file_output` section.
You can specify the output directory, select variables for export, and optionally
define regional boundaries using lat/lon cropboxes. The output `format` is determined
by the chosen IO backend class. Two backends, `KVBackend` and `XarrayBackend`, can be
instantiated without additional arguments and are automatically written to netCDF files
after inference. For `ZarrBackend` and `NetCDF4Backend`, chunking configuration is
crucial for write performance, and they require the `_partial_` flag set to `True`
since file names are determined during runtime.

Given the substantial data volume generated by large ensembles, the `cropboxes` parameter
enables restricting output to one or more lat/lon regions.

```yaml
file_output:
    path: 'outputs'  # directory to which files are written
    output_vars: ["t2m", 'u10m', 't850', 'z500'] # variables to write out
    format:               # io backend class
        _target_: earth2studio.io.NetCDF4Backend
        _partial_: True
        backend_kwargs:
            mode: 'w'
            diskless: False
            persist: False
            chunks:
                ensemble: 1
                time: 1
                lead_time: 1
    cropboxes:
        gulf_of_mexico:
            lat_min: 10
            lat_max: 40
            lon_min: 250
            lon_max: 310
```

### Execution

To execute the inference pipeline, tailor the config file to your needs and run:

```bash
uv run main.py --config-name=your_config.yaml
```

The pipeline supports multi-GPU and multi-node execution, with individual inferences
being fully independent. The parallelism strategy minimizes the overhead of loading
models to devices by pairing initial conditions with model checkpoints.

> [!Note]
> The maximum number of GPUs that can be effectively utilized equals the product of
the number of checkpoints and initial conditions. Additional GPUs will remain idle
during inference.

To run the pipeline in a multi-GPU or multi-node environment:

```bash
mpirun -n 2 uv run python main.py --config-name=your_config.yaml
```

> [!Note]
> CorrDiff output files are named as `corrdiff_{corrdiff_model}_pkg_{pkg}_{ic}_batch_{batch_id}.nc`
for clarity and traceability.

## References

- [Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators][hens-paper]
- [Spherical Fourier Neural Operators](https://arxiv.org/abs/2306.03838)
- [NERSC Registry][nersc-registry]

<!--Common Links-->
[hens-paper]: https://arxiv.org/abs/2408.03100 "Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators"
[nersc-registry]: https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
[hydra-docs]: https://hydra.cc/docs/advanced/instantiate_objects/overview/
[helene-wiki]: https://en.wikipedia.org/wiki/Hurricane_Helene
[bernd-wiki]: https://en.wikipedia.org/wiki/2021_European_flood_event
