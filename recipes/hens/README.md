# Earth2Studio Huge Ensembles (HENS) Recipe

This project implements a multi-checkpoint inference pipeline for large-scale
ensemble weather forecasting. The pipeline enables parallel processing of multiple
model checkpoints, providing a flexible framework for uncertainty quantification
in weather prediction systems.

A key application is the recovery of the HENS method, as described in
[Huge Ensembles Part I][hens-paper], which provides a calibrated ensemble forecasting
system based on Spherical Fourier Neural Operators (SFNO). HENS uses customised
bred vectors for initial condition
perturbations and multiple independently trained model checkpoints to represent
model uncertainty, specifically designed for better representation of extreme
weather events.

This approach enables unprecedented sampling of the tails of forecast
distributions for improved extreme event prediction.
It offers better uncertainty quantification and reduces the probability
of outlier events while requiring orders of magnitude fewer computational
resources than traditional physics-based models.

Beyond the core HENS functionality, the pipeline includes additional features
that are demonstrated through various configuration files, as detailed in
[section 5](#5-reference-workflows). Key features include:

- Multi-GPU inference
- Reproducibility of individual inference batches
- Integration of one or more Diagnostic models in the pipeline
- Tropical cyclone tracking
- Regional output for efficient data storage

> [!TIP]
> For a simple example of running the HENS checkpoints, see the dedicated
> [example](https://nvidia.github.io/earth2studio/examples/11_huge_ensembles.html#sphx-glr-examples-11-huge-ensembles-py)
> for a basic introduction.

## Prerequisites

### Software

To run HENS, we need to download the pre-trained checkpoints, kindly provided
by the authors of the method.
The following command can be used to download the entire model registry (241 Gb) from [NERSC][nersc-registry]:

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

- GPU: CUDA Compute Compatability >8.0
- GPU Memory: >40Gb
- Storage: >512Gb NVMe SSD

## Quick Start

Add a quick start to get the user running with the user up and running as fast as
possible, e.g.

Start by installing the required packages with pip:

```bash
pip install -r requirements.txt
```

Or set up a uv virtual environment:

```bash
uv sync
```

Run the template

```bash
uv run python main.py

>> Hello
>> 0.6.0rc0

uv run python main.py print.hello False

>> 0.6.0rc0
```

## Documentation

### Configuration

The pipeline follows earth2studio's modular design approach. It is highly
customisable via [instantiating through Hydra][hydra-docs].
This allows for flexible customisation of components. Additional functionality
can be provided through custom functions and instantiated via Hydra, as
demonstrated in the configuration of the HENS perturbation.

#### Project Basics

Select a project name to uniquely identify your run. This prevents overwriting
files from previous runs and helps with organisation. Note that `nensemble`
refers to the number of ensemble members per (IC × number of checkpoints).
When used with the HENS perturbation, `nensemble` and `batch_size` have to be even
as the perurbation method is symmetric, ie the perturbation is once added and once
subtracted from the IC.

```yaml
project: 'project_name' # project name

start_times: ["2024-09-24", "2024-09-24 12:00:00"] # list of ICs
nsteps: 40     # number of forecasting steps
nensemble: 32  # ensemble size **per checkpoint, per IC**
batch_size: 2  # inference batch size
```

Alternatively, you can specify a block of equidistant ICs. In the following
example, individual ICs are spaced 48 hours apart:

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
perturbation method can be implemented. Note that some perturbation methods
require runtime information that is only available during pipeline execution.
In such cases, set `_partial_` to `True` and complete the instantiation within
the Python script.

```yaml
perturbation:
    _target_: hens_perturbation.HENSPerturbation
    _partial_: True
    skill_path: '/path/to/channel/specific/skill.nc' # downloaded in section 2
    noise_amplification: .35  # scaling amplitude of noise vector
    perturbed_var: 'z500'     # variable to perturb in seeding step
    integration_steps: 3      # vector breeding steps
```

#### Tropical Cyclone Tracking

Cyclone tracking can be triggered by providing the `cyclone_tracking` section in
the config. The pipeline utilises `CycloneTrackingVorticity` model.
While alternative trackers are available in earth2studio, they require synchronisation
to enable seamless switching between different approaches (this feature is currently
in development). The tracking results are exported as CSV files to the directory
specified by `out_dir`. The tracker supports regional analysis through the
`cropboxes` parameter defined in the `file_output` section.

```yaml
cyclone_tracking:
    out_dir: './outputs'
```

#### Writing Fields to Disk

The pipeline supports writing forecast fields to disk through the `file_output` section.
Users can specify the output directory, select variables for export, and optionally
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
    path: './outputs'  # directory to which files are written
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

To execute the pipeline, tailor the config file to your needs and run:

```bash
python hens.py --config-name=your_config.yaml
```

The pipeline supports multi-GPU and multi-node execution, with individual inferences
being fully independent. The parallelism strategy minimises the overhead of loading
models to devices by pairing initial conditions with model checkpoints.

> [!Note]
> The maximum number of GPUs that can be effectively utilised equals the product of
the number of checkpoints and initial conditions. Additional GPUs will remain idle
during inference.

To run the pipeline in a multi-GPU or multi-node environment:

```bash
mpirun -n 2 python hens.py --config-name=your_config.yaml
```

### Reference Workflows

#### Hurricane Helene

[Hurricane Helene][helene-wiki] was a significant tropical cyclone that made landfall
in September 2024,
causing widespread impacts across the southeastern United States. The storm's rapid
intensification and complex structure made it a challenging case for weather prediction.

This workflow demonstrates ensemble inference for Helene, with the model
initialised approximately two and a half days before landfall. To run this example,
first download the model packages and skill file as described in
[section 2](#2-prerequisites).
In the configuration file `helene.yaml`, specify the path to the model packages under
`forecast_model.package` and the skill file under `perturbation.skill_path`. The current
configuration is set up for two checkpoints, one initial condition, and four ensemble
members per checkpoint-IC pair, resulting in eight ensemble members in total. Once the
small configuration is verified, you can expand the number of checkpoints, ICs, and
ensemble members per checkpoint-IC pair.

Execute the ensemble inference by running:

```bash
[mpirun -n XX] python hens.py --config-name=helene.yaml
```

The current configuration uses two checkpoints, one initial condition, and four
ensemble members per checkpoint-IC pair, resulting in eight ensemble members total.
You can expand these parameters once the configuration is verified.

#### Reproducing Individual Batches of the Helene Ensemble

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

##### Steps to Reproduce a Batch

1. **Identify the Batch to Reproduce**
   - Determine the batch ID(s) you want to reproduce
   - Set these IDs in the `batch_ids_reproduce` parameter

2. **Configure the Reproduction Run**
   - Use the same configuration as `helene.yaml` for all parameters except file output
   - If the original run didn't specify a random seed, you can find it in:
     - The cyclone tracker file
     - The netCDF output file (when using `KVBackend` or `XarrayBackend`)
     - Other IObackends, including the netcdf4 backend do not provide the
       seed in the output file
   - If the original run didn't specify a random seed, different processes will
     generate different seeds

3. **Execute the Reproduction**

   ```bash
   python hens.py --config-name=reproduce_helene_batches.yaml
   ```

4. **Verify the Output**
   - Compare the output tracks with the selected batches of the original run
   - The entries should be identical, with only the track ID differing
     (as this depends on the total number of ensemble members)

#### Rain or Shine?

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
python hens.py --config-name=storm_bernd.yaml
```

## References

- [Huge Ensembles Part I Paper][hens-paper]
- [NERSC Registry][nersc-registry]

<!--Common Links-->
[hens-paper]: https://arxiv.org/abs/2408.03100 "Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators"
[nersc-registry]: https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
[hydra-docs]: https://hydra.cc/docs/advanced/instantiate_objects/overview/
[helene-wiki]: https://en.wikipedia.org/wiki/Hurricane_Helene
[bernd-wiki]: https://en.wikipedia.org/wiki/2021_European_flood_event
