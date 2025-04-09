# Recovering HENS through Parallel Inference using Multiple Checkpoints (Python)

## Table of Contents

- [1. Pipeline overview](#1-pipeline-overview)
- [2. Prerequisites](#2-prerequisites)
- [3. Configuring the pipeline](#3-configuring-the-pipeline)
- [4. Executing the pipeline](#4-executing-the-pipeline)
- [5. Reference Workflows](#5-reference-workflows)
  - [5.1 Hurricane Helene](#51-hurricane-helene)
  - [5.2 Reproducing individual Batches of the Helene Ensemble](#52-reproducing-individual-batches-of-the-helene-ensemble)
  - [5.3 Precipitation Forecast](#53-precipitation-forecast)

## 1. Method overview

This project implements a multi-checkpoint inference pipeline designed for
large-scale ensemble weather forecasting.
The pipeline enables parallel processing of multiple model checkpoints, providing
a flexible framework for uncertainty quantification in weather prediction systems.

A key application of this pipeline is the recovery of the HENS (Huge Ensemble) method, as described in
[Huge Ensembles Part I][hens-paper].

HENS provides a calibrated ensemble forecasting
system based on Spherical Fourier Neural Operators (SFNO).
It uses customised bred vectors for initial condition perturbations and multiple
independently trained model checkpoints to represent model uncertainty. This is
specifically designed to better represent extreme weather events.

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

## 2. Prerequisites

To run HENS, we need to download the pre-trained checkpoints, kindly provided
by the authors of the method. We also require the skill file for scaling the
noise vector in the breeding steps of the initial condition (IC) perturbation.
Optionally, we can pre-download IC data in a file structure compatible with the
DataArrayDirectory data source.

- Download model packages provided by [NERSC][nersc-registry]

  ```bash
  wget --recursive --no-parent --no-host-directories --cut-dirs=4 \
    --reject="index.html*" --directory-prefix=hens_model_registry \
    https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
  ```

- Download [channel-specific skill][skill-file]
- Download data [optional, script might be provided]

## 3. Configuring the pipeline

The pipeline follows earth2studio's modular design approach. It is highly
customisable via [instantiating through Hydra][hydra-docs].
This allows for flexible customisation of components. Additional functionality
can be provided through custom functions and instantiated via Hydra, as
demonstrated in the configuration of the HENS perturbation.

### Project Basics

Select a project name to uniquely identify your run. This prevents overwriting
files from previous runs and helps with organisation. Note that `nensemble`
refers to the number of ensemble members per (IC × number of checkpoints).

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

### Forecast Model

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

### Diagnostic Models

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

### IC Data Source

The IC data source configuration determines where the pipeline retrieves
its input data. Earth2studio supports various data sources, including
common online repositories and on-prem solutions. Users can also implement
custom data sources to meet specific requirements.

```yaml
data_source:
    _target_: earth2studio.data.NCAR_ERA5
```

### IC Perturbation

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

### Tropical Cyclone Tracking

- TC tracking through...
- blah

```yaml
cyclone_tracking:
    out_dir: './outputs'
```

### Writing Fields to Disk

- crop output in lat/lon boxes

```yaml
file_output:
    path: './outputs'  # directory to which files are written
    output_vars: ["t2m", 'u10m', 't850', 'z500'] # variables to write out
    format:               # io backend class
        _target_: earth2studio.io.KVBackend
    cropboxes:
        gulf_of_mexico:
            lat_min: 10
            lat_max: 40
            lon_min: 250
            lon_max: 310
```

## 4. Executing the pipeline

The method described in [Huge Ensembles Part I][hens-paper] can be recovered by
using the `HemisphericCentredBredVector` perturbation.

Additionally, diagnostic models and tropical cyclone tracking can be added to the
pipeline by setting `mode` to `diagnostic` or to `cyclone_tracking` in the config.

To run an inference, tailor the config file to your needs and do:

```bash
python hens.py
```

To run on multiple GPUs using your personal config file, do:

```bash
mpirun -n 2 python hens.py --config-name=custom_config.yaml
```

## 5. Reference Workflows

### 5.1 Hurricane Helene

- Does 234583 member ensemble of [Hurricane Helene][helene-wiki]
- Use config `helene.yaml`
- Run in parallel, use up to X GPUs

### 5.2 Reproducing individual Batches of the Helene Ensemble

### 5.3 Precipitation Forecast

- Shows use of diagnostic models
- Various diagnostic models can be used simultaneously
- Order is important
- Show use of data directory data loader and data dl script

[hens-paper]: https://arxiv.org/abs/2408.03100 "Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators"
[nersc-registry]: https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/
[skill-file]: https://portal.nersc.gov/cfs/m4416/hens/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc
[hydra-docs]: https://hydra.cc/docs/advanced/instantiate_objects/overview/
[helene-wiki]: https://en.wikipedia.org/wiki/Hurricane_Helene
