# Cyclone Tracking

![Helene Prediction](helene_pred.gif)

> [!Note]
> This recipe serves as an example of how to integrate custom downstream models into a forecasting pipeline. While TempestExtremes is used here for tropical cyclone tracking, the patterns demonstrated—such as asynchronous CPU/GPU execution, in-memory file handling, and diagnostic model integration—can be adapted for other downstream analysis tools.

This pipeline produces ensemble prediction and tracks tropical cyclones (TCs) within the prediction. Here, we demonstrate how to integrate downstream analysis tools like [TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes) into a prediction workflow while avoiding the need to store large atmospheric field data to disk, which can grow prohibitively large and significantly impact inference speed.

Although TempestExtremes is not GPU-accelerated, its execution time on CPU is typically shorter than the time required for the GPU to produce the next prediction. We therefore implement an asynchronous mode where the CPU executes TempestExtremes on completed time steps while the GPU continues producing subsequent predictions, resulting in virtually no computational overhead from the tracking process.

## Table of Contents

1. [Setting up the Environment](#1-setting-up-the-environment)
   - [1.1 Container (Recommended)](#11-container-recommended)
   - [1.2 UV Environment](#12-uv-environment)
2. [Configure and Execute Runs](#2-configure-and-execute-runs)
   - [2.1 Generate Ensemble](#21-generate-ensemble)
   - [2.2 Reproduce Individual Ensemble Members](#22-reproduce-individual-ensemble-members)
   - [2.3 Extract Reference Tracks from ERA5](#23-extract-reference-tracks-from-era5-using-ibtracs-as-ground-truth)
3. [Visualisation](#3-visualisation)
4. [TempestExtremes Integration](#4-tempestextremes-integration)
5. [Example Workflow](#5-example-workflow)
   - [5.1 Extract Baseline](#51-extract-baseline-optional)
   - [5.2 Produce Ensemble Forecasts](#52-produce-ensemble-forecasts)
   - [5.3 Analyse Tracks](#53-analyse-tracks)
   - [5.4 Reproduce Interesting Members](#54-reproduce-interesting-members-to-extract-fields)


## 1. Setting up the Environment

There are two ways to set up the environment: either directly using a uv environment, or using uv to install dependencies inside a container. Since flash-attention, torch-harmonics and TempestExtremes all require compilation, **the container approach is strongly recommended**.

### 1.1 Container (Recommended)

A Dockerfile is provided in the top-level directory of the repository. [Building an image from this Dockerfile](https://docs.docker.com/get-started/docker-concepts/building-images/build-tag-and-publish-an-image/#build-an-image) is highly recommended as it avoids the need to compile flash-attention (which otherwise can take hours) and installs TempestExtremes. Note that compiling torch-harmonics and therefore building the container might still take around 20min.

### 1.2 UV Environment

> [!Note]
> When installing through uv, note that compiling `flash-attn` (required for AIFS-ENS) can take several hours. See the [troubleshooting notes](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html#flash-attention-has-long-build-time-for-aifs-models) for more information. Consider using the container instead, where `flash-attn` is pre-installed in the base image.
>
> Additionally, you will need to install TempestExtremes yourself and point the configuration files to the TempestExtremes executables.

The project contains a `pyproject.toml` and a `uv.lock` file for setting up the environment. To install with locked dependencies:

```bash
uv sync --frozen
```

This creates a uv environment in the `.venv` directory. The `--frozen` flag ensures the exact versions specified in the lock file are used. Optionally, activate the virtual environment:

```bash
source .venv/bin/activate
```

Additionally, you need to [install TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes?tab=readme-ov-file#installation-via-cmake-recommended) and either add the executables for `DetectNodes` and `StitchNodes` to the `PATH` or point the associated commands in the configs to these executables.

#### Potential Issues with torch-harmonics

If the `--frozen` flag cannot be used (e.g. when updating dependencies to their latest versions), torch-harmonics may fail during the dependency resolution phase. During this phase, uv must build torch-harmonics to resolve dependencies before installation, but the build can fail if other dependencies are not yet installed. Additionally, a Git LFS error occasionally appears. To address these issues, split the installation into a two-step process and skip Git LFS:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --no-install-package torch-harmonics
GIT_LFS_SKIP_SMUDGE=1 uv sync
```


## 2. Configure and Execute Runs

Runs are configured through YAML files located in `tc_tracking/cfg`. To execute the pipeline:

```bash
python tc_hunt.py --config-name=config.yaml
```

The script can also be executed in distributed settings using Slurm, MPI, or torchrun. For example:

```bash
torchrun --nproc-per-node=2 tc_hunt.py --config-name=config.yaml
```

The pipeline has three operational modes:

- **`generate_ensemble`**: Generate an ensemble prediction and extract tropical cyclones.
- **`reproduce_members`**: Reproduce individual ensemble members to store atmospheric fields of interesting tracks. Note: Currently only works with FCN3, as AIFS-ENS does not expose a method to set the model's internal random state.
- **`extract_baseline`**: Extract tropical cyclone tracks from historical reanalysis data (e.g. ERA5) for validation purposes.

In the following we will explain how to configure the yaml files for those three modes. You can find example configs in `./cfg`.

### 2.1 Generate Ensemble

Here we explain how to configure a yaml file to produce an ensemble prediction with either FCN3 or AIFS-ENS and track tropical cyclones within the predictions. A full example can be found at `cfg/helene.yaml`.

**Basics**

```yaml
project: 'helene'          # project name used for output names
mode: 'generate_ensemble'  # choose mode
model: 'aifs'              # 'fcn3' or 'aifs'
random_seed: 7777          # optional: random seed (DOES NOT directly control model-internal random state)
```

**<a name="config_ens"></a>Ensemble Configuration**

```yaml
ics: ["2024-09-24 00:00:00", "2024-09-25 18:00:00"]  # list of initial conditions
n_steps: 28                                          # number of forecast steps
ensemble_size: 16                                    # number of ensemble members FOR EACH initial condition
batch_size: 2                                        # batch size for inference
```

**GPU Memory Requirements**: For a batch size of 1, AIFS-ENS requires approximately 50GB of GPU memory and scales nearly linearly with increasing batch size. FCN3 requires over 60GB for batch size 1, but memory usage increases only marginally with larger batch sizes, making it more memory-efficient for batched inference.

**IO Configuration**

```yaml
data_source:                                      # data source for initial conditions
    _target_: earth2studio.data.NCAR_ERA5         # choose any of the online or on-prem sources

store_type: "netcdf" # 'none', 'netcdf', 'zarr'   # file type for field data output; 'none' produces no field output
store_dir: "./outputs_${project}"                 # directory to which all outputs are written
out_vars: ['msl', 'z300', 'z500', 'u10m', 'v10m'] # field variables to write to file
```

The data source can be any of the Earth-2 Studio data sources or a custom data source implementation.

**Storage Considerations**: Exercise caution when storing field variables, as storage requirements grow rapidly (proportional to number of ICs × ensemble size × forecast length). Additionally, writing large volumes of data to disk can significantly slow down inference. For FCN3 workflows, consider storing only track data initially and using the [reproducibility feature](#22-reproduce-individual-ensemble-members) to regenerate fields for ensemble members with interesting tracks.

**Cyclone Tracking with TempestExtremes** (optional)

```yaml
cyclone_tracking:
    asynchronous: True                             # let GPU produce next prediction while CPU executes TempestExtremes
    vars: ['msl', 'u10m', 'v10m', 'z300', 'z500']  # variables required by TempestExtremes
    detect_cmd: 'DetectNodes --verbosity 0 --mergedist 6 --closedcontourcmd _DIFF(z300,z500),-58.8,6.5,1.0;msl,200.,5.5,0 --searchbymin msl --outputcmd msl,min,0;_VECMAG(u10m,v10m),max,5;height,min,0'
    stitch_cmd: 'StitchNodes --in_fmt lon,lat,msl,wind_speed,height --range 8.0 --mintime 54h --maxgap 4 --out_file_format csv --threshold wind_speed,>=,10.,10;lat,<=,50.,10;lat,>=,-50.,10;height,<=,150.,10'
                                                   # detect node and stitch node commands from TempestExtremes
    orography_path: './orography.nc'               # path to orography file (required only if height is relevant to TempestExtremes)
    keep_raw_data: False                           # whether to keep raw field data used for tracking
    use_ram: True                                  # write in-memory files to avoid disk I/O
    task_timeout_seconds: 120                      # timeout for tracking tasks
    print_te_output: False                         # print TempestExtremes output to terminal
```

For more information on [asynchronous mode](#async), see the dedicated section below.

**Debugging TempestExtremes Commands**: To debug `detect_cmd` and `stitch_cmd`, set `print_te_output` to `True` to see TempestExtremes logs. However, iterating through full inference runs for debugging can be cumbersome. For more efficient debugging, set `keep_raw_data` to `True` and `use_ram` to `False` to write field snapshots as NetCDF files to `${store_dir}/cyclone_tracks_te/raw_data`. You can then debug the TempestExtremes commands directly from the command line using these files. Once the commands work correctly, copy them back to the configuration file. When running TempestExtremes from the command line, you need to specify file paths explicitly (see [TempestExtremes documentation](https://climate.ucdavis.edu/tempestextremes.php)): use `--in_data_list` and `--out_file_list` for `DetectNodes`, and `--in` and `--out` for `StitchNodes`.

**Stability Check** (optional)

FCN3 can perform stable rollouts on subseasonal timescales (weeks to months), but occasionally experiences numerical instabilities on seasonal timescales. A simple stability check has been implemented to detect and handle these cases:

**Key Points**:
- **Applicable to**: FCN3 for (sub)seasonal forecasts. The long-term behaviour of AIFS-ENS is still under investigation.
- **Detection mechanism**: Instabilities manifest in global averages of field variables, which typically build up over ~10 days before diverging rapidly.
- **Implementation**: Provide a list of variables to monitor and a threshold for each. Global averages are computed at initialisation and at each time step. If any variable's global average exceeds its threshold, the simulation is interrupted and restarted with a new random seed.
- **Limitation**: This is a basic implementation. More advanced detection methods that don't require 10 extra simulation days, and better load-balancing when many ensemble members become unstable, would be beneficial improvements.

```yaml
stability_check:
    variables: ['t1000', 'u250', 'v850']
    thresholds: [10, 10, 10]
```

**Hydra Configuration** (optional)

```yaml
hydra:
    job:
        name: ${project}
    run:
        dir: ${store_dir}/${project}_logs/${now:%Y-%m-%d_%H-%M-%S}
```

This propagates the project name and output directory to Hydra's logging system. You will most likely never need to modify this section.


### 2.2 Reproduce Individual Ensemble Members

A common workflow is to run ensemble forecasts without storing atmospheric fields, since track data is approximately five orders of magnitude smaller than the associated field data. This enables sending results of hundreds of hypothetical cyclone seasons as an email attachment. However, for ensemble members with particularly interesting tracks, examining the atmospheric fields can help understand the storm's evolution.

FCN3 provides access to its internal random state, enabling exact reproduction of individual ensemble members. In theory, AIFS-ENS should have this capability as well, but the internal random state is not exposed through the Anemoi models interface, preventing reproducibility of AIFS-ENS runs.

Note that [reproducibility is challenging](https://arxiv.org/abs/1605.04339) and only works reliably when using the same environment on the same system to produce the predictions.

To reproduce a run, you need:
- Configuration file from the original run (ideally)
- Ensemble member ID and random seed for all members to be reproduced (both provided in the track CSV filename)
- Batch size (also provided in the track CSV filename)

**Configuration Changes**

The following describes what to change in the configuration file to reproduce ensemble members. You can compare `cfg/helene.yaml` and `cfg/reproduce_helene.yaml` as reference examples. In the original configuration, consider changing the project name to avoid overwriting files, and set the mode to:

```yaml
mode: 'reproduce_members'
```

In the [**ensemble configuration section**](#config_ens), remove the `ics` variable and replace it with a list of ensemble members to reproduce. For each ensemble member, provide the initial condition, member ID, and random seed:

```yaml
reproduce_members: [
                    ['2024-09-24 00:00:00',  5, 2148531792],
                    ['2024-09-24 00:00:00',  8, 1342799488],
                    ['2024-09-24 12:00:00',  4, 2528887706],
                    ['2024-09-24 12:00:00',  5, 2528887706],
                    ['2024-09-24 12:00:00', 13, 1968734508],
                   ]
```

The example above would reproduce members 5 and 8 from the ensemble starting at midnight, and members 4, 5, and 13 from the ensemble starting at midday.

**Batch Reproduction Behaviour**: Note that the internal random state can only be set for a complete batch. While individual members within a batch will produce independent predictions, they share the same random seed. This means a full batch must always be reproduced. For example, with a batch size of 2 in the example above, the run would reproduce members 4, 5, 8, and 9 from the midnight ensemble, and members 4, 5, 12, and 13 from the midday ensemble.


### 2.3 Extract Reference Tracks from ERA5 Using IBTrACS as Ground Truth

[IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) (International Best Track Archive for Climate Stewardship) is considered the gold standard for tropical cyclone observation data. However, it represents point observations that a global model at quarter-degree resolution cannot reproduce accurately. Therefore, a meaningful validation of model predictions must be performed against ERA5. To do so, we first have to extract tropical cyclone tracks directly from reanalysis data (such as ERA5) and compare the predictions against these reference tracks. This pipeline demonstrates how to extract reference tracks for named storms, using IBTrACS to identify the temporal windows and approximate locations of storms of interest.

**Basics**

```yaml
project: 'reference_tracks'
mode: 'extract_baseline'
```

**Cases**

Specify which storms to extract by providing their name, year, and basin for each case:

```yaml
cases:
    hato:
        year: 2017
        basin: west_pacific
    helene:
        year: 2024
        basin: north_atlantic
```

The script queries the IBTrACS database to determine when each storm was active, retrieves the corresponding atmospheric data from the specified data source, and applies TempestExtremes to extract tropical cyclone tracks. Since multiple storms may be active globally at any given time, a second matching step compares all extracted tracks against the IBTrACS ground truth to identify the correct track for each named storm.

**IO Configuration**

**IBTrACS Data**:
- Download the IBTrACS data in CSV format using:
  ```bash
  wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv
  ```
  Note: This file is over 300MB and may take several minutes to download.
- For testing with only Helene and Hato, you can use the subset provided at `./test/aux_data/ibtracs.HATO_HELENE.list.v04r01.csv`.

**Reanalysis Data**:
- For extracting multiple storms, we recommend using an on-premise dataset to avoid lengthy download times.
- The data source can be configured using the standard approach (as shown in the ensemble generation section).
- Alternatively, different data sources can be specified for different year ranges, which is particularly useful when datasets are partitioned into training/testing/out-of-sample periods.
- The example below shows how to configure different ERA5 sources for different time periods:

```yaml
store_dir: "./outputs_${project}"
ibtracs_source_data: "/path/to/ibtracs.ALL.list.v04r01.csv"

data_source:
    era5_train:
        years: "1980-2016"
        source:
            _target_: earth2studio.data.NCAR_ERA5

    era5_oos:
        years: "2017-2022"
        source:
            _target_: earth2studio.data.CDS
```


## 3. Visualisation

Two Jupyter notebooks are provided in `./plotting` for analysing and visualising tropical cyclone tracking results:

- **`tracks_slayground.ipynb`**: Ensemble track analysis including spaghetti plots (trajectory visualisation), absolute and relative intensity metrics (wind speed, MSLP), comparisons against ERA5 reference tracks and IBTrACS observations, extreme value statistics, and error moment analysis over lead time.

- **`plot_tracks_n_fields.ipynb`**: Create animated visualisations of storm tracks overlaid on atmospheric field data.

## 4. TempestExtremes Integration

This pipeline uses TempestExtremes to demonstrate how to integrate custom downstream analysis tools into a prediction workflow. TempestExtremes requires file-based input, does not run on GPU, and cannot be installed as a Python library—making it a useful reference case, as tools with fewer constraints should be simpler to integrate. The current implementation avoids writing atmospheric field data to disk and reading it back, which would otherwise slow down inference significantly and become prohibitive given the large data volumes involved.

**General Integration Approach**

TempestExtremes is implemented as a diagnostic model class but without returning any data, since results are written directly to disk. The diagnostic model class spawns a subprocess to execute TempestExtremes command-line tools. Track files are produced directly by TempestExtremes and written to the specified output directory.

Since TempestExtremes requires file-based input, we must provide atmospheric data through files. To avoid the overhead of writing and reading large volumes of data to/from disk, we write temporary files to `/dev/shm` (shared memory filesystem), effectively creating in-memory files that TempestExtremes can access while avoiding physical disk I/O.

**<a name="async"></a>Asynchronous Mode**

On many platforms, executing TempestExtremes on the CPU is faster than producing the prediction on the GPU. We therefore implemented an asynchronous mode where a CPU thread executes TempestExtremes on completed time steps while the GPU continues producing the next prediction, enabling tracking with virtually no overhead.

**Implementation Details**:
- In asynchronous mode, each ensemble member in a batch is assigned its own processing thread, allowing parallel tracking across the batch.
- If TempestExtremes execution becomes slower than GPU prediction (which might occur with AIFS-ENS on newer GPUs like B200), consider increasing the batch size. GPU prediction time scales approximately linearly with batch size, while the time for a single CPU thread to process one member remains constant.
- If you encounter issues, first verify whether they also occur with asynchronous mode disabled before debugging further.


## 5. Example Workflow

This section walks through an example workflow demonstrating the pipeline's core functionality: generating an ensemble prediction, reproducing individual members, visualising results, and comparing predicted tracks against reference tracks extracted from ERA5 data.

### 5.1 Extract Baseline (Optional)

This step extracts reference tracks from ERA5 reanalysis for comparison with forecast predictions. Note that downloading the ERA5 field data for Hato and Helene is required and can take some time. If it takes too long, you can skip this step and use the pre-computed reference tracks provided in `./test/aux_data` instead.

First, download the IBTrACS data, then extract the baseline tracks:

```bash
python tc_hunt.py --config-name=extract_era5.yaml
```

This should produce a folder called `outputs_reference_tracks/` containing two reference track files:
- `reference_track_hato_2017_west_pacific.csv`
- `reference_track_helene_2024_north_atlantic.csv`

### 5.2 Produce Ensemble Forecasts

Run the forecast loop for Helene using FCN3 and for Hato using AIFS-ENS:

```bash
python tc_hunt.py --config-name=helene.yaml
python tc_hunt.py --config-name=hato.yaml
```

This should produce two output folders: `outputs_helene` and `outputs_hato`, each containing tracked tropical cyclone trajectories.

### 5.3 Analyse Tracks

Visualise the results using the notebook `plotting/tracks_slayground.ipynb`.

**For Helene**, configure the first cell of the notebook with:
```python
case = 'helene_2024_north_atlantic'
pred_track_dir = '/path/to/outputs_helene/cyclone_tracks_te'
tru_track_dir = '/path/to/outputs_reference_tracks'
# If you skipped baseline extraction, use:
# tru_track_dir = '/path/to/test/aux_data'
```

**For Hato**, use:
```python
case = 'hato_2017_west_pacific'
pred_track_dir = '/path/to/outputs_hato/cyclone_tracks_te'
tru_track_dir = '/path/to/outputs_reference_tracks'
# If you skipped baseline extraction, use:
# tru_track_dir = '/path/to/test/aux_data'
```

### 5.4 Reproduce Interesting Members to Extract Fields

Suppose that after conducting the above analysis you want to take a closer look at ensemble members 0, 8, 9, and 12 of the Helene prediction. Since the forecast was produced with FCN3, we can reproduce these members and store the atmospheric fields for detailed analysis.

**Reproduce with Field Storage**:

1. In `cfg/reproduce_helene.yaml`, verify that the random seeds in the `reproduce_members` variable match the seeds from the track filenames in `outputs_helene/cyclone_tracks_te`. If they do not, update the seeds in the config accordingly.
2. Double-check that the batch size, ensemble size, and data source are identical to the original run.
3. Execute the pipeline:
   ```bash
   python tc_hunt.py --config-name=reproduce_helene.yaml
   ```
4. This should produce a folder `outputs_reproduce_helene` containing both track files and a NetCDF file named `reproduce_helene_2024-09-24T00.00.00_mems0000-0013.nc` with the atmospheric fields.

**Visualise Tracks and Fields**:

Use the notebook `plotting/plot_tracks_n_fields.ipynb` to create animated visualisations:

1. Configure the file paths:
   ```python
   field_data = '/path/to/outputs_reproduce_helene/helene_2024-09-24T00.00.00_mems0000-0015.nc'
   track_dir = '/path/to/outputs_reproduce_helene/cyclone_tracks_te'
   ```

2. Select visualisation parameters:
   ```python
   variable = 'wind_speed'   # or 'msl', etc.
   ensemble_member = 8       # choose from the reproduced subset
   region = 'gulf_of_mexico'
   ```

3. Run the notebook to generate an interactive animation where you can step through time.

**Verification**: To verify that reproducibility worked correctly, point `track_dir` to the original Helene run (`outputs_helene/cyclone_tracks_te`) and confirm that the tracks are identical.

