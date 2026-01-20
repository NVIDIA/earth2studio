# cyclone tracking

- Pipeline to produce an ensemble forecast and track tropical cyclones (TCs).
- pipeline shows on example of [TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes) how to implement downstream models in a prediction pipeline and avoid storing atmospheric fields to disc, as such data can grow quite large and just the process of writing to disk can significantly impact inference speed.
- TempestExtremes is not GPU-accelerated. Yet, the run time is usually shorter than what the CPU takes to produce forecast. We therefore implemented it such that the CPU looks for tracks while the GPU produces the next prediction, resulting in virually no overhead of tracking.


## Setting up the environment

There are two ways to set up the environment, either directly using an uv environment or using uv to install inside a container.
Since flash-attention, torch-harmonics and TmepestExtremes have to be compiled, the container is the recommended way.

1. **container** (recommended)

    There is a Dockerfile provided in the top level directory of the repository.
    Highly recommend using the Dockerfile to build an image, comes with pre-nstalled flash-attention which otherwise takes a sweet age to compile. Also, TempestExtremes will be installed into the image.

2. **uv**

    > [!Note]
    > When installing through uv note that compiling `flash-attn` (for AIFS-ENS) might take
    > up to hours to compile, more info in the [troubleshooting notes](https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html#flash-attention-has-long-build-time-for-aifs-models). Consider using the container instead, where `flash-attn` is pre-installed
    > in the base image.
    > Additionally, you need to install TempestExtremes yourself and point the config files to the
    > executibles.

    The project contains a pyproject.toml and a uv.lock file for setting up the envirnment. All you need to do is run:

    ```bash
    uv sync --frozen
    ```

    This will create a uv environment in the .venv directory. The fronzen flag will make sure the exact version specified in the lock file is used.
    Optionally, trigger the virtual environment with:

    ```bash
    source .venv/bin/activate
    ```

    **issues with torch-harmoics**
    if `--frozen` flag in `uv sync` cannot be used eg for updating the dependencies to the latest versions, it can happen that torch harmonics fails during dependency resolution phase. In that phase, uv has to build th to resolve dependencies, but does not yet install it. In these cases it can happen that th build fails as other dependencies are not installed yet, Installation of packages happens in a second phase, the install phase. Additionally, there is a git_lfs error that appears every now and then. To adress both or either, split the installation up in a two-step process and skip GIT_LFS:
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 uv sync --no-install-package torch-harmonics
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    ```


## Configure and Execute runs

Runs are configured through yaml files. to run the pipeline, do:

```bash
python tc_hunt.py --config-name=config.yaml
```

The pipleine has three different flavours:
- `generate_ensemble`: generate an ensemble forecast to extract storms from
- `reproduce_members`: reproduce individual members, e.g. to store atmospheric fields of interesting tracks (NOTE: currently only works with FCN3, as AIFS-ENS is missing a method to set the random state of the model)
- `extract_baseline`: extract storms from historic data


### generate ensemble

Produce an ensemble forecast with either FCN3 or AIFS-ENS and track TCs.

**basics**

```yaml
project: 'helene'          # project name used for file names
mode: 'generate_ensemble'  # choose mode
model: 'aifs'              # 'fcn3' 'aifs'
random_seed: 7777          # optional: random seed (will not directly impact model-internal random state)
```

**ensemble**

```yaml
ics: ["2024-09-24 00:00:00, 2024-09-25 18:00:00"] # list of initial conditions
n_steps: 28                                       # number of forecast steps
ensemble_size: 16                                 # number of ensemble members for each initial condition
batch_size: 2                                     # batch size of inference
```

Note that for a batch size of 1, AIFS-ENS requires around 50GB of GPU memory but scales almost linearly with batch size.
FCN3 in contrast requires over 60GB for batch size of 1, but hardly increases when increasing batch size.

**IO**

```yaml
data_source:                                      # data source for initial conditions
    _target_: earth2studio.data.NCAR_ERA5

store_type: "netcdf" # 'none' 'netcdf', 'zarr'    # file type of field data output. no field output if none
store_dir: "./outputs_${project}"                 # directory to which all outputs are written to
out_vars: ['msl', 'z300', 'z500', 'u10m', 'v10m'] # field variables to write to file

```

**Cyclone Tracking with TmepestExtremes**

```yaml

cyclone_tracking:
    asynchronous: True                             # use asyncronos mode
    vars: ['msl', 'u10m', 'v10m', 'z300', 'z500']  # variables which are required by TE
    detect_cmd: 'DetectNodes --verbosity 0 --mergedist 6 --closedcontourcmd _DIFF(z300,z500),-58.8,6.5,0;msl,200.,5.5,0 --searchbymin msl --outputcmd msl,min,0;_VECMAG(u10m,v10m),max,5;height,min,0'
    stitch_cmd: 'StitchNodes --in_fmt lon,lat,msl,wind_speed,height --range 8.0 --mintime 54h --maxgap 4 --out_file_format csv --threshold wind_speed,>=,10.,10;lat,<=,50.,10;lat,>=,-50.,10;height,<=,150.,10'
           # detect node and stitch node commands from TE
    orography_path: './orography.nc'               # path to orograohy file (required only if height relevant to TE)
    keep_raw_data: False                           # keep raw field data for tracking
    use_ram: True                                  # write in-memory files to avoid writing to and reading from disk
    task_timeout_seconds: 120                      # timeout for tracking
    print_te_output: False                         # write TE output to terminal
```


#### configuration

**now you got me messed up**

**please believe me**


### reproduce individual ensemble members

```yaml
mode: 'reproduce_members'
```

### extract Reference tracks from ERA5 using IBTrACS as ground truth

```yaml
mode: 'extract_baseline'
```
- download ibtracs file with following command...
- point the config to the loacation
- extract storms from field data to enable fair comparison

IBTrACS is used as ground truth for the reference tracks. However, due to various reasons simulation data is compared against extracting the tracks directly from ERA5...

Only works if exact same batch is reproduced, that means:
- produces full batches
- important to set batch size
- ensemble size required if final batch of ensemble shall be reproduced
- random seed has to be provided for each ensemble member, but the global random seed won't have effect here, as it mainly impacts the generation of random seeds for members
- works only on same machine using identical environment. No guarantees otherwise
- works only with FCN3


## visualise

Have a stab at the notebook...


## example workflow

### extract baseline

- download ibtraccs data
- extract baseline

### produce ensemble

- run forecast loop

### visualise members

- visualise in notebook

### re-run interesting members to extract fields

- reproduce, this time store fields
- visualise track and field

