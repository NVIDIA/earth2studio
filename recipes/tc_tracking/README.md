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

Runs are configured through yaml files which are put in `tc_tracking/cfg`. to run the pipeline, do:

```bash
python tc_hunt.py --config-name=config.yaml
```

The script can also be exectued in distributed settings using slurm, mpi, or torchrun. eg:

```bash
torchrun --nproc-per-node=2 tc_hunt.py --config-name=config.yaml
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

**<a name="config_ens"></a>ensemble**

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
    _target_: earth2studio.data.NCAR_ERA5         # choose any of the online or on-prem sources

store_type: "netcdf" # 'none' 'netcdf', 'zarr'    # file type of field data output. no field output if none
store_dir: "./outputs_${project}"                 # directory to which all outputs are written to
out_vars: ['msl', 'z300', 'z500', 'u10m', 'v10m'] # field variables to write to file

```
- data source can provide any of the e2studio data sources or a custom data source

- be careful with storing field varibales, as the storage requirements grow quickly and also writing the data to disc can signifcantly slow down the inference
- Note that with FCN3, runs can be reproduced (see section YY), opening the option to store and analyse tracks only, and then reproduce those ensemble members with interesting tracks where you might want to see the fields

**Cyclone Tracking with TmepestExtremes** (optional)

```yaml
cyclone_tracking:
    asynchronous: True                             # Let GPU produce next prediction while CPU executes TempestExtremes
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
- for more infos on [asynchronous mode](#async) have a look at the section below
- tip: For debugging `detect_cmd` and `stitch_cmd` you can set `print_te_output` to `True` to see where `TempestExtremes` fails. However, if more iterations are required to get TmepestExtremes to run, debugging through inference runs can be a rather cumbersome. Consider setting `keep_raw_data` to `True` and `use_ram` to `False` to write a field snapshot to a netcdf file to `${store_dir}/cyclone_tracks_te/raw_data`. Then you can debug directly pasting the commands in the command line. Once you have found the correct commands, just copy the working commands to the config file. When operating in the command line, you need to point TempestExtremes to the files (see [documentation](https://climate.ucdavis.edu/tempestextremes.php) for more details): for `detect_cmd`, this will be `--in_data_list` and `--out_file_list`, for `stitch_cmd` this will be `--in` and `--out`.

**stability check** (optional)

- FCN3 can do stable rollouts on subseasonal time scales, stability check is only relevant for (sub)seasonal time scales
- THe long term behaviour of AIFS-ENS is still something we need to investigate, so the current method is for FCN3
- however blows up some time
- simple stability check was implemented
    - instability can be seen in global averages of field variables
    - build up over like 10 days, the escape towards infinity
    - so it is easy to chatch one of them if yo ucan allow letting the simulation run for 10 extra days in (sub)seasonal
    - if cautch, the ensemble member can be re-run with differrent random seed
    - to implement stability check, provide list of variables to monitor and a threshold for each var. from these varables, global averages will be taken at initial conditions. At each time step global average is computed again and if it surpasses the threshold, the run will be interrupted and re-started with new random seed.
    - Note that this all currently is rather basic implementation, more advanced methods for detection that might not require 10 extra days of simulation and method for better load-balancing in case a lot of runs get unstable would definitely be beneficial.

```yaml
stability_check:
    variables: ['t1000', 'u250', 'v850']
    thresholds: [10, 10, 10]
```


**hydra** (optional)
```yaml
hydra:
    job:
        name: ${project}
    run:
        dir: ${store_dir}/${project}_logs/${now:%Y-%m-%d_%H-%M-%S}
```
- propagating project name and output directory for logs to hydra.
- most likely never need to change that section


### reproduce individual ensemble members

- run without storing fields, as tracks contain roughly 5 orders of magnitude less data that the associated field data required to extract the tracks
- send results of hundreds of hypothetical cyclone seasons as an email attachment
- however, for interesting tracks it can be that looking at field could be interesting

- FCN3 allows accessing its random internal state, enabling reproducing individual ensemble members
- AIFS should have this capability as well in theory, however, the internal random state cannot is not exposed through anemoi models, which is why we cannot reproduce AIFS-ENS runs
- not that reproducibility [is hard](https://arxiv.org/abs/1605.04339) only works when the same environment on the same system is used to produce the prediction.

- to reproduce a run you need the following
    - config file of the original run (ideally)
    - ensemble member ID and random seed of all members which shall be reproduced (both are provided in file name of track csv file)
    - batch size (also provided in name of track csv file)

here we describe what to change in the file to reproduce, you can also compare `cfg/helene.yaml` and `cfg/reproduce_helene.yaml`.
in the original config, consider switching the project name to avoid ovewriting files and change the mode to:

```yaml
mode: 'reproduce_members'
```

in the [**ensembles section**](#config_ens), remove the ic varaible and replace it with a list of ensemble members you want to reproduce.
For each ensemble member, provide intitial condition, member ID and random seed:

```yaml
reproduce_members: [
                    ['2024-09-24 00:00:00',  5, 2148531792],
                    ['2024-09-24 00:00:00',  8, 1342799488],
                    ['2024-09-24 12:00:00',  4, 2528887706],
                    ['2024-09-24 12:00:00',  5, 2528887706],
                    ['2024-09-24 12:00:00', 13, 1968734508],
                   ]
```

The above example would reproduce members 4 and 8 from the ensemble starting at midnight and members 4, 5 and 13 from the ensemble starting midday.
Not that the internal random state can only be set for the full batch. While the individual members of a batch will provide independend predictions, they have the same random seed. This also means that always a full batch has to be reporduced:
Consider a batch size of 2 for the example above. In that case, the run would reproduce members 4, 5, 8 and 9 of the ensemble starting at midnight, and members 4, 5, 12 and 13 of the ensemble starting midday.


### extract Reference tracks from ERA5 using IBTrACS as ground truth

- IBTrACS is considered gold standard for TC data
- However, it is observation data which a global model at quartely degree is not able to reproduce
- Hence for comparison it can make sense to extract storms directly from analysis data like ERA5 and score prediction results against these
- This pipeline shows how to extract reference tracks for named storms using IBtracs as reference to identify tracks

**basics**

```yaml
project: 'reference_tracks'
mode: 'extract_baseline'
```

**cases**

- choose storms. for each storm which shall be extracted, provide name, year and basin as per below
- the script will look in IBTRACS data during which time the storm was active, pull the relevant data and extracts TC tracks using TempestExtremes
- at any given time there might be multiple storms on the globe, hence in a second step all extracted tracks are compared if they are close to the IBTRACS groundtruth to find the right track.

```yaml
cases:
    hato:
        year: 2017
        basin: west_pacific
    helene:
        year: 2024
        basin: north_atlantic
```

**IO**
ibtracs data:
- download ibtracs data in csv format with following command: `wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv` Note that file is over 300MB and might take a couple of minutes to download
- note that `./test/aux_data/ibtracs.HATO_HELENE.list.v04r01.csv` can be used if just testing with Helene and Hato.

analysis data:
- for extracting esp if more storms shall be extracted we recommend using an on-prem data set, as otherwise download times will be limiting here
- either, data source can just be implemented the standard way as above
- or different data sets can be prescribed for differnet years, esp helpfull when available data set is split up in train/test/outofsample
- as an example here we show with different versions of ERA5

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


## visualise

Have a stab at the notebook...

## TempestExtremes integration

- using TempestExtremes as example to demonstrate how to implement custom downstream models in pipeline
- increased complexity, as TE does not run on GPU and cannot be installed as python library.
- implementing that way to avoid writing atmospheric field data to and reading it again from disk, as this slows down inference and can be prohibitive with available storage

**general integration**

- implemented like a diagnostic model, but without return type
- that diagnostic model class then spawns a subprocess to call TE
- result will be tracks file directly written to disk
- data access for TE has to be provided though file
- to avoid reading/writing large amounts of data to/from disk, we write to `/dev/shm` effectively producing in-memory file


**<a name="async"></a>asynchronos mode**

- on many platforms, running TmepestExtremes on the CPU is still faster than producing the forecast on the GPU. Hence, we let a CPU core execute TempestExtremes while the GPU already starts producing the next prediction
- note that in asynchronos mode each member of a batch gets its own process assigned. Hence, if TE is slower than the prediction (could e.g. happen when using AIFS on B200), consider increading the batch size. The time the GPU takes to produce a batched predcition scales more or less linearly with batch size, while the time that a single CPU core takes to porcess one member is constant.
- if you see issues, first see if they also appear when asyncronos mode is switched off.


## example workflow

### extract baseline (optional)

- note that this requires downloading the field data for Hato and Helene. you can skip this step and use the refernce tracks under `./test/aux_data` instead
- download ibtraccs data
- extract baseline by running
```bash
python tc_hunt.py --config-name=extract_era5.yaml
```

- That should produce a folder called `outputs_reference_tracks/` that includes the two reference tracks:
    - `reference_track_hato_2017_west_pacific.csv`
    - `reference_track_helene_2024_north_atlantic.csv`

### produce ensemble

- run forecast loop for helen with FCN3 and for hato using AIFS-ENS:
    - `python tc_hunt.py --config-name=helene.yaml`
    - `python tc_hunt.py --config-name=hato.yaml`
- this should produce two folders, `outputs_helene` and `outputs_hato`

### analyse tracks

- visualise in notebook `plotting/tracks_slayground.ipynb`
- in first cell of notebook, set for helene
    - `case = 'helene_2024_north_atlantic'`
    - `pred_track_dir = '/path/to/outputs_helene/cyclone_tracks_te'`
    - `tru_track_dir = '/path/to/outputs_reference_tracks'`
    - in case you did not run the extraction yourself you can use `tru_track_dir = '/path/to/test/aux_data'`
- for hato
    - `case = 'hato_2017_west_pacific'`
    - `pred_track_dir = '/path/to/outputs_hato/cyclone_tracks_te'`
    - `tru_track_dir = '/path/to/outputs_reference_tracks'`

### re-produce interesting members to extract fields

- let's assume the above analysis has shown that we want to have a closer look at members 0, 8, 9 and 12 of the Helene prediction
- luckily, that was run with FCN3 so we can reproduce and store the fields
- reproduce, this time store fields
    - in `cfg/reproduce_helene.yaml` make sure random seeds provided to the `reproduce_members` variable match the seeds from the track files in `outputs_helene/cyclone_tracks_te`
    - double-check that batch size and ensemble size and data source are identical to the original run
    - execute the pipeline: `python tc_hunt.py --config-name=reproduce_helene.yaml`
    - should produce folder `outputs_reprduce_helene` that now also includes a netcdf file called `reproduce_helene_2024-09-24T00.00.00_mems0000-0013.nc`
- visualise track and field in `plotting/plot_track_n_fields.ipynb`
    - TODO: remove base dir there
    - set `field_data = '/path/to/outputs_reproduce_helene/helene_2024-09-24T00.00.00_mems0000-0015.nc'`
    - set `track_dir = '/path/to/outputs_reproduce_helene/cyclone_tracks_te'`

    - select a variable you want to visualise for the field, eg `variable = 'wind_speed'` or `variable = 'msl'`
    - select the ensemble member you want to look at (note that you only reproduced a subset)
    - select the region for plotting: `region = 'gulf_of_mexico'`
    - at the end of the notebook you will have be rewarded with an animation where you can click through time
    - as test if reproducibility worked, point `track_dir` to the original Helene run

