# Tests

## Test 1: Generate Ensemble Forecast with Cyclone Tracking

> [!Note]
> `test_tc_hunt.sh` runs Tests 1 and 2 automatically, with FCN3 by default or
> AIFS-ENS via `./test_tc_hunt.sh aifs` (`aifs-v2` for version 2). If, at the
> end you see a message containing `all good, yay (:`, the test finished
> successfully. The steps below describe the FCN3 test.

This case tests whether the pipeline can generate an ensemble forecast and
track tropical cyclones using TempestExtremes. Furthermore, it is tested if
those exact tracks can be reproduced. For AIFS-ENS, this requires
`deterministic=True` and, given its memory needs, `batch_size=1`, which the
script sets.

First, let us produce a five member ensemble of a 90h prediction of Hurricane Helene

```bash
cd earth2studio/recipes/tc_tracking/test
python ../main.py --config-path=$(pwd)/cfg --config-name=baseline_helene.yaml
```

### Expected Result: Generate Ensemble

You can find the track files in `outputs_baseline_helene/cyclone_tracks_te`.
The directory should contain five csv files with naming pattern according to
`tracks_2024-09-24T00:00:00_mem_000X_seed_YYYYYYYYY_bs_2.csv`
This pattern provides information about initial condition, ensemble member ID,
random seed and batch size.

## Test 2: Reproducibility of Helene Ensemble Members

Next, reproduce members three and four of the previous run. Their seeds are
in the names of their track files, `cfg/reproduce_helene.yaml` expects them
in place of the `'???'` placeholders. Instead of editing the file, they can
also be passed on the command line:

```bash
python ../main.py --config-path=$(pwd)/cfg --config-name=reproduce_helene.yaml \
    reproduce_members.0.2=<seed of member 3> reproduce_members.1.2=<seed of member 4>
```

Note that the baseline run was produced with batch size two. For reproducibility,
we need to apply the same batch size in the reproduction run and reproduce the full
batch that contained the ensemble member in question. For the present case, this means
that member two will also be reproduced, as it is in the same batch as member three.

### Expected Result: Reproduce Helene

There should now be three track files in `outputs_reproduce_helene/cyclone_tracks_te`,
with file names identical to the associated files in `outputs_baseline_helene/cyclone_tracks_te`.
Whether the content of the files (i.e., the generated tracks) is also identical can be tested through
a simple `diff` call:

<!-- markdownlint-disable MD013 -->

```bash
diff outputs_baseline_helene/cyclone_tracks_te/tracks_2024-09-24T00:00:00_mem_0002_seed_4045540270_bs_2.csv \
     outputs_reproduce_helene/cyclone_tracks_te/tracks_2024-09-24T00:00:00_mem_0002_seed_4045540270_bs_2.csv
```

<!-- markdownlint-enable MD013 -->

Repeat the diff call for all reproduced ensemble members, the return should always be empty.

## Test 3: Extracting individual storms from historic data

> [!Note]
> The test can be done manually following the steps below. Alternatively,
> `test_historic_tc_extraction.sh` provides an automated way for running
> the full test. If, at the end you see a message containing `all good, yay (:`,
> the test finished successfully.

This mode lets users extract storms from historic data sets. The user can choose
a storm, the script will look for that storm in the IBTrACS data base, obtain data
around the storm's life time from a data source (on-prem or online) and then extract
that storm using TempestExtremes.

The IBTrACS data file is downloaded automatically on first use (~300 MB).

For the current test, let us extract Typhoon Hato (2017) and Hurricane Helene (2024)
from ERA5. Unlike the ensemble forecast tests above, `extract_baseline` is CPU-only
and does not use `torchrun`; the two storms are processed in parallel via a
`ProcessPoolExecutor` (`num_workers: 2` in `cfg/extract_era5.yaml`):

```bash
cd earth2studio/recipes/tc_tracking/test
python ../main.py --config-path=$(pwd)/cfg --config-name=extract_era5.yaml
```

To exercise the serial fallback, override the worker count from the command line:

```bash
python ../main.py --config-path=$(pwd)/cfg --config-name=extract_era5.yaml \
    num_workers=1
```

### Expected Result — Historic Extraction

> [!Important]
> The reference tracks in `../aux_data` were extracted before the extraction
> included the last 6-hourly time step of a storm. Until they are regenerated
> with the current code, the comparison below shows that additional time step.

The run should produce two reference tracks in `outputs_reference_tracks/`.
Output is deterministic regardless of the `num_workers` setting (each storm writes
its own CSV independently). Now, let us compare the extracted tracks with the
baseline, in both cases the files should be identical:

<!-- markdownlint-disable MD013 -->

```bash
diff outputs_reference_tracks/reference_track_hato_2017_west_pacific.csv \
     ../aux_data/reference_track_hato_2017_west_pacific.csv
diff outputs_reference_tracks/reference_track_helene_2024_north_atlantic.csv \
     ../aux_data/reference_track_helene_2024_north_atlantic.csv
```

<!-- markdownlint-enable MD013 -->
