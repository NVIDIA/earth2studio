# Tests

Run the test scripts from `earth2studio/recipes/tc_tracking/test`. A test
passed if its last message contains `all good, yay (:`.

## Tests 1 and 2: Generate an Ensemble with Cyclone Tracking, Reproduce Members

```bash
./test_tc_hunt.sh          # FCN3
./test_tc_hunt.sh aifs     # AIFS-ENS v1 (aifs-v2 for version 2)
```

The script runs two steps:

1. It generates a five-member ensemble of a 90 h forecast of Hurricane Helene
   with `cfg/baseline_helene.yaml` and tracks the cyclones with
   TempestExtremes.
2. It reproduces members three and four with `cfg/reproduce_helene.yaml` and
   checks that their track files are identical to those of the first step.

For AIFS-ENS, reproduction requires `deterministic=True` and, given its memory
needs, `batch_size=1`, which the script sets.

### Expected Result: Generate and Reproduce

`outputs_baseline_helene/cyclone_tracks_te` holds five track files named
`tracks_2024-09-24T00:00:00_mem_000X_seed_YYYYYYYYY_bs_2.csv`, i.e. by initial
condition, ensemble member, random seed and batch size. The seed is the one
the member was run with: if its batch became unstable and was retried (see the
stability check in the main README), it is the seed of the retry. With field
output (`store_type` `netcdf` or `zarr`), the same seed is stored in the array
`random_seed`; NetCDF output is one file per batch.

`outputs_reproduce_helene/cyclone_tracks_te` holds three track files, identical
in name and content to those of the baseline. There are three because members
are reproduced as whole batches: the seeds of all batches are derived from
`random_seed: 7777`, and member two shares its batch with member three. A
reproduction therefore needs the batch size, ensemble size and random seed of
the original run.

To reproduce members by hand, take their seeds from the names of their track
files and pass them in place of the `'???'` placeholders of
`cfg/reproduce_helene.yaml`:

```bash
python ../main.py --config-path=$(pwd)/cfg --config-name=reproduce_helene.yaml \
    reproduce_members.0.2=<seed of member 3> reproduce_members.1.2=<seed of member 4>
```

## Test 3: Retries of Unstable Batches

```bash
./test_retries.sh          # FCN3
./test_retries.sh aifs     # AIFS-ENS v1
```

The script runs three members on one GPU with stability thresholds no forecast
can meet and `max_retries=1`, so every attempt of every batch is unstable. It
checks that each batch was attempted twice and then ended as failed: recorded
in `outputs_retries/retries_progress/failed-*.log`, with NaN fields, seed `-1`
and no track files. The run takes one to two minutes, most of it loading the
model.

With several GPUs (`torchrun`), a retry runs on whichever rank is free, see
the main README. How the ranks pass retries to each other is covered by unit
tests, which need neither a GPU nor a model and run in seconds:

```bash
pytest test/test_schedule.py
```

## Test 4: Extracting individual storms from historic data

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
