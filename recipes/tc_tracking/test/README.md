# Tests

## Test 1: Reproducibility of Helene Ensemble Members

> [!Note]
> The test can be done manually following the steps below. Alternatively,
> `test_tc_hunt.sh` provides an automated way for running the full test.
> If, at the end you see a message containing `all good, yay (:`, the test
> finished successfully.

This case will test the whether the pipeline can genreate an ensemble forecast
using FCN3 and track tropical cylones using TempestExtremes.
Furthermore, it is tested if those exact tracks can be reproduced.
Note that reproducing with AIFS-ENS won't work as the model does not provide
access to its internal random state.

First, let us produce a five member ensemble of a 90h prediction of Hurricane Helene
```bash
cd earth2studio/recipes/tc_tracking/test
python ../tc_hunt.py --config-path=$(pwd) --config-name=baseline_helene.yaml
```

You can find the track files in `outputs_baseline_helene/cyclone_tracks_te`.
The directory should contain five csv files with naming pattern accrding to
`tracks_2024-09-29T00:00:00_mem_000X_seed_YYYYYYYYY_bs_2.csv`
This pattern provides information about initial condition, ensemble member ID
and batch size.

Next, check if the random seeds for members three and four provided in `reproduce_helene.yaml`
are identical to the seeds in the csv file names. If not, replace the seeds for both members
in the yaml file with the seeds from the csv file names.

Having made sure that the configurations for reproduicing the ensemble members
is correct, let us now reproduce members three and four of the previous run:
```bash
python ../tc_hunt.py --config-path=$(pwd) --config-name=reproduce_helene.yaml
```
Note that the baseline run was produced with batch size two. For reproducibility,
we need to apply the same batch size in the reproduction run and reproduce the full
batch that contained the ensemble member in question. For the present case, this means
that member two will also be reproduced, as it is in the same batch as member three.

### Expected Result

There should now be three track files in `outputs_reproduce_helene/cyclone_tracks_te`,
with file names identical to the associated files in `outputs_baseline_helene/cyclone_tracks_te`.
If also the content of the files, ie the genereted tracks, is identical can be tested trhough
a simple `diff` call:

```bash
diff outputs_baseline_helene/cyclone_tracks_te/tracks_2024-09-29T00:00:00_mem_0002_seed_4045540270_bs_2.csv \
     outputs_reproduce_helene/cyclone_tracks_te/tracks_2024-09-29T00:00:00_mem_0002_seed_4045540270_bs_2.csv
```

Repeat the diff call for all reproduced ensemble members, the return should always be empty.


## Test 2: Extracting indivdual storms from historic data
