# Tests

## Test 1: Generate Ensemble Forecast with Cyclone Tracking

> [!Note]
> The test can be done manually following the steps below. Alternatively,
> `test_tc_hunt.sh` provides an automated way for running the full test.
> If, at the end you see a message containing `all good, yay (:`, the test
> finished successfully.

This case will test whether the pipeline can generate an ensemble forecast
using FCN3 and track tropical cyclones using TempestExtremes.

First, let us produce a five member ensemble of a 90h prediction of Hurricane Helene

```bash
cd earth2studio/recipes/tc_tracking/test
python ../tc_hunt.py --config-path=$(pwd)/cfg --config-name=baseline_helene.yaml
```

### Expected Result

You can find the track files in `outputs_baseline_helene/cyclone_tracks_te`.
The directory should contain five csv files with naming pattern according to
`tracks_2024-09-24T00:00:00_mem_000X_seed_YYYYYYYYY_bs_2.csv`
This pattern provides information about initial condition, ensemble member ID
and batch size.
