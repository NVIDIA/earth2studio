# Tests

> [!Note]
> Here, we demonstrate how to run the recommended way through uv.
If you prefer running in a different python environment, simply
replace `uv run script.py` commands with `python script.py`.

## Test 1: DLESyM ensemble verification

Run the PNW heatwave DLESyM ensemble, limiting the total number of checkpoints to reduce
runtime:

```bash
uv run main.py --config-name pnw_dlesym ncheckpoints=2
```

This will produce outputs in the relative path `outputs/pnw_heatwave_dlesym_4a4o4p_gaussian/north_america`.
Verify the correctness of these outputs by running the testing script against the relative
path:

```bash
cd earth2studio/recipes/s2s/test
uv run test_skill.py --model dlesym --path outputs/pnw_heatwave_dlesym_4a4o4p_gaussian/north_america/forecast.zarr
```

### Expected DLESyM Result

If successful, the test script will report

```bash
Expected skill verified for dlesym
```

Otherwise, the script will report

```bash
Expected skill not verified for dlesym:
...
```

If a failure is encountered, the script will report which variables and lead times are
producing unexpected results

## Test 2: HENS-SFNO ensemble verification

Run the PNW heatwave SFNO ensemble, limiting the total number of checkpoints to reduce
runtime:

```bash
uv run main.py --config-name pnw_sfno ncheckpoints=2
```

This will produce outputs in the relative path `outputs/pnw_heatwave_sfnohens_16c4p/north_america`.
Verify the correctness of these outputs by running the testing script against the relative
path:

```bash
cd earth2studio/recipes/s2s/test
uv run test_skill.py --model sfno --path outputs/pnw_heatwave_sfnohens_16c4p/north_america/forecast.zarr
```

### Expected SFNO Result

If successful, the test script will report

```bash
Expected skill verified for sfno
```

Otherwise, the script will report

```bash
Expected skill not verified for sfno:
...
```

If a failure is encountered, the script will report which variables and lead times are producing
unexpected results
