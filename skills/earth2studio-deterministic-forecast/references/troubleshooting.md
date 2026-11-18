# Troubleshooting Guide

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError` on variable | Lexicon missing variable | Check compat; pick different source |
| `OutOfMemoryError` | VRAM exceeded | Use smaller model or free cache |
| `FileNotFoundError` package | Weights not cached | Call `load_default_package()` first |
| `ValueError` about forcing / slots | Model has forcing slots | Use `Pipeline.from_model` or a manual loop; see `forcing_coords()` |
| `ValueError` on store extent / write | Output template does not cover the written labels | Rebuild with `output_template(...)` covering all times and leads |
| `TimeoutError` data fetch | API slow/unreachable | Retry or use cached source |
| `ValueError: nsteps` | Horizon < model step | Increase horizon or finer model |

## Model-Data Source Compatibility

Common pairings:

Start from `model.default_sources()`. If you need another source, check that
its lexicon (`earth2studio/lexicon/`) has every variable in
`model.input_coords().coords["variable"]`, and that regional models get a source
covering their region. Reanalysis sources suit historical runs; operational
sources have limited history.

## IO Backend Selection

| Backend | Best for |
|---------|----------|
| ZarrBackend | Large outputs, chunked storage, recommended default |
| AsyncZarrBackend | Same as Zarr but async writes for performance |
| NetCDF4Backend | Compatibility with legacy tools |
| XarrayBackend | In-memory, small runs, interactive exploration |
| KVBackend | Key-value dict, debugging |

## Forecast + Diagnostic Compatibility

- Check diagnostic input variables, units/statistics, temporal dimensions, grid,
  and slot count against the full prognostic output. `Pipeline.from_model`
  applies each single-input, single-output diagnostic independently to forecast
  fields; it does not fetch missing diagnostic inputs or chain diagnostics.
- Keep inputs until after diagnosis. Pipeline's `output_variables` and
  `regridder` operate on published streams, so filtering the saved forecast does
  not remove diagnostic inputs. Output regridding cannot repair a mismatch
  between the forecast grid and the diagnostic input grid.
- If a diagnostic needs variables present only in forecast outputs, inspect
  `PrognosticRunner.output_coords` and `_publish` for the installed release.
  This checkout applies every diagnostic to a published initial condition, so
  missing lead-zero inputs fail. Upstream `1.0.0-rc` skips those diagnostics at
  lead zero and starts their streams at the first forecast. Do not promise
  identical lead axes for every model pair or fill missing inputs with zeros.
- Use one backend per named stream. `forecast` is reserved; a diagnostic key
  must match its `output` entry to be saved. Check `runner.output_coords(horizon)`
  for planned stream variables and leads before fetching data.

## Limitations

- Only deterministic (single-member) forecasts; use ensemble workflow for probabilistic runs
- Cannot train or fine-tune models — inference only
- Model weights require first-time download (several GB depending on model)
- Regional models (e.g. StormCast) require matching regional data sources
- GPU required; CPU-only inference is not supported for most models
