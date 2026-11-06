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

## Limitations

- Only deterministic (single-member) forecasts; use ensemble workflow for probabilistic runs
- Cannot train or fine-tune models — inference only
- Model weights require first-time download (several GB depending on model)
- Regional models (e.g. StormCast) require matching regional data sources
- GPU required; CPU-only inference is not supported for most models
