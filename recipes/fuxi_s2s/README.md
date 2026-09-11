# FuXi-S2S inference from ARCO ERA5

This recipe runs the FuXi-S2S prognostic model from public hourly ARCO ERA5
data. `FuXiS2SERA5` creates the two UTC daily means required by the model and
samples ERA5's 0.25-degree grid onto FuXi-S2S's native 1.5-degree grid.

The same entry point supports one stochastic trajectory or an arbitrary
ensemble size. FuXi-S2S draws stochastic perturbations inside its ONNX graph,
so the ensemble workflow uses `Zero` and does not perturb the initial
conditions a second time.

## Prerequisites

- A CUDA-capable NVIDIA GPU.
- Network access to the FuXi-S2S checkpoint and public ARCO ERA5 store.
- Sufficient disk space for the checkpoint, cached ERA5 chunks, and Zarr
  forecast output.

The FuXi-S2S checkpoint has a CC BY-NC-ND 4.0 license and is restricted to
non-commercial research use. Review its model-card terms before running it.

## Quick start

From this directory, create the recipe environment:

```bash
uv sync
```

Run one stochastic trajectory for 14 forecast days:

```bash
uv run python main.py \
  --issue-time 2020-06-03T00:00:00Z \
  --nsteps 14 \
  --members 1 \
  --output outputs/fuxi_s2s_single.zarr
```

Run an 11-member, six-week ensemble while retaining one member on the GPU at a
time:

```bash
uv run python main.py \
  --issue-time 2020-06-03T00:00:00Z \
  --nsteps 42 \
  --members 11 \
  --batch-size 1 \
  --output outputs/fuxi_s2s_11_member.zarr
```

Create the two weekly plot products from that Zarr store:

```bash
uv run python plot.py \
  --input outputs/fuxi_s2s_11_member.zarr \
  --output-dir outputs/fuxi_s2s_11_member_plots
```

The plotting command writes:

- `fuxi_s2s_weekly_ensemble_mean.png`: weekly ensemble-mean maps of 7-day
  mean 2-m temperature and 7-day accumulated precipitation, wrapped into a
  readable three-week-wide layout.
- `fuxi_s2s_ensemble_member_distributions.png`: every member's global
  area-weighted weekly values, with the sample median and either the
  interquartile range or member range. A one-member run is labeled as a single
  stochastic trajectory without sample-distribution statistics.
- `fuxi_s2s_plot_summary.json`: input provenance, aggregation semantics,
  weekly statistics, and image-validation metadata.

Only complete seven-day periods are plotted. FuXi-S2S `tp` is a daily mean of
24 one-hour accumulations, so `plot.py` multiplies it by 24 before summing each
week. Use `--tp-semantics daily-total` only for a store whose `tp` field has
already been converted to daily totals. The figures describe raw stochastic
members; they are not calibrated probabilities or verification against
observations.

Use `--variables all` to save every model field. Otherwise the default output
contains `t2m`, `z500`, `msl`, and `tp`; changing the saved variables does not
reduce the 76 fields needed for model initialization.

## Time convention

`--issue-time` is the strict forecast issue time. The recipe subtracts one day
before calling Earth2Studio because FuXi-S2S consumes two completed calendar-day
means. For issue time `2020-06-03T00:00:00Z`, the data source supplies means
labeled June 1 and June 2. The June 2 `tp` and `ttr` means use interval-ending
samples through June 3 at 00 UTC, so no future observations enter the forecast.

## Ensemble behavior

- `--members 1` uses `run.deterministic`, but the result is still one
  stochastic FuXi-S2S trajectory.
- `--members N`, for `N > 1`, uses `run.ensemble` and writes an `ensemble`
  dimension of size `N`.
- `--batch-size` controls how many members are resident in each inference batch;
  it is not a limit on total ensemble size.
- The released checkpoint exposes neither a deterministic random seed nor a
  special control member.

Runtime and output storage grow approximately linearly with the ensemble size.
The first run also downloads the checkpoint and many ERA5 chunks, so it may take
substantially longer than a cached run. Set `EARTH2STUDIO_CACHE` to choose a
persistent cache location.

The supported `--nsteps` range is 1–42 days. The output Zarr root records both
`forecast_issue_time_utc` and `latest_complete_daily_mean_utc`, so downstream
processing does not need to reconstruct the one-day label offset.

The checkpoint exposes no checkpoint-level or per-member seed input. This CLI
does not configure ONNX Runtime's process-wide random seed, so trajectories are
not intended to be bitwise reproducible between invocations.

The script refuses to replace an existing Zarr store unless `--overwrite` is
provided.

## Tests

The unit tests mock checkpoint and data downloads while verifying single-member
and ensemble workflow routing, weekly aggregation, member statistics, and plot
rendering:

```bash
uv run pytest
```

## References

- [Earth2Studio](https://nvidia.github.io/earth2studio/)
- [FuXi-S2S](https://github.com/tpys/FuXi-S2S)
- [ARCO ERA5](https://github.com/google-research/arco-era5)
