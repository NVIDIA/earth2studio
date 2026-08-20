---
# Site-wide scorecard defaults, read by docs/generate_scorecard.py.
#
# labels: fallback display names for models that have no config/<model>.md
#         (a per-model config's `label:` always wins).
# metrics.lower_is_better: metrics whose skill curves read "lower is better";
#         drives the plot's subtitle. Anything not listed reads higher-better
#         (acc) or has its own rule (spread_skill targets 1.0).
labels:
  fcn3: FCN3
  aurora: Aurora
  sfno: SFNO
  fengwu: FengWu
  ucast: UCast
  graphcast: GraphCast
  graphcast_small: GraphCast-small
  pangu3: Pangu (3 h)
  pangu6: Pangu (6 h)
  pangu24: Pangu (24 h)
# data_sources: display names for earth2studio data source classes shown in
#         the Evaluation table (falls back to the raw class name).
data_sources:
  ARCO: ERA5 (ARCO)
  GFS: GFS
  HRRR: HRRR
metrics:
  lower_is_better: [rmse, mae, lsd, ensemble_mean_mse, crps, ensemble_variance]
---
