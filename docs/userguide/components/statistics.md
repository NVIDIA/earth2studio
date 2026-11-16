# Statistics and Metrics { #statistics_model_userguide }

Statistics summarize a field, such as its mean or variance. Metrics compare a
prediction with observations, such as root-mean-square error (RMSE) or continuous
ranked probability score (CRPS). Browse the
[statistics API](../../modules/statistics.md) for available operations.

## Interface

| Member | Statistic | Metric |
| --- | --- | --- |
| `reduction_dimensions` | Names of dimensions to reduce | Names of dimensions to reduce |
| `output_coords(x)` | Plan from the input signature | Plan from the forecast signature |
| `__call__(...)` | `statistic(x)` | `metric(x, y)` |

Inputs and outputs are DataArrays. Coordinate planning uses allocation-free
signatures; see [Coordinate Systems](../about/overview.md#coordinates_userguide).
The `Statistic` and `Metric` protocols live in `earth2studio.statistics.base`;
inheritance is optional. Weights, streaming updates, and ensemble handling are
options of particular implementations rather than universal requirements.

## Specification

The [statistics specification][statistics-spec] defines the following requirements
without numbered compliance codes.

- **Reduce by name.** Reduced dimensions must exist. Surviving dimensions retain
  their input order. Some methods also add dimensions, such as `threshold`,
  `window_size`, or `wavenumber`; inspect `output_coords` rather than assuming the
  result is always a subset of input dimensions.
  Missing dimensions raise `ValueError` during execution and `KeyError` in planning.
  Added `threshold`, `window_size`, and `wavenumber` axes trail; `metric`,
  `histogram_data`, and `bin` lead. Method-reserved dimension names cannot appear in inputs.
- **Align comparisons exactly.** Metrics require the same dimension names and
  coordinate values, apart from a declared ensemble dimension present only on the
  forecast. Observation axes may be transposed to match, but there is no implicit
  coordinate alignment or interpolation.
- **Match weight shape.** Where supported, weights are NumPy arrays or Torch tensors
  with one axis per reduced dimension, matching their sizes and input-dimension
  order. They are not arbitrary broadcastable arrays. `None` gives uniform weights.
  Incorrect weight rank raises `ValueError`; incorrect shape raises `AssertionError`.
- **Keep streams separate.** `mean`, `var`, and `std` with `batch_update=True` retain
  running accumulators. Each call returns the statistic over all batches seen;
  create a new instance for an independent stream.
- **Plan for the output backend.** Computation uses PyTorch. The `auto` backend
  policy returns NumPy on CPU and CuPy on CUDA; `torch` retains tensors and autograd
  history. Inputs are not modified.
- **Match planning to execution.** `output_coords(x)` describes the result's dimensions,
  shape, and coordinates. Calls accept DataArrays with no separate coordinate argument
  and return one DataArray. Weights are placed on the input device; uniform weights
  use float32, and float16/bfloat16 weights are promoted to float32 for reductions.

[statistics-spec]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/dev/spec/STATISTICS_SPEC.md

## Usage

For aligned forecast and observation fields with `lat` and `lon` dimensions:

```python
from earth2studio.statistics import mean, rmse

spatial_mean = mean(["lat", "lon"])
average = spatial_mean(forecast)

error = rmse(["lat", "lon"])
planned = error.output_coords(forecast)
score = error(forecast, observations)
```

For a regular latitude-longitude grid ordered `..., lat, lon`, cosine-latitude
weights give an area-weighted mean:

```python
import numpy as np

lat = forecast.coords["lat"].values
weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, forecast.sizes["lon"]))
area_mean = mean(["lat", "lon"], weights=weights)
average = area_mean(forecast)
```

Use actual cell areas for other geometries. For ensemble forecasts, metrics such as
`rmse` accept `ensemble_dimension="ensemble"` to compare the ensemble mean with
observations; CRPS instead evaluates the predictive distribution.

To accumulate across time batches with consistent surviving coordinates:

```python
running_mean = mean(["time"], batch_update=True)
for batch in batches:
    average_so_far = running_mean(batch)
```

## Create Your Own Statistic or Metric

Implement the interface in a local module. Declare reductions and any added axes in
`output_coords` without reading values; ensure execution returns exactly those
coordinates. Validate metric input compatibility before arithmetic so Xarray's
automatic alignment cannot silently change the comparison. If you support weights
or streaming updates, document their shapes and lifetime. See the
[extension examples](../../examples/index.md#extend) for custom components.
