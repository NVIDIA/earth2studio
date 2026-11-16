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
signatures; see [Data and Metadata](coordinates.md).
The `Statistic` and `Metric` protocols live in `earth2studio.statistics.base`;
inheritance is optional. Weights, streaming updates, and ensemble handling are
options of particular implementations rather than universal requirements.

See [E2S-09: Statistics][statistics-spec] for dimension handling, input alignment,
weights, streaming state, and backend requirements.

[statistics-spec]: ../../specs/E2S-09_STATISTICS_SPEC.md

## Usage

First align forecast and observation labels and grids; metrics do not interpolate
or implicitly align coordinates. For fields with `lat` and `lon` dimensions:

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

Match weight axes to the reduced dimensions in input-dimension order; use actual
cell areas for other geometries. For ensemble forecasts, metrics such as
`rmse` accept `ensemble_dimension="ensemble"` to compare the ensemble mean with
observations; CRPS instead evaluates the predictive distribution.

To accumulate across time batches with consistent surviving coordinates:

```python
running_mean = mean(["time"], batch_update=True)
for batch in batches:
    average_so_far = running_mean(batch)
```

Each result covers all batches seen by that instance. Create a new instance to
start an independent stream.

## Create Your Own Statistic or Metric

Implement the interface in a local module. Declare reductions and any added axes in
`output_coords` without reading values; ensure execution returns exactly those
coordinates. Validate metric input compatibility before arithmetic so Xarray's
automatic alignment cannot silently change the comparison. If you support weights
or streaming updates, document their shapes and lifetime. See the
[extension examples](../../examples/index.md#extend) for custom components.
