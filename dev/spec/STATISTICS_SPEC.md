# Earth2Studio Statistics and Metrics

Statistics reduce a single field; metrics compare a forecast against an
observation. Both operate on Xarray `DataArray`s, matching the diagnostic model (DX)
interface.

## Protocol

```python
class Statistic(Protocol):
    @property
    def reduction_dimensions(self) -> list[str]: ...
    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem: ...
    def __call__(self, x: xr.DataArray) -> xr.DataArray: ...

class Metric(Protocol):
    @property
    def reduction_dimensions(self) -> list[str]: ...
    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem: ...
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray: ...
```

- `x` is the forecast, `y` the observation (`Metric` only).
- `CoordinateSystem` is an allocation-free `DataArray` signature (see
  `earth2studio.utils.coords`). `output_coords(x)` has the same dims, shape, and
  coordinates as the result of `__call__`.
- There is no `coords` argument and no tuple return. Tensor/`CoordSystem` calls are
  unsupported.

## Dimensions and Alignment

- Reduction dimensions are located by name in `x.dims`. A missing dimension raises
  `ValueError` in `__call__` and `KeyError` in `output_coords`.
- Reduced dimensions are removed; surviving dimensions keep their order and
  coordinates. Added dimensions (`threshold`, `window_size`, `wavenumber`) are
  appended; `metric`, `histogram_data`, and `bin` lead.
- `y` must have the same dimensions as `x` (minus `ensemble_dimension` when given;
  the ensemble dimension must exist in `x` and not in `y`). `y` is transposed to
  `x` order and coordinates are compared by value. Mismatches raise `ValueError`.
- Dimension names reserved by a method (`threshold`, `window_size`) may not appear
  in the inputs (`ValueError`).

## Weights

Weights are constructor arguments (`torch.Tensor` or `np.ndarray`, for example from
`lat_weight`). They have `ndim == len(reduction_dimensions)` (`ValueError`) and a
shape equal to the sizes of the reduction dimensions in `x` order (`AssertionError`).
`None` means uniform weights.

## Backends

- Inputs may be NumPy, CuPy, or Torch backed. Torch-backed input is converted with
  `to_backend('auto')` before NumPy/CuPy computation.
- NumPy/CuPy are used directly by `mean`, `variance`, `std`, `acc`, `brier_score`,
  `crps` (unfair), `rmse`, `mae`, `spread_skill_ratio`, `skill_spread`.
- Torch is used by `crps` (fair, `kcrps`), `rank_histogram`, `energy_score`, `fss`,
  and `log_spectral_distance`. Inputs are converted with `x.e2s.to_torch()` and the
  result returned with `from_torch`.
- Outputs are NumPy/CuPy-backed `DataArray`s on the device of the input.
- `crps`, `rank_histogram`, and `log_spectral_distance` require the `statistics`
  extra (PhysicsNeMo).

## Methods

| Method | Notes |
| --- | --- |
| `mean`, `std`, `variance` | `batch_update` keeps running sums across calls; `mean` with no reduction dims is the identity; `variance`/`std` require at least one |
| `acc` | requires `time` and `variable`; climatology is a `DataSource` (fetched with `fetch_data` when `lead_time` is present) or `None` for zero |
| `brier_score` | `thresholds`; optional `ensemble_dimension`; trailing `threshold` dim |
| `crps` | `ensemble_dimension` (str); optional `reduction_dimensions`, `weights`; `fair` selects the unbiased estimator |
| `rank_histogram` | leading `histogram_data` (`bin_centers`, `bin_counts`) and `bin`; bins default to ensemble size + 1; `randomize_ties` |
| `energy_score` | `ensemble_dimension`, non-empty `multivariate_dimensions`; `fair` needs at least 2 members (`ValueError`) |
| `fss` | `window_sizes`, `thresholds`; `spatial_dimensions` default to the last two; spatial dims must be reduced; spatial coordinates are cropped by the window margin; trailing `threshold`, `window_size` |
| `log_spectral_distance` | `spatial_dimensions` default to the last two; adds `wavenumber` before reduction; optional `wavenumber_cutoff` |
| `rmse`, `mae` | optional `ensemble_dimension` (ensemble mean taken first) |
| `spread_skill_ratio`, `skill_spread` | `skill_spread` has a leading `metric` dim (`mse`, `variance`) |

## Custom Statistics

Any object with the members above satisfies the protocol. A `Statistic` used by
workflows (for example the seasonal statistics example) is called as `stat(x)` and
its `output_coords(x)` describes the result without computing it.
