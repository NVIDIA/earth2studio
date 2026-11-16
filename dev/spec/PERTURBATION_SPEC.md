# Earth2Studio Perturbations

Perturbations generate ensemble initial conditions from a state. They operate on
Xarray `DataArray`s, matching the diagnostic model (DX) interface.

## Protocol

```python
class Perturbation(Protocol):
    def __call__(self, x: xr.DataArray) -> xr.DataArray: ...
```

- Input is one `DataArray` holding the state; dimensions and coordinates describe
  the array (for example `ensemble, time, lead_time, variable, lat, lon`).
- Output has the same dimensions, shape, coordinates, and attributes as the input.
  Coordinates are never changed by a perturbation. Dtype matches the input, except
  `HemisphericCentredBredVector`, which follows the dtype of its data source.
- The input is not modified in place.
- There is no `coords` argument and no tuple return. Tensor/`CoordSystem` calls are
  unsupported.

## Dimensions

Perturbations locate dimensions by name using `handshake_dim` / `handshake_size`
and raise `KeyError` (missing dimension) or `ValueError` (wrong position or size).

- `Zero`, `Gaussian`: none.
- `Brown`, `SphericalGaussian`, `CorrelatedSphericalGaussian`: `lat` (-2), `lon` (-1).
- `BredVector`: none; `lead_time` of size > 1 warns.
- `HemisphericCentredBredVector`: `time` (-5, size 1), `lead_time` (-4),
  `variable` (-3), `lat` (-2), `lon` (-1). Must have 5 or 6 dims with an optional
  leading batch dim (`ValueError` otherwise).
- `LaggedEnsemble`: `ensemble` (0, size equals number of lags), `time` (1),
  `lead_time` (2).

Spherical methods require a lat/lon grid with aspect ratio N:2N or N+1:2N
(`ValueError` otherwise).

## Backends and Execution

- Array backends (NumPy, CuPy, Torch) are accepted. Perturbations convert with
  `x.e2s.to_torch()`, compute with Torch on the data's device, and return
  `from_torch(out, x)`. The output backend follows the configured array backend
  policy (`auto` unless set with `backend(...)`).
- `noise_amplitude` is a `float` or a `torch.Tensor` broadcastable with the data.
- Stochastic methods draw from Torch random state (`torch.manual_seed`).
  `Gaussian` additionally owns a `torch.Generator` that supports checkpointing.
- Perturbations that call a model (`BredVector`, `HemisphericCentredBredVector`)
  take a callable `DataArray -> DataArray` (for example a prognostic model's
  forward). Model outputs of any backend are accepted.
- `LaggedEnsemble` and `HemisphericCentredBredVector` fetch data with `fetch_data`
  on the device of the input.

## Runner Integration

`run.ensemble` expands the initial state with an `ensemble` dimension and applies
`x = perturbation(x)` once before the first model step.

## Custom Perturbations

Any callable with the signature above satisfies the protocol. Wrappers that perturb
a subset of variables select by coordinate, for example
`out.loc[{"variable": vars}] = pm(x).sel(variable=vars)`.
