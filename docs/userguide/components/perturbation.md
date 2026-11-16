# Perturbations { #perturbation_userguide }

Perturbations generate alternative initial conditions for ensemble forecasts, from
simple Gaussian noise to spatially correlated noise and bred vectors. Browse the
[perturbation API](../../modules/perturbation.md) for available methods.

## Interface

```python
perturbed = perturbation(x)  # DataArray -> DataArray
```

A perturbation accepts a field and returns the **perturbed field**, not just noise.
The `earth2studio.perturbation.base.Perturbation` protocol requires only this call;
inheritance is optional. Built-in methods accept NumPy-, CuPy-, or Torch-backed
DataArrays and use PyTorch for computation.

## Specification

The [perturbation specification][perturbation-spec] defines the following requirements
without numbered compliance codes.

- **Preserve structure.** Output dimensions, shape, coordinates, and attributes match
  the input. Input values and metadata are not modified in place.
- **Use physical amplitudes.** Noise is applied to unnormalized fields. A scalar
  `noise_amplitude` applies one scale everywhere; a broadcastable tensor can assign
  different scales to variables. Any internal normalization must be reversed.
- **Meet method-specific geometry.** Gaussian and Zero impose no spatial layout.
  Brown and spherical methods require trailing `lat, lon` dimensions; spherical
  methods require an `N × 2N` or `(N+1) × 2N` grid. `LaggedEnsemble` requires leading
  `ensemble, time, lead_time` dimensions, with one ensemble member per lag.
  `HemisphericCentredBredVector` requires trailing `time, lead_time, variable, lat, lon`,
  one time, and at most one leading batch axis. Missing dimensions raise `KeyError`;
  wrong positions or sizes raise `ValueError`. `BredVector` warns for multiple lead times.
- **Control randomness.** Use `torch.manual_seed` for reproducible stochastic
  perturbations. Gaussian also supports its own generator state; consult its API
  when checkpointing or managing independent random streams.
- **Account for backend policy.** The default `auto` policy returns NumPy on CPU and
  CuPy on CUDA; `torch` preserves tensors. Dtype is preserved except for
  `HemisphericCentredBredVector`, which uses its fetched source field's dtype.
- **Match the call contract.** Accept and return one DataArray, with no separate
  coordinate argument or tuple return. Model-based perturbations accept a callable
  `DataArray -> DataArray`. Lagged and hemispheric methods fetch on the input device.

[perturbation-spec]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/dev/spec/PERTURBATION_SPEC.md

## Usage

Given a `t2m` field `x` in kelvin, construct four members with independent noise:

```python
import numpy as np
import torch

from earth2studio.perturbation import Gaussian

torch.manual_seed(42)
ensemble = x.expand_dims(ensemble=np.arange(4))
perturbation = Gaussian(noise_amplitude=0.1)  # 0.1 K for this single-variable field
perturbed = perturbation(ensemble)
```

The built-in `earth2studio.run.ensemble` workflow expands the ensemble dimension
and applies the perturbation once to the initial input. For bred vectors, provide
a model-like callable accepting and returning DataArrays; see the method's API
for its integration and rescaling options.

## Create Your Own Perturbation

Implement a callable in your own project. For example, a multiplicative perturbation
can preserve metadata and leave its input untouched:

```python
import xarray as xr


class Scale:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return x.copy(data=x.data * self.factor)
```

Check that your method preserves coordinates and attributes, does not modify its
input, and produces the intended amplitude in physical units. See the
[extension examples](../../examples/index.md#extend) for workflow integrations.
