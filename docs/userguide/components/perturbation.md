# Perturbations { #perturbation_userguide }

Perturbations generate alternative initial conditions for ensemble forecasts, from
simple Gaussian noise to spatially correlated noise and bred vectors. Browse the
[perturbation API](../../modules/perturbation.md) for available methods.

## Interface

```python
perturbed = perturbation(x)  # DataArray -> DataArray
```

A perturbation accepts a field and returns the **perturbed field**, not just noise.
The [`earth2studio.perturbation.base.Perturbation`][interface] protocol requires only this call;
inheritance is optional. Built-in methods accept NumPy-, CuPy-, or Torch-backed
DataArrays and use PyTorch for computation.

See [E2S-08: Perturbations][perturbation-spec] for field preservation,
method-specific dimensions, and backend behavior.

[perturbation-spec]: ../../specs/E2S-08_PERTURBATION_SPEC.md
[interface]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/perturbation/base.py

## Usage

Given a `t2m` field `x` in kelvin, construct four members with independent noise:

```python
import numpy as np

from earth2studio.perturbation import Gaussian

ensemble = x.expand_dims(ensemble=np.arange(4))
perturbation = Gaussian(noise_amplitude=0.1, seed=42)  # 0.1 K for this field
perturbed = perturbation(ensemble)
```

`Gaussian` uses a private random generator: pass `seed` for reproducible runs, or
call `perturbation.set_rng(42)` to restart its stream. `torch.manual_seed` does not
control every perturbation; check the selected method's seeding interface.

Choose amplitudes in the field's physical units. For multiple variables, a
broadcastable tensor can assign a different scale to each variable. Check the
selected method's grid and dimension requirements before applying it; Gaussian
does not impose a spatial layout.

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
