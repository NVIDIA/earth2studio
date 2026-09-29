# Array Backends and Gradients

Earth2Studio's `from_torch()` and `.e2s.to_torch()` bridge Torch tensors and
xarray DataArrays. Choose the Torch backend to retain autograd history across
component boundaries and supported xarray operations.

## Select a backend

Set the environment default **before importing Earth2Studio**:

```bash
EARTH2STUDIO_ARRAY_BACKEND=torch python your_script.py
```

| Setting | Payload |
| --- | --- |
| `auto` (default) | NumPy for CPU input, CuPy for CUDA input |
| `numpy` | NumPy on CPU; CUDA inputs are transferred |
| `cupy` | CuPy on CUDA; CPU inputs are transferred |
| `torch` | Torch-backed xarray adapter retaining the tensor and its graph |

A context manager overrides that default for new arrays; an explicit `backend=`
argument overrides both. Existing arrays keep their backend. Nested scopes restore
the previous setting on exit.

```python
import torch

from earth2studio.utils.cupy import backend, from_torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
with backend("torch"):
    array = from_torch(x, {"sample": [0, 1, 2]})
    result = (array * 2).mean("sample", skipna=False)
    result.e2s.to_torch()[0].backward()
torch.testing.assert_close(x.grad, torch.full_like(x, 2 / 3))
```

The runnable `dev/examples/05_array_backends.py` tutorial demonstrates all four
settings, nested scopes, explicit overrides, and a two-component gradient chain.

## Supported operations

The adapter implements a bounded subset of NumPy dispatch using Torch operations.
**Supported does not necessarily mean differentiable.** Gradients follow Torch's
rules for the operation and dtype; comparisons and index choices do not acquire
gradients merely because their input is tracked.

| Operations | Gradient behavior |
| --- | --- |
| Selection and indexing | Gradients reach selected values, not indices or labels |
| Transpose, reshape, moveaxis, expand/squeeze, broadcast | Preserve graph connections |
| Concat/stack, copies, batching/unbatching | Preserve graph connections, including through copies |
| Add, subtract, multiply, true divide, power, unary +/− | Native Torch derivatives where defined |
| Absolute, square, sqrt, exp, log | Use native Torch derivatives and domain restrictions |
| Sum, mean, nansum, nanmean | Use native Torch reductions; NaN-aware behavior follows Torch |
| Three-argument `where` | Gradients flow through selected value branches, not the condition |
| Casts, CPU/CUDA transfers | Float/complex casts preserve graphs; integer/bool casts do not |
| Remainder | Uses Torch's derivative where supported; discontinuities remain |
| Floor divide | Forward dispatch is supported; Torch does not implement its backward |
| Comparisons, isnan/isfinite, logical operations, invert | Discrete outputs; no gradients |
| `zeros_like` | Creates constant data without a connection to the input graph |

Selection includes `isel`, `sel`, basic/reversed slices, and orthogonal/vectorized
indexing. Batching uses `.e2s.batch()` and `.e2s.unbatch()`. Broadcast backward
accumulates contributions from the expanded dimensions.

Arithmetic follows Torch promotion rules. Reductions support axis, dtype and
keepdims, including empty-axis reductions; `out` is unsupported. Use `skipna=False`
for known NaN-free inputs to avoid NaN-aware reduction overhead. Dtype support is
limited to Torch types with NumPy representations; bfloat16 is not supported.
Individual Torch operations may impose additional dtype/device restrictions.

The implementation's `_TorchArray._operations`, `_reductions`, and `_parameters`
tables in `earth2studio/utils/cupy.py` list the dispatched NumPy functions and
accepted arguments. Indexing and array methods are implemented alongside them.
When extending these tables, update this reference and add value/gradient checks
in `test/utils/test_cupy.py` against the corresponding Torch operation.

## Graph boundaries

- `.e2s.to_torch()` extracts the original Torch tensor, preserving its history.
  `requires_grad=True` on a NumPy/CuPy import creates a fresh leaf; it cannot
  reconstruct a graph lost during an earlier export.
- Exports to NumPy/CuPy (`from_torch` with a non-Torch backend,
  `.e2s.as_numpy()`, `.e2s.as_cupy()`, or `.e2s.to_backend(...)`) warn and detach
  when the source requires gradients. `auto` is also a non-Torch export.
- Implicit NumPy conversion and unsupported adapter operations raise errors
  rather than silently exporting. Arbitrary NumPy/CuPy kernels and in-place
  updates are outside the supported interface.
- Models retain control of `no_grad` and `inference_mode`. Selecting the Torch
  backend does not make an inference-only model differentiable.
- Coordinates are CPU metadata, not differentiable tensors. Backend-specific
  consumers may still reject Torch-backed data; `handshake_device()` currently
  accepts only NumPy/CuPy payloads.
