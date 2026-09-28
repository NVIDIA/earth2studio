# Configurable xarray payload backends

## Scope

Preserve PyTorch graphs across existing `from_torch()` / `.e2s.to_torch()`
handoffs and supported xarray operations. Production changes live in
`earth2studio/utils/cupy.py`; tests in `test/utils/test_cupy.py`.
Models retain ownership of inference mode and gradient policy.

Worktree: `/localhome/local-ngeneva/workspace/earth2studio-gradient-flow`.
Branch: `ngeneva/gradient-flow`, stacked on `ngeneva/projected-grid-followup`.

## Backend policy

Precedence: explicit `backend=` → context-local `backend(...)` →
`EARTH2STUDIO_ARRAY_BACKEND` (validated once at import) → `auto`.
Nested scopes restore their previous value even after exceptions.
Policy changes affect future wrapping, not existing arrays.

| Backend | Output |
| --- | --- |
| `auto` | NumPy for CPU sources, CuPy for CUDA sources |
| `numpy` | CPU NumPy, transferring from CUDA if needed |
| `cupy` | CUDA CuPy, transferring from CPU if needed |
| `torch` | Private xarray duck array holding the original Torch tensor |

```python
from earth2studio.utils.cupy import backend, from_torch

with backend("torch"):
    result = pipeline(x)

array = from_torch(tensor, coords, backend="torch")
array = array.e2s.to_backend("torch", device="cuda:0")
```

For configuration-only use, set `EARTH2STUDIO_ARRAY_BACKEND=torch` before
import. `from_torch`'s backend argument is keyword-only; existing calls remain
valid. `to_backend(backend, *, device=None)` explicitly converts existing data
and retains coordinates, name, attributes and encoding. CPU and CUDA are
supported; integer devices mean CUDA indices. NumPy rejects CUDA targets;
CuPy rejects CPU targets. CuPy defaults to the source CUDA device or the
current device for CPU inputs. `auto` resolves from the source device.
CuPy stays lazily imported; invalid backend names raise `ValueError`.

### Minimal gradient example

Save this as `backend_example.py` and run it with the environment setting applied
before import:

```bash
EARTH2STUDIO_ARRAY_BACKEND=torch uv run backend_example.py
```

```python
import torch
import xarray as xr

from earth2studio.utils.cupy import from_torch


def double(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor * 2, array)


def square(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor.square(), array)


x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
array = from_torch(x, {"sample": [0, 1, 2]})
result = square(double(array))
result.sum(skipna=False).e2s.to_torch()[0].backward()
torch.testing.assert_close(x.grad, 8 * x.detach())  # [8, 16, 24]
```

For a scoped setting instead, put wrapping and both component calls inside
`with backend("torch"):` after importing `backend` from the same module.
Both components must execute with autograd enabled; the backend setting does
not override `torch.no_grad()` or `torch.inference_mode()`.

## Autograd semantics

- Torch extraction returns the underlying tensor, independently of policy.
  Torch wrapping and CPU/CUDA transfers preserve existing history.
- Every export of gradient-tracked data to NumPy/CuPy emits one filterable
  `UserWarning`, then detaches. This includes `auto`, `as_numpy`, `as_cupy`
  and `to_backend`. Untracked exports and Torch handoffs do not warn.
- `requires_grad=False` preserves existing tracking. `True` enables it on a
  new alias when needed, without mutating the caller's flag. Only floating
  and complex tensors can track gradients. Reimported NumPy/CuPy arrays may
  start fresh leaves but cannot recover an exported graph.
- Coordinates remain CPU metadata; graph-bearing data lives in the payload,
  never in attributes. Existing metadata behavior is retained.

## Supported operations and limits

The private adapter runs selection (basic, reversed, orthogonal and vectorized),
transpose/reshape, expand/squeeze/broadcast, concat/stack, shallow/deep copies,
arithmetic/comparisons, sum/mean, casts, and shared batching with Torch.
Differentiable results retain graph connections. Reductions support floating
skip-NaN paths, explicit supported dtypes, empty-axis and keepdims behavior.
Batching uses views when `contiguous=False`, rejecting copy-required layouts.

Unsupported functions/options and implicit NumPy coercion raise errors.
Explicit `.e2s.as_numpy()` is the export boundary. The initial contract excludes
in-place updates, arbitrary NumPy/CuPy kernels and Torch dtypes without NumPy
representations (such as bfloat16). Arithmetic follows Torch promotion rules.
This is bounded xarray interoperability, not a general NumPy replacement.

Dispatch tables are shared; NumPy dtype lookup is cached per Torch dtype using
an explicit CPU scalar on first use. Wrapping never exports the tensor payload.
Basic slicing, transpose and broadcast use Torch views. Advanced/reversed
indexing, concatenation and requested contiguous layouts can allocate copies.
NumPy constants/index arrays used on CUDA require transfers; reuse Torch-backed
constants when practical. For known NaN-free data, `skipna=False` avoids the
additional work inherent in NaN-aware reductions.

Backend-specific consumers still require their expected payload. In particular,
`utils.coords.handshake_device()` currently accepts only NumPy/CuPy and rejects
the Torch adapter; supporting it requires a separate coordinate-utility change.
Models using `inference_mode` still disable autograd. The environment setting
controls these conversion helpers, not third-party DataArray constructors.

## Verification

Tests cover policy precedence/isolation, round-trip identity and gradients,
metadata/storage sharing, supported xarray operations against direct Torch,
NaNs and reduction dtypes, explicit rejection, batching and a two-component
chain with unchanged conversion calls. CUDA tests cover differentiable device
transfers and detached CuPy exports/reimports. Environment tests use monkeypatch,
not subprocesses or module reloads. See the plan for commands and results.
