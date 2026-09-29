# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare backend settings from the repository root:

EARTH2STUDIO_ARRAY_BACKEND=auto uv run dev/examples/05_array_backends.py
EARTH2STUDIO_ARRAY_BACKEND=numpy uv run dev/examples/05_array_backends.py
EARTH2STUDIO_ARRAY_BACKEND=torch uv run dev/examples/05_array_backends.py
EARTH2STUDIO_ARRAY_BACKEND=cupy uv run dev/examples/05_array_backends.py

The CuPy setting requires CUDA and CuPy. Auto uses NumPy for CPU tensors and
CuPy for CUDA tensors. Set the environment variable before importing Earth2Studio.
Context-manager overrides below work independently of that environment default.
Components must execute with autograd enabled; the backend does not override
no_grad or inference_mode.
"""

import os
from importlib.util import find_spec

import torch
import xarray as xr

from earth2studio.utils.cupy import backend, from_torch


def double(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor * 2, array)


def square(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor.square(), array)


if __name__ == "__main__":
    coords = {"sample": [0, 1, 2]}
    source = torch.tensor([1.0, 2.0, 3.0])
    cupy_available = torch.cuda.is_available() and find_spec("cupy") is not None
    policy = os.getenv("EARTH2STUDIO_ARRAY_BACKEND", "auto")
    if policy == "cupy" and not cupy_available:
        raise SystemExit("The cupy environment setting requires CUDA and CuPy.")

    # No explicit backend: use the environment default, read at import time.
    default_array = from_torch(source, coords)
    print(f"Environment ({policy}), CPU input: {type(default_array.data)}")

    # A context manager overrides the environment for new arrays only.
    for setting in ("auto", "numpy", "torch", "cupy"):
        if setting == "cupy" and not cupy_available:
            print("Context (cupy): skipped; requires CUDA and CuPy")
            continue
        with backend(setting):
            array = from_torch(source, coords)
            tensor, _ = array.e2s.to_torch()
            print(f"Context ({setting}): {type(array.data)}, {tensor.device}")

    # Auto follows the source device, rather than always choosing NumPy.
    if cupy_available:
        with backend("auto"):
            array = from_torch(source.cuda(), coords)
            print(f"Context (auto), CUDA input: {type(array.data)}")

    # Nested scopes restore the outer setting; explicit arguments take precedence.
    with backend("torch"):
        with backend("numpy"):
            array = from_torch(source, coords)
            print(f"Nested numpy context: {type(array.data)}")
            explicit = from_torch(source, coords, backend="torch")
            print(f"Explicit torch override: {type(explicit.data)}")
        outer = from_torch(source, coords)
        print(f"Restored outer torch context: {type(outer.data)}")
    restored = from_torch(source, coords)
    print(f"Restored environment default: {type(restored.data)}")

    # Existing conversion calls preserve gradients throughout a Torch scope.
    x = source.clone().requires_grad_(True)
    with backend("torch"):
        array = from_torch(x, coords)
        result = square(double(array))
        result.sum(skipna=False).e2s.to_torch()[0].backward()
    torch.testing.assert_close(x.grad, 8 * x.detach())
    print(f"Input gradients: {x.grad}")  # tensor([8., 16., 24.])

    # Explicit exports detach and warn when autograd history is being dropped.
    exported = result.e2s.to_backend("numpy")
    print(f"Detached NumPy export: {exported.data}")
    print(f"Export tracks gradients: {exported.e2s.to_torch()[0].requires_grad}")
