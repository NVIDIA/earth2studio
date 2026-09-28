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

"""Run from the repository root:

EARTH2STUDIO_ARRAY_BACKEND=torch uv run dev/scratch/backend_example.py

Set the environment variable before import. Components must execute with
autograd enabled; the backend does not override no_grad or inference_mode.
"""

import torch
import xarray as xr

from earth2studio.utils.cupy import from_torch


def double(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor * 2, array)


def square(array: xr.DataArray) -> xr.DataArray:
    tensor, _ = array.e2s.to_torch()
    return from_torch(tensor.square(), array)


if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    array = from_torch(x, {"sample": [0, 1, 2]})
    result = square(double(array))
    result.sum(skipna=False).e2s.to_torch()[0].backward()
    torch.testing.assert_close(x.grad, 8 * x.detach())
    print(f"Input gradients: {x.grad}")  # tensor([8., 16., 24.])
