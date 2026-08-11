# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.utils.cupy import from_torch

cp = pytest.importorskip("cupy", reason="CuPy is required for GPU integration tests")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cupy_torch_and_batch_round_trip():
    array = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        dims=("member", "time", "variable"),
        coords={
            "member": np.arange(2),
            "time": np.arange(3),
            "variable": ["a", "b", "c", "d"],
        },
    ).e2s.as_cupy(device=0)
    assert array.e2s.is_cupy
    assert array.e2s.as_cupy().data is array.data

    tensor, coords = array.e2s.to_torch()
    assert tensor.data_ptr() == array.data.data.ptr
    tensor[0, 0, 0] = -1
    assert int(array.data[0, 0, 0]) == -1

    restored = from_torch(tensor, coords)
    assert restored.data.data.ptr == tensor.data_ptr()
    assert restored.e2s.is_cupy

    batched = restored.e2s.batch(("member", "time"), contiguous=False)
    assert cp.shares_memory(batched.data, restored.data)
    unbatched = batched.e2s.unbatch(contiguous=False)
    assert cp.shares_memory(unbatched.data, batched.data)
    cp.testing.assert_array_equal(unbatched.data, restored.data)

    host = unbatched.e2s.as_numpy()
    assert isinstance(host.data, np.ndarray)
    np.testing.assert_array_equal(host.data, cp.asnumpy(restored.data))

    reordered = restored.transpose("variable", "member", "time")
    copied = reordered.e2s.batch(("variable", "time"))
    assert copied.data.flags.c_contiguous
    cp.testing.assert_array_equal(copied.e2s.unbatch().data, reordered.data)
