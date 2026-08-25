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

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.utils.cupy import from_torch


def test_numpy_torch_and_batch_round_trip():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    array = xr.DataArray(
        data,
        dims=("member", "time", "variable"),
        coords={
            "member": [0, 1],
            "time": np.arange(3),
            "variable": ["a", "b", "c", "d"],
            "valid_time": ("time", np.arange(3) + 10),
        },
        name="state",
        attrs={"units": "K"},
    )

    tensor, coords = array.e2s.to_torch()
    assert tensor.data_ptr() == data.ctypes.data
    assert list(coords) == list(array.dims)
    tensor[0, 0, 0] = -1
    assert data[0, 0, 0] == -1

    restored = from_torch(tensor, coords, name=array.name, attrs=array.attrs)
    assert restored.data.ctypes.data == tensor.data_ptr()
    assert restored.name == array.name
    assert array.e2s.as_numpy().data is array.data

    unlabeled = xr.DataArray(np.ones((2, 3)), dims=("x", "y"))
    _, generated_coords = unlabeled.e2s.to_torch()
    np.testing.assert_array_equal(generated_coords["x"], np.arange(2))

    batched = array.e2s.batch(("member", "time"), contiguous=False)
    assert batched.dims == ("batch", "variable")
    assert np.shares_memory(batched.data, array.data)
    unbatched = batched.e2s.unbatch(contiguous=False)
    assert unbatched.dims == array.dims
    assert np.shares_memory(unbatched.data, batched.data)
    xr.testing.assert_identical(unbatched, array)


def test_batch_copy_and_validation():
    array = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("a", "b", "c"),
        coords={"a": np.arange(2), "b": np.arange(3), "c": np.arange(4)},
    )

    with pytest.raises(ValueError, match="requires a copy"):
        array.e2s.batch(("a", "c"), contiguous=False)
    batched = array.e2s.batch(("a", "c"))
    assert batched.data.flags.c_contiguous
    xr.testing.assert_identical(batched.e2s.unbatch(), array)

    with pytest.raises(ValueError, match="At least one"):
        array.e2s.batch(())
    with pytest.raises(ValueError, match="unique"):
        array.e2s.batch(("a", "a"))
    with pytest.raises(ValueError, match="not found"):
        array.e2s.batch(("missing",))
    with pytest.raises(ValueError, match="already exists"):
        array.e2s.batch(("a",), batch_dim="b")
    with pytest.raises(ValueError, match="does not contain"):
        array.e2s.unbatch()
    with pytest.raises(NotImplementedError, match="Gradient-preserving"):
        from_torch(
            torch.zeros(2, requires_grad=True),
            OrderedDict((("a", np.arange(2)),)),
            preserve_grad=True,
        )
    with pytest.raises(NotImplementedError, match="Gradient-preserving"):
        array.e2s.to_torch(preserve_grad=True)
    with pytest.raises(ValueError, match="rank"):
        from_torch(torch.zeros(2, 3), OrderedDict((("a", np.arange(2)),)))
    with pytest.raises(ValueError, match="dimension size"):
        from_torch(
            torch.zeros(2, 3),
            OrderedDict((("a", np.arange(2)), ("b", np.arange(2)))),
        )
    with pytest.raises(TypeError, match="Unsupported Torch device"):
        from_torch(
            torch.empty(2, device="meta"),
            OrderedDict((("a", np.arange(2)),)),
        )

    batched = array.e2s.batch(("a",))
    with pytest.raises(ValueError, match="already contains"):
        batched.e2s.batch(("batch",))
    with pytest.raises(ValueError, match="leading dimension"):
        batched.transpose("b", "batch", "c").e2s.unbatch()
    with pytest.raises(ValueError, match="size does not match"):
        batched.isel(batch=slice(1)).e2s.unbatch()
