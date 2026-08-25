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

import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.utils import cupy as cupy_utils
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
    assert restored.name == array.name and restored.attrs == array.attrs
    assert array.e2s.as_numpy().data is array.data

    _, generated_coords = xr.DataArray(np.ones((2, 3)), dims=("x", "y")).e2s.to_torch()
    np.testing.assert_array_equal(generated_coords["x"], np.arange(2))

    batched = array.e2s.batch(("member", "time"), contiguous=False)
    assert batched.dims == ("batch", "variable")
    assert np.shares_memory(batched.data, array.data)
    unbatched = batched.e2s.unbatch(contiguous=False)
    assert np.shares_memory(unbatched.data, batched.data)
    xr.testing.assert_identical(unbatched, array)


def test_batch_copy_and_validation(monkeypatch):
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

    lazy = xr.DataArray(da.arange(2, chunks=1), dims=("a",))
    assert isinstance(lazy.e2s.as_numpy().data, np.ndarray)
    for operation in (lambda: lazy.e2s.batch(("a",)), lazy.e2s.to_torch):
        with pytest.raises(TypeError, match="only NumPy- or CuPy"):
            operation()

    def missing_cupy(_: str) -> None:
        raise ImportError

    monkeypatch.setattr(cupy_utils, "import_module", missing_cupy)
    with pytest.raises(ImportError, match="CuPy is required"):
        array.e2s.as_cupy()
    assert not array.e2s.is_cupy

    with pytest.raises(ValueError, match="At least one"):
        array.e2s.batch(())
    with pytest.raises(ValueError, match="unique"):
        array.e2s.batch(("a", "a"))
    with pytest.raises(ValueError, match="not found"):
        array.e2s.batch(("missing",))
    mixed_coord = array.assign_coords(mixed=(("a", "c"), np.ones((2, 4))))
    with pytest.raises(NotImplementedError, match="batched and unbatched"):
        mixed_coord.e2s.batch(("a", "b"))
    with pytest.raises(ValueError, match="Recursive batching"):
        array.e2s.batch(("a",), batch_dim="b")
    with pytest.raises(ValueError, match="coordinate 'batch' already exists"):
        array.assign_coords(batch=1).e2s.batch(("a",))
    with pytest.raises(ValueError, match="does not contain"):
        array.e2s.unbatch()
    coords = OrderedDict((("a", np.arange(2)),))
    with pytest.raises(NotImplementedError, match="requires_grad=True"):
        from_torch(torch.zeros(2, requires_grad=True), coords, requires_grad=True)
    with pytest.raises(NotImplementedError, match="requires_grad=True"):
        array.e2s.to_torch(requires_grad=True)
    with pytest.raises(ValueError, match="rank"):
        from_torch(torch.zeros(2, 3), coords)
    with pytest.raises(ValueError, match="dimension size"):
        from_torch(
            torch.zeros(2, 3),
            OrderedDict((("a", np.arange(2)), ("b", np.arange(2)))),
        )
    with pytest.raises(TypeError, match="Unsupported Torch device"):
        from_torch(torch.empty(2, device="meta"), coords)

    batched = array.e2s.batch(("a",))
    with pytest.raises(ValueError, match="Recursive batching"):
        batched.e2s.batch(("missing",))
    with pytest.raises(ValueError, match="leading dimension"):
        batched.transpose("b", "batch", "c").e2s.unbatch()
    with pytest.raises(ValueError, match="size does not match"):
        batched.isel(batch=slice(1)).e2s.unbatch()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cupy_torch_and_batch_round_trip():
    cp = pytest.importorskip(
        "cupy", reason="CuPy is required for GPU integration tests"
    )
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
