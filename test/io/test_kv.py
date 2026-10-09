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

"""Backend-specific tests for :class:`earth2studio.io.KVBackend`.

Protocol behavior is covered by test_io_contract.py.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.io import KVBackend
from earth2studio.utils.coords import coord_array
from earth2studio.utils.cupy import from_torch


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_kv_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    io = KVBackend(device=device)
    assert io.device == device

    template = coord_array(
        ("time", "lat"),
        {"time": np.array(["2024-01-01"], dtype="datetime64[ns]"), "lat": [0.0, 1.0]},
        name="field",
    )
    io.add_array(template)
    if device == "cpu" or torch.cuda.is_available():
        assert io["field"].device.type == device.split(":")[0]


def test_kv_dataset() -> None:
    io = KVBackend()
    assert isinstance(io.root, dict)
    assert len(io) == 0

    template = coord_array(
        ("time", "variable", "lat"),
        {
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "variable": ["t2m", "u10m"],
            "lat": [0.0, 1.0],
        },
    )
    io.add_array(template)
    assert len(io) == 2
    assert list(io) == ["t2m", "u10m"]
    assert "t2m" in io and "lat" in io and "v10m" not in io
    assert isinstance(io["t2m"], torch.Tensor)
    assert io["t2m"].shape == (1, 2)

    ds = io.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert list(ds.data_vars) == ["t2m", "u10m"]
    assert ds["t2m"].dims == ("time", "lat")


def test_kv_integer_fill() -> None:
    io = KVBackend()
    io.add_array(coord_array(("sample",), {"sample": [0, 1]}, dtype=np.int32, name="n"))
    assert io["n"].dtype == torch.int32
    np.testing.assert_array_equal(io["n"].numpy(), [0, 0])


def test_kv_read_by_label() -> None:
    io = KVBackend()
    template = coord_array(
        ("time", "variable", "lat"),
        {
            "time": np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[h]"),
            "variable": ["t2m", "u10m"],
            "lat": [0.0, 1.0],
        },
    )
    io.add_array(template)
    field = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dims=("time", "variable", "lat"),
        coords={
            "time": template.coords["time"].values,
            "variable": ["t2m", "u10m"],
            "lat": [0.0, 1.0],
        },
    )
    io.write(field)

    selection = {
        "variable": ["u10m"],
        "time": template.coords["time"].values[::-1],
        "lat": [1.0, 0.0],
    }
    result = io.read(selection)
    assert result.dims == ("variable", "time", "lat")
    np.testing.assert_array_equal(
        result.values, field.sel(selection).transpose(*selection).values
    )

    # Reads return host-resident NumPy payloads by default
    assert isinstance(result.values, np.ndarray)


def test_kv_torch_backed_write() -> None:
    io = KVBackend()
    template = coord_array(
        ("time", "lat"),
        {"time": np.array(["2024-01-01"], dtype="datetime64[ns]"), "lat": [0.0, 1.0]},
        name="field",
    )
    io.add_array(template)
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    io.write(
        from_torch(
            tensor,
            {"time": template.coords["time"].values, "lat": [0.0, 1.0]},
            name="field",
        )
    )
    np.testing.assert_array_equal(io["field"].numpy(), [[1.0, 2.0]])
