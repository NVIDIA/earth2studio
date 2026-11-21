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
import xarray as xr

from earth2studio.io import XarrayBackend
from earth2studio.utils.coords import coord_array

# Protocol behavior is covered by test_io_contract.py


def test_xarray_dataset() -> None:
    io = XarrayBackend(attrs={"title": "forecast"})
    assert isinstance(io.root, xr.Dataset)
    assert io.root.attrs == {"title": "forecast"}
    assert len(io) == 0

    io.add_array(
        coord_array(
            ("time", "variable", "lat"),
            {
                "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "variable": ["t2m", "u10m"],
                "lat": [0.0, 1.0],
            },
        )
    )
    assert len(io) == 2
    assert list(io) == ["t2m", "u10m"]
    assert "t2m" in io and "lat" in io and "v10m" not in io
    assert isinstance(io["t2m"], xr.DataArray)
    assert io["t2m"].dims == ("time", "lat")


def test_xarray_integer_fill() -> None:
    io = XarrayBackend()
    io.add_array(coord_array(("sample",), {"sample": [0, 1]}, dtype=np.int32, name="n"))
    assert io["n"].dtype == np.int32
    np.testing.assert_array_equal(io["n"].values, [0, 0])


def test_xarray_dataset_kwargs() -> None:
    # Dataset keyword arguments, such as coordinates, seed the store
    io = XarrayBackend(coords={"lat": [0.0, 1.0]})
    io.add_array(coord_array(("lat",), {"lat": [0.0, 1.0]}, name="mask"))
    assert list(io) == ["mask"]
    with pytest.raises(ValueError, match="lat"):
        io.add_array(coord_array(("lat",), {"lat": [0.0, 2.0]}, name="other"))
