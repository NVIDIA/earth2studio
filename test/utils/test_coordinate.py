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

from earth2studio import coord_array
from earth2studio.grids import CurvilinearGrid, HEALPixGrid
from earth2studio.utils.coordinate import handshake_dataarray, handshake_dataarrays


def test_coordinate_signature_and_handshake():
    signature = coord_array(
        ("batch", "lead_time", "variable", "x"),
        {"lead_time": [np.timedelta64(0, "h")], "variable": ["a"], "x": [0, 1]},
        dynamic=("batch",),
        statistics={"a": "mean:24h"},
    )
    assert signature.shape == (0, 1, 1, 2) and signature.data.nbytes == 0
    assert signature.e2s.dynamic_dims == ("batch",)
    assert signature.e2s.get_statistic("a") == "mean:24h"

    array = xr.DataArray(
        np.zeros((3, 1, 1, 2)),
        dims=("time", "lead_time", "variable", "x"),
        coords={"lead_time": signature.lead_time, "variable": ["a"], "x": [0, 1]},
        attrs=signature.attrs,
    )
    handshake_dataarray(array, signature)
    with pytest.raises(ValueError, match="trailing dimensions"):
        handshake_dataarray(
            array.transpose("time", "lead_time", "x", "variable"), signature
        )
    with pytest.raises(ValueError, match="is missing"):
        handshake_dataarray(array.drop_indexes("x").drop_vars("x"), signature)
    with pytest.raises(ValueError, match="statistics metadata"):
        handshake_dataarray(array.assign_attrs(earth2studio_statistics={}), signature)
    with pytest.raises(ValueError, match="must lead"):
        coord_array(("x", "batch"), {"x": [0]}, dynamic=("batch",))


def test_coordinate_grid_and_collection_handshake():
    signature = coord_array(
        ("batch", "lead_time", "variable", "lat", "lon"),
        {"lead_time": [np.timedelta64(0, "h")], "variable": ["a"]},
        dynamic=("batch",),
        grid="fcn1",
    )
    assert signature.e2s.get_grid().shape == (720, 1440)
    array = xr.DataArray(
        np.zeros((1, 1, 1, 720, 1440), dtype=np.float32),
        dims=signature.dims,
        coords=signature.coords,
        attrs=signature.attrs,
    )
    handshake_dataarrays((array, array), (signature, signature))
    with pytest.raises(ValueError, match="Expected 2 DataArrays"):
        handshake_dataarrays((array,), (signature, signature))
    with pytest.raises(ValueError, match="grid metadata"):
        handshake_dataarray(array.assign_attrs(earth2studio_grid={}), signature)


def test_coordinate_non_rectilinear_grids():
    curvilinear = CurvilinearGrid(
        np.array([[40.0, 40.1], [39.0, 39.1]]),
        np.array([[250.0, 251.0], [250.1, 251.1]]),
    )
    signature = coord_array(
        ("variable", "y", "x"), {"variable": ["a"]}, grid=curvilinear
    )
    assert signature.coords["lat"].dims == ("y", "x")
    assert signature.e2s.get_grid().fingerprint() == curvilinear.fingerprint()

    healpix = HEALPixGrid(level=2, ordering="xy", layout="face")
    signature = coord_array(
        ("variable", *healpix.dims), {"variable": ["a"]}, grid=healpix
    )
    assert signature.e2s.get_grid().fingerprint() == healpix.fingerprint()
