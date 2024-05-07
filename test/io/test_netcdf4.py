# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import tempfile
from collections import OrderedDict

import netCDF4
import numpy as np
import pytest
import torch

from earth2studio.io import NetCDF4Backend
from earth2studio.utils.coords import split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31T00:00:00")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["t2m", "tcwv"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_netcdf4_fields(
    time: list[np.datetime64],
    lead_time: list[np.datetime64],
    variable: list[str],
    device: str,
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "lead_time": lead_time,
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    nc = NetCDF4Backend("inmemory.nc", diskless=True, persist=False)
    assert isinstance(nc.root, netCDF4.Dataset)

    # Instantiate
    array_name = "fields"
    nc.add_array(total_coords, array_name)

    # Check instantiation
    for dim in total_coords:
        assert dim in nc
        assert dim in nc.coords
        assert nc[dim].shape == total_coords[dim].shape

    # Test __contains__
    assert array_name in nc

    # Test __getitem__
    shape = tuple([len(dim) for dim in total_coords.values()])
    assert nc[array_name].shape == shape

    # Test __len__
    assert len(nc) == 6

    # Test __iter__
    for array in nc:
        assert array in ["fields", "time", "lead_time", "variable", "lat", "lon"]

    # Test add_array with torch.Tensor
    nc.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in nc
    assert nc["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    nc.write(x, total_coords, "fields_1")
    assert "fields_1" in nc
    assert nc["fields_1"].shape == x.shape

    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "lead_time": np.asarray(lead_time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 1, 180, 180), device=device)
    nc.write(partial_data, partial_coords, array_name)
    assert np.allclose(nc[array_name][0, 0, 0, :, :180], partial_data.to("cpu").numpy())

    xx, _ = nc.read(partial_coords, array_name, device=device)
    assert torch.allclose(partial_data, xx)

    nc.close()

    # Test Directory Store
    with tempfile.TemporaryDirectory() as td:
        file_name = os.path.join(td, "temp_nc.nc")

        nc = NetCDF4Backend(file_name)
        assert isinstance(nc.root, netCDF4.Dataset)

        # Instantiate
        array_name = "fields"
        nc.add_array(total_coords, array_name)

        # Check instantiation
        for dim in total_coords:
            assert dim in nc
            assert dim in nc.coords
            assert nc[dim].shape == total_coords[dim].shape

        assert array_name in nc
        assert nc[array_name].shape == tuple(
            [len(val) for val in total_coords.values()]
        )

        # Test writing
        partial_coords = OrderedDict(
            {
                "time": np.asarray(time)[:1],
                "lead_time": np.asarray(lead_time)[:1],
                "variable": np.asarray(variable)[:1],
                "lat": total_coords["lat"],
                "lon": total_coords["lon"][:180],
            }
        )
        partial_data = torch.randn((1, 1, 1, 180, 180), device=device)
        nc.write(partial_data, partial_coords, array_name)
        assert np.allclose(
            nc[array_name][0, 0, 0, :, :180], partial_data.to("cpu").numpy()
        )
        nc.close()

        nc = NetCDF4Backend(file_name)
        for coord in nc.coords:
            assert np.all(total_coords[coord] == nc.coords[coord])


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31T00:00:00")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["t2m", "tcwv"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_netcdf4_variable(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Remove var names
    coords = total_coords.copy()
    var_names = coords.pop("variable")

    # Test Memory Store
    nc = NetCDF4Backend(
        "inmemory.nc",
        diskless=True,
        persist=False,
    )
    assert isinstance(nc.root, netCDF4.Dataset)

    nc.add_array(coords, var_names)
    for dim in coords:
        assert nc.coords[dim].shape == coords[dim].shape

    for var_name in var_names:
        assert var_name in nc
        assert nc[var_name].shape == tuple([len(values) for values in coords.values()])

    # Test writing
    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)
    nc.write(*split_coords(partial_data, partial_coords, "variable"))
    assert np.allclose(nc[variable[0]][0, :, :180], partial_data.to("cpu").numpy())
    nc.close()

    # Test Directory Store
    with tempfile.TemporaryDirectory() as td:
        file_name = os.path.join(td, "temp_nc.nc")
        nc = NetCDF4Backend(file_name)
        assert isinstance(nc.root, netCDF4.Dataset)

        nc.add_array(coords, var_names)
        for dim in coords:
            assert nc.coords[dim].shape == coords[dim].shape

        for var_name in var_names:
            assert var_name in nc
            assert nc[var_name].shape == tuple(
                [len(values) for values in coords.values()]
            )

        # Test writing
        partial_coords = OrderedDict(
            {
                "time": np.asarray(time)[:1],
                "variable": np.asarray(variable)[:1],
                "lat": total_coords["lat"],
                "lon": total_coords["lon"][:180],
            }
        )
        partial_data = torch.randn((1, 1, 180, 180), device=device)
        nc.write(*split_coords(partial_data, partial_coords, "variable"))
        assert np.allclose(nc[variable[0]][0, :, :180], partial_data.to("cpu").numpy())
        nc.close()


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_netcdf4_exceptions(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    nc = NetCDF4Backend(
        "inmemory.nc",
        diskless=True,
        persist=False,
    )
    assert isinstance(nc.root, netCDF4.Dataset)

    # Test mismatch between len(array_names) and len(data)
    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        nc.add_array(total_coords, array_name, data=[dummy] * 2)

    # Test trying to add the same array twice.
    nc.add_array(
        total_coords,
        ["dummy_1"],
        data=[dummy],
    )
    with pytest.raises(AssertionError):
        nc.add_array(
            total_coords,
            ["dummy_1"],
            data=[dummy],
        )

    # Try to write with bad coords
    bad_coords = {"ensemble": np.arange(0)} | total_coords
    bad_shape = (1,) + shape
    dummy = torch.randn(bad_shape, device=device, dtype=torch.float32)
    with pytest.raises(AssertionError):
        nc.write(dummy, bad_coords, "dummy_1")

    # Try to write with too many array names
    with pytest.raises(ValueError):
        nc.write([dummy, dummy], bad_coords, "dummy_1")

    nc.close()
