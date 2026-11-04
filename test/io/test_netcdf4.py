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

import os
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch
import xarray

from earth2studio.io import NetCDF4Backend
from earth2studio.utils.coords import convert_multidim_to_singledim, split_coords


@pytest.mark.parametrize(
    "unit, values, stored_units, stored_values",
    [
        ("h", [-6, 0, 6], "hours", [-6, 0, 6]),
        ("ms", [-1001, 0, 1001], "milliseconds", [-1001, 0, 1001]),
        ("us", [-1001, 0, 1001], "microseconds", [-1001, 0, 1001]),
        ("ns", [-1, 0, 2**53 + 1], "nanoseconds", [-1, 0, 2**53 + 1]),
        ("6h", [-1, 0, 1], "hours", [-6, 0, 6]),
    ],
)
def test_netcdf4_lead_time_roundtrip(
    tmp_path: Path,
    unit: str,
    values: list[int],
    stored_units: str,
    stored_values: list[int],
) -> None:
    lead_time = np.array(values, dtype=f"timedelta64[{unit}]")
    coords = OrderedDict(
        time=np.array([np.datetime64("2024-01-01")]), lead_time=lead_time
    )
    expected = torch.arange(len(lead_time), dtype=torch.float32).reshape(1, -1)
    file_name = tmp_path / "lead_time.nc"
    nc = NetCDF4Backend(file_name, backend_kwargs={"mode": "w"})
    try:
        nc.add_array(coords, "fields", data=expected)
        assert nc["lead_time"].units == stored_units
        np.testing.assert_array_equal(nc["lead_time"][:], stored_values)
    finally:
        nc.close()

    with xarray.open_dataset(file_name, decode_timedelta=True) as dataset:
        np.testing.assert_array_equal(dataset.lead_time.values, lead_time)
        np.testing.assert_array_equal(dataset.fields.values, expected.numpy())
        decoded_coords = OrderedDict(
            time=dataset.time.values, lead_time=dataset.lead_time.values
        )

    nc = NetCDF4Backend(file_name)
    try:
        np.testing.assert_array_equal(nc.coords["lead_time"], lead_time)
        actual, _ = nc.read(decoded_coords, "fields")
        torch.testing.assert_close(actual, expected)
    finally:
        nc.close()

    # Coordinates decoded by xarray must also be accepted for a new output file.
    nc = NetCDF4Backend(tmp_path / "rewritten.nc", backend_kwargs={"mode": "w"})
    try:
        nc.add_array(decoded_coords, "fields", data=expected)
    finally:
        nc.close()
    with xarray.open_dataset(
        tmp_path / "rewritten.nc", decode_timedelta=True
    ) as dataset:
        np.testing.assert_array_equal(dataset.lead_time.values, lead_time)


def test_netcdf4_write_is_visible_before_close(tmp_path: Path) -> None:
    coords = OrderedDict({"lat": np.arange(2), "lon": np.arange(2)})
    source_path = tmp_path / "source.nc"
    copied_path = tmp_path / "copied.nc"
    expected = np.arange(4, dtype=np.float32).reshape(2, 2)
    nc = NetCDF4Backend(source_path, backend_kwargs={"mode": "w"})

    try:
        nc.add_array(coords, "fields")
        nc.write(torch.from_numpy(expected), coords, "fields")
        shutil.copyfile(source_path, copied_path)

        with xarray.open_dataset(copied_path) as dataset:
            assert np.array_equal(dataset["fields"].values, expected)
    finally:
        nc.close()


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
    nc = NetCDF4Backend(
        "inmemory.nc", backend_kwargs={"mode": "w", "diskless": True, "persist": False}
    )
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
        "inmemory.nc", backend_kwargs={"mode": "w", "diskless": True, "persist": False}
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
    "mode",
    ["r+", "w"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])  #
def test_netcdf4_file(mode: str, device: str, tmp_path: str) -> None:
    time = [np.datetime64("1958-01-31T00:00:00")]
    variable = ["t2m", "tcwv"]
    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test File Store
    nc = NetCDF4Backend(tmp_path / "test.nc", backend_kwargs={"mode": mode})
    assert isinstance(nc.root, netCDF4.Dataset)

    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    nc.add_array(total_coords, array_name, data=dummy)
    nc.close()

    # Check to see if write mode overwrites netCDF
    nc = NetCDF4Backend(tmp_path / "test.nc", backend_kwargs={"mode": mode})
    nc.add_array(total_coords, array_name, data=dummy)
    nc.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])  #
def test_netcdf4_calendar(device: str, tmp_path: str) -> None:
    time = [np.datetime64("1958-01-31T00:00:00")]
    lead_time = [np.timedelta64(1, "h")]
    variable = ["t2m", "tcwv"]
    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "lead_time": np.asarray(lead_time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test File Store
    nc = NetCDF4Backend(tmp_path / "test.nc", backend_kwargs={"mode": "w"})
    assert isinstance(nc.root, netCDF4.Dataset)

    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    nc.add_array(total_coords, array_name, data=dummy)
    nc.close()

    # Check that the calendar matches with xarray and netCDF4
    ds = xarray.open_dataset(tmp_path / "test.nc")
    xarray_time = ds.time[0].values
    xarray_lead_time = ds.lead_time[0].values
    assert xarray_time == np.asarray(time)
    assert xarray_lead_time == np.asarray(lead_time)

    # Convert to datetime
    _unix = np.datetime64(0, "s")
    _ds = np.timedelta64(1, "s")
    xarray_datetime = datetime.utcfromtimestamp((xarray_time - _unix) / _ds)

    # Get the time from netCDF4
    netcdf4_hours = netCDF4.Dataset(tmp_path / "test.nc")["time"][0].data.item()
    netcdf4_datetime = datetime(1, 1, 1) + timedelta(hours=netcdf4_hours)
    assert xarray_datetime == netcdf4_datetime


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
        "inmemory.nc", backend_kwargs={"mode": "w", "diskless": True, "persist": False}
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
    nc.add_array(
        total_coords,
        ["dummy_1"],
        data=[dummy + 1.0],
    )
    assert np.allclose(nc["dummy_1"][:], dummy.cpu().numpy())

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
def test_netcdf4_fields_multidim(
    time: list[np.datetime64],
    lead_time: list[np.datetime64],
    variable: list[str],
    device: str,
) -> None:

    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(0, 360, 360, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "lead_time": lead_time,
            "variable": np.asarray(variable),
            "lat": LAT,
            "lon": LON,
        }
    )

    adjusted_coords, _ = convert_multidim_to_singledim(total_coords)

    # Test Memory Store
    nc = NetCDF4Backend(
        "inmemory.nc", backend_kwargs={"mode": "w", "diskless": True, "persist": False}
    )
    assert isinstance(nc.root, netCDF4.Dataset)

    # Instantiate
    array_name = "fields"
    nc.add_array(total_coords, array_name)

    # Check instantiation
    for dim in adjusted_coords:
        assert dim in nc
        assert dim in nc.coords
        assert nc[dim].shape == adjusted_coords[dim].shape

    # Test __contains__
    assert array_name in nc

    # Test __getitem__
    shape = tuple([len(dim) for dim in adjusted_coords.values()])
    assert nc[array_name].shape == shape

    # Test __len__
    assert len(nc) == 8

    # Test __iter__
    for array in nc:
        assert array in [
            "fields",
            "time",
            "lead_time",
            "variable",
            "ilat",
            "ilon",
            "lat",
            "lon",
        ]

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
    nc.write(x, adjusted_coords, array_name)

    xx, _ = nc.read(adjusted_coords, array_name, device=device)
    assert torch.allclose(x, xx)

    # Test separate write
    nc.write(x, total_coords, "fields_1")
    assert "fields_1" in nc
    assert nc["fields_1"].shape == x.shape

    xx, _ = nc.read(total_coords, "fields_1", device=device)
    assert torch.allclose(x, xx)

    nc.close()


def test_netcdf4_write_uninitialized_array(tmp_path: str) -> None:
    lat = np.arange(4)
    lon = np.arange(10)
    coords = OrderedDict({"lat": lat, "lon": lon})

    nc_path = tmp_path / "uninitialized.nc"
    nc = NetCDF4Backend(nc_path, backend_kwargs={"mode": "w"})
    nc.add_array(coords, ["init"])

    nc.write(
        torch.zeros((4, 2), dtype=torch.float32),
        {"lat": lat, "lon": lon[:2]},
        "init",
    )
    nc.write(
        torch.ones((4, 5), dtype=torch.float32),
        {"lat": lat, "lon": lon[:5]},
        "newvar",
    )
    nc.close()

    # Verify via xarray
    ds = xarray.open_dataset(nc_path)
    assert "init" in ds.data_vars
    assert "newvar" in ds.data_vars
    assert np.allclose(ds["init"].isel(lon=slice(0, 2)).values, 0.0)
    assert np.allclose(ds["newvar"].isel(lon=slice(0, 5)).values, 1.0)
