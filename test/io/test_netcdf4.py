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

import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.io import NetCDF4Backend
from earth2studio.io.netcdf4 import LEAD_TIME_UNITS, TIME_CALENDAR, TIME_UNITS
from earth2studio.utils.coords import coord_array

# Protocol behavior is covered by test_io_contract.py

TIMES = np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[h]")
LEADS = np.arange(3) * np.timedelta64(6, "h")
VARIABLES = ["t2m", "tp:sum:6h"]
GRID = LatLonGrid(np.array([1.0, 0.0, -1.0]), np.arange(4.0))


def make_template() -> xr.DataArray:
    return coord_array(
        ("time", "lead_time", "variable", "lat", "lon"),
        {"time": TIMES, "lead_time": LEADS, "variable": VARIABLES},
        grid=GRID,
        attrs={"units": "K"},
    )


def make_field(time: np.ndarray = TIMES, seed: int = 0) -> xr.DataArray:
    shape = (len(time), len(LEADS), len(VARIABLES), 3, 4)
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lead_time", "variable", "lat", "lon"),
        coords={
            "time": time,
            "lead_time": LEADS,
            "variable": VARIABLES,
            **GRID.coords(),
        },
    )


def test_netcdf4_dataset(tmp_path: Path) -> None:
    io = NetCDF4Backend(tmp_path / "store.nc", backend_kwargs={"mode": "w"})
    assert isinstance(io.root, netCDF4.Dataset)
    assert len(io) == 0

    io.add_array(make_template())
    # Dunder access covers every netCDF4 variable, coordinates included
    assert len(io) == 6
    assert set(io) == {"time", "lead_time", "lat", "lon", *VARIABLES}
    assert "t2m" in io and "lat" in io and "u10m" not in io
    assert isinstance(io["t2m"], netCDF4.Variable)
    assert io["t2m"].dimensions == ("time", "lead_time", "lat", "lon")
    assert io["t2m"].shape == (2, 3, 3, 4)
    io.close()


def test_netcdf4_memory_store() -> None:
    # Diskless stores never touch the file system
    io = NetCDF4Backend(
        "inmemory.nc", backend_kwargs={"mode": "w", "diskless": True, "persist": False}
    )
    io.add_array(make_template())
    field = make_field()
    io.write(field)
    io.flush()
    np.testing.assert_array_equal(
        io.to_xarray()["t2m"].values, field.sel(variable="t2m").values
    )
    io.close()
    assert not Path("inmemory.nc").exists()


def test_netcdf4_write_is_visible_before_close(tmp_path: Path) -> None:
    source_path = tmp_path / "source.nc"
    copied_path = tmp_path / "copied.nc"
    io = NetCDF4Backend(source_path, backend_kwargs={"mode": "w"})
    field = make_field()
    try:
        io.add_array(make_template())
        io.write(field)
        io.flush()
        shutil.copyfile(source_path, copied_path)

        with xr.open_dataset(copied_path) as dataset:
            np.testing.assert_array_equal(
                dataset["t2m"].values, field.sel(variable="t2m").values
            )
    finally:
        io.close()


def test_netcdf4_reopen(tmp_path: Path) -> None:
    # A reopened store keeps its coordinates, so add_array is a no-op and
    # writes continue where they left off
    path = tmp_path / "store.nc"
    io = NetCDF4Backend(path, backend_kwargs={"mode": "w"})
    first = make_field(time=TIMES[:1], seed=1)
    io.add_array(make_template())
    io.write(first)
    io.close()

    io = NetCDF4Backend(path)
    np.testing.assert_array_equal(io.to_xarray()["time"].values, TIMES)
    io.add_array(make_template())
    assert len(io) == 6
    second = make_field(time=TIMES[1:], seed=2)
    io.write(second)
    io.close()

    with xr.open_dataset(path) as dataset:
        stored = dataset["t2m"]
        assert stored.time.dtype == np.dtype("datetime64[ns]")
        np.testing.assert_array_equal(
            stored.sel(time=TIMES[0]).values, first.sel(variable="t2m").values[0]
        )
        np.testing.assert_array_equal(
            stored.sel(time=TIMES[1]).values, second.sel(variable="t2m").values[0]
        )


def test_netcdf4_cf_time_encoding(tmp_path: Path) -> None:
    # Labels are stored as CF nanosecond integers, which cftime also reads
    path = tmp_path / "store.nc"
    io = NetCDF4Backend(path, backend_kwargs={"mode": "w"})
    io.add_array(make_template())
    io.close()

    with netCDF4.Dataset(path) as dataset:
        time = dataset["time"]
        assert time.units == TIME_UNITS
        assert time.calendar == TIME_CALENDAR
        np.testing.assert_array_equal(
            time[:], TIMES.astype("datetime64[ns]").astype("int64")
        )
        assert dataset["lead_time"].units == LEAD_TIME_UNITS

    with xr.open_dataset(path) as dataset:
        assert dataset["time"].dtype == np.dtype("datetime64[ns]")
        assert dataset["lead_time"].dtype == np.dtype("timedelta64[ns]")
        np.testing.assert_array_equal(dataset["time"].values, TIMES)
        np.testing.assert_array_equal(dataset["lead_time"].values, LEADS)


def test_netcdf4_json_attributes(tmp_path: Path) -> None:
    # netCDF cannot store lists or dictionaries, so they become JSON strings
    io = NetCDF4Backend(tmp_path / "store.nc", backend_kwargs={"mode": "w"})
    io.add_array(make_template().assign_attrs(levels=[100, 200], extra={"a": 1}))
    io.flush()
    stored = io.to_xarray()["t2m"]
    assert stored.attrs["units"] == "K"
    assert json.loads(stored.attrs["levels"]) == [100, 200]
    assert json.loads(stored.attrs["extra"]) == {"a": 1}
    assert json.loads(stored.attrs["dims"]) == ["lat", "lon"]
    io.close()


def test_netcdf4_auxiliary_coordinates(tmp_path: Path) -> None:
    # Auxiliary coordinates are listed in the CF coordinates attribute
    io = NetCDF4Backend(tmp_path / "store.nc", backend_kwargs={"mode": "w"})
    latitude = np.arange(12.0).reshape(3, 4)
    io.add_array(
        coord_array(
            ("y", "x"),
            {
                "lat": (("y", "x"), latitude),
                "lon": (("y", "x"), latitude + 100.0),
            },
            name="t2m",
        )
    )
    assert io["t2m"].coordinates == "lat lon"
    io.flush()
    stored = io.to_xarray()["t2m"]
    assert set(stored.coords) == {"lat", "lon"}
    np.testing.assert_array_equal(stored.coords["lat"].values, latitude)
    io.close()


def test_netcdf4_dimension_size_conflict(tmp_path: Path) -> None:
    io = NetCDF4Backend(tmp_path / "store.nc", backend_kwargs={"mode": "w"})
    io.add_array(coord_array(("sample",), sizes={"sample": 3}, name="seed"))
    with pytest.raises(ValueError, match="size 3"):
        io.add_array(coord_array(("sample",), sizes={"sample": 4}, name="other"))
    io.close()


def test_netcdf4_closed(tmp_path: Path) -> None:
    path = tmp_path / "store.nc"
    io = NetCDF4Backend(path, backend_kwargs={"mode": "w"})
    io.add_array(make_template())
    field = make_field()
    io.write(field)
    io.close()
    io.close()

    with pytest.raises(RuntimeError):
        io.write(field)
    with pytest.raises(RuntimeError):
        io.add_array(make_template())
    with pytest.raises(RuntimeError):
        io.read(make_template())
    # A closed store is reopened from its file
    np.testing.assert_array_equal(
        io.to_xarray()["t2m"].values, field.sel(variable="t2m").values
    )


def test_netcdf4_read_partial(tmp_path: Path) -> None:
    io = NetCDF4Backend(tmp_path / "store.nc", backend_kwargs={"mode": "w"})
    io.add_array(make_template())
    field = make_field()
    io.write(field)
    selection = {
        "time": TIMES[::-1],
        "lead_time": LEADS[[2, 0]],
        "variable": ["t2m"],
        "lat": GRID.coords()["lat"].values,
        "lon": np.arange(2.0),
    }
    result = io.read(selection)
    expected = field.sel(selection)
    assert result.dims == tuple(selection)
    np.testing.assert_array_equal(result.values, expected.values)
    assert result.attrs["units"] == "K"
    io.close()
