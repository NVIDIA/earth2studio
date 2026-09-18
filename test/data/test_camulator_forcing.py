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

import asyncio
import pathlib
import shutil
from datetime import datetime, timedelta

import cftime
import numpy as np
import pytest
import xarray as xr

from earth2studio.data import CAMulatorForcing
from earth2studio.data.camulator import CAMULATOR_GRID_LAT, CAMULATOR_GRID_LON

N_LAT, N_LON = len(CAMULATOR_GRID_LAT), len(CAMULATOR_GRID_LON)
# The shipped files run south to north; the data source serves north to south
FILE_LAT = CAMULATOR_GRID_LAT[::-1].astype(np.float32)
FILE_LON = CAMULATOR_GRID_LON.astype(np.float32)
FORCING_VARS = ["mtdwswrf", "sst", "sic", "global_mean_co2"]
CO2_MOL_MOL = 3.7e-4

_TEST_TIME = datetime(year=2001, month=1, day=1, hour=0)


def _write_forcing_file(
    path: pathlib.Path, times: list, lat: np.ndarray = FILE_LAT
) -> str:
    """Write a small CAMulator-like forcing file with a no-leap time axis.

    SOLIN encodes the time index, SST the file latitude, ICEFRAC is constant
    and co2vmr_3d is a constant in mol mol-1.
    """
    nt = len(times)
    solin = np.broadcast_to(
        np.arange(nt, dtype=np.float32)[:, None, None], (nt, N_LAT, N_LON)
    )
    sst = np.broadcast_to(lat[None, :, None], (nt, len(lat), N_LON))
    solin = np.broadcast_to(solin[:, :1, :], (nt, len(lat), N_LON))
    ds = xr.Dataset(
        {
            "SOLIN": (["time", "latitude", "longitude"], solin.copy()),
            "SST": (["time", "latitude", "longitude"], sst.copy()),
            "ICEFRAC": (
                ["time", "latitude", "longitude"],
                np.full((nt, len(lat), N_LON), 0.5, dtype=np.float32),
            ),
            "co2vmr_3d": (
                ["time", "latitude", "longitude"],
                np.full((nt, len(lat), N_LON), CO2_MOL_MOL, dtype=np.float32),
            ),
        },
        coords={"time": times, "latitude": lat, "longitude": FILE_LON},
    )
    ds.to_netcdf(path)
    return str(path)


def _noleap_times(year: int) -> list:
    """Jan 1, Feb 28, Mar 1 and Dec 31 at 6 h in a no-leap year."""
    days = [(1, 1), (2, 28), (3, 1), (12, 31)]
    return [
        cftime.DatetimeNoLeap(year, m, d, h) for m, d in days for h in (0, 6, 12, 18)
    ]


@pytest.fixture(scope="module")
def cyclic_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("camulator") / "cyclic.nc"
    return _write_forcing_file(path, _noleap_times(2000))


@pytest.fixture(scope="module")
def transient_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("camulator") / "transient.nc"
    return _write_forcing_file(path, _noleap_times(1980) + _noleap_times(1981))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "time",
    [
        _TEST_TIME,
        [_TEST_TIME, _TEST_TIME + timedelta(hours=6)],
    ],
)
@pytest.mark.parametrize("variable", ["sst", FORCING_VARS])
def test_camulator_forcing_fetch(time, variable):
    ds = CAMulatorForcing(cache=True)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(variable), N_LAT, N_LON)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    np.testing.assert_array_equal(data.coords["lat"].values, CAMULATOR_GRID_LAT)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(900)
@pytest.mark.parametrize("cache", [True, False])
def test_camulator_forcing_cache(cache):
    ds = CAMulatorForcing(cache=cache)
    data = ds(_TEST_TIME, ["sst"])
    assert not np.isnan(data.values).any()
    # With cache=False the per-instance temp directory is removed after the call
    assert pathlib.Path(ds.cache).is_dir() == cache or not cache
    data2 = ds(_TEST_TIME, ["sst"])
    assert data2.shape == data.shape
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(2001, 1, 1, 6),
        [datetime(1999, 3, 1, 0), datetime(2010, 12, 31, 18)],
        np.array([np.datetime64("2001-01-01T12:00")]),
    ],
)
@pytest.mark.parametrize("variable", ["sst", FORCING_VARS])
def test_camulator_forcing_call_mock(cyclic_file, time, variable):
    ds = CAMulatorForcing(forcing_file=cyclic_file, verbose=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(variable), N_LAT, N_LON)
    assert list(data.dims) == ["time", "variable", "lat", "lon"]
    assert data.dtype == np.float32
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert data.coords["time"].dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(data.coords["lat"].values, CAMULATOR_GRID_LAT)
    np.testing.assert_array_equal(data.coords["lon"].values, CAMULATOR_GRID_LON)

    # SST encodes the file latitude: after the flip the value at lat=90 is 90
    sst = data.sel(variable="sst").values
    np.testing.assert_allclose(sst[:, 0, 0], 90.0)
    np.testing.assert_allclose(sst[:, -1, 0], -90.0)
    np.testing.assert_allclose(sst[0, :, 0], CAMULATOR_GRID_LAT, rtol=1e-6)

    if "global_mean_co2" in variable:
        co2 = data.sel(variable="global_mean_co2").values
        np.testing.assert_allclose(co2, CO2_MOL_MOL * 1.0e6, rtol=1e-6)  # ppm
    if "sic" in variable:
        np.testing.assert_allclose(data.sel(variable="sic").values, 0.5)


@pytest.mark.timeout(60)
def test_camulator_forcing_time_lookup(cyclic_file, transient_file):
    # Cyclic: matched on (month, day, hour) regardless of the requested year.
    # SOLIN holds the record index: Jan 1 00Z -> 0, Feb 28 06Z -> 5, Dec 31 18Z -> 15
    ds = CAMulatorForcing(forcing_file=cyclic_file, verbose=False)
    solin = ds(
        [datetime(1985, 1, 1, 0), datetime(2003, 2, 28, 6), datetime(2020, 12, 31, 18)],
        "mtdwswrf",
    ).values[:, 0, 0, 0]
    np.testing.assert_array_equal(solin, [0.0, 5.0, 15.0])

    # Transient: matched on the year too (record 1980 then 1981, 16 steps each)
    ds = CAMulatorForcing(mode="transient", forcing_file=transient_file, verbose=False)
    solin = ds(
        [datetime(1980, 1, 1, 0), datetime(1981, 1, 1, 0), datetime(1981, 3, 1, 12)],
        "mtdwswrf",
    ).values[:, 0, 0, 0]
    np.testing.assert_array_equal(solin, [0.0, 16.0, 26.0])


@pytest.mark.timeout(60)
def test_camulator_forcing_leap_day(cyclic_file):
    # Default: 29 February reuses the 28 February forcing
    ds = CAMulatorForcing(forcing_file=cyclic_file, verbose=False)
    feb28 = ds(datetime(2000, 2, 28, 12), "mtdwswrf").values
    feb29 = ds(datetime(2000, 2, 29, 12), "mtdwswrf").values
    np.testing.assert_array_equal(feb28, feb29)

    ds = CAMulatorForcing(forcing_file=cyclic_file, leap_day="raise", verbose=False)
    with pytest.raises(ValueError):
        ds(datetime(2000, 2, 29, 12), "mtdwswrf")


@pytest.mark.timeout(60)
def test_camulator_forcing_exceptions(cyclic_file, transient_file):
    with pytest.raises(ValueError):
        CAMulatorForcing(mode="annual")
    with pytest.raises(ValueError):
        CAMulatorForcing(leap_day="skip")

    ds = CAMulatorForcing(forcing_file=cyclic_file, verbose=False)
    # Unknown variable and a CAMulator variable that is not a forcing
    with pytest.raises(KeyError):
        ds(_TEST_TIME, "nonexistent_variable")
    with pytest.raises(KeyError):
        ds(_TEST_TIME, "t2m")
    # Off the 6-hourly grid
    with pytest.raises(ValueError):
        ds(datetime(2001, 1, 1, 3), "sst")
    with pytest.raises(ValueError):
        ds(datetime(2001, 1, 1, 0, 30), "sst")
    # Date not in the (reduced) cyclic record
    with pytest.raises(ValueError):
        ds(datetime(2001, 6, 1, 0), "sst")

    # Transient record: year outside the file
    ds = CAMulatorForcing(mode="transient", forcing_file=transient_file, verbose=False)
    with pytest.raises(ValueError):
        ds(datetime(1990, 1, 1, 0), "sst")


@pytest.mark.timeout(60)
def test_camulator_forcing_grid_validation(tmp_path):
    # North-to-south file: silently flipping it would misplace every row
    path = _write_forcing_file(
        tmp_path / "flipped.nc", _noleap_times(2000)[:2], lat=FILE_LAT[::-1].copy()
    )
    with pytest.raises(ValueError):
        CAMulatorForcing(forcing_file=path, verbose=False)(_TEST_TIME, "sst")
    # Wrong latitude grid
    path = _write_forcing_file(
        tmp_path / "coarse.nc",
        _noleap_times(2000)[:2],
        lat=np.linspace(-90, 90, 96, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        CAMulatorForcing(forcing_file=path, verbose=False)(_TEST_TIME, "sst")


@pytest.mark.timeout(10)
def test_camulator_forcing_available():
    assert CAMulatorForcing.available(datetime(2001, 1, 1, 0)) is True
    assert CAMulatorForcing.available(datetime(2001, 1, 1, 18)) is True
    assert CAMulatorForcing.available(np.datetime64("2001-01-01T06:00")) is True
    assert CAMulatorForcing.available(datetime(2001, 1, 1, 13)) is False
    assert CAMulatorForcing.available(datetime(2001, 1, 1, 0, 30)) is False
    assert CAMulatorForcing.available(np.datetime64("2001-01-01T00:01")) is False


@pytest.mark.timeout(60)
def test_camulator_forcing_fetch_async(cyclic_file):
    ds = CAMulatorForcing(forcing_file=cyclic_file, verbose=False)
    data = asyncio.run(ds.fetch(_TEST_TIME, ["sst", "sic"]))
    assert data.shape == (1, 2, N_LAT, N_LON)
    assert not np.isnan(data.values).any()


@pytest.mark.timeout(60)
def test_camulator_forcing_no_cache_cleanup(cyclic_file, monkeypatch, tmp_path):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    ds = CAMulatorForcing(forcing_file=cyclic_file, cache=False, verbose=False)
    cache_dir = pathlib.Path(ds.cache)
    assert cache_dir.is_dir()
    assert "tmp_camulator_" in cache_dir.name

    data = ds(_TEST_TIME, "sst")
    assert data.shape == (1, 1, N_LAT, N_LON)
    # Dataset closed and temp directory removed after the call; a second call reopens
    assert ds._ds is None
    assert not cache_dir.is_dir()
    data2 = ds(_TEST_TIME, "sst")
    np.testing.assert_array_equal(data.values, data2.values)
