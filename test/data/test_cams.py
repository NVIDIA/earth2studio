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

import datetime
import pathlib
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import CAMS_FX

YESTERDAY = datetime.datetime.now(datetime.UTC).replace(
    hour=0, minute=0, second=0, microsecond=0
) - datetime.timedelta(days=1)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("variable", ["aod550", ["aod550", "tcco"]])
@pytest.mark.parametrize(
    "lead_time",
    [
        datetime.timedelta(hours=0),
        [datetime.timedelta(hours=0), datetime.timedelta(hours=24)],
    ],
)
def test_cams_fx_fetch(variable, lead_time):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    ds = CAMS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, datetime.timedelta):
        lead_time = [lead_time]

    assert shape[0] == 1  # time
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert len(data.coords["lat"]) > 0
    assert len(data.coords["lon"]) > 0
    assert not np.isnan(data.values).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_cams_fx_cache(cache):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    lead_time = datetime.timedelta(hours=0)
    ds = CAMS_FX(cache=cache)
    data = ds(time, lead_time, ["aod550", "tcco"])
    shape = data.shape

    assert shape[0] == 1
    assert shape[2] == 2
    assert not np.isnan(data.values).all()
    assert pathlib.Path(ds.cache).is_dir() == cache

    data = ds(time, lead_time, "aod550")
    assert data.shape[2] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
def test_cams_fx_invalid():
    with pytest.raises((ValueError, KeyError)):
        ds = CAMS_FX()
        ds(YESTERDAY, datetime.timedelta(hours=0), "nonexistent_var")


def test_cams_fx_time_validation():
    with pytest.raises(ValueError, match="CAMS Global forecast"):
        CAMS_FX._validate_time([datetime.datetime(2014, 1, 1)])


def test_cams_fx_available():
    assert CAMS_FX.available(datetime.datetime(2024, 1, 1))
    assert not CAMS_FX.available(datetime.datetime(2010, 1, 1))
    assert CAMS_FX.available(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC))


def test_cams_fx_api_vars_dedup():
    """Variables sharing the same API name must not produce duplicate requests."""
    from earth2studio.data.cams import _resolve_variable

    info_a = _resolve_variable("aod550", 0)
    info_b = _resolve_variable("aod550", 1)
    api_vars = list(dict.fromkeys([info_a.api_name, info_b.api_name]))
    assert len(api_vars) == 1


def test_cams_fx_call_mock(tmp_path: pathlib.Path):
    """Test CAMS_FX __call__ with surface and pressure-level variables (mocked)."""
    lat = CAMS_FX.CAMS_LAT
    lon = CAMS_FX.CAMS_LON
    forecast_period = np.array([0, 3, 6], dtype=np.float64)
    pressure_level = np.array([500.0, 850.0])

    # Surface NetCDF
    mock_surface_ds = xr.Dataset(
        {
            "aod550": (
                ["forecast_period", "latitude", "longitude"],
                np.random.rand(len(forecast_period), len(lat), len(lon)),
            ),
            "tcco": (
                ["forecast_period", "latitude", "longitude"],
                np.random.rand(len(forecast_period), len(lat), len(lon)),
            ),
        },
        coords={
            "forecast_period": forecast_period,
            "latitude": lat,
            "longitude": lon,
        },
    )
    surface_path = tmp_path / "mock_surface.nc"
    mock_surface_ds.to_netcdf(surface_path)

    # Pressure-level NetCDF
    mock_pressure_ds = xr.Dataset(
        {
            "u": (
                ["forecast_period", "pressure_level", "latitude", "longitude"],
                np.random.rand(
                    len(forecast_period), len(pressure_level), len(lat), len(lon)
                ),
            ),
            "t": (
                ["forecast_period", "pressure_level", "latitude", "longitude"],
                np.random.rand(
                    len(forecast_period), len(pressure_level), len(lat), len(lon)
                ),
            ),
        },
        coords={
            "forecast_period": forecast_period,
            "pressure_level": pressure_level,
            "latitude": lat,
            "longitude": lon,
        },
    )
    pressure_path = tmp_path / "mock_pressure.nc"
    mock_pressure_ds.to_netcdf(pressure_path)

    with patch("earth2studio.data.cams.cdsapi") as mock_cdsapi:
        mock_cdsapi.Client = MagicMock()
        time = datetime.datetime(2024, 6, 1, 0, 0)
        lead_time = [datetime.timedelta(hours=0), datetime.timedelta(hours=6)]

        # --- surface-only fetch: single download ---
        with patch("earth2studio.data.cams._download_cams_netcdf") as mock_dl:
            mock_dl.return_value = surface_path
            data = CAMS_FX(cache=False)(time, lead_time, ["aod550", "tcco"])

            assert data.shape == (1, 2, 2, len(lat), len(lon))
            assert list(data.coords["variable"].values) == ["aod550", "tcco"]
            mock_dl.assert_called_once()

        # --- pressure-level-only fetch: single download ---
        with patch("earth2studio.data.cams._download_cams_netcdf") as mock_dl:
            mock_dl.return_value = pressure_path
            data = CAMS_FX(cache=False)(time, lead_time, ["u500", "t850"])

            assert data.shape == (1, 2, 2, len(lat), len(lon))
            assert list(data.coords["variable"].values) == ["u500", "t850"]
            mock_dl.assert_called_once()

        # --- mixed fetch: two separate downloads ---
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return surface_path if call_count == 1 else pressure_path

        with patch("earth2studio.data.cams._download_cams_netcdf") as mock_dl:
            mock_dl.side_effect = _side_effect
            data = CAMS_FX(cache=False)(time, lead_time, ["aod550", "u500"])

            assert data.shape == (1, 2, 2, len(lat), len(lon))
            assert list(data.coords["variable"].values) == ["aod550", "u500"]
            assert mock_dl.call_count == 2


def test_cams_fx_pressure_level_leadtime_validation():
    """Pressure-level vars at non-3h lead times must raise ValueError."""
    with patch("earth2studio.data.cams.cdsapi") as mock_cdsapi:
        mock_cdsapi.Client = MagicMock()
        ds = CAMS_FX(cache=False)
        with pytest.raises(ValueError, match="multiple of 3 hours"):
            ds(
                datetime.datetime(2024, 6, 1, 0, 0),
                datetime.timedelta(hours=1),
                "u500",
            )
