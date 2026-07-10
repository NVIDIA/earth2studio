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

import inspect
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import (
    EarthMoverBrightBandIFS,
    EarthMoverBrightBandIFS_FX,
    EarthMoverERA5,
)
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.utils.imports import OptionalDependencyFailure

LAT = np.linspace(90, -90, 9)
LON = np.array([-180.0, -90.0, 0.0, 90.0])
E2S_LON = np.array([0.0, 90.0, 180.0, 270.0])
TIMES = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")

ERA5_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ERA5_SINGLE_VARIABLES = (
    "blh",
    "cape",
    "cp",
    "d2m",
    "fdir",
    "fg10m",
    "fsr",
    "hcc",
    "ie",
    "lcc",
    "lsp",
    "mcc",
    "msl",
    "sd",
    "sf",
    "skt",
    "slhf",
    "sp",
    "ssr",
    "ssrd",
    "sst",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "swvl1",
    "t2m",
    "tcc",
    "tcw",
    "tcwv",
    "tisr",
    "tp",
    "tsr",
    "u10m",
    "u100m",
    "v10m",
    "v100m",
    "zust",
)
ERA5_PRESSURE_VARIABLES = tuple(
    f"{name}{level}"
    for name in ("pv", "q", "r", "t", "u", "v", "w", "z")
    for level in ERA5_LEVELS
)
ERA5_VARIABLES = ERA5_SINGLE_VARIABLES + ERA5_PRESSURE_VARIABLES

IFS_FORECAST_DATASET_VARIABLES = (
    "100u",
    "100v",
    "10u",
    "10v",
    "2d",
    "2t",
    "cp",
    "fdir",
    "hcc",
    "lcc",
    "mcc",
    "msl",
    "sd",
    "ssrd",
    "tp",
)
IFS_FORECAST_VARIABLES = (
    "u100m",
    "v100m",
    "u10m",
    "v10m",
    "d2m",
    "t2m",
    "cp",
    "fdir",
    "hcc",
    "lcc",
    "mcc",
    "msl",
    "sd",
    "ssrd",
    "tp",
)
IFS_ANALYSIS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
IFS_ANALYSIS_SURFACE_VARIABLES = (
    "u100m",
    "v100m",
    "u10m",
    "v10m",
    "d2m",
    "t2m",
    "hcc",
    "lcc",
    "mcc",
    "msl",
    "skt",
    "sp",
    "sst",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
    "tcc",
    "tcw",
    "tcwv",
)
IFS_ANALYSIS_PRESSURE_VARIABLES = tuple(
    f"{name}{level}"
    for name in ("q", "t", "u", "v", "w", "z")
    for level in IFS_ANALYSIS_LEVELS
)
IFS_ANALYSIS_VARIABLES = (
    IFS_ANALYSIS_SURFACE_VARIABLES + IFS_ANALYSIS_PRESSURE_VARIABLES
)
TEST_TIME = datetime(2022, 1, 1, 0)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        TEST_TIME,
        [TEST_TIME, TEST_TIME + timedelta(hours=6)],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl", "z500"]])
def test_earthmover_era5_fetch(time, variable):
    ds = EarthMoverERA5(cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(variable), 721, 1440)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        TEST_TIME,
        [TEST_TIME, TEST_TIME + timedelta(hours=6)],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])
def test_earthmover_brightband_ifs_fetch(time, variable):
    ds = EarthMoverBrightBandIFS(cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(variable), 721, 1440)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,lead_time",
    [
        (TEST_TIME, timedelta(hours=0)),
        (TEST_TIME, [timedelta(hours=0), timedelta(hours=6)]),
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])
def test_earthmover_brightband_ifs_fx_fetch(time, lead_time, variable):
    ds = EarthMoverBrightBandIFS_FX(cache=False)
    data = ds(time, lead_time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    assert data.shape == (1, len(lead_time), len(variable), 1801, 3600)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


def _grid(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def mock_earthmover_era5() -> dict[str | None, xr.Dataset]:
    levels = np.array(ERA5_LEVELS[::-1], dtype="float64")
    single_coords = {
        "valid_time": ("valid_time", TIMES),
        "lat": LAT,
        "lon": LON,
    }
    pressure_coords = {
        **single_coords,
        "pressure_level": ("pressure_level", levels),
    }
    single_dims = ("valid_time", "lat", "lon")
    pressure_dims = ("valid_time", "pressure_level", "lat", "lon")
    single_shape = (TIMES.size, LAT.size, LON.size)
    pressure_shape = (TIMES.size, levels.size, LAT.size, LON.size)
    single = xr.Dataset(
        {
            "t2m": (single_dims, _grid(single_shape)),
            "msl": (single_dims, _grid(single_shape) + 10.0),
            "fdir": (single_dims, _grid(single_shape) + 20.0),
            "sf": (
                single_dims,
                np.full(single_shape, 0.002, dtype=np.float32),
            ),
            "stl1": (single_dims, _grid(single_shape) + 40.0),
        },
        coords=single_coords,
    )
    pressure = xr.Dataset(
        {
            "q": (pressure_dims, _grid(pressure_shape)),
            "z": (pressure_dims, _grid(pressure_shape) + 30.0),
        },
        coords=pressure_coords,
    )
    pressure["pressure_level"].attrs = {
        "standard_name": "air_pressure",
        "units": "hPa",
        "axis": "Z",
    }
    single["t2m"].attrs = {"GRIB_shortName": "t2m", "units": "K"}
    single["msl"].attrs = {
        "GRIB_paramId": 151,
        "GRIB_shortName": "msl",
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
    }
    single["fdir"].attrs = {
        "GRIB_paramId": 228021,
        "GRIB_shortName": "fdir",
        "units": "J m**-2",
    }
    single["sf"].attrs = {
        "GRIB_paramId": 144,
        "GRIB_shortName": "sf",
        "standard_name": "lwe_thickness_of_snowfall_amount",
        "units": "m of water equivalent",
    }
    single["stl1"].attrs = {
        "GRIB_paramId": 139,
        "GRIB_shortName": "stl1",
        "standard_name": "surface_temperature",
        "units": "K",
    }
    pressure["q"].attrs = {
        "GRIB_paramId": 133,
        "GRIB_shortName": "q",
        "standard_name": "specific_humidity",
        "units": "kg kg**-1",
    }
    pressure["z"].attrs = {
        "GRIB_paramId": 129,
        "GRIB_shortName": "z",
        "standard_name": "geopotential",
        "units": "m**2 s**-2",
    }
    return {
        "single/spatial": single,
        "pressure/spatial": pressure,
    }


def mock_earthmover_brightband_ifs() -> xr.Dataset:
    lead_time = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    levels = np.array(IFS_ANALYSIS_LEVELS[::-1], dtype="float64")
    valid_time = TIMES[np.newaxis, :] + lead_time[:, np.newaxis]
    coords = {
        "lead_time": ("lead_time", lead_time),
        "init_time": ("init_time", TIMES),
        "valid_time": (("lead_time", "init_time"), valid_time),
        "level": ("level", levels),
        "latitude": LAT,
        "longitude": LON,
    }
    surface_dims = ("lead_time", "init_time", "latitude", "longitude")
    level_dims = ("lead_time", "init_time", "level", "latitude", "longitude")
    surface_shape = (lead_time.size, TIMES.size, LAT.size, LON.size)
    level_shape = (lead_time.size, TIMES.size, levels.size, LAT.size, LON.size)
    ds = xr.Dataset(
        {
            "t2m": (surface_dims, _grid(surface_shape)),
            "msl": (surface_dims, _grid(surface_shape) + 10.0),
            "q": (level_dims, _grid(level_shape)),
        },
        coords=coords,
    )
    ds["level"].attrs = {
        "standard_name": "air_pressure",
        "units": "hPa",
        "axis": "Z",
    }
    ds["t2m"].attrs = {
        "GRIB_paramId": 167,
        "GRIB_shortName": "2t",
        "GRIB_cfVarName": "t2m",
        "standard_name": "air_temperature",
        "units": "K",
    }
    ds["msl"].attrs = {
        "GRIB_paramId": 151,
        "GRIB_shortName": "msl",
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
    }
    ds["q"].attrs = {
        "GRIB_paramId": 133,
        "GRIB_shortName": "q",
        "standard_name": "specific_humidity",
        "units": "kg kg**-1",
    }
    return ds


def mock_earthmover_brightband_ifs_fx() -> xr.Dataset:
    coords = {
        "time": ("time", TIMES),
        "step": ("step", LEAD_TIMES),
        "latitude": LAT,
        "longitude": LON,
    }
    dims = ("time", "step", "latitude", "longitude")
    shape = (TIMES.size, LEAD_TIMES.size, LAT.size, LON.size)
    ds = xr.Dataset(
        {
            "2t": (dims, _grid(shape)),
            "fdir": (dims, _grid(shape) + 10.0),
            "msl": (dims, _grid(shape) + 20.0),
        },
        coords=coords,
    )
    ds["2t"].attrs = {
        "GRIB_paramId": 167,
        "GRIB_shortName": "2t",
        "units": "K",
    }
    ds["fdir"].attrs = {
        "GRIB_paramId": 228021,
        "GRIB_shortName": "fdir",
        "units": "J m**-2",
    }
    ds["msl"].attrs = {
        "GRIB_paramId": 151,
        "GRIB_shortName": "msl",
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
    }
    return ds


_MockStore = xr.Dataset | dict[str | None, xr.Dataset]


class _FakeSession:
    def __init__(self, store: _MockStore):
        self.store = store


class _FakeRepo:
    def __init__(self, store: _MockStore):
        self.store = store

    def readonly_session(self, branch="main"):
        return _FakeSession(self.store)


@pytest.fixture
def patch_earthmover(monkeypatch):
    def _patch(store: _MockStore, client_cls=None):
        import earth2studio.data.earthmover as earthmover

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                return _FakeRepo(store)

        def _open_zarr(_store, group=None, **kwargs):
            if isinstance(store, dict):
                return store[group]
            return store

        client_cls_ = client_cls or _FakeAsyncClient
        monkeypatch.setenv("EARTHMOVER_API_KEY", "test-key")
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, earthmover.__file__, raising=False
        )
        monkeypatch.setattr(
            earthmover, "arraylake", SimpleNamespace(AsyncClient=client_cls_)
        )
        monkeypatch.setattr(earthmover.xr, "open_zarr", _open_zarr)
        return earthmover

    return _patch


def assert_analysis_data_array(out: xr.DataArray, variables: list[str]) -> None:
    assert list(out.dims) == ["time", "variable", "lat", "lon"]
    assert out.shape == (1, len(variables), LAT.size, LON.size)
    assert list(out.coords["variable"].values) == variables
    np.testing.assert_allclose(out.lon.values, E2S_LON)
    assert np.isfinite(out.values).all()


def assert_forecast_data_array(
    out: xr.DataArray, variables: list[str], lead_time_count: int
) -> None:
    assert list(out.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    assert out.shape == (1, lead_time_count, len(variables), LAT.size, LON.size)
    assert list(out.coords["variable"].values) == variables
    np.testing.assert_allclose(out.lon.values, E2S_LON)
    assert np.isfinite(out.values).all()


class TestEarthMoverSources:
    def test_constructor_signatures(self):
        for cls in (
            EarthMoverERA5,
            EarthMoverBrightBandIFS,
            EarthMoverBrightBandIFS_FX,
        ):
            assert list(inspect.signature(cls.__init__).parameters) == [
                "self",
                "repo",
                "branch",
                "client",
                "cache",
                "verbose",
            ]

    def test_protocols(self):
        assert isinstance(EarthMoverERA5("org/repo"), DataSource)
        assert isinstance(EarthMoverBrightBandIFS("org/repo"), DataSource)
        assert isinstance(EarthMoverBrightBandIFS_FX("org/repo"), ForecastSource)

    def test_era5_supported_variables_match_marketplace_listing(self):
        assert EarthMoverERA5.VARIABLES == ERA5_VARIABLES
        assert "z500" in EarthMoverERA5.VARIABLES
        assert "pv500" in EarthMoverERA5.VARIABLES
        assert "fdir" in EarthMoverERA5.VARIABLES
        assert "sf" in EarthMoverERA5.VARIABLES
        assert "stl1" in EarthMoverERA5.VARIABLES

    def test_analysis_supported_variables_match_marketplace_listing(self):
        assert EarthMoverBrightBandIFS.VARIABLES == IFS_ANALYSIS_VARIABLES
        assert "q500" in EarthMoverBrightBandIFS.VARIABLES
        assert "fdir" not in EarthMoverBrightBandIFS.VARIABLES

    def test_forecast_supported_variables_match_marketplace_listing(self):
        assert (
            EarthMoverBrightBandIFS_FX.DATASET_VARIABLES
            == IFS_FORECAST_DATASET_VARIABLES
        )
        assert EarthMoverBrightBandIFS_FX.VARIABLES == IFS_FORECAST_VARIABLES
        assert "fdir" in EarthMoverBrightBandIFS_FX.VARIABLES
        assert "q500" not in EarthMoverBrightBandIFS_FX.VARIABLES

    def test_earthmover_era5_call_mock(self, patch_earthmover):
        patch_earthmover(mock_earthmover_era5())
        variables = ["t2m", "msl", "sf", "stl1", "q500", "z500"]
        ds = EarthMoverERA5("vandelay-industries/era5")

        out = ds(datetime(2022, 1, 1), variables)

        assert_analysis_data_array(out, variables)
        assert float(out.sel(variable="t2m").isel(time=0, lat=0, lon=0)) == 2.0
        assert float(out.sel(variable="sf").isel(time=0, lat=0, lon=0)) == 2.0

    def test_earthmover_brightband_ifs_call_mock(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs())
        variables = ["t2m", "msl", "q500"]
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        out = ds(datetime(2022, 1, 1), variables)

        assert_analysis_data_array(out, variables)
        assert float(out.sel(variable="t2m").isel(time=0, lat=0, lon=0)) == 2.0

    def test_earthmover_brightband_ifs_fx_call_mock(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs_fx())
        variables = ["t2m", "fdir", "msl"]
        ds = EarthMoverBrightBandIFS_FX("vandelay-industries/ifs")

        out = ds(
            datetime(2022, 1, 1),
            [timedelta(hours=0), timedelta(hours=6)],
            variables,
        )

        assert_forecast_data_array(out, variables, lead_time_count=2)
        assert (
            float(out.sel(variable="t2m").isel(time=0, lead_time=0, lat=0, lon=0))
            == 2.0
        )

    def test_era5_available(self):
        assert EarthMoverERA5.available(datetime(1940, 1, 1, 0))
        assert EarthMoverERA5.available(np.datetime64("2022-01-01T06:00:00"))
        assert not EarthMoverERA5.available(datetime(1939, 12, 31, 23))
        assert not EarthMoverERA5.available(datetime(2077, 1, 1))
        assert not EarthMoverERA5.available(datetime(2022, 1, 1, 0, 30))

    def test_analysis_available(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs())
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        assert ds.available(datetime(2022, 1, 1, 0))
        assert ds.available(np.datetime64("2022-01-01T06:00:00"))
        assert not ds.available(datetime(1999, 1, 1, 0))

    def test_forecast_available(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs_fx())
        ds = EarthMoverBrightBandIFS_FX("vandelay-industries/ifs")

        assert ds.available(datetime(2022, 1, 1, 0))
        assert ds.available(np.datetime64("2022-01-01T06:00:00"))
        assert not ds.available(datetime(1999, 1, 1, 0))


class TestEarthMoverConfig:
    def test_repo_from_organization_env(self, monkeypatch):
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        assert EarthMoverERA5()._repo_name == "my-org/era5-subscription"
        assert (
            EarthMoverBrightBandIFS()._repo_name
            == "my-org/ecmwf-ifs-initial-conditions-open-subscription"
        )
        assert (
            EarthMoverBrightBandIFS_FX()._repo_name
            == "my-org/ecmwf-ifs-15-day-forecast-open-subscription"
        )

    def test_explicit_repo_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        assert (
            EarthMoverERA5("other-org/custom-repo")._repo_name
            == "other-org/custom-repo"
        )
        assert (
            EarthMoverBrightBandIFS("other-org/custom-repo")._repo_name
            == "other-org/custom-repo"
        )
        assert (
            EarthMoverBrightBandIFS_FX("other-org/custom-repo")._repo_name
            == "other-org/custom-repo"
        )

    def test_missing_repo_requires_config(self, monkeypatch):
        monkeypatch.delenv("EARTHMOVER_ORGANIZATION", raising=False)

        with pytest.raises(ValueError, match="EARTHMOVER_ORGANIZATION"):
            EarthMoverERA5()

        with pytest.raises(ValueError, match="EARTHMOVER_ORGANIZATION"):
            EarthMoverBrightBandIFS()

        with pytest.raises(ValueError, match="EARTHMOVER_ORGANIZATION"):
            EarthMoverBrightBandIFS_FX()

    def test_env_api_key_used(self, patch_earthmover):
        seen = {}

        class _APIKeyClient:
            def __init__(self, token=None):
                seen["api_key"] = token

        patch_earthmover(mock_earthmover_era5(), client_cls=_APIKeyClient)
        ds = EarthMoverERA5("vandelay-industries/era5")

        ds._make_client()

        assert seen["api_key"] == os.environ["EARTHMOVER_API_KEY"]

    def test_auth_precedence(self, monkeypatch, patch_earthmover):
        earthmover = patch_earthmover(mock_earthmover_era5())
        sentinel = earthmover.arraylake.AsyncClient()
        monkeypatch.delenv("EARTHMOVER_API_KEY", raising=False)
        ds = EarthMoverERA5("vandelay-industries/era5", client=sentinel)

        assert ds._make_client() is sentinel

    def test_missing_api_key_requires_env(self, monkeypatch, patch_earthmover):
        patch_earthmover(mock_earthmover_era5())
        monkeypatch.delenv("EARTHMOVER_API_KEY", raising=False)
        ds = EarthMoverERA5("vandelay-industries/era5")

        with pytest.raises(ValueError, match="EARTHMOVER_API_KEY"):
            ds._make_client()

    def test_subscription_error(self, monkeypatch, patch_earthmover):
        class _DeniedClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                raise RuntimeError("403 Forbidden: access denied")

        patch_earthmover(mock_earthmover_era5(), client_cls=_DeniedClient)
        monkeypatch.setenv("EARTHMOVER_API_KEY", "test-key")
        ds = EarthMoverERA5("vandelay-industries/era5")

        with pytest.raises(PermissionError, match="subscription"):
            ds(datetime(2022, 1, 1), "t2m")


class TestEarthMoverErrors:
    def test_era5_exceptions(self, patch_earthmover):
        patch_earthmover(mock_earthmover_era5())
        ds = EarthMoverERA5("vandelay-industries/era5")

        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), "definitely_not_a_var")
        with pytest.raises(ValueError, match="Could not resolve"):
            ds(datetime(2022, 1, 1), "d2m")
        with pytest.raises(ValueError, match="on or after January 1st, 1940"):
            ds(datetime(1939, 12, 31, 23), "t2m")
        with pytest.raises(ValueError, match="1 hour interval"):
            ds(datetime(2022, 1, 1, 0, 30), "t2m")

    def test_analysis_exceptions(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs())
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), "definitely_not_a_var")
        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), "fdir")
        with pytest.raises(ValueError, match="Could not resolve"):
            ds(datetime(2022, 1, 1), "d2m")
        with pytest.raises(ValueError, match="not available"):
            ds(datetime(1999, 1, 1), "t2m")

    def test_forecast_exceptions(self, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs_fx())
        ds = EarthMoverBrightBandIFS_FX("vandelay-industries/ifs")

        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), timedelta(hours=0), "definitely_not_a_var")
        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), timedelta(hours=0), "q500")
        with pytest.raises(ValueError, match="not available"):
            ds(datetime(1999, 1, 1), timedelta(hours=0), "t2m")
