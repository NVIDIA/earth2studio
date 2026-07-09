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

from earth2studio.data import EarthMoverBrightBandIFS, EarthMoverBrightBandIFS_FX
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.utils.imports import OptionalDependencyFailure

LAT = np.linspace(90, -90, 9)
LON = np.array([-180.0, -90.0, 0.0, 90.0])
E2S_LON = np.array([0.0, 90.0, 180.0, 270.0])
TIMES = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
FORECAST_DATASET_VARIABLES = (
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
FORECAST_VARIABLES = (
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
ANALYSIS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ANALYSIS_SURFACE_VARIABLES = (
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
ANALYSIS_PRESSURE_VARIABLES = tuple(
    f"{name}{level}"
    for name in ("q", "t", "u", "v", "w", "z")
    for level in ANALYSIS_LEVELS
)
ANALYSIS_VARIABLES = ANALYSIS_SURFACE_VARIABLES + ANALYSIS_PRESSURE_VARIABLES
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


def mock_earthmover_brightband_ifs() -> xr.Dataset:
    lead_time = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    levels = np.array(ANALYSIS_LEVELS[::-1], dtype="float64")
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


class _FakeSession:
    def __init__(self, dataset: xr.Dataset):
        self.store = object()
        self.dataset = dataset


class _FakeRepo:
    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

    def readonly_session(self, branch="main"):
        return _FakeSession(self.dataset)


@pytest.fixture
def patch_earthmover(monkeypatch):
    def _patch(dataset: xr.Dataset, client_cls=None):
        import earth2studio.data.earthmover as earthmover

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                return _FakeRepo(dataset)

        client_cls_ = client_cls or _FakeAsyncClient
        monkeypatch.setenv("EARTHMOVER_API_KEY", "test-key")
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, earthmover.__file__, raising=False
        )
        monkeypatch.setattr(
            earthmover, "arraylake", SimpleNamespace(AsyncClient=client_cls_)
        )
        monkeypatch.setattr(earthmover.xr, "open_zarr", lambda *args, **kwargs: dataset)
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
        for cls in (EarthMoverBrightBandIFS, EarthMoverBrightBandIFS_FX):
            assert list(inspect.signature(cls.__init__).parameters) == [
                "self",
                "repo",
                "branch",
                "client",
                "cache",
                "verbose",
            ]

    def test_protocols(self):
        assert isinstance(EarthMoverBrightBandIFS("org/repo"), DataSource)
        assert isinstance(EarthMoverBrightBandIFS_FX("org/repo"), ForecastSource)

    def test_analysis_supported_variables_match_marketplace_listing(self):
        assert EarthMoverBrightBandIFS.VARIABLES == ANALYSIS_VARIABLES
        assert "q500" in EarthMoverBrightBandIFS.VARIABLES
        assert "fdir" not in EarthMoverBrightBandIFS.VARIABLES

    def test_forecast_supported_variables_match_marketplace_listing(self):
        assert (
            EarthMoverBrightBandIFS_FX.DATASET_VARIABLES == FORECAST_DATASET_VARIABLES
        )
        assert EarthMoverBrightBandIFS_FX.VARIABLES == FORECAST_VARIABLES
        assert "fdir" in EarthMoverBrightBandIFS_FX.VARIABLES
        assert "q500" not in EarthMoverBrightBandIFS_FX.VARIABLES

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
            EarthMoverBrightBandIFS()

        with pytest.raises(ValueError, match="EARTHMOVER_ORGANIZATION"):
            EarthMoverBrightBandIFS_FX()

    def test_env_api_key_used(self, patch_earthmover):
        seen = {}

        class _APIKeyClient:
            def __init__(self, token=None):
                seen["api_key"] = token

        patch_earthmover(mock_earthmover_brightband_ifs(), client_cls=_APIKeyClient)
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        ds._make_client()

        assert seen["api_key"] == os.environ["EARTHMOVER_API_KEY"]

    def test_auth_precedence(self, monkeypatch, patch_earthmover):
        earthmover = patch_earthmover(mock_earthmover_brightband_ifs())
        sentinel = earthmover.arraylake.AsyncClient()
        monkeypatch.delenv("EARTHMOVER_API_KEY", raising=False)
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs", client=sentinel)

        assert ds._make_client() is sentinel

    def test_missing_api_key_requires_env(self, monkeypatch, patch_earthmover):
        patch_earthmover(mock_earthmover_brightband_ifs())
        monkeypatch.delenv("EARTHMOVER_API_KEY", raising=False)
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        with pytest.raises(ValueError, match="EARTHMOVER_API_KEY"):
            ds._make_client()

    def test_subscription_error(self, monkeypatch, patch_earthmover):
        class _DeniedClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                raise RuntimeError("403 Forbidden: access denied")

        patch_earthmover(mock_earthmover_brightband_ifs(), client_cls=_DeniedClient)
        monkeypatch.setenv("EARTHMOVER_API_KEY", "test-key")
        ds = EarthMoverBrightBandIFS("vandelay-industries/ifs")

        with pytest.raises(PermissionError, match="subscription"):
            ds(datetime(2022, 1, 1), "t2m")


class TestEarthMoverErrors:
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
