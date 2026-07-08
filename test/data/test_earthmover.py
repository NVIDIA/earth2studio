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
LON = np.linspace(0, 357.5, 10)
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


def _grid(extra_dims=()):
    shape = tuple(d.size for _, d in extra_dims) + (LAT.size, LON.size)
    return np.random.rand(*shape).astype("float32")


def era5_surface() -> xr.Dataset:
    """Create a single-level IFS-like dataset with GRIB metadata."""
    coords = {"valid_time": ("valid_time", TIMES), "latitude": LAT, "longitude": LON}
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "t2m": (dims, _grid([("valid_time", xr.DataArray(TIMES))])),
            "u10": (dims, _grid([("valid_time", xr.DataArray(TIMES))])),
            "msl": (dims, _grid([("valid_time", xr.DataArray(TIMES))])),
        },
        coords=coords,
    )
    ds["t2m"].attrs = {
        "GRIB_paramId": 167,
        "GRIB_shortName": "2t",
        "GRIB_cfVarName": "t2m",
        "standard_name": "unknown",
        "units": "K",
    }
    ds["u10"].attrs = {
        "GRIB_paramId": 165,
        "GRIB_shortName": "10u",
        "GRIB_cfVarName": "u10",
        "units": "m s**-1",
    }
    ds["msl"].attrs = {
        "GRIB_paramId": 151,
        "GRIB_shortName": "msl",
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
    }
    return ds


def era5_pressure() -> xr.Dataset:
    """Create an IFS-like pressure-level dataset."""
    levels = np.array([1000.0, 850.0, 500.0, 250.0])
    coords = {
        "valid_time": ("valid_time", TIMES),
        "pressure_level": ("pressure_level", levels),
        "latitude": LAT,
        "longitude": LON,
    }
    coords_da = xr.DataArray(levels, dims="pressure_level")
    dims = ("valid_time", "pressure_level", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "t": (dims, _grid([("valid_time", TIMES), ("pressure_level", coords_da)])),
            "z": (dims, _grid([("valid_time", TIMES), ("pressure_level", coords_da)])),
        },
        coords=coords,
    )
    ds["pressure_level"].attrs = {
        "standard_name": "air_pressure",
        "units": "hPa",
        "axis": "Z",
    }
    ds["t"].attrs = {
        "GRIB_paramId": 130,
        "GRIB_shortName": "t",
        "standard_name": "air_temperature",
        "units": "K",
    }
    ds["z"].attrs = {
        "GRIB_paramId": 129,
        "GRIB_shortName": "z",
        "standard_name": "geopotential",
        "units": "m**2 s**-2",
    }
    return ds


def ifs_analysis() -> xr.Dataset:
    """Create a single IFS-like analysis store with surface and pressure fields."""
    return xr.merge([era5_surface(), era5_pressure()])


def ifs_forecast() -> xr.Dataset:
    """Create an IFS forecast-like dataset with a step axis."""
    coords = {
        "time": ("time", TIMES),
        "step": ("step", LEAD_TIMES),
        "latitude": LAT,
        "longitude": LON,
    }
    dims = ("time", "step", "latitude", "longitude")
    data = np.random.rand(TIMES.size, LEAD_TIMES.size, LAT.size, LON.size).astype(
        "float32"
    )
    ds = xr.Dataset(
        {
            "2t": (dims, data.copy()),
            "fdir": (dims, data.copy()),
            "msl": (dims, data.copy()),
        },
        coords=coords,
    )
    ds["2t"].attrs = {"long_name": "2 metre temperature", "units": "K"}
    ds["fdir"].attrs = {
        "long_name": "Total sky direct solar radiation at surface",
        "units": "J m**-2",
    }
    ds["msl"].attrs = {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Mean sea level pressure",
        "units": "Pa",
    }
    return ds


def hrrr_celsius() -> xr.Dataset:
    """Create a regular-grid dataset with CF metadata and Celsius units."""
    coords = {"time": ("time", TIMES), "latitude": LAT, "longitude": LON}
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset({"temperature_2m": (dims, _grid([("time", TIMES)]))}, coords=coords)
    ds["temperature_2m"].attrs = {
        "standard_name": "air_temperature",
        "long_name": "2 metre temperature",
        "units": "degree_Celsius",
    }
    return ds


def projected_grid() -> xr.Dataset:
    """Create a projected grid with 2-D latitude and longitude coordinates."""
    y = np.arange(6)
    x = np.arange(7)
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {"temperature_2m": (dims, np.random.rand(TIMES.size, y.size, x.size))},
        coords={
            "time": ("time", TIMES),
            "latitude": (("y", "x"), np.random.rand(y.size, x.size)),
            "longitude": (("y", "x"), np.random.rand(y.size, x.size)),
            "y": y,
            "x": x,
        },
    )
    ds["temperature_2m"].attrs = {"standard_name": "air_temperature", "units": "K"}
    return ds


def ambiguous_winds() -> xr.Dataset:
    """Create a dataset with ambiguous CF-only surface wind metadata."""
    coords = {"time": ("time", TIMES), "latitude": LAT, "longitude": LON}
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "wind_u_10m": (dims, _grid([("time", TIMES)])),
            "wind_u_80m": (dims, _grid([("time", TIMES)])),
        },
        coords=coords,
    )
    ds["wind_u_10m"].attrs = {"standard_name": "eastward_wind", "units": "m s-1"}
    ds["wind_u_80m"].attrs = {"standard_name": "eastward_wind", "units": "m s-1"}
    return ds


class _FakeSession:
    def __init__(self, datasets):
        self.store = object()
        self._datasets = datasets


class _FakeRepo:
    def __init__(self, datasets):
        self._datasets = datasets

    def readonly_session(self, branch="main"):
        return _FakeSession(self._datasets)


@pytest.fixture
def patch_earthmover(monkeypatch):
    """Patch arraylake and xr.open_zarr for no-network Earthmover tests."""

    def _patch(datasets, client_cls=None):
        import earth2studio.data.earthmover as earthmover

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                return _FakeRepo(datasets)

        client_cls_ = client_cls or _FakeAsyncClient
        monkeypatch.setenv("EARTHMOVER_API_TOKEN", "test-token")
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, earthmover.__file__, raising=False
        )
        monkeypatch.setattr(
            earthmover, "arraylake", SimpleNamespace(AsyncClient=client_cls_)
        )

        calls = {"i": 0}

        def fake_open_zarr(store, group=None, **kwargs):
            ds = datasets[calls["i"]]
            calls["i"] += 1
            return ds

        monkeypatch.setattr(earthmover.xr, "open_zarr", fake_open_zarr)
        return earthmover

    return _patch


def assert_analysis_data_array(out: xr.DataArray, variables: list[str]) -> None:
    """Check the standard Earth2Studio analysis datasource shape."""
    assert list(out.dims) == ["time", "variable", "lat", "lon"]
    assert out.shape == (1, len(variables), LAT.size, LON.size)
    assert list(out.coords["variable"].values) == variables
    assert np.isfinite(out.values).all()


def assert_forecast_data_array(
    out: xr.DataArray, variables: list[str], lead_time_count: int
) -> None:
    """Check the standard Earth2Studio forecast datasource shape."""
    assert list(out.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    assert out.shape == (1, lead_time_count, len(variables), LAT.size, LON.size)
    assert list(out.coords["variable"].values) == variables
    assert np.isfinite(out.values).all()


class TestEarthMoverMockSources:
    """Mock datasource tests that run without network access."""

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

    def test_protocol(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        assert isinstance(EarthMoverBrightBandIFS("org/repo"), DataSource)
        assert isinstance(EarthMoverBrightBandIFS_FX("org/repo"), ForecastSource)

    def test_forecast_supported_variables_match_marketplace_listing(self):
        assert (
            EarthMoverBrightBandIFS_FX.DATASET_VARIABLES == FORECAST_DATASET_VARIABLES
        )
        assert EarthMoverBrightBandIFS.VARIABLES == FORECAST_VARIABLES
        assert EarthMoverBrightBandIFS_FX.VARIABLES == FORECAST_VARIABLES

    def test_analysis_repo_from_organization_env(self, monkeypatch, patch_earthmover):
        patch_earthmover([era5_surface()])
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        ds = EarthMoverBrightBandIFS()

        assert ds._repo_name == "my-org/ecmwf-ifs-initial-conditions-open-subscription"

    def test_analysis_explicit_repo_takes_precedence(
        self, monkeypatch, patch_earthmover
    ):
        patch_earthmover([era5_surface()])
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        ds = EarthMoverBrightBandIFS("other-org/custom-repo")

        assert ds._repo_name == "other-org/custom-repo"

    def test_forecast_repo_from_organization_env(self, monkeypatch, patch_earthmover):
        patch_earthmover([ifs_forecast()])
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        ds = EarthMoverBrightBandIFS_FX()

        assert ds._repo_name == "my-org/ecmwf-ifs-15-day-forecast-open-subscription"

    def test_forecast_explicit_repo_takes_precedence(
        self, monkeypatch, patch_earthmover
    ):
        patch_earthmover([ifs_forecast()])
        monkeypatch.setenv("EARTHMOVER_ORGANIZATION", "my-org")

        ds = EarthMoverBrightBandIFS_FX("other-org/custom-repo")

        assert ds._repo_name == "other-org/custom-repo"

    def test_analysis_fetch(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        variables = ["t2m", "msl", "u10m"]
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        out = ds(datetime(2022, 1, 1), variables)

        assert_analysis_data_array(out, variables)

    @pytest.mark.asyncio
    async def test_analysis_fetch_async(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        out = await ds.fetch(datetime(2022, 1, 1), "t2m")

        assert_analysis_data_array(out, ["t2m"])

    def test_analysis_celsius_conversion(self, patch_earthmover):
        patch_earthmover([hrrr_celsius()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/hrrr")

        out = ds(datetime(2022, 1, 1), "t2m")

        assert_analysis_data_array(out, ["t2m"])
        assert float(out.min()) > 200.0

    def test_paramid_disambiguation(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        out = ds(datetime(2022, 1, 1), "u10m")

        assert_analysis_data_array(out, ["u10m"])

    def test_forecast_fetch(self, patch_earthmover):
        patch_earthmover([ifs_forecast()])
        variables = ["t2m", "fdir", "msl"]
        ds = EarthMoverBrightBandIFS_FX("vandelay-industries/ifs")

        out = ds(
            datetime(2022, 1, 1),
            [timedelta(hours=0), timedelta(hours=6)],
            variables,
        )

        assert_forecast_data_array(out, variables, lead_time_count=2)

    @pytest.mark.asyncio
    async def test_forecast_fetch_async(self, patch_earthmover):
        patch_earthmover([ifs_forecast()])
        ds = EarthMoverBrightBandIFS_FX("vandelay-industries/ifs")

        out = await ds.fetch(datetime(2022, 1, 1), timedelta(hours=0), "t2m")

        assert_forecast_data_array(out, ["t2m"], lead_time_count=1)

    def test_available(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        assert ds.available(datetime(2022, 1, 1, 0))
        assert not ds.available(datetime(1999, 1, 1, 0))


class TestEarthMoverErrors:
    def test_unknown_variable(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
            ds(datetime(2022, 1, 1), "definitely_not_a_var")

    def test_unresolved_variable(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        with pytest.raises(ValueError, match="Could not resolve"):
            ds(datetime(2022, 1, 1), "d2m")

    def test_time_not_available(self, patch_earthmover):
        patch_earthmover([era5_surface()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        with pytest.raises(ValueError, match="not available"):
            ds(datetime(1999, 1, 1), "t2m")

    def test_ambiguous_resolution(self, patch_earthmover):
        patch_earthmover([ambiguous_winds()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/hrrr")

        with pytest.raises(ValueError, match="ambiguous"):
            ds(datetime(2022, 1, 1), "u10m")

    def test_projected_grid_error(self, patch_earthmover):
        patch_earthmover([projected_grid()])
        ds = EarthMoverBrightBandIFS("vandelay-industries/hrrr-analysis")

        with pytest.raises(ValueError, match="projected"):
            ds(datetime(2022, 1, 1), "t2m")

    def test_subscription_error(self, monkeypatch, patch_earthmover):
        class _DeniedClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_repo(self, name):
                raise RuntimeError("403 Forbidden: access denied")

        patch_earthmover([era5_surface()], client_cls=_DeniedClient)
        monkeypatch.setenv("EARTHMOVER_API_TOKEN", "test-token")
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        with pytest.raises(PermissionError, match="subscription"):
            ds(datetime(2022, 1, 1), "t2m")

    def test_env_token_used(self, patch_earthmover):
        seen = {}

        class _TokenClient:
            def __init__(self, token=None):
                seen["token"] = token

        patch_earthmover([era5_surface()], client_cls=_TokenClient)
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        ds._make_client()

        assert seen["token"] == os.environ["EARTHMOVER_API_TOKEN"]

    def test_auth_precedence(self, monkeypatch, patch_earthmover):
        earthmover = patch_earthmover([era5_surface()])
        sentinel = earthmover.arraylake.AsyncClient()
        monkeypatch.delenv("EARTHMOVER_API_TOKEN", raising=False)
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5", client=sentinel)

        assert ds._make_client() is sentinel

    def test_missing_token_requires_env(self, monkeypatch, patch_earthmover):
        patch_earthmover([era5_surface()])
        monkeypatch.delenv("EARTHMOVER_API_TOKEN", raising=False)
        ds = EarthMoverBrightBandIFS("vandelay-industries/era5")

        with pytest.raises(ValueError, match="EARTHMOVER_API_TOKEN"):
            ds._make_client()

    def test_missing_repo_requires_config(self, monkeypatch, patch_earthmover):
        patch_earthmover([era5_surface()])
        monkeypatch.delenv("EARTHMOVER_ORGANIZATION", raising=False)
        monkeypatch.delenv("EARTHMOVER_ORGINIZATION", raising=False)

        with pytest.raises(ValueError, match="EARTHMOVER_ORGANIZATION"):
            EarthMoverBrightBandIFS()
