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

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import Arraylake, ArraylakeForecast
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.lexicon.arraylake import (
    ArraylakeLexicon,
    make_modifier,
    normalize_units,
)

# ---------------------------------------------------------------------------
# Synthetic datasets reproducing the three metadata conventions observed across
# Earthmover Marketplace repositories. These let us exercise the metadata-driven
# resolver without network access.
# ---------------------------------------------------------------------------

LAT = np.linspace(90, -90, 9)
LON = np.linspace(0, 357.5, 10)
TIMES = np.array(["2022-01-01T00:00:00", "2022-01-01T06:00:00"], dtype="datetime64[ns]")


def _grid(extra_dims=()):
    shape = tuple(d.size for _, d in extra_dims) + (LAT.size, LON.size)
    return np.random.rand(*shape).astype("float32")


def era5_surface() -> xr.Dataset:
    """ERA5-style: ECMWF cfVarName variable names + full GRIB_* attrs (SI units)."""
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
    """ERA5-style pressure group: single var + pressure_level coordinate."""
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
    # z stored as geopotential (m2 s-2) -> should resolve with identity modifier
    ds["z"].attrs = {
        "GRIB_paramId": 129,
        "GRIB_shortName": "z",
        "standard_name": "geopotential",
        "units": "m**2 s**-2",
    }
    return ds


def ifs_forecast() -> xr.Dataset:
    """IFS-style: ECMWF GRIB short names as variable names, step (lead) axis."""
    steps = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    coords = {
        "time": ("time", TIMES),
        "step": ("step", steps),
        "latitude": LAT,
        "longitude": LON,
    }
    dims = ("time", "step", "latitude", "longitude")
    data = np.random.rand(TIMES.size, steps.size, LAT.size, LON.size).astype("float32")
    ds = xr.Dataset(
        {"2t": (dims, data.copy()), "msl": (dims, data.copy())}, coords=coords
    )
    ds["2t"].attrs = {"long_name": "2 metre temperature", "units": "K"}
    ds["msl"].attrs = {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Mean sea level pressure",
        "units": "Pa",
    }
    return ds


def hrrr_celsius() -> xr.Dataset:
    """HRRR-style: descriptive names, CF standard_name, Celsius units, lat/lon grid.

    (A regular lat/lon variant; the real HRRR archive is on a projected grid,
    which is covered separately in the projected-grid test.)
    """
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
    """Projected (Lambert) grid: 2-D lat/lon over y/x dims."""
    y = np.arange(6)
    x = np.arange(7)
    lat2d = np.random.rand(y.size, x.size)
    lon2d = np.random.rand(y.size, x.size)
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {"temperature_2m": (dims, np.random.rand(TIMES.size, y.size, x.size))},
        coords={
            "time": ("time", TIMES),
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
            "y": y,
            "x": x,
        },
    )
    ds["temperature_2m"].attrs = {"standard_name": "air_temperature", "units": "K"}
    return ds


def ambiguous_winds() -> xr.Dataset:
    """Two surface vars share standard_name eastward_wind (10 m vs 80 m).

    Mirrors HRRR, where height is encoded in the variable name, not metadata.
    """
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


def _patch_open(monkeypatch, datasets):
    """Patch arraylake client + xr.open_zarr to serve `datasets` per group order."""
    import earth2studio.data.arraylake as almod

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_repo(self, name):
            return _FakeRepo(datasets)

    monkeypatch.setattr(almod.arraylake, "Client", _FakeClient)

    calls = {"i": 0}

    def fake_open_zarr(store, group=None, **kwargs):
        ds = datasets[calls["i"]]
        calls["i"] += 1
        return ds

    monkeypatch.setattr(almod.xr, "open_zarr", fake_open_zarr)


# ---------------------------------------------------------------------------
# Lexicon / unit-conversion unit tests (no network, no mocking)
# ---------------------------------------------------------------------------


def test_arraylake_lexicon_specs():
    # Pressure-level id encodes variable + level; surface ids are distinct.
    assert ArraylakeLexicon.spec("t850").param_id == 130
    assert ArraylakeLexicon.spec("t850").level == 850
    assert ArraylakeLexicon.spec("z500").short_name == "z"
    # The classic name collision: E2S u10 (10 hPa) != ECMWF u10 (10 m wind).
    assert ArraylakeLexicon.spec("u10").level_type == "isobaric"
    assert ArraylakeLexicon.spec("u10m").level_type == "surface"
    # The two must NOT share a paramId (131 = u on pressure level, 165 = 10 m wind).
    assert (
        ArraylakeLexicon.spec("u10").param_id != ArraylakeLexicon.spec("u10m").param_id
    )
    with pytest.raises(KeyError):
        ArraylakeLexicon.spec("not_a_variable")


def test_arraylake_unit_normalization():
    assert normalize_units("m s**-1") == normalize_units("m s-1")
    assert normalize_units("degree_Celsius") == "degc"
    assert normalize_units("(0 - 1)") == "1"
    assert normalize_units("percent") == "percent"


def test_arraylake_make_modifier():
    t2m = ArraylakeLexicon.spec("t2m")
    celsius_to_k = make_modifier(t2m, "degree_Celsius")
    assert np.isclose(celsius_to_k(np.array([0.0]))[0], 273.15)
    # Geopotential height (m) -> geopotential (m2 s-2)
    z500 = ArraylakeLexicon.spec("z500")
    gh_mod = make_modifier(z500, "m")
    assert np.isclose(gh_mod(np.array([1.0]))[0], 9.80665)
    # Already geopotential -> identity
    assert np.isclose(make_modifier(z500, "m2 s-2")(np.array([5.0]))[0], 5.0)
    # Cloud cover percent -> fraction
    tcc = ArraylakeLexicon.spec("tcc")
    assert np.isclose(make_modifier(tcc, "percent")(np.array([50.0]))[0], 0.5)


# ---------------------------------------------------------------------------
# Mocked resolution / fetch tests
# ---------------------------------------------------------------------------


def test_arraylake_protocol():
    assert isinstance(Arraylake("org/repo"), DataSource)
    assert isinstance(ArraylakeForecast("org/repo"), ForecastSource)


def test_arraylake_call_mock_era5(monkeypatch):
    _patch_open(monkeypatch, [era5_surface(), era5_pressure()])
    ds = Arraylake("vandelay-industries/era5", group=["single", "pressure"])
    out = ds(datetime(2022, 1, 1), ["t2m", "msl", "t850", "z500"])
    assert list(out.dims) == ["time", "variable", "lat", "lon"]
    assert out.shape == (1, 4, LAT.size, LON.size)
    assert list(out.coords["variable"].values) == ["t2m", "msl", "t850", "z500"]
    assert np.isfinite(out.values).all()


def test_arraylake_call_mock_celsius_conversion(monkeypatch):
    _patch_open(monkeypatch, [hrrr_celsius()])
    ds = Arraylake("vandelay-industries/hrrr")
    out = ds(datetime(2022, 1, 1), "t2m")
    # degree_Celsius source values (in [0,1)) must be shifted to Kelvin range.
    assert float(out.min()) > 200.0


def test_arraylake_paramid_disambiguation(monkeypatch):
    # u10m (10 m wind, paramId 165) must match cfVarName "u10", NOT be confused
    # with the E2S "u10" (10 hPa wind). Resolver keys on paramId/shortName.
    _patch_open(monkeypatch, [era5_surface()])
    ds = Arraylake("vandelay-industries/era5", group="single")
    out = ds(datetime(2022, 1, 1), "u10m")
    assert out.shape == (1, 1, LAT.size, LON.size)


def test_arraylake_forecast_mock_ifs(monkeypatch):
    _patch_open(monkeypatch, [ifs_forecast()])
    ds = ArraylakeForecast("vandelay-industries/ifs")
    out = ds(
        datetime(2022, 1, 1),
        [timedelta(hours=0), timedelta(hours=6)],
        ["t2m", "msl"],
    )
    assert list(out.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    assert out.shape == (1, 2, 2, LAT.size, LON.size)


def test_arraylake_available(monkeypatch):
    _patch_open(monkeypatch, [era5_surface()])
    ds = Arraylake("vandelay-industries/era5", group="single")
    assert ds.available(datetime(2022, 1, 1, 0))
    assert not ds.available(datetime(1999, 1, 1, 0))


def test_arraylake_exceptions(monkeypatch):
    # Unknown E2S variable.
    _patch_open(monkeypatch, [era5_surface()])
    ds = Arraylake("vandelay-industries/era5", group="single")
    with pytest.raises(ValueError, match="not a known Earth2Studio variable"):
        ds(datetime(2022, 1, 1), "definitely_not_a_var")

    # Variable not present in the repo (no matching metadata).
    _patch_open(monkeypatch, [era5_surface()])
    ds2 = Arraylake("vandelay-industries/era5", group="single")
    with pytest.raises(ValueError, match="Could not resolve"):
        ds2(datetime(2022, 1, 1), "q500")

    # Time not available.
    _patch_open(monkeypatch, [era5_surface()])
    ds3 = Arraylake("vandelay-industries/era5", group="single")
    with pytest.raises(ValueError, match="not available"):
        ds3(datetime(1999, 1, 1), "t2m")


def test_arraylake_ambiguous_resolution(monkeypatch):
    # When only standard_name matches and multiple vars share it, refuse to guess.
    _patch_open(monkeypatch, [ambiguous_winds()])
    ds = Arraylake("vandelay-industries/hrrr")
    with pytest.raises(ValueError, match="ambiguous"):
        ds(datetime(2022, 1, 1), "u10m")


def test_arraylake_projected_grid_error(monkeypatch):
    _patch_open(monkeypatch, [projected_grid()])
    ds = Arraylake("vandelay-industries/hrrr-analysis")
    with pytest.raises(ValueError, match="projected"):
        ds(datetime(2022, 1, 1), "t2m")


def test_arraylake_subscription_error(monkeypatch):
    import earth2studio.data.arraylake as almod

    class _DeniedClient:
        def __init__(self, *a, **k):
            pass

        def get_repo(self, name):
            raise RuntimeError("403 Forbidden: access denied")

    monkeypatch.setattr(almod.arraylake, "Client", _DeniedClient)
    ds = Arraylake("vandelay-industries/era5")
    with pytest.raises(PermissionError, match="subscription"):
        ds(datetime(2022, 1, 1), "t2m")


def test_arraylake_auth_precedence(monkeypatch):
    # Explicit client takes precedence; no network/credentials needed.
    _patch_open(monkeypatch, [era5_surface()])
    import earth2studio.data.arraylake as almod

    sentinel = almod.arraylake.Client()
    ds = Arraylake("vandelay-industries/era5", group="single", client=sentinel)
    assert ds._make_client() is sentinel
