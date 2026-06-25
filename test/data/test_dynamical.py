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

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import Dynamical, DynamicalForecast

# Un-normalized synthetic grid: ascending latitude and -180..180 longitude,
# mirroring how dynamical.org serves coordinates.
_LAT = np.linspace(-90.0, 90.0, 7)
_LON = np.linspace(-180.0, 135.0, 8)

# Variable -> (unit, raw value) used to build the synthetic store and to verify
# STAC-unit-driven conversion to the Earth2Studio convention.
_VARS = {
    "temperature_2m": ("degree_Celsius", 20.0),
    "wind_u_10m": ("m s-1", 3.0),
    "total_cloud_cover_atmosphere": ("percent", 50.0),
    "geopotential_height_500hpa": ("m", 5500.0),
}


def _fake_collection(dims: dict) -> dict:
    return {
        "id": "test-collection",
        "cube:dimensions": dims,
        "cube:variables": {
            name: {"dimensions": list(dims), "unit": unit}
            for name, (unit, _) in _VARS.items()
        },
        "assets": {
            "icechunk": {
                "href": "s3://test-bucket/test-prefix/v0.icechunk/",
                "xarray:storage_options": {
                    "anon": True,
                    "client_kwargs": {"region_name": "us-west-2"},
                },
            }
        },
    }


def _fake_catalog(collection_id: str) -> dict:
    return {
        "links": [
            {
                "rel": "child",
                "href": f"https://stac.dynamical.org/{collection_id}/collection.json",
            }
        ]
    }


def _analysis_dims(times: np.ndarray) -> dict:
    return {
        "time": {
            "type": "temporal",
            "extent": [str(times[0]) + "Z", None],
        },
        "latitude": {"type": "spatial"},
        "longitude": {"type": "spatial"},
    }


def _make_dataset(coords: dict, dims: tuple) -> xr.Dataset:
    shape = tuple(len(coords[d]) for d in dims)
    data_vars = {
        name: (dims, np.full(shape, value, dtype=np.float32))
        for name, (_, value) in _VARS.items()
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _patch(monkeypatch, klass, collection_id, dims, dataset):
    """Patch STAC fetch + Icechunk open to serve synthetic data (no network)."""

    def fake_fetch_json(url: str) -> dict:
        if url.endswith("catalog.json"):
            return _fake_catalog(collection_id)
        return _fake_collection(dims)

    monkeypatch.setattr("earth2studio.data.dynamical._fetch_json", fake_fetch_json)
    monkeypatch.setattr(klass, "_open_icechunk", lambda self, asset: dataset)


@pytest.mark.timeout(30)
def test_dynamical_call_mock(monkeypatch):
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, Dynamical, "test-analysis", dims, ds_obj)

    source = Dynamical("test-analysis")
    variables = ["t2m", "u10m", "tcc", "z500"]
    data = source(times, variables)

    assert data.shape == (2, 4, len(_LAT), len(_LON))
    assert list(data.coords["variable"].values) == variables
    # Grid normalized: latitude descending, longitude in [0, 360) ascending
    assert data.coords["lat"].values[0] == pytest.approx(90.0)
    assert data.coords["lat"].values[-1] == pytest.approx(-90.0)
    assert (data.coords["lon"].values >= 0).all()
    assert (np.diff(data.coords["lon"].values) > 0).all()
    # STAC-unit-driven conversions
    np.testing.assert_allclose(data.sel(variable="t2m").values, 20.0 + 273.15)
    np.testing.assert_allclose(data.sel(variable="u10m").values, 3.0)
    np.testing.assert_allclose(data.sel(variable="tcc").values, 0.5)
    np.testing.assert_allclose(data.sel(variable="z500").values, 5500.0 * 9.80665)


@pytest.mark.timeout(30)
def test_dynamical_native_passthrough(monkeypatch):
    times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, Dynamical, "test-analysis", dims, ds_obj)

    source = Dynamical("test-analysis")
    # Native dynamical.org variable name not in the lexicon
    data = source(times, ["wind_u_10m"])
    np.testing.assert_allclose(data.values, 3.0)


@pytest.mark.timeout(30)
def test_dynamical_forecast_call_mock(monkeypatch):
    init_times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    leads = np.array([0, 6 * 3600, 24 * 3600], dtype="timedelta64[s]").astype(
        "timedelta64[ns]"
    )
    dims = {
        "init_time": {"type": "temporal", "extent": [str(init_times[0]) + "Z", None]},
        "lead_time": {"type": "other"},
        "latitude": {"type": "spatial"},
        "longitude": {"type": "spatial"},
    }
    ds_obj = _make_dataset(
        {
            "init_time": init_times,
            "lead_time": leads,
            "latitude": _LAT,
            "longitude": _LON,
        },
        ("init_time", "lead_time", "latitude", "longitude"),
    )
    _patch(monkeypatch, DynamicalForecast, "test-forecast", dims, ds_obj)

    source = DynamicalForecast("test-forecast")
    lead_list = [datetime.timedelta(hours=h) for h in (0, 6, 24)]
    data = source(init_times, lead_list, ["t2m", "z500"])

    assert data.shape == (1, 3, 2, len(_LAT), len(_LON))
    assert list(data.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    np.testing.assert_allclose(data.sel(variable="t2m").values, 20.0 + 273.15)
    np.testing.assert_allclose(data.sel(variable="z500").values, 5500.0 * 9.80665)


@pytest.mark.timeout(30)
def test_dynamical_exceptions(monkeypatch):
    times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, Dynamical, "test-analysis", dims, ds_obj)

    # Unknown variable (not in lexicon, not native to collection)
    source = Dynamical("test-analysis")
    with pytest.raises(KeyError):
        source(times, ["definitely_not_a_variable"])

    # Variable in lexicon but not served by this collection
    source = Dynamical("test-analysis")
    with pytest.raises(KeyError):
        source(times, ["t850"])

    # Time before the collection's temporal extent
    source = Dynamical("test-analysis")
    with pytest.raises(ValueError):
        source(np.array(["1900-01-01T00:00"], dtype="datetime64[ns]"), ["t2m"])


@pytest.mark.timeout(30)
def test_dynamical_unknown_collection(monkeypatch):
    monkeypatch.setattr(
        "earth2studio.data.dynamical._fetch_json",
        lambda url: _fake_catalog("noaa-gfs-analysis"),
    )
    source = Dynamical("does-not-exist")
    with pytest.raises(ValueError):
        source(np.array(["2024-01-01T00:00"], dtype="datetime64[ns]"), ["t2m"])


@pytest.mark.timeout(30)
def test_dynamical_projected_grid(monkeypatch):
    projected_dims = {
        "time": {"type": "temporal", "extent": ["2024-01-01T00:00:00Z", None]},
        "x": {"type": "spatial"},
        "y": {"type": "spatial"},
    }

    def fake_fetch_json(url: str) -> dict:
        if url.endswith("catalog.json"):
            return _fake_catalog("noaa-hrrr-analysis")
        return _fake_collection(projected_dims)

    monkeypatch.setattr("earth2studio.data.dynamical._fetch_json", fake_fetch_json)
    source = Dynamical("noaa-hrrr-analysis")
    with pytest.raises(ValueError):
        source(np.array(["2024-01-01T00:00"], dtype="datetime64[ns]"), ["t2m"])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2024, month=1, day=1),
        np.array([np.datetime64("2024-06-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "u10m"], "msl"])
def test_dynamical_fetch(time, variable):
    source = Dynamical("noaa-gfs-analysis")
    data = source(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime.datetime):
        time = [time]

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(variable)
    assert data.shape[2] == 721
    assert data.shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()
    # Temperature converted to Kelvin
    if "t2m" in variable:
        t2m = data.sel(variable="t2m").values
        assert np.nanmin(t2m) > 180.0 and np.nanmax(t2m) < 350.0


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "collection,variable",
    [
        # noaa-gfs-forecast is surface-only (no pressure levels)
        ("noaa-gfs-forecast", ["t2m", "u10m"]),
        # ecmwf-aifs-single-forecast carries pressure-level fields
        ("ecmwf-aifs-single-forecast", ["t2m", "z500"]),
    ],
)
def test_dynamical_forecast_fetch(collection, variable):
    source = DynamicalForecast(collection)
    # 2024-06-01 is valid for both GFS (from 2021) and AIFS (from 2024-04-01)
    time = [datetime.datetime(year=2024, month=6, day=1)]
    lead_time = [datetime.timedelta(hours=0), datetime.timedelta(hours=24)]
    data = source(time, lead_time, variable)

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(lead_time)
    assert data.shape[2] == len(variable)
    assert data.shape[3] == 721
    assert data.shape[4] == 1440
    assert not np.isnan(data.values).any()
    # Geopotential conversion (height m -> m2 s-2) yields ~5e4 at 500 hPa
    if "z500" in variable:
        z500 = data.sel(variable="z500").values
        assert np.nanmin(z500) > 4.0e4 and np.nanmax(z500) < 6.0e4
