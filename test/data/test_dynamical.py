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

from earth2studio.data import (
    DynamicalAIFS,
    DynamicalAIFS_ENS,
    DynamicalAIFS_FX,
    DynamicalAIFSENS_FX,
    DynamicalGEFS,
    DynamicalGEFS_FX,
    DynamicalGFS,
    DynamicalGFS_FX,
    DynamicalHRRR,
    DynamicalHRRR_FX,
    DynamicalICON_EU_FX,
    DynamicalIFS_ENS,
    DynamicalIFS_ENS_FX,
    DynamicalMRMS,
)
from earth2studio.data.dynamical import _DynamicalBase

# Un-normalized synthetic grid: ascending latitude and -180..180 longitude,
# mirroring how dynamical.org serves coordinates.
_LAT = np.linspace(-90.0, 90.0, 7)
_LON = np.linspace(-180.0, 135.0, 8)
_Y = np.array([2.0, 1.0, 0.0])
_X = np.array([-1.0, 0.0, 1.0, 2.0])
_LAT_2D = np.array(
    [[50.0, 50.1, 50.2, 50.3], [49.0, 49.1, 49.2, 49.3], [48.0, 48.1, 48.2, 48.3]],
    dtype=np.float32,
)
_LON_2D = np.array(
    [
        [-105.0, -104.9, -104.8, -104.7],
        [-105.1, -105.0, -104.9, -104.8],
        [-105.2, -105.1, -105.0, -104.9],
    ],
    dtype=np.float32,
)

# Variable -> (unit, raw value) used to build the synthetic store and to verify
# STAC-unit-driven conversion to the Earth2Studio convention.
_VARS = {
    "temperature_2m": ("degree_Celsius", 20.0),
    "wind_u_10m": ("m s-1", 3.0),
    "precipitation_surface": ("kg m-2 s-1", 0.001),
    "precipitation_rate_surface": ("kg m-2 s-1", 0.002),
    "total_cloud_cover_atmosphere": ("percent", 50.0),
    "geopotential_height_500hpa": ("m", 5500.0),
    "temperature_500hpa": ("degree_Celsius", -10.0),
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


def _projected_analysis_dims(times: np.ndarray) -> dict:
    return {
        "time": {
            "type": "temporal",
            "extent": [str(times[0]) + "Z", None],
        },
        "y": {"type": "spatial"},
        "x": {"type": "spatial"},
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
    monkeypatch.setattr(
        klass,
        "_open_icechunk",
        lambda self, href, region=None, virtual_containers=None: dataset,
    )


@pytest.mark.timeout(30)
def test_dynamical_call_mock(monkeypatch):
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, DynamicalGFS, "noaa-gfs-analysis", dims, ds_obj)

    source = DynamicalGFS()
    variables = ["t2m", "u10m", "tpf", "tcc", "z500", "t500"]
    data = source(times, variables)

    assert data.shape == (2, len(variables), len(_LAT), len(_LON))
    assert list(data.coords["variable"].values) == variables
    # Grid normalized: latitude descending, longitude in [0, 360) ascending
    assert data.coords["lat"].values[0] == pytest.approx(90.0)
    assert data.coords["lat"].values[-1] == pytest.approx(-90.0)
    assert (data.coords["lon"].values >= 0).all()
    assert (np.diff(data.coords["lon"].values) > 0).all()
    # STAC-unit-driven conversions
    np.testing.assert_allclose(data.sel(variable="t2m").values, 20.0 + 273.15)
    np.testing.assert_allclose(data.sel(variable="u10m").values, 3.0)
    np.testing.assert_allclose(data.sel(variable="tpf").values, 0.001)
    np.testing.assert_allclose(data.sel(variable="tcc").values, 0.5)
    np.testing.assert_allclose(data.sel(variable="z500").values, 5500.0 * 9.80665)
    np.testing.assert_allclose(data.sel(variable="t500").values, -10.0 + 273.15)


@pytest.mark.timeout(30)
def test_dynamical_native_passthrough(monkeypatch):
    times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, DynamicalGFS, "noaa-gfs-analysis", dims, ds_obj)

    source = DynamicalGFS()
    # Native dynamical.org variable name not in the lexicon
    data = source(times, ["wind_u_10m"])
    np.testing.assert_allclose(data.values, 3.0)


@pytest.mark.timeout(30)
def test_dynamical_available_variables(monkeypatch):
    times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    ).drop_vars("precipitation_surface")
    _patch(monkeypatch, DynamicalGFS, "noaa-gfs-analysis", dims, ds_obj)

    source = DynamicalGFS()

    assert source.available_variables() == ["u10m", "t2m", "tcc", "tpf", "z500", "t500"]
    assert source.available_variables(native=True) == sorted(ds_obj.data_vars)


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
    _patch(monkeypatch, DynamicalGFS_FX, "noaa-gfs-forecast", dims, ds_obj)

    source = DynamicalGFS_FX()
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
    _patch(monkeypatch, DynamicalGFS, "noaa-gfs-analysis", dims, ds_obj)

    # Unknown variable (not in lexicon, not native to collection)
    source = DynamicalGFS()
    with pytest.raises(KeyError):
        source(times, ["definitely_not_a_variable"])

    # Variable in lexicon but not served by this collection
    source = DynamicalGFS()
    with pytest.raises(KeyError):
        source(times, ["t850"])

    # Time before the collection's temporal extent
    source = DynamicalGFS()
    with pytest.raises(ValueError):
        source(np.array(["1900-01-01T00:00"], dtype="datetime64[ns]"), ["t2m"])

    # Time after the last timestamp in the store (STAC extent is open-ended, so
    # the upper bound falls back to the store's actual last coordinate)
    source = DynamicalGFS()
    with pytest.raises(ValueError):
        source(np.array(["2030-01-01T00:00"], dtype="datetime64[ns]"), ["t2m"])


@pytest.mark.timeout(30)
def test_dynamical_available(monkeypatch):
    # Store spans 2024-01-01 .. 2024-12-01; STAC extent end is open (None)
    times = np.array(
        ["2024-01-01T00:00", "2024-06-01T00:00", "2024-12-01T00:00"],
        dtype="datetime64[ns]",
    )
    dims = _analysis_dims(times)
    ds_obj = _make_dataset(
        {"time": times, "latitude": _LAT, "longitude": _LON},
        ("time", "latitude", "longitude"),
    )
    _patch(monkeypatch, DynamicalGFS, "noaa-gfs-analysis", dims, ds_obj)

    source = DynamicalGFS()
    # Within the store's actual time span
    assert source.available(np.datetime64("2024-06-01T00:00")) is True
    assert source.available(datetime.datetime(2024, 1, 1)) is True
    assert source.available(np.datetime64("2024-12-01T00:00")) is True
    # Before the store's start
    assert source.available(np.datetime64("1900-01-01T00:00")) is False
    # After the store's last coordinate (open-ended STAC extent must not admit it)
    assert source.available(np.datetime64("2030-01-01T00:00")) is False


@pytest.mark.timeout(30)
def test_dynamical_forecast_available(monkeypatch):
    init_times = np.array(
        ["2024-01-01T00:00", "2024-06-01T00:00"], dtype="datetime64[ns]"
    )
    leads = np.array([0, 6 * 3600], dtype="timedelta64[s]").astype("timedelta64[ns]")
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
    _patch(monkeypatch, DynamicalGFS_FX, "noaa-gfs-forecast", dims, ds_obj)

    source = DynamicalGFS_FX()
    # Availability is checked against the init_time coordinate span
    assert source.available(np.datetime64("2024-03-01T00:00")) is True
    assert source.available(np.datetime64("2024-06-01T00:00")) is True
    # After the last init_time in the store
    assert source.available(np.datetime64("2024-12-01T00:00")) is False
    assert source.available(np.datetime64("1900-01-01T00:00")) is False


@pytest.mark.timeout(30)
def test_dynamical_unknown_collection(monkeypatch):
    monkeypatch.setattr(
        "earth2studio.data.dynamical._fetch_json",
        lambda url: _fake_catalog("noaa-gfs-analysis"),
    )
    source = _DynamicalBase("does-not-exist")
    with pytest.raises(ValueError):
        source(
            np.array(["2024-01-01T00:00"], dtype="datetime64[ns]"),
            [datetime.timedelta(0)],
            ["t2m"],
        )


@pytest.mark.timeout(30)
def test_dynamical_projected_grid_call_mock(monkeypatch):
    times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    dims = _projected_analysis_dims(times)
    ds_obj = _make_dataset(
        {
            "time": times,
            "y": _Y,
            "x": _X,
            "latitude": (("y", "x"), _LAT_2D),
            "longitude": (("y", "x"), _LON_2D),
        },
        ("time", "y", "x"),
    )
    _patch(monkeypatch, DynamicalHRRR, "noaa-hrrr-analysis", dims, ds_obj)

    source = DynamicalHRRR()
    data = source(times, ["t2m", "u10m"])

    assert data.shape == (1, 2, len(_Y), len(_X))
    assert list(data.dims) == ["time", "variable", "y", "x"]
    assert data.coords["lat"].shape == (len(_Y), len(_X))
    assert data.coords["lon"].shape == (len(_Y), len(_X))
    assert data.coords["_lat"].shape == (len(_Y), len(_X))
    assert data.coords["_lon"].shape == (len(_Y), len(_X))
    assert (data.coords["lon"].values >= 0).all()
    np.testing.assert_allclose(data.sel(variable="t2m").values, 20.0 + 273.15)


@pytest.mark.timeout(30)
def test_dynamical_regional_greenwich_longitude_order(monkeypatch):
    init_times = np.array(["2026-02-10T00:00"], dtype="datetime64[ns]")
    leads = np.array([0, 3600], dtype="timedelta64[s]").astype("timedelta64[ns]")
    lon = np.array([-1.0, 0.0, 1.0])
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
            "longitude": lon,
        },
        ("init_time", "lead_time", "latitude", "longitude"),
    )
    _patch(monkeypatch, DynamicalICON_EU_FX, "dwd-icon-eu-forecast-5-day", dims, ds_obj)

    source = DynamicalICON_EU_FX()
    data = source(init_times, [datetime.timedelta(0)], ["t2m"])

    np.testing.assert_allclose(data.coords["lon"].values, lon)


@pytest.mark.timeout(30)
def test_dynamical_projected_forecast_call_mock(monkeypatch):
    init_times = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    leads = np.array([0, 3600], dtype="timedelta64[s]").astype("timedelta64[ns]")
    dims = {
        "init_time": {"type": "temporal", "extent": [str(init_times[0]) + "Z", None]},
        "lead_time": {"type": "other"},
        "y": {"type": "spatial"},
        "x": {"type": "spatial"},
    }
    ds_obj = _make_dataset(
        {
            "init_time": init_times,
            "lead_time": leads,
            "y": _Y,
            "x": _X,
            "latitude": (("y", "x"), _LAT_2D),
            "longitude": (("y", "x"), _LON_2D),
        },
        ("init_time", "lead_time", "y", "x"),
    )
    ds_obj = ds_obj.drop_vars("precipitation_surface")
    _patch(
        monkeypatch,
        DynamicalHRRR_FX,
        "noaa-hrrr-forecast-48-hour-virtual",
        dims,
        ds_obj,
    )

    source = DynamicalHRRR_FX()
    data = source(
        init_times, [datetime.timedelta(0), datetime.timedelta(hours=1)], ["t2m", "tpf"]
    )

    assert data.shape == (1, 2, 2, len(_Y), len(_X))
    assert list(data.dims) == ["time", "lead_time", "variable", "y", "x"]
    assert data.coords["lat"].shape == (len(_Y), len(_X))
    assert (data.coords["lon"].values >= 0).all()
    np.testing.assert_allclose(data.sel(variable="tpf").values, 0.002)


# Concrete named data source -> the STAC collection id it must resolve to.
_CONCRETE_COLLECTIONS = [
    (DynamicalGFS, "noaa-gfs-analysis"),
    (DynamicalGEFS, "noaa-gefs-analysis"),
    (DynamicalHRRR, "noaa-hrrr-analysis"),
    (DynamicalMRMS, "noaa-mrms-conus-analysis-hourly"),
    (DynamicalAIFS, "ecmwf-aifs-single-forecast"),
    (DynamicalAIFS_ENS, "ecmwf-aifs-ens-forecast"),
    (DynamicalIFS_ENS, "ecmwf-ifs-ens-forecast-15-day-0-25-degree"),
    (DynamicalGFS_FX, "noaa-gfs-forecast"),
    (DynamicalGEFS_FX, "noaa-gefs-forecast-35-day"),
    (DynamicalHRRR_FX, "noaa-hrrr-forecast-48-hour-virtual"),
    (DynamicalICON_EU_FX, "dwd-icon-eu-forecast-5-day"),
    (DynamicalIFS_ENS_FX, "ecmwf-ifs-ens-forecast-15-day-0-25-degree"),
    (DynamicalAIFS_FX, "ecmwf-aifs-single-forecast"),
    (DynamicalAIFSENS_FX, "ecmwf-aifs-ens-forecast"),
]

_CONCRETE_SINGLE_FETCH_CASES = [
    (DynamicalGEFS, "noaa-gefs-analysis", "analysis", "t2m", False),
    (DynamicalHRRR, "noaa-hrrr-analysis", "analysis_projected", "t2m", False),
    (DynamicalMRMS, "noaa-mrms-conus-analysis-hourly", "analysis", "tpf", False),
    (DynamicalAIFS, "ecmwf-aifs-single-forecast", "analysis_forecast", "t2m", False),
    (
        DynamicalAIFS_ENS,
        "ecmwf-aifs-ens-forecast",
        "analysis_forecast",
        "u10m",
        True,
    ),
    (
        DynamicalIFS_ENS,
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "analysis_forecast",
        "u10m",
        True,
    ),
    (DynamicalGEFS_FX, "noaa-gefs-forecast-35-day", "forecast", "u10m", True),
    (
        DynamicalHRRR_FX,
        "noaa-hrrr-forecast-48-hour-virtual",
        "forecast_projected",
        "t2m",
        False,
    ),
    (DynamicalICON_EU_FX, "dwd-icon-eu-forecast-5-day", "forecast", "t2m", False),
    (
        DynamicalIFS_ENS_FX,
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "forecast",
        "u10m",
        True,
    ),
    (DynamicalAIFS_FX, "ecmwf-aifs-single-forecast", "forecast", "t2m", False),
    (DynamicalAIFSENS_FX, "ecmwf-aifs-ens-forecast", "forecast", "u10m", True),
]


def _forecast_dims(init_times: np.ndarray, projected: bool, member: bool) -> dict:
    dims = {
        "init_time": {"type": "temporal", "extent": [str(init_times[0]) + "Z", None]},
        "lead_time": {"type": "other"},
    }
    if member:
        dims["ensemble_member"] = {"type": "other"}
    if projected:
        dims.update({"y": {"type": "spatial"}, "x": {"type": "spatial"}})
    else:
        dims.update({"latitude": {"type": "spatial"}, "longitude": {"type": "spatial"}})
    return dims


def _forecast_dataset(init_times: np.ndarray, projected: bool, member: bool):
    leads = np.array([0], dtype="timedelta64[s]").astype("timedelta64[ns]")
    coords = {"init_time": init_times, "lead_time": leads}
    dims = ["init_time", "lead_time"]
    if member:
        coords["ensemble_member"] = np.array([0, 1, 2])
        dims.append("ensemble_member")
    if projected:
        coords.update(
            {
                "y": _Y,
                "x": _X,
                "latitude": (("y", "x"), _LAT_2D),
                "longitude": (("y", "x"), _LON_2D),
            }
        )
        dims.extend(["y", "x"])
    else:
        coords.update({"latitude": _LAT, "longitude": _LON})
        dims.extend(["latitude", "longitude"])

    if not member:
        return _make_dataset(coords, tuple(dims))

    members = coords["ensemble_member"]
    shape = tuple(len(coords[d]) for d in dims)
    member_axis = dims.index("ensemble_member")
    member_field = members.reshape(
        [-1 if i == member_axis else 1 for i in range(len(dims))]
    )
    data_vars = {
        name: (tuple(dims), np.broadcast_to(member_field, shape).astype(np.float32))
        for name in _VARS
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "klass,collection",
    _CONCRETE_COLLECTIONS,
)
def test_dynamical_concrete_collection_ids(klass, collection):
    # Concrete sources bake in their STAC collection id and take no collection arg
    source = klass()
    assert source.collection == collection
    assert isinstance(source, _DynamicalBase)


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "klass,collection,kind,variable,member",
    _CONCRETE_SINGLE_FETCH_CASES,
)
def test_dynamical_concrete_single_fetch_mock(
    monkeypatch, klass, collection, kind, variable, member
):
    time = np.array(["2025-08-01T00:00"], dtype="datetime64[ns]")
    projected = kind.endswith("projected")
    if kind == "analysis":
        dims = _analysis_dims(time)
        ds_obj = _make_dataset(
            {"time": time, "latitude": _LAT, "longitude": _LON},
            ("time", "latitude", "longitude"),
        )
    elif kind == "analysis_projected":
        dims = _projected_analysis_dims(time)
        ds_obj = _make_dataset(
            {
                "time": time,
                "y": _Y,
                "x": _X,
                "latitude": (("y", "x"), _LAT_2D),
                "longitude": (("y", "x"), _LON_2D),
            },
            ("time", "y", "x"),
        )
    else:
        dims = _forecast_dims(time, projected, member)
        ds_obj = _forecast_dataset(time, projected, member)
    _patch(monkeypatch, klass, collection, dims, ds_obj)

    source = klass(member=2) if member else klass()
    if kind.startswith("analysis"):
        data = source(time, [variable])
    else:
        data = source(time, [datetime.timedelta(0)], [variable])

    expected_shape = (len(_Y), len(_X)) if projected else (len(_LAT), len(_LON))
    expected = 2.0 if member else (20.0 + 273.15 if variable == "t2m" else 0.001)
    assert data.shape[-len(expected_shape) :] == expected_shape
    assert list(data.coords["variable"].values) == [variable]
    np.testing.assert_allclose(data.values, expected)


@pytest.mark.timeout(30)
def test_dynamical_cache():
    assert DynamicalGFS(cache=True).cache == DynamicalGFS(cache=False).cache


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_dynamical_fetch():
    source = DynamicalGFS()
    time = [datetime.datetime(year=2024, month=6, day=1)]
    variable = ["t2m", "u10m"]
    data = source(time, variable)

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
def test_dynamical_forecast_fetch():
    source = DynamicalGFS_FX()
    variable = ["t2m"]
    time = [datetime.datetime(year=2024, month=6, day=1)]
    lead_time = [datetime.timedelta(hours=0)]
    data = source(time, lead_time, variable)

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(lead_time)
    assert data.shape[2] == len(variable)
    assert data.shape[3] == 721
    assert data.shape[4] == 1440
    assert not np.isnan(data.values).any()
