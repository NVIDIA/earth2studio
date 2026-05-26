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
from unittest.mock import AsyncMock, patch

import numpy as np
import pyarrow as pa
import pytest

from earth2studio.data import GOESGLM
from earth2studio.data.goes_glm import _GOESGLMFile

netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 not installed")


# ---------------------------------------------------------------------------
# Helpers to build synthetic GLM L2 LCFA NetCDF files
# ---------------------------------------------------------------------------
def _write_glm_netcdf(
    path: pathlib.Path,
    lats: list[float],
    lons: list[float],
    energies: list[float],
    offsets: list[float],
    epoch: datetime = datetime(2024, 6, 1, 18, 0, 0),
) -> None:
    """Write a minimal GLM L2 LCFA NetCDF that the source can parse."""
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.time_coverage_start = epoch.strftime("%Y-%m-%dT%H:%M:%S.0Z")
        ds.createDimension("number_of_events", len(lats))
        lat_v = ds.createVariable("event_lat", "f4", ("number_of_events",))
        lat_v[:] = np.asarray(lats, dtype=np.float32)
        lon_v = ds.createVariable("event_lon", "f4", ("number_of_events",))
        lon_v[:] = np.asarray(lons, dtype=np.float32)
        en_v = ds.createVariable("event_energy", "f4", ("number_of_events",))
        en_v[:] = np.asarray(energies, dtype=np.float32)
        off_v = ds.createVariable("event_time_offset", "f8", ("number_of_events",))
        off_v[:] = np.asarray(offsets, dtype=np.float64)


# ---------------------------------------------------------------------------
# Mock tests — exercise __call__ end-to-end without network
# ---------------------------------------------------------------------------
def test_goes_glm_call_mock(tmp_path):
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    s3_uri = (
        "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/"
        "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc"
    )

    async def _no_op_fetch(self, uri):  # type: ignore[no-untyped-def]
        return None

    async def _fake_discover(self, time_list):  # type: ignore[no-untyped-def]
        return [_GOESGLMFile(s3_uri=s3_uri, satellite="G16", file_start=epoch)]

    ds = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(5, "m"),
        cache=False,
        verbose=False,
    )
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    _write_glm_netcdf(
        pathlib.Path(ds._cache_path(s3_uri)),
        lats=[35.0, 40.0, 60.0],
        lons=[-120.0, -100.0, 10.0],
        energies=[1.5e-15, 2.5e-15, 3.5e-15],
        offsets=[0.0, 30.0, 60.0],
        epoch=epoch,
    )

    try:
        with (
            patch.object(GOESGLM, "_discover_files", _fake_discover),
            patch.object(GOESGLM, "_fetch_remote_file", _no_op_fetch),
        ):
            df = ds(epoch, ["flashe", "flashc"])

            assert list(df.columns) == ds.SCHEMA.names
            assert len(df) == 6  # 3 events x 2 variables
            assert set(df["variable"].unique()) == {"flashe", "flashc"}
            assert df["lat"].between(-90, 90).all()
            assert df["lon"].between(0, 360).all()
            assert set(df["satellite"].unique()) == {"G16"}
            flashe = df[df["variable"] == "flashe"]
            assert flashe["observation"].max() == pytest.approx(3.5e-15)
            flashc = df[df["variable"] == "flashc"]
            assert (flashc["observation"].astype(float) == 1.0).all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_call_mock_fields_subset(tmp_path):
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    s3_uri = (
        "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/"
        "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc"
    )

    async def _no_op_fetch(self, uri):  # type: ignore[no-untyped-def]
        return None

    async def _fake_discover(self, time_list):  # type: ignore[no-untyped-def]
        return [_GOESGLMFile(s3_uri=s3_uri, satellite="G16", file_start=epoch)]

    ds = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(5, "m"),
        cache=False,
        verbose=False,
    )
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    _write_glm_netcdf(
        pathlib.Path(ds._cache_path(s3_uri)),
        lats=[35.0],
        lons=[-100.0],
        energies=[2.5e-15],
        offsets=[0.0],
        epoch=epoch,
    )
    try:
        subset = ["time", "lat", "lon", "observation", "variable"]
        with (
            patch.object(GOESGLM, "_discover_files", _fake_discover),
            patch.object(GOESGLM, "_fetch_remote_file", _no_op_fetch),
        ):
            df = ds(epoch, ["flashe"], fields=subset)
        assert list(df.columns) == subset
        assert (df["variable"] == "flashe").all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_call_mock_empty():
    ds = GOESGLM(satellite="east", cache=False, verbose=False)

    async def _no_discover(self, time_list):  # type: ignore[no-untyped-def]
        return []

    with patch.object(GOESGLM, "_discover_files", _no_discover):
        df = ds(datetime(2024, 6, 1, 18, 0), ["flashe"])
    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names


def test_goes_glm_call_mock_bbox(tmp_path):
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    s3_uri = (
        "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/"
        "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc"
    )

    async def _no_op_fetch(self, uri):  # type: ignore[no-untyped-def]
        return None

    async def _fake_discover(self, time_list):  # type: ignore[no-untyped-def]
        return [_GOESGLMFile(s3_uri=s3_uri, satellite="G16", file_start=epoch)]

    ds = GOESGLM(
        satellite="east",
        lat_lon_bbox=(24.5, -125.0, 49.5, -66.0),  # CONUS
        time_tolerance=np.timedelta64(5, "m"),
        cache=False,
        verbose=False,
    )
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    _write_glm_netcdf(
        pathlib.Path(ds._cache_path(s3_uri)),
        lats=[35.0, 40.0, 60.0],
        lons=[-120.0, -100.0, 10.0],
        energies=[1.0e-15, 2.0e-15, 3.0e-15],
        offsets=[0.0, 30.0, 60.0],
        epoch=epoch,
    )
    try:
        with (
            patch.object(GOESGLM, "_discover_files", _fake_discover),
            patch.object(GOESGLM, "_fetch_remote_file", _no_op_fetch),
        ):
            df = ds(epoch, ["flashe"])
        assert len(df) == 2  # only the two CONUS events
        assert df["lat"].max() < 50.0
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# Network integration tests (slow, xfail)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_goes_glm_fetch():
    ds = GOESGLM(
        satellite="east",
        lat_lon_bbox=(24.5, -125.0, 49.5, -66.0),
        time_tolerance=np.timedelta64(1, "m"),
        cache=False,
        verbose=False,
    )
    df = ds(datetime(2024, 6, 1, 18, 0), ["flashe", "flashc"])
    assert list(df.columns) == ds.SCHEMA.names
    assert not df.empty
    assert set(df["variable"].unique()).issubset({"flashe", "flashc"})


# ---------------------------------------------------------------------------
# Unit tests — exceptions, resolve_fields, time/satellite/parse helpers
# ---------------------------------------------------------------------------
def test_goes_glm_exceptions():
    with pytest.raises(ValueError):
        GOESGLM(satellite="unknown")
    with pytest.raises(ValueError):
        GOESGLM(lat_lon_bbox=(50.0, -120.0, 40.0, -110.0))  # lat_min >= lat_max

    ds = GOESGLM(satellite="G16", cache=False, verbose=False)
    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 18, 0), ["not_a_var"])
    with pytest.raises(KeyError):
        GOESGLM.resolve_fields(["does_not_exist"])
    with pytest.raises(TypeError):
        GOESGLM.resolve_fields(pa.schema([pa.field("time", pa.string())]))


def test_goes_glm_resolve_fields():
    assert GOESGLM.resolve_fields(None).names == GOESGLM.SCHEMA.names
    assert GOESGLM.resolve_fields("observation").names == ["observation"]
    subset = ["time", "lat", "lon", "observation", "variable"]
    assert GOESGLM.resolve_fields(subset).names == subset


def test_goes_glm_available():
    assert GOESGLM.available(datetime(2024, 6, 1, 18, 0))
    assert GOESGLM.available(np.datetime64("2024-06-01T18:00"))
    assert not GOESGLM.available(datetime(1990, 1, 1))


def test_goes_glm_validate_time():
    GOESGLM._validate_time([datetime(2024, 6, 1, 18, 0)])
    with pytest.raises(ValueError):
        GOESGLM._validate_time([datetime(1990, 1, 1)])


def test_goes_glm_tolerance_conversion():
    ds = GOESGLM(time_tolerance=np.timedelta64(2, "m"), cache=False, verbose=False)
    assert ds._tolerance_lower == timedelta(minutes=-2)
    assert ds._tolerance_upper == timedelta(minutes=2)
    asym = GOESGLM(
        time_tolerance=(np.timedelta64(-30, "s"), np.timedelta64(90, "s")),
        cache=False,
        verbose=False,
    )
    assert asym._tolerance_lower == timedelta(seconds=-30)
    assert asym._tolerance_upper == timedelta(seconds=90)


def test_goes_glm_satellite_routing():
    ds_e = GOESGLM(satellite="east", cache=False, verbose=False)
    assert ds_e._satellite_for_time(datetime(2024, 1, 1)) == "G16"
    assert ds_e._satellite_for_time(datetime(2025, 6, 1)) == "G19"

    ds_w = GOESGLM(satellite="west", cache=False, verbose=False)
    assert ds_w._satellite_for_time(datetime(2020, 6, 1)) == "G17"
    assert ds_w._satellite_for_time(datetime(2024, 6, 1)) == "G18"

    ds_pin = GOESGLM(satellite="G16", cache=False, verbose=False)
    assert ds_pin._satellite_for_time(datetime(2026, 1, 1)) == "G16"


def test_goes_glm_lat_lon_bbox_accepts_360_convention():
    # The GLM source filters in [-180, 180); a bbox passed in [0, 360]
    # should be auto-normalised under the hood.
    ds = GOESGLM(lat_lon_bbox=(24.5, 235.0, 49.5, 294.0), cache=False, verbose=False)
    lat_min, lon_min, lat_max, lon_max = ds._lat_lon_bbox  # type: ignore[misc]
    assert lat_min == 24.5 and lat_max == 49.5
    assert lon_min == pytest.approx(-125.0)
    assert lon_max == pytest.approx(-66.0)


def test_goes_glm_parse_file(tmp_path):
    f = tmp_path / "events.nc"
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    _write_glm_netcdf(
        f,
        lats=[35.0, 40.0, 60.0],
        lons=[-120.0, -100.0, 10.0],
        energies=[1e-15, 2e-15, 3e-15],
        offsets=[0.0, 5.123, 19.5],
        epoch=epoch,
    )
    df = GOESGLM._parse_glm_file(str(f), lat_lon_bbox=None)
    assert df is not None and len(df) == 3
    # CONUS bbox keeps 2 of 3 events
    df_conus = GOESGLM._parse_glm_file(str(f), lat_lon_bbox=(24.5, -125.0, 49.5, -66.0))
    assert df_conus is not None and len(df_conus) == 2
    # All-out-of-box returns None
    assert (
        GOESGLM._parse_glm_file(str(f), lat_lon_bbox=(-10.0, -10.0, -5.0, -5.0)) is None
    )


# ---------------------------------------------------------------------------
# Internal-plumbing tests (mocked S3 filesystem)
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


def test_goes_glm_discover_files():
    ds = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(1, "m"),
        cache=False,
        verbose=False,
    )

    bucket_prefix = "noaa-goes16/GLM-L2-LCFA/2024/153/18/"
    keys = [
        bucket_prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc",
        bucket_prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800200_e20241531800400_c20241531800420.nc",
        bucket_prefix
        + "OR_GLM-L2-LCFA_G16_s20241531805000_e20241531805200_c20241531805220.nc",
        bucket_prefix + "junk.txt",
        bucket_prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc",
    ]
    fake_fs = AsyncMock()
    fake_fs._ls = AsyncMock(return_value=keys)
    ds.fs = fake_fs

    files = _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)]))
    assert len(files) == 2
    assert {f.satellite for f in files} == {"G16"}

    # Missing prefix → empty result, no exception.
    fake_fs._ls = AsyncMock(side_effect=FileNotFoundError())
    assert _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)])) == []


def test_goes_glm_fetch_remote_file(tmp_path):
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    fake_fs = AsyncMock()
    fake_fs._cat_file = AsyncMock(return_value=b"fake-netcdf-bytes")
    ds.fs = fake_fs
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)

    try:
        uri = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/file.nc"
        _run(ds._fetch_remote_file(uri))
        assert pathlib.Path(ds._cache_path(uri)).read_bytes() == b"fake-netcdf-bytes"
        fake_fs._cat_file.assert_awaited_once_with(uri)

        # Second call is a no-op (cache hit).
        fake_fs._cat_file.reset_mock()
        _run(ds._fetch_remote_file(uri))
        fake_fs._cat_file.assert_not_called()

        # Missing file in S3 is swallowed (warn-only).
        missing = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/missing.nc"
        fake_fs._cat_file = AsyncMock(side_effect=FileNotFoundError())
        _run(ds._fetch_remote_file(missing))
        assert not pathlib.Path(ds._cache_path(missing)).exists()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)

    # Requires an initialised filesystem.
    ds2 = GOESGLM(satellite="east", cache=False, verbose=False)
    assert ds2.fs is None
    with pytest.raises(ValueError):
        _run(ds2._fetch_remote_file("s3://noaa-goes16/anything.nc"))
