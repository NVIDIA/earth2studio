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
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import GOESGLM
from earth2studio.data.goes_glm import (
    _GOESGLMFile,
    _parse_filename_start,
    _parse_glm_file,
)

netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 not installed")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
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
        off_v.units = f"seconds since {epoch.strftime('%Y-%m-%d %H:%M:%S')}"
        off_v[:] = np.asarray(offsets, dtype=np.float64)


# ----------------------------------------------------------------------
# Pure unit tests
# ----------------------------------------------------------------------
def test_filename_start_parse():
    key = (
        "noaa-goes16/GLM-L2-LCFA/2024/153/18/"
        "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc"
    )
    assert _parse_filename_start(key) == datetime(2024, 6, 1, 18, 0, 0)
    assert _parse_filename_start("nope.nc") is None
    assert _parse_filename_start("OR_GLM-L2-LCFA_G16_sbad__e__c.nc") is None


def test_parse_glm_file(tmp_path):
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
    # Without bbox: 3 events
    df = _parse_glm_file(str(f), bbox=None)
    assert df is not None
    assert len(df) == 3
    # Sub-second precision preserved
    assert df["time"].iloc[1] == pd.Timestamp(epoch + timedelta(seconds=5.123))
    # With CONUS bbox: only 2 of 3 events
    df_conus = _parse_glm_file(str(f), bbox=(24.5, 49.5, -125.0, -66.0))
    assert df_conus is not None
    assert len(df_conus) == 2
    # All-out-of-box returns None
    df_empty = _parse_glm_file(str(f), bbox=(-10.0, -5.0, -10.0, -5.0))
    assert df_empty is None


def test_satellite_routing():
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    # G16 era
    assert ds._satellite_for_time(datetime(2024, 1, 1)) == "G16"
    # G19 era
    assert ds._satellite_for_time(datetime(2025, 6, 1)) == "G19"

    ds_w = GOESGLM(satellite="west", cache=False, verbose=False)
    assert ds_w._satellite_for_time(datetime(2020, 6, 1)) == "G17"
    assert ds_w._satellite_for_time(datetime(2024, 6, 1)) == "G18"

    ds_pin = GOESGLM(satellite="G16", cache=False, verbose=False)
    assert ds_pin._satellite_for_time(datetime(2024, 1, 1)) == "G16"
    # Pinning bypasses slot history (caller's responsibility)
    assert ds_pin._satellite_for_time(datetime(2026, 1, 1)) == "G16"


def test_constructor_validation():
    with pytest.raises(ValueError):
        GOESGLM(satellite="unknown")
    with pytest.raises(ValueError):
        GOESGLM(bbox=(50.0, 40.0, -120.0, -110.0))  # lat_min >= lat_max


def test_tolerance_normalisation():
    ds = GOESGLM(time_tolerance=np.timedelta64(2, "m"), cache=False, verbose=False)
    assert ds._tolerance_lower == timedelta(minutes=-2)
    assert ds._tolerance_upper == timedelta(minutes=2)
    ds_asym = GOESGLM(
        time_tolerance=(np.timedelta64(-30, "s"), np.timedelta64(90, "s")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(seconds=-30)
    assert ds_asym._tolerance_upper == timedelta(seconds=90)


def test_validate_time_and_available():
    GOESGLM._validate_time([datetime(2024, 6, 1, 18, 0)])
    with pytest.raises(ValueError):
        GOESGLM._validate_time([datetime(1990, 1, 1)])

    assert GOESGLM.available(datetime(2024, 6, 1, 18, 0))
    assert not GOESGLM.available(datetime(1990, 1, 1))
    assert GOESGLM.available(np.datetime64("2024-06-01T18:00"))


def test_resolve_fields():
    full = GOESGLM.resolve_fields(None)
    assert full.names == GOESGLM.SCHEMA.names

    subset = GOESGLM.resolve_fields(["time", "lat", "lon", "observation", "variable"])
    assert subset.names == ["time", "lat", "lon", "observation", "variable"]

    one = GOESGLM.resolve_fields("observation")
    assert one.names == ["observation"]

    with pytest.raises(KeyError):
        GOESGLM.resolve_fields(["does_not_exist"])

    bad_schema = pa.schema([pa.field("does_not_exist", pa.float32())])
    with pytest.raises(KeyError):
        GOESGLM.resolve_fields(bad_schema)

    wrong_type = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        GOESGLM.resolve_fields(wrong_type)


# ----------------------------------------------------------------------
# Mock end-to-end test (no network)
# ----------------------------------------------------------------------
def test_goes_glm_mock_call(tmp_path):
    """Exercise the full __call__ → fetch → compile path with mocked I/O.

    A synthetic NetCDF is dropped into the deterministic cache path so the
    real ``_parse_glm_file`` runs. ``_discover_files`` and
    ``_fetch_remote_file`` are stubbed to avoid any S3 traffic.
    """
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    s3_uri = (
        "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/"
        "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc"
    )

    async def _no_op_fetch(self, uri):  # type: ignore[no-untyped-def]
        # File already placed in cache by the test; nothing to do.
        return None

    async def _fake_discover(self, time_list):  # type: ignore[no-untyped-def]
        return [
            _GOESGLMFile(
                s3_uri=s3_uri,
                satellite="G16",
                file_start=epoch,
            )
        ]

    ds = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(5, "m"),
        cache=False,
        verbose=False,
    )
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    # Pre-populate the cache with the synthetic NetCDF at the expected path.
    local_path = ds._cache_path(s3_uri)
    _write_glm_netcdf(
        pathlib.Path(local_path),
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

            # Default field set is the full schema.
            assert list(df.columns) == ds.SCHEMA.names
            # 3 events × 2 requested variables = 6 rows.
            assert len(df) == 6
            assert set(df["variable"]) == {"flashe", "flashc"}
            # Longitude normalised to [0, 360)
            assert (df["lon"] >= 0).all()
            assert (df["lon"] < 360).all()
            # CONUS lon -120 → 240, -100 → 260, 10 → 10
            assert np.any(np.isclose(df["lon"].values, 240.0))
            assert np.any(np.isclose(df["lon"].values, 260.0))
            # flashc is constantly 1.0
            flashc_rows = df[df["variable"] == "flashc"]
            assert (flashc_rows["observation"].astype(float) == 1.0).all()
            # flashe carries energy values
            flashe_rows = df[df["variable"] == "flashe"]
            assert flashe_rows["observation"].max() == pytest.approx(3.5e-15)
            # Satellite column is the routed platform
            assert set(df["satellite"]) == {"G16"}

            # Subset fields path
            subset = ds(
                epoch,
                ["flashe"],
                fields=["time", "lat", "lon", "observation", "variable"],
            )
            assert list(subset.columns) == [
                "time",
                "lat",
                "lon",
                "observation",
                "variable",
            ]
            assert (subset["variable"] == "flashe").all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_bbox_filter_call(tmp_path):
    """Mock-mode call with a CONUS bbox should drop out-of-region events."""
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
        bbox=(24.5, 49.5, -125.0, -66.0),
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

    with (
        patch.object(GOESGLM, "_discover_files", _fake_discover),
        patch.object(GOESGLM, "_fetch_remote_file", _no_op_fetch),
    ):
        df = ds(epoch, ["flashe"])

    # Only the two CONUS events survive bbox filter; one variable requested → 2 rows
    assert len(df) == 2
    # The 60N / 10E event is excluded
    assert df["lat"].max() < 50.0


def test_goes_glm_invalid_variable():
    # Variable validation happens before any S3 access, so no patching needed.
    ds = GOESGLM(satellite="G16", cache=False, verbose=False)
    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 18, 0), ["not_a_var"])


# ----------------------------------------------------------------------
# Tests for the internal S3 plumbing (mocked filesystem)
# ----------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


def test_discover_files_filters_window_and_dedups():
    ds = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(1, "m"),
        cache=False,
        verbose=False,
    )

    # Simulate s3fs._ls for one hour prefix returning a mix of valid GLM
    # keys (one inside the window, one outside, one duplicate, plus a
    # non-.nc entry that should be ignored).
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
        + "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc",  # dup
    ]

    fake_fs = AsyncMock()
    fake_fs._ls = AsyncMock(return_value=keys)
    ds.fs = fake_fs

    files = _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)]))
    # 2 keys fall inside [-1m, +1m]; the 18:05 file does not; junk.txt is
    # ignored; duplicate collapsed.
    assert len(files) == 2
    assert all(f.satellite == "G16" for f in files)
    assert all(f.s3_uri.startswith("s3://noaa-goes16/") for f in files)


def test_discover_files_missing_prefix_returns_empty():
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    fake_fs = AsyncMock()
    fake_fs._ls = AsyncMock(side_effect=FileNotFoundError("missing"))
    ds.fs = fake_fs
    files = _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)]))
    assert files == []


def test_fetch_remote_file_writes_cache_and_skips_existing(tmp_path):
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    fake_fs = AsyncMock()
    fake_fs._cat_file = AsyncMock(return_value=b"fake-netcdf-bytes")
    ds.fs = fake_fs

    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    uri = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/file.nc"
    _run(ds._fetch_remote_file(uri))
    cached = pathlib.Path(ds._cache_path(uri))
    assert cached.read_bytes() == b"fake-netcdf-bytes"
    fake_fs._cat_file.assert_awaited_once_with(uri)

    # Second call should not re-fetch
    fake_fs._cat_file.reset_mock()
    _run(ds._fetch_remote_file(uri))
    fake_fs._cat_file.assert_not_called()

    # Missing file in S3 is swallowed (warn-only)
    missing_uri = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/missing.nc"
    fake_fs._cat_file = AsyncMock(side_effect=FileNotFoundError())
    _run(ds._fetch_remote_file(missing_uri))
    assert not pathlib.Path(ds._cache_path(missing_uri)).exists()
    shutil.rmtree(ds.cache, ignore_errors=True)


def test_fetch_remote_file_requires_filesystem():
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    assert ds.fs is None
    with pytest.raises(ValueError):
        _run(ds._fetch_remote_file("s3://noaa-goes16/anything.nc"))


def test_empty_dataframe_path():
    """With no discovered files, the source returns an empty DF with the
    requested fields."""
    ds = GOESGLM(satellite="east", cache=False, verbose=False)

    async def _no_discover(self, time_list):  # type: ignore[no-untyped-def]
        return []

    with patch.object(GOESGLM, "_discover_files", _no_discover):
        df = ds(
            datetime(2024, 6, 1, 18, 0),
            ["flashe"],
            fields=["time", "lat", "lon", "observation", "variable"],
        )
    assert list(df.columns) == ["time", "lat", "lon", "observation", "variable"]
    assert df.empty
    # Empty result still carries schema-typed columns, not all-object.
    assert df["time"].dtype.kind == "M"
    assert df["lat"].dtype == np.float32
    assert df["lon"].dtype == np.float32
    assert df["observation"].dtype == np.float32
    assert df["variable"].dtype == object


def test_resolve_event_times_fallback(tmp_path):
    """When event_time_offset has no ``units`` attribute, the parser
    falls back to ``time_coverage_start``."""
    f = tmp_path / "events_no_units.nc"
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    import netCDF4 as nc

    with nc.Dataset(f, "w", format="NETCDF4") as ds:
        ds.time_coverage_start = epoch.strftime("%Y-%m-%dT%H:%M:%S.0Z")
        ds.createDimension("number_of_events", 1)
        for name in ("event_lat", "event_lon", "event_energy"):
            v = ds.createVariable(name, "f4", ("number_of_events",))
            v[:] = np.array([10.0], dtype=np.float32)
        # event_time_offset WITHOUT units → triggers fallback path
        off = ds.createVariable("event_time_offset", "f8", ("number_of_events",))
        off[:] = np.array([7.5], dtype=np.float64)

    df = _parse_glm_file(str(f), bbox=None)
    assert df is not None
    assert df["time"].iloc[0] == pd.Timestamp(epoch + timedelta(seconds=7.5))


# ----------------------------------------------------------------------
# Real-S3 smoke test (network)
# ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_goes_glm_real_fetch():
    ds = GOESGLM(
        satellite="east",
        bbox=(24.5, 49.5, -125.0, -66.0),
        time_tolerance=np.timedelta64(1, "m"),
        cache=False,
        verbose=False,
    )
    df = ds(datetime(2024, 6, 1, 18, 0), ["flashe", "flashc"])
    assert list(df.columns) == ds.SCHEMA.names
    assert not df.empty
    assert set(df["variable"]).issubset({"flashe", "flashc"})
