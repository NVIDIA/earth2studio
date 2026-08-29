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
    groups: tuple[list[float], list[float], list[float], list[float]] | None = None,
    flashes: tuple[list[float], list[float], list[float], list[float]] | None = None,
) -> None:
    """Write a minimal GLM L2 LCFA NetCDF that the source can parse.

    ``groups`` / ``flashes``, when given, are
    ``(lats, lons, energies, offsets)`` tuples written alongside the
    events so the group and flash tiers of the hierarchy can be
    exercised too.
    """
    tiers = [("event", (lats, lons, energies, offsets))]
    if groups is not None:
        tiers.append(("group", groups))
    if flashes is not None:
        tiers.append(("flash", flashes))

    # Flashes span many frames, so their time variable is named for the
    # first constituent event rather than the flash itself.
    time_names = {
        "event": "event_time_offset",
        "group": "group_time_offset",
        "flash": "flash_time_offset_of_first_event",
    }

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.time_coverage_start = epoch.strftime("%Y-%m-%dT%H:%M:%S.0Z")
        for level, (t_lats, t_lons, t_energies, t_offsets) in tiers:
            dim = f"number_of_{level}s" if level != "flash" else "number_of_flashes"
            ds.createDimension(dim, len(t_lats))
            lat_v = ds.createVariable(f"{level}_lat", "f4", (dim,))
            lat_v[:] = np.asarray(t_lats, dtype=np.float32)
            lon_v = ds.createVariable(f"{level}_lon", "f4", (dim,))
            lon_v[:] = np.asarray(t_lons, dtype=np.float32)
            en_v = ds.createVariable(f"{level}_energy", "f4", (dim,))
            en_v[:] = np.asarray(t_energies, dtype=np.float32)
            off_v = ds.createVariable(time_names[level], "f8", (dim,))
            off_v[:] = np.asarray(t_offsets, dtype=np.float64)


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
            df = ds(epoch, ["lightning_event_energy", "lightning_event_count"])

            assert list(df.columns) == ds.SCHEMA.names
            assert len(df) == 6  # 3 events x 2 variables
            assert set(df["variable"].unique()) == {
                "lightning_event_energy",
                "lightning_event_count",
            }
            assert df["lat"].between(-90, 90).all()
            assert df["lon"].between(0, 360).all()
            assert set(df["satellite"].unique()) == {"G16"}
            lightning_event_energy = df[df["variable"] == "lightning_event_energy"]
            assert lightning_event_energy["observation"].max() == pytest.approx(3.5e-15)
            lightning_event_count = df[df["variable"] == "lightning_event_count"]
            assert (lightning_event_count["observation"].astype(float) == 1.0).all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_call_mock_hierarchy(tmp_path):
    """Event, group and flash ids can be mixed in a single request."""
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
        groups=([35.0, 40.0], [-120.0, -100.0], [4.5e-15, 5.5e-15], [0.0, 30.0]),
        flashes=([35.0], [-120.0], [9.5e-15], [0.0]),
    )

    try:
        with (
            patch.object(GOESGLM, "_discover_files", _fake_discover),
            patch.object(GOESGLM, "_fetch_remote_file", _no_op_fetch),
        ):
            df = ds(
                epoch,
                [
                    "lightning_event_count",
                    "lightning_group_energy",
                    "lightning_flash_energy",
                    "lightning_flash_count",
                ],
            )

        counts = df.groupby("variable").size().to_dict()
        # Row counts follow each level's record count, not the event count
        assert counts == {
            "lightning_event_count": 3,
            "lightning_group_energy": 2,
            "lightning_flash_energy": 1,
            "lightning_flash_count": 1,
        }

        # Energies are read from the matching level's native variable
        groups = df[df["variable"] == "lightning_group_energy"]
        assert groups["observation"].max() == pytest.approx(5.5e-15)
        flashes = df[df["variable"] == "lightning_flash_energy"]
        assert flashes["observation"].iloc[0] == pytest.approx(9.5e-15)
        # Counts stay synthetic at every level
        assert (
            df[df["variable"] == "lightning_flash_count"]["observation"].astype(float)
            == 1.0
        ).all()
        assert df["lon"].between(0, 360).all()
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
            df = ds(epoch, ["lightning_event_energy"], fields=subset)
        assert list(df.columns) == subset
        assert (df["variable"] == "lightning_event_energy").all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_call_mock_empty():
    ds = GOESGLM(satellite="east", cache=False, verbose=False)

    async def _no_discover(self, time_list):  # type: ignore[no-untyped-def]
        return []

    with patch.object(GOESGLM, "_discover_files", _no_discover):
        df = ds(datetime(2024, 6, 1, 18, 0), ["lightning_event_energy"])
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
            df = ds(epoch, ["lightning_event_energy"])
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
    df = ds(
        datetime(2024, 6, 1, 18, 0), ["lightning_event_energy", "lightning_event_count"]
    )
    assert list(df.columns) == ds.SCHEMA.names
    assert not df.empty
    assert set(df["variable"].unique()).issubset(
        {"lightning_event_energy", "lightning_event_count"}
    )


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
    records = GOESGLM._parse_glm_file(str(f), lat_lon_bbox=None)
    assert set(records) == {"event"}
    assert len(records["event"]) == 3
    # CONUS bbox keeps 2 of 3 events
    conus = GOESGLM._parse_glm_file(str(f), lat_lon_bbox=(24.5, -125.0, 49.5, -66.0))
    assert len(conus["event"]) == 2
    # A level with nothing inside the box is dropped entirely
    assert (
        GOESGLM._parse_glm_file(str(f), lat_lon_bbox=(-10.0, -10.0, -5.0, -5.0)) == {}
    )


def test_goes_glm_parse_file_levels(tmp_path):
    f = tmp_path / "hierarchy.nc"
    epoch = datetime(2024, 6, 1, 18, 0, 0)
    _write_glm_netcdf(
        f,
        lats=[35.0, 40.0, 60.0],
        lons=[-120.0, -100.0, 10.0],
        energies=[1e-15, 2e-15, 3e-15],
        offsets=[0.0, 5.123, 19.5],
        epoch=epoch,
        groups=([35.0, 40.0], [-120.0, -100.0], [4e-15, 5e-15], [0.0, 5.0]),
        flashes=([35.0], [-120.0], [9e-15], [0.0]),
    )

    records = GOESGLM._parse_glm_file(
        str(f), lat_lon_bbox=None, levels=("event", "group", "flash")
    )
    assert set(records) == {"event", "group", "flash"}
    assert len(records["event"]) == 3
    assert len(records["group"]) == 2
    assert len(records["flash"]) == 1
    # Every level exposes the same normalised column set
    for frame in records.values():
        assert list(frame.columns) == ["time", "lat", "lon", "energy"]
    assert records["flash"]["energy"].iloc[0] == pytest.approx(9e-15)
    # Flash timestamps come from the first-event offset variable
    assert records["flash"]["time"].iloc[0] == pd.Timestamp(epoch)

    # Only the requested levels are read off disk
    assert set(GOESGLM._parse_glm_file(str(f), None, levels=("flash",))) == {"flash"}


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

    # obstore listing entries are bucket-relative paths
    prefix = "GLM-L2-LCFA/2024/153/18/"
    paths = [
        prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc",
        prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800200_e20241531800400_c20241531800420.nc",
        prefix
        + "OR_GLM-L2-LCFA_G16_s20241531805000_e20241531805200_c20241531805220.nc",
        prefix + "junk.txt",
        prefix
        + "OR_GLM-L2-LCFA_G16_s20241531800000_e20241531800200_c20241531800220.nc",
    ]

    # A plain fake satisfying the obspec ListAsync protocol — no obstore
    # internals need patching, the store is simply injected.
    class _FakeListStore:
        def __init__(self, listing: list[str]):
            self.listing = listing
            self.calls = 0

        def list_async(self, prefix=None, **kwargs):
            self.calls += 1

            async def _gen():
                yield [{"path": p} for p in self.listing]

            return _gen()

    fake = _FakeListStore(paths)
    ds.stores["noaa-goes16"] = fake

    files = _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)]))
    assert len(files) == 2
    assert {f.satellite for f in files} == {"G16"}
    # The ±1 min tolerance window spans two hour directories (17 and 18)
    assert fake.calls == 2

    # The requested hours are complete (in the past), so their listings
    # are memoized: a second discovery issues no further LIST requests.
    files2 = _run(ds._discover_files([datetime(2024, 6, 1, 18, 0, 0)]))
    assert len(files2) == 2
    assert fake.calls == 2

    # Missing prefix → empty listing → empty result, no exception.
    ds2 = GOESGLM(
        satellite="east",
        time_tolerance=np.timedelta64(1, "m"),
        cache=False,
        verbose=False,
    )
    ds2.stores["noaa-goes16"] = _FakeListStore([])
    assert _run(ds2._discover_files([datetime(2024, 6, 1, 18, 0, 0)])) == []


def test_goes_glm_fetch_remote_file(tmp_path):
    ds = GOESGLM(satellite="east", cache=False, verbose=False)
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)

    fake_read = AsyncMock(return_value=b"fake-netcdf-bytes")

    try:
        uri = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/file.nc"
        with patch("earth2studio.data.goes_glm.obstore_read_range", fake_read):
            _run(ds._fetch_remote_file(uri))
        assert pathlib.Path(ds._cache_path(uri)).read_bytes() == b"fake-netcdf-bytes"
        # The obstore read receives the store-relative key (bucket stripped)
        fake_read.assert_awaited_once()
        assert fake_read.await_args.args[1] == "GLM-L2-LCFA/2024/153/18/file.nc"

        # Second call is a no-op (cache hit).
        fake_read.reset_mock()
        with patch("earth2studio.data.goes_glm.obstore_read_range", fake_read):
            _run(ds._fetch_remote_file(uri))
        fake_read.assert_not_called()

        # Missing file in S3 is swallowed (warn-only).
        missing = "s3://noaa-goes16/GLM-L2-LCFA/2024/153/18/missing.nc"
        fake_missing = AsyncMock(side_effect=FileNotFoundError("missing"))
        with patch("earth2studio.data.goes_glm.obstore_read_range", fake_missing):
            _run(ds._fetch_remote_file(missing))
        assert not pathlib.Path(ds._cache_path(missing)).exists()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# Deprecated variable id aliases (flashe / flashc)
# ---------------------------------------------------------------------------
def test_goes_glm_lexicon_deprecated_aliases():
    """Deprecated ids resolve to their canonical name and warn."""
    from earth2studio.lexicon import GOESGLMLexicon

    for alias, canonical in [
        ("flashe", "lightning_event_energy"),
        ("flashc", "lightning_event_count"),
    ]:
        with pytest.warns(FutureWarning, match=alias):
            assert GOESGLMLexicon.resolve_alias(alias) == canonical
        with pytest.warns(FutureWarning):
            assert GOESGLMLexicon[alias][0] == GOESGLMLexicon.VOCAB[canonical]

    # Canonical ids resolve unchanged and must not warn
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert (
            GOESGLMLexicon.resolve_alias("lightning_event_energy")
            == "lightning_event_energy"
        )


def test_goes_glm_call_mock_deprecated_aliases(tmp_path):
    """`flashe`/`flashc` still select the right measurement end-to-end.

    Guards the aliasing bug where a deprecated id falls through to the
    count branch and silently returns 1.0 instead of event energy.
    """
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
            with pytest.warns(FutureWarning):
                df = ds(epoch, ["flashe", "flashc"])

            # The requested (deprecated) id is echoed back unchanged
            assert set(df["variable"].unique()) == {"flashe", "flashc"}
            assert len(df) == 6

            # flashe must carry energy, not the constant 1.0 of flashc
            flashe = df[df["variable"] == "flashe"]
            assert flashe["observation"].max() == pytest.approx(3.5e-15)
            flashc = df[df["variable"] == "flashc"]
            assert (flashc["observation"].astype(float) == 1.0).all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_goes_glm_call_mock_unknown_variable_still_raises():
    """A non-alias unknown id raises rather than silently passing through."""
    ds = GOESGLM(cache=False, verbose=False)
    try:
        with pytest.raises(KeyError):
            ds(datetime(2024, 6, 1, 18, 0), ["not_a_variable"])
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)
