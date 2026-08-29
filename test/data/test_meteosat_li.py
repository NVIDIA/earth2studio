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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import MeteosatLI
from earth2studio.data.meteosat_li import (
    _LI_EPOCH,
    _merge_windows,
    _MeteosatLIGranule,
    _normalize_lat_lon_bbox,
)

netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 not installed")


TEST_TIME = datetime(2025, 6, 15, 12, 0, 0)


def _seconds_since_epoch(t: datetime) -> float:
    """Seconds between the LI epoch (2000-01-01) and ``t``."""
    return float((np.datetime64(t, "us") - _LI_EPOCH) / np.timedelta64(1, "s"))


def _write_flat_li_netcdf(
    path: pathlib.Path,
    product: str,
    lats: list[float],
    lons: list[float],
    radiances: list[float],
    times: list[datetime],
    platform: str = "MTI1",
) -> None:
    """Write a minimal flat LI L2 product (LFL or LGR) the source can parse."""
    record_dim, time_var = (
        ("flashes", "flash_time") if product == "LFL" else ("groups", "group_time")
    )
    n = len(lats)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.platform = platform
        ds.createDimension(record_dim, n)
        t_v = ds.createVariable(time_var, "f8", (record_dim,))
        t_v.units = "seconds since 2000-01-01 00:00:00.0"
        t_v[:] = np.asarray([_seconds_since_epoch(t) for t in times], dtype=np.float64)
        ds.createVariable("latitude", "f4", (record_dim,))[:] = np.asarray(
            lats, dtype=np.float32
        )
        ds.createVariable("longitude", "f4", (record_dim,))[:] = np.asarray(
            lons, dtype=np.float32
        )
        ds.createVariable("radiance", "f4", (record_dim,))[:] = np.asarray(
            radiances, dtype=np.float32
        )
        ds.createVariable("flash_id", "u4", (record_dim,))[:] = np.arange(
            1, n + 1, dtype=np.uint32
        )
        if product == "LFL":
            ds.createVariable("flash_duration", "f4", (record_dim,))[:] = np.full(
                n, 100.0, dtype=np.float32
            )
            ds.createVariable("flash_footprint", "f4", (record_dim,))[:] = np.full(
                n, 5.0, dtype=np.float32
            )


def _write_lef_netcdf(
    path: pathlib.Path,
    heads: dict[str, tuple[list[float], list[float], list[float], list[float]]],
    epoch: datetime,
    platform: str = "MTI1",
) -> None:
    """Write a minimal nested LEF product with one group per optical head.

    ``heads`` maps head name to ``(lats, lons, radiances, time_offsets)``.
    """
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.platform = platform
        data = ds.createGroup("data")
        for head, (lats, lons, radiances, offsets) in heads.items():
            grp = data.createGroup(head)
            grp.createDimension("events", len(lats))
            e_v = grp.createVariable("epoch_time", "f8", ())
            e_v.units = "seconds since 2000-01-01 00:00:00.0"
            e_v[...] = _seconds_since_epoch(epoch)
            grp.createVariable("time_offset", "f4", ("events",))[:] = np.asarray(
                offsets, dtype=np.float32
            )
            grp.createVariable("latitude", "f4", ("events",))[:] = np.asarray(
                lats, dtype=np.float32
            )
            grp.createVariable("longitude", "f4", ("events",))[:] = np.asarray(
                lons, dtype=np.float32
            )
            grp.createVariable("radiance", "f4", ("events",))[:] = np.asarray(
                radiances, dtype=np.float32
            )
            grp.createVariable("flash_id", "u4", ("events",))[:] = np.arange(
                1, len(lats) + 1, dtype=np.uint32
            )


def _granule(product: str, start: datetime = TEST_TIME) -> _MeteosatLIGranule:
    return _MeteosatLIGranule(
        product=product,
        product_id=f"TEST-{product}-{start:%Y%m%d%H%M%S}",
        start=start,
        end=start + timedelta(minutes=10),
    )


def _stage(ds: MeteosatLI, granule: _MeteosatLIGranule, writer) -> None:
    """Pre-populate the cache with a synthetic granule so no download runs."""
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    writer(pathlib.Path(ds._cache_path(granule)))


# ---------------------------------------------------------------------------
# Mock tests - exercise __call__ end-to-end without network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "product,variables",
    [
        ("LFL", ["lightning_flash_radiance", "lightning_flash_count"]),
        ("LGR", ["lightning_group_radiance", "lightning_group_count"]),
    ],
)
def test_meteosat_li_call_mock(product, variables, monkeypatch):
    granule = _granule(product)
    ds = MeteosatLI(cache=False, verbose=False)

    _stage(
        ds,
        granule,
        lambda p: _write_flat_li_netcdf(
            p,
            product,
            lats=[45.0, 10.0, -20.0],
            lons=[5.0, -30.0, 20.0],
            radiances=[100.0, 200.0, 300.0],
            times=[TEST_TIME + timedelta(seconds=s) for s in (0, 60, 120)],
        ),
    )

    async def _fake_discover(self, time_list, products):
        return [granule]

    async def _fake_fetch(self, g):
        return self._cache_path(g)

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    monkeypatch.setattr(MeteosatLI, "_fetch_granule", _fake_fetch)

    try:
        df = ds(TEST_TIME, variables)

        assert list(df.columns) == ds.SCHEMA.names
        assert len(df) == 6  # 3 detections x 2 variables
        assert set(df["variable"].unique()) == set(variables)
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        assert set(df["platform"].unique()) == {"MTI1"}
        assert df.attrs["source"] == ds.SOURCE_ID

        radiance = df[df["variable"] == variables[0]]
        assert radiance["observation"].max() == pytest.approx(300.0)
        count = df[df["variable"] == variables[1]]
        assert (count["observation"].astype(float) == 1.0).all()
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_call_mock_lef(monkeypatch):
    """LEF records are concatenated across all four optical heads."""
    granule = _granule("LEF")
    ds = MeteosatLI(cache=False, verbose=False)

    _stage(
        ds,
        granule,
        lambda p: _write_lef_netcdf(
            p,
            heads={
                "north": ([45.0, 46.0], [5.0, 6.0], [10.0, 20.0], [0.0, 30.0]),
                "south": ([-25.0], [1.0], [30.0], [60.0]),
                "east": ([5.0], [40.0], [40.0], [90.0]),
                "west": ([-5.0], [-40.0], [50.0], [120.0]),
            },
            epoch=TEST_TIME,
        ),
    )

    async def _fake_discover(self, time_list, products):
        return [granule]

    async def _fake_fetch(self, g):
        return self._cache_path(g)

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    monkeypatch.setattr(MeteosatLI, "_fetch_granule", _fake_fetch)

    try:
        df = ds(TEST_TIME, ["lightning_event_radiance"])
        assert len(df) == 5  # 2 + 1 + 1 + 1 events across heads
        assert df["observation"].max() == pytest.approx(50.0)
        # Times decode as epoch + per-event offset
        assert df["time"].min() == pd.Timestamp(TEST_TIME)
        assert df["time"].max() == pd.Timestamp(TEST_TIME + timedelta(seconds=120))
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_call_mock_fields_subset(monkeypatch):
    granule = _granule("LFL")
    ds = MeteosatLI(cache=False, verbose=False)
    _stage(
        ds,
        granule,
        lambda p: _write_flat_li_netcdf(
            p,
            "LFL",
            lats=[45.0],
            lons=[5.0],
            radiances=[100.0],
            times=[TEST_TIME],
        ),
    )

    async def _fake_discover(self, time_list, products):
        return [granule]

    async def _fake_fetch(self, g):
        return self._cache_path(g)

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    monkeypatch.setattr(MeteosatLI, "_fetch_granule", _fake_fetch)

    try:
        df = ds(
            TEST_TIME,
            ["lightning_flash_radiance"],
            fields=["time", "lat", "lon", "observation"],
        )
        assert list(df.columns) == ["time", "lat", "lon", "observation"]
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_call_mock_bbox(monkeypatch):
    """Detections outside the bounding box are dropped at parse time."""
    granule = _granule("LFL")
    ds = MeteosatLI(
        lat_lon_bbox=(35.0, -12.0, 60.0, 30.0),  # Europe
        cache=False,
        verbose=False,
    )
    _stage(
        ds,
        granule,
        lambda p: _write_flat_li_netcdf(
            p,
            "LFL",
            lats=[45.0, -20.0, 50.0],
            lons=[5.0, 20.0, 100.0],  # only rows 0 and 2 are in the lat range
            radiances=[100.0, 200.0, 300.0],
            times=[TEST_TIME] * 3,
        ),
    )

    async def _fake_discover(self, time_list, products):
        return [granule]

    async def _fake_fetch(self, g):
        return self._cache_path(g)

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    monkeypatch.setattr(MeteosatLI, "_fetch_granule", _fake_fetch)

    try:
        df = ds(TEST_TIME, ["lightning_flash_radiance"])
        assert len(df) == 1  # only (45.0, 5.0) is inside the box
        assert df["lat"].iloc[0] == pytest.approx(45.0)
        assert df["lon"].iloc[0] == pytest.approx(5.0)
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_call_mock_time_tolerance(monkeypatch):
    """Only detections inside the tolerance window are returned."""
    granule = _granule("LFL")
    ds = MeteosatLI(time_tolerance=np.timedelta64(1, "m"), cache=False, verbose=False)
    _stage(
        ds,
        granule,
        lambda p: _write_flat_li_netcdf(
            p,
            "LFL",
            lats=[45.0, 46.0, 47.0],
            lons=[5.0, 6.0, 7.0],
            radiances=[100.0, 200.0, 300.0],
            times=[
                TEST_TIME,
                TEST_TIME + timedelta(seconds=30),
                TEST_TIME + timedelta(minutes=5),  # outside +/- 1 min
            ],
        ),
    )

    async def _fake_discover(self, time_list, products):
        return [granule]

    async def _fake_fetch(self, g):
        return self._cache_path(g)

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    monkeypatch.setattr(MeteosatLI, "_fetch_granule", _fake_fetch)

    try:
        df = ds(TEST_TIME, ["lightning_flash_radiance"])
        assert len(df) == 2
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_call_mock_empty(monkeypatch):
    """No granules discovered yields an empty, correctly typed frame."""
    ds = MeteosatLI(cache=False, verbose=False)

    async def _fake_discover(self, time_list, products):
        return []

    monkeypatch.setattr(MeteosatLI, "_discover_granules", _fake_discover)
    try:
        df = ds(TEST_TIME, ["lightning_flash_radiance"])
        assert df.empty
        assert list(df.columns) == ds.SCHEMA.names
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# Remote fetch (requires EUMETSAT Data Store credentials)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
def test_meteosat_li_fetch():
    ds = MeteosatLI(
        lat_lon_bbox=(35.0, -12.0, 60.0, 30.0),
        time_tolerance=np.timedelta64(2, "m"),
        cache=False,
        verbose=False,
    )
    df = ds(TEST_TIME, ["lightning_flash_radiance", "lightning_flash_count"])
    assert not df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()) == {
        "lightning_flash_radiance",
        "lightning_flash_count",
    }
    assert df["lat"].between(35.0, 60.0).all()
    assert (df[df["variable"] == "lightning_flash_radiance"]["observation"] > 0).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
def test_meteosat_li_cache():
    ds = MeteosatLI(
        lat_lon_bbox=(35.0, -12.0, 60.0, 30.0),
        time_tolerance=np.timedelta64(2, "m"),
        cache=True,
        verbose=False,
    )
    df = ds(TEST_TIME, ["lightning_flash_radiance"])
    assert not df.empty
    # Second call is served from the on-disk cache
    cached = list(pathlib.Path(ds.cache).glob("*.nc"))
    assert len(cached) > 0
    df2 = ds(TEST_TIME, ["lightning_flash_radiance"])
    assert len(df) == len(df2)
    shutil.rmtree(ds.cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_meteosat_li_exceptions():
    ds = MeteosatLI(cache=False, verbose=False)

    with pytest.raises(KeyError):
        ds(TEST_TIME, ["not_a_variable"])

    # Before the LI Level-2 archive start
    with pytest.raises(ValueError):
        ds(datetime(2023, 1, 1), ["lightning_flash_radiance"])

    # Future time
    with pytest.raises(ValueError):
        ds(datetime.now() + timedelta(days=2), ["lightning_flash_radiance"])

    with pytest.raises(ValueError):
        MeteosatLI(lat_lon_bbox=(60.0, -12.0, 35.0, 30.0))  # lat_min > lat_max

    # Crosses the antimeridian: 170 E -> 190 E normalises to (170, -170),
    # which is not expressible as a single [-180, 180) interval
    with pytest.raises(ValueError):
        MeteosatLI(lat_lon_bbox=(35.0, 170.0, 60.0, 190.0))

    # Crossing the *prime* meridian is fine: 350 E -> 370 E is contiguous
    # once normalised to (-10, 10)
    assert MeteosatLI(lat_lon_bbox=(35.0, 350.0, 60.0, 370.0)) is not None


def test_meteosat_li_available():
    assert MeteosatLI.available(TEST_TIME)
    assert MeteosatLI.available(np.datetime64("2025-06-15T12:00:00"))
    assert not MeteosatLI.available(datetime(2023, 1, 1))
    assert not MeteosatLI.available(datetime(2024, 7, 4, 13, 0))  # archive start


def test_meteosat_li_resolve_fields():
    assert MeteosatLI.resolve_fields(None) is MeteosatLI.SCHEMA
    assert MeteosatLI.resolve_fields("lat").names == ["lat"]
    assert MeteosatLI.resolve_fields(["time", "lat"]).names == ["time", "lat"]

    with pytest.raises(KeyError):
        MeteosatLI.resolve_fields(["nope"])
    with pytest.raises(KeyError):
        MeteosatLI.resolve_fields(pa.schema([pa.field("nope", pa.string())]))
    with pytest.raises(TypeError):
        MeteosatLI.resolve_fields(pa.schema([pa.field("lat", pa.string())]))


def test_meteosat_li_tolerance_conversion():
    ds = MeteosatLI(time_tolerance=np.timedelta64(3, "m"))
    assert ds._tolerance_lower == timedelta(minutes=-3)
    assert ds._tolerance_upper == timedelta(minutes=3)

    ds = MeteosatLI(time_tolerance=(np.timedelta64(0, "m"), np.timedelta64(10, "m")))
    assert ds._tolerance_lower == timedelta(0)
    assert ds._tolerance_upper == timedelta(minutes=10)


def test_meteosat_li_lat_lon_bbox_accepts_360_convention():
    assert _normalize_lat_lon_bbox(None) is None
    # [0, 360) input is normalised to the native [-180, 180) convention
    box = _normalize_lat_lon_bbox((35.0, 340.0, 60.0, 350.0))
    assert box[1] == pytest.approx(-20.0)
    assert box[3] == pytest.approx(-10.0)
    # Already in [-180, 180) is passed through
    assert _normalize_lat_lon_bbox((35.0, -12.0, 60.0, 30.0)) == (
        35.0,
        -12.0,
        60.0,
        30.0,
    )


def test_meteosat_li_merge_windows():
    pad = timedelta(minutes=10)
    t = TEST_TIME
    # Two overlapping windows collapse into one
    merged = _merge_windows(
        [
            (t, t + timedelta(minutes=5)),
            (t + timedelta(minutes=2), t + timedelta(minutes=8)),
        ],
        pad,
    )
    assert len(merged) == 1
    assert merged[0] == (t - pad, t + timedelta(minutes=8))

    # Widely separated windows stay distinct
    merged = _merge_windows(
        [
            (t, t + timedelta(minutes=1)),
            (t + timedelta(hours=5), t + timedelta(hours=6)),
        ],
        pad,
    )
    assert len(merged) == 2


def test_meteosat_li_cache_path_is_stable_and_unique():
    ds = MeteosatLI(cache=False, verbose=False)
    try:
        lfl, lgr = _granule("LFL"), _granule("LGR")
        assert ds._cache_path(lfl) == ds._cache_path(_granule("LFL"))
        assert ds._cache_path(lfl) != ds._cache_path(lgr)
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_parse_granule(tmp_path):
    path = tmp_path / "lfl.nc"
    _write_flat_li_netcdf(
        path,
        "LFL",
        lats=[45.0, -20.0],
        lons=[5.0, 20.0],
        radiances=[100.0, 200.0],
        times=[TEST_TIME, TEST_TIME + timedelta(seconds=30)],
    )

    frame = MeteosatLI._parse_granule(str(path), "LFL", None)
    assert len(frame) == 2
    assert frame["time"].iloc[0] == pd.Timestamp(TEST_TIME)
    assert frame["radiance"].iloc[1] == pytest.approx(200.0)
    assert frame["platform"].iloc[0] == "MTI1"
    assert "flash_duration" in frame.columns

    # A box that excludes every record returns None
    assert MeteosatLI._parse_granule(str(path), "LFL", (80.0, 0.0, 85.0, 10.0)) is None


def test_meteosat_li_parse_granule_empty(tmp_path):
    path = tmp_path / "empty.nc"
    _write_flat_li_netcdf(path, "LFL", lats=[], lons=[], radiances=[], times=[])
    assert MeteosatLI._parse_granule(str(path), "LFL", None) is None


def test_meteosat_li_discover_granules(monkeypatch):
    """Discovery deduplicates granules returned for overlapping windows."""
    ds = MeteosatLI(time_tolerance=np.timedelta64(5, "m"), cache=False, verbose=False)
    granule = _granule("LFL")
    calls = []

    async def _fake_search(self, product, start, end):
        calls.append((product, start, end))
        return [granule]

    monkeypatch.setattr(MeteosatLI, "_search_window", _fake_search)
    try:
        found = asyncio.run(
            ds._discover_granules(
                [TEST_TIME, TEST_TIME + timedelta(minutes=2)], ["LFL"]
            )
        )
        # Both timestamps fall in one merged window -> a single search
        assert len(calls) == 1
        # The same granule seen twice is returned once
        assert found == [granule]
    finally:
        shutil.rmtree(ds.cache, ignore_errors=True)


def test_meteosat_li_parse_granule_drops_fill_records(tmp_path, recwarn):
    """Fill-valued padding records are dropped, not turned into junk times.

    Real LEF granules pad sparsely-populated optical heads with a record
    whose coordinates and time are the NetCDF fill value (~1e37). Reading
    that through the microsecond cast overflows int64 and yields an
    undefined timestamp, so the record must be discarded on parse.
    """
    path = tmp_path / "lef_fill.nc"
    fill = netCDF4.default_fillvals["f4"]
    epoch = TEST_TIME

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.platform = "MTI1"
        data = ds.createGroup("data")
        # 'north' holds two real detections, 'south' only a padding record
        for head, lats, lons, rads, offs in [
            ("north", [45.0, 46.0], [5.0, 6.0], [12.0, 18.0], [1.0, 2.0]),
            ("south", [fill], [fill], [fill], [fill]),
        ]:
            grp = data.createGroup(head)
            grp.createDimension("events", len(lats))
            e_v = grp.createVariable("epoch_time", "f8", ())
            e_v.units = "seconds since 2000-01-01 00:00:00.0"
            e_v[...] = _seconds_since_epoch(epoch)
            for name, values in [
                ("time_offset", offs),
                ("latitude", lats),
                ("longitude", lons),
                ("radiance", rads),
            ]:
                var = grp.createVariable(name, "f4", ("events",), fill_value=fill)
                var[:] = np.asarray(values, dtype=np.float32)
            grp.createVariable("flash_id", "u4", ("events",))[:] = np.arange(
                1, len(lats) + 1, dtype=np.uint32
            )

    frame = MeteosatLI._parse_granule(str(path), "LEF", None)

    # Only the two real detections survive
    assert frame is not None and len(frame) == 2
    assert frame["lat"].tolist() == [45.0, 46.0]
    # Their times are the epoch plus the per-record offset, not garbage
    assert frame["time"].min() == pd.Timestamp(epoch) + pd.Timedelta(seconds=1)
    assert frame["time"].max() == pd.Timestamp(epoch) + pd.Timedelta(seconds=2)
    assert frame["time"].notna().all()

    # The overflowing cast used to raise "invalid value encountered in cast"
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
