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
import pytest

from earth2studio.data import GFS, GFS_FX


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2021, month=1, day=1),  # Lower limit
        [
            datetime(year=2022, month=1, day=1, hour=6),
            datetime(year=2023, month=2, day=3, hour=18),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["msl", "tp"]])
def test_gfs_fetch(time, variable):

    ds = GFS(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time",
    [
        (datetime(year=2022, month=12, day=25), timedelta(hours=1)),
        (
            datetime(year=2022, month=12, day=25),
            [timedelta(hours=2), timedelta(hours=3)],
        ),
        (
            np.array(
                [np.datetime64("2024-01-01T00:00"), np.datetime64("2024-02-01T00:00")]
            ),
            np.array([np.timedelta64(0, "h")]),
        ),
    ],
)
def test_gfs_fx_fetch(time, lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    ds = GFS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "msl"]])
@pytest.mark.parametrize("cache", [True, False])
def test_gfs_cache(time, variable, cache):

    ds = GFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "source,time,variable,valid",
    [
        ("ncep", datetime.now() - timedelta(days=1), "t2m", True),
        ("foo", datetime.now() - timedelta(days=1), "t2m", False),
    ],
)
def test_gfs_sources(source, time, variable, valid):

    if not valid:
        with pytest.raises(ValueError):
            ds = GFS(source=source, cache=False)
        return
    # Get nearest 6 hour mark
    time = time.replace(second=0, microsecond=0, minute=0)
    time = time.replace(hour=6 * (time.hour // 6))

    ds = GFS(source=source, cache=False)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2020, month=12, day=31),
        datetime(year=2023, month=1, day=1, hour=13),
        datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_gfs_available(time, variable):
    assert not GFS.available(time)
    with pytest.raises(ValueError):
        ds = GFS()
        ds(time, variable)


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(385, "h")]),
    ],
)
def test_gfs_fx_available(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GFS_FX()
        ds(time, lead_time, variable)


# ----------------------------------------------------------------------
# Index parser (offline, no network)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("hour,uses_atmos", [(6, False), (12, True)])
def test_gfs_layout_cutoff(hour, uses_atmos):
    time = datetime(2021, 3, 22, hour)
    ds = GFS(cache=False)
    uris = [
        ds._grib_uri(time, timedelta()),
        ds._grib_index_uri(time, timedelta()),
    ]
    assert all(("/atmos/" in uri) is uses_atmos for uri in uris)


_MOCK_PGRB2_IDX = (
    "1:0:d=2024010100:PRMSL:mean sea level:anl:\n"
    "2:65300:d=2024010100:HGT:500 mb:anl:\n"
    "3:96515:d=2024010100:TMP:2 m above ground:anl:\n"
    "4:147274:d=2024010100:UGRD:10 m above ground:anl:\n"
)


@pytest.mark.timeout(10)
def test_gfs_index_parser(tmp_path):
    # Write a fake .idx file and patch _fetch_remote_file to return it.
    idx_path = tmp_path / "fake.pgrb2.idx"
    idx_path.write_text(_MOCK_PGRB2_IDX)

    ds = GFS(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        table = asyncio.run(ds._fetch_index("dummy-uri"))

    # Last record is dropped (no next line to compute its length from)
    assert len(table) == 3
    assert table["1::PRMSL::mean sea level::anl"] == (0, 65300)
    assert table["2::HGT::500 mb::anl"] == (65300, 96515 - 65300)
    assert table["3::TMP::2 m above ground::anl"] == (96515, 147274 - 96515)


@pytest.mark.timeout(10)
def test_gfs_index_parser_max_byte_size(tmp_path):
    # A record spanning more than MAX_BYTE_SIZE must raise.
    idx_path = tmp_path / "fake.pgrb2.idx"
    idx_path.write_text(
        "1:0:d=2024010100:PRMSL:mean sea level:anl:\n"
        "2:6000000:d=2024010100:HGT:500 mb:anl:\n"
    )

    ds = GFS(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        with pytest.raises(ValueError):
            asyncio.run(ds._fetch_index("dummy-uri"))


# ----------------------------------------------------------------------
# Mock end-to-end (no network, exercises __call__ path)
# ----------------------------------------------------------------------


@pytest.mark.timeout(10)
@pytest.mark.parametrize("record_number", [450, 596])
def test_gfs_fx_tp_uses_accumulation_window(record_number):
    fake_index = {f"{record_number}::APCP::surface::6-7 hour acc fcst": (100, 10)}
    ds = GFS_FX(cache=False, verbose=False)
    offset, length, modifier = ds._lookup_record(
        fake_index, "tp", datetime(2021, 1, 1), timedelta(hours=7)
    )
    assert (offset, length) == (100, 10)
    assert modifier(np.array(1000.0)) == 1.0


@pytest.mark.timeout(15)
def test_gfs_missing_variable_mock(tmp_path, monkeypatch):
    # An absent variable leaves the output slot detectably missing.
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Index only has t2m; z500 is missing on purpose.
    partial_index = {"1::TMP::2 m above ground": (0, 65300)}
    fake_grid = np.random.rand(721, 1440).astype(np.float32) + 1.0

    async def _fake_fetch_index(uri):
        return partial_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.pgrb2")

    ds = GFS(source="aws", cache=True, verbose=False)
    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gfs.decode_grib_message", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1), ["t2m", "z500"])

    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    assert np.isnan(data.sel(variable="z500").values).all()


@pytest.mark.timeout(30)
def test_gfs_call_mock(mock_statistic_grib):
    ds = GFS(source="aws", cache=True, verbose=True)
    variables = ["t2m", "t2m:mean:12h", "t2m:sum:12h", "t2m:max:12h", "t2m:min:12h"]
    index_calls = mock_statistic_grib(ds)
    data = ds(datetime(2024, 1, 2), variables)
    # Window [-12h, 0h) holds the 6-hourly analyses at 12z (12h) and 18z (18h)
    assert data.shape == (1, 5, 2, 2)
    values = [float(data.sel(variable=v).values[0, 0, 0]) for v in variables]
    assert values == [24.0, 15.0, 30.0, 18.0, 12.0]
    assert set(data.attrs["earth2studio_statistics"]) == set(variables[1:])
    # Instantaneous plus the two extra analysis files, nothing else
    assert len(set(index_calls)) == 3
    assert len(index_calls) == 3
    # Analysis windows move across cycles, always fetching lead zero.
    assert all("f000" in uri for uri in index_calls)
    index_calls.clear()
    data = ds([datetime(2024, 1, 2, 6), datetime(2024, 1, 2)], variables)
    np.testing.assert_allclose(
        data.values[:, :, 0, 0], [[30, 21, 42, 24, 18], [24, 15, 30, 18, 12]]
    )
    assert len(index_calls) == len(set(index_calls)) == 4
    assert all("f000" in uri for uri in index_calls)
    from earth2studio.data.utils import fetch_data

    data = fetch_data(
        ds,
        np.array(["2024-01-02"], dtype="datetime64[ns]"),
        np.array(["TMP::2 m above ground"]),
    )
    assert data.values[0, 0, 0, 0, 0] == 24
    data = ds(datetime(2024, 1, 2), ["t2m", "z500", "tp06"])
    np.testing.assert_allclose(data.values[0, :, 0, 0], [24, 24 * 9.81, 0.021])
    assert list(data.coords["variable"].values) == ["t2m", "z500", "tp06"]
    mock_statistic_grib(ds, missing=(12,))
    data = ds(datetime(2024, 1, 2), variables)
    np.testing.assert_allclose(data.values[0, :, 0, 0], [24, 18, 18, 18, 18])
    np.testing.assert_allclose(
        data.values[0, :, 1, 0], [np.nan, np.nan, 0, np.nan, np.nan], equal_nan=True
    )


@pytest.mark.timeout(30)
def test_gfs_fx_call_mock(mock_statistic_grib, monkeypatch, tmp_path):
    fx = GFS_FX(source="aws", cache=True, verbose=True)
    variables = [
        "t2m",
        "t2m:mean:3h",
        "t2m:sum:+0h:+3h",
        "t2m:max:-2h:+0h",
        "t2m:min:3h",
    ]
    index_calls = mock_statistic_grib(fx)
    data = fx(datetime(2024, 1, 1), [timedelta(hours=6), timedelta(hours=7)], variables)
    with pytest.raises(ValueError, match="lead time"):
        fx(datetime(2024, 1, 1), [timedelta(hours=0)], ["t2m:mean:3h"])
    assert data.shape == (1, 2, 5, 2, 2)
    # lead 6: mean(3,4,5)=4, sum(6,7,8)=21, max(4,5)=5; lead 7 is shifted by 1h
    np.testing.assert_allclose(
        data.values[0, :, :, 0, 0], [[6, 4, 21, 5, 3], [7, 5, 24, 6, 4]]
    )
    assert set(data.attrs["earth2studio_statistics"]) == set(variables[1:])
    # Leads 3..9 are each fetched once despite overlapping windows
    assert len(set(index_calls)) == 7
    assert len(index_calls) == 7
    np.testing.assert_allclose(data.values[0, :, 1, 0, 1], [4, 5])
    np.testing.assert_allclose(data.values[0, :, 2, 0, 1], [14, 8])

    # Equal valid times from different initializations remain distinct requests.
    index_calls.clear()
    data = fx(
        [datetime(2024, 1, 1, 6), datetime(2024, 1, 1)],
        [timedelta(hours=7), timedelta(hours=6)],
        variables,
    )
    np.testing.assert_allclose(
        data.values[:, :, :, 0, 0],
        [
            [[13, 11, 42, 12, 10], [12, 10, 39, 11, 9]],
            [[7, 5, 24, 6, 4], [6, 4, 21, 5, 3]],
        ],
    )
    assert len(index_calls) == len(set(index_calls)) == 14
    assert sum("gfs.t06z" in uri for uri in index_calls) == 7
    assert sum("gfs.t00z" in uri for uri in index_calls) == 7

    # True trailing totals at a complete bucket and across the reset at hour 6.
    data = fx(
        [datetime(2024, 1, 1), datetime(2024, 1, 1, 6)],
        [timedelta(hours=h) for h in (6, 7, 11, 12, 13)],
        ["tp", "tp06", "tp06:sum:+0h:+2h", "t2m"],
    )
    assert list(data.coords["variable"].values) == [
        "tp",
        "tp06",
        "tp06:sum:+0h:+2h",
        "t2m",
    ]
    np.testing.assert_allclose(
        data.sel(variable="t2m").values[:, :, 0, 0],
        [[6, 7, 11, 12, 13], [12, 13, 17, 18, 19]],
    )
    np.testing.assert_allclose(
        data.sel(variable="tp").values[:, :, 0, 0],
        [[0.021, 0.007, 0.045, 0.057, 0.013]] * 2,
    )
    np.testing.assert_allclose(
        data.sel(variable="tp06").values[:, :, 0, 0],
        [[0.021, 0.027, 0.051, 0.057, 0.063]] * 2,
    )
    np.testing.assert_allclose(
        data.sel(variable="tp06:sum:+0h:+2h").values[:, :, 0, 0],
        [[0.048, 0.060, 0.108, 0.120, 0.132]] * 2,
    )
    with pytest.raises(ValueError, match="at least 6 hours"):
        fx(datetime(2024, 1, 1), timedelta(hours=5), "tp06")
    with pytest.raises(ValueError, match="at least 6 hours"):
        fx(datetime(2024, 1, 1), timedelta(hours=6), "tp06:mean:3h")
    mock_statistic_grib(fx, missing=(6,))
    data = fx(datetime(2024, 1, 1), timedelta(hours=7), ["tp", "tp06", "msl"])
    assert data.sel(variable="tp").values[0, 0, 0, 0] == 0.007
    assert np.isnan(data.sel(variable=["tp06", "msl"]).values).all()
    _check_shared_download(monkeypatch, tmp_path)


def _check_shared_download(monkeypatch, tmp_path):
    """Concurrent requests for one byte range share a single atomic download."""
    import earth2studio.data.gfs as gfs_module

    calls = []

    async def fake_fetch(
        store,
        key,
        cache_dir,
        byte_offset=0,
        byte_length=None,
        cache_key=None,
        chunked=False,
        atomic=False,
    ):
        calls.append((key, byte_offset, atomic))
        await asyncio.sleep(0.01)
        return str(tmp_path / cache_key)

    monkeypatch.setattr(gfs_module, "obstore_fetch_to_cache", fake_fetch)
    ds = GFS_FX(source="aws", cache=False, verbose=False)
    ds.store = object()

    async def run():
        args = ("noaa-gfs-bdp-pds/gfs.20240101/00/atmos/gfs.t00z.pgrb2.0p25.f006", 4)
        same = await asyncio.gather(
            *[ds._fetch_remote_file(args[0], byte_offset=args[1]) for _ in range(4)]
        )
        other = await ds._fetch_remote_file(args[0], byte_offset=8)
        return same, other

    same, other = asyncio.run(run())
    assert len(set(same)) == 1 and other not in same
    assert [c[1:] for c in calls] == [(4, True), (8, True)]
    assert not ds._inflight_downloads
