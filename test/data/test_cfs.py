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
import hashlib
import pathlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from earth2studio.data import CFS_FX, CFS_FX_Flux

# Recent AWS-archived cycle used by the slow fetch tests.  Picked to be well
# inside the 2023-04-22+ AWS history bound and well outside the NOMADS rolling
# window so the AWS branch is always exercised.
_TEST_CYCLE = datetime(year=2024, month=6, day=1, hour=0)


# ----------------------------------------------------------------------
# Network fetch tests (slow, hit the public AWS mirror)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (
            _TEST_CYCLE,
            timedelta(hours=6),
            "msl",
        ),
        (
            _TEST_CYCLE,
            [timedelta(hours=0), timedelta(hours=6)],
            ["z500", "u500", "v500"],
        ),
        (
            np.array([np.datetime64(_TEST_CYCLE.isoformat())]),
            np.array([np.timedelta64(12, "h")]),
            np.array(["t850", "q850", "d2m"]),
        ),
    ],
)
def test_cfs_pgbf_fetch(time, lead_time, variable):
    ds = CFS_FX(member=1, source="aws", cache=False)
    data = ds(time, lead_time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(lead_time), len(variable), 181, 360)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert float(data.lat.min()) == -90.0
    assert float(data.lat.max()) == 90.0


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cfs_flxf_fetch():
    ds = CFS_FX_Flux(member=1, source="aws", cache=False)
    data = ds(_TEST_CYCLE, timedelta(hours=6), ["t2m", "u10m", "v10m", "tpf", "tcc"])

    assert data.shape == (1, 1, 5, 190, 384)
    assert not np.isnan(data.values).any()
    # T126 Gaussian latitudes are non-uniform; the most-poleward node is at
    # ~89.28 N, never exactly 90.
    assert 89.0 < float(data.lat.max()) < 90.0
    assert -90.0 < float(data.lat.min()) < -89.0
    # Sanity-check physical ranges.
    t2m = data.sel(variable="t2m").values
    assert 180.0 < t2m.min() and t2m.max() < 340.0


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_cfs_cache(tmp_path, monkeypatch, cache):
    # Redirect the cache root so the test does not touch ~/.cache.
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    ds = CFS_FX(member=1, source="aws", cache=cache)
    ds(_TEST_CYCLE, timedelta(hours=6), ["msl"])
    cache_dir = pathlib.Path(ds.cache)
    if cache:
        # Persistent cache should remain populated after the call.
        assert cache_dir.is_dir()
        assert any(cache_dir.iterdir())
    else:
        # Synchronous __call__ tears the per-instance temp dir down in finally.
        assert not cache_dir.exists()


# ----------------------------------------------------------------------
# Offline validation tests (no network)
# ----------------------------------------------------------------------


@pytest.mark.timeout(5)
@pytest.mark.parametrize("member", [0, 5, -1])
def test_cfs_invalid_member(member):
    with pytest.raises(ValueError):
        CFS_FX(member=member)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("source", ["ftp", "", "AWS", "NOMADS"])
def test_cfs_invalid_source(source):
    with pytest.raises(ValueError):
        CFS_FX(source=source)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2023, month=4, day=21),  # day before AWS archive starts
        datetime(year=2024, month=1, day=1, hour=3),  # not on 6-h cycle
    ],
)
def test_cfs_invalid_time_aws(time):
    ds = CFS_FX(source="aws", cache=False)
    with pytest.raises(ValueError):
        ds(time, timedelta(hours=6), "msl")


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-6),
        timedelta(hours=3),  # not on 6-h cycle
        timedelta(days=200),  # past the 180-day cap
    ],
)
def test_cfs_invalid_leadtime(lead_time):
    ds = CFS_FX(source="aws", cache=False)
    with pytest.raises(ValueError):
        ds(_TEST_CYCLE, lead_time, "msl")


@pytest.mark.timeout(5)
def test_cfs_invalid_variable():
    ds = CFS_FX(source="aws", cache=False)
    # pgbf does not carry t2m; that's a flxf record.
    with pytest.raises(KeyError):
        ds(_TEST_CYCLE, timedelta(hours=6), "t2m")


@pytest.mark.timeout(5)
def test_cfs_nomads_history_range():
    # NOMADS rolling window: anything more than 8 days in the past raises.
    ds = CFS_FX(source="nomads", cache=False)
    old_time = datetime(year=2020, month=1, day=1)
    with pytest.raises(ValueError, match="NOMADS"):
        ds(old_time, timedelta(hours=6), "msl")


@pytest.mark.timeout(5)
def test_cfs_uri_construction():
    # Exercise both URI branches (NOMADS HTTPS + AWS bucket-path) so the
    # _join_uri helper is covered for both sources.
    t = datetime(2024, 6, 1, 6)
    lt = timedelta(hours=12)

    ds_nomads = CFS_FX(source="nomads", cache=False)
    uri = ds_nomads._grib_uri(t, lt)
    assert uri.startswith("https://nomads.ncep.noaa.gov/")
    assert uri.endswith(
        "cfs.20240601/06/6hrly_grib_01/pgbf2024060118.01.2024060106.grb2"
    )

    ds_aws = CFS_FX(source="aws", cache=False)
    uri = ds_aws._grib_uri(t, lt)
    assert uri.startswith("noaa-cfs-pds/")


@pytest.mark.timeout(5)
def test_cfs_available_numpy_datetime():
    # Cover the np.datetime64 conversion branch in available().
    assert not CFS_FX.available(np.datetime64("2023-04-21T00:00"))  # pre-archive
    assert not CFS_FX.available(np.datetime64("2024-01-01T03:00"))  # off-cycle


# ----------------------------------------------------------------------
# Index parser (offline, no network)
# ----------------------------------------------------------------------


_MOCK_PGBF_IDX = (
    # Scalar record followed by a vector-pair (UGRD/VGRD share the same byte
    # offset and disambiguate via the .1 / .2 recno suffix).
    "1:0:d=2024010100:PRMSL:mean sea level:6 hour fcst:\n"
    "2:65300:d=2024010100:HGT:500 mb:6 hour fcst:\n"
    "3.1:96515:d=2024010100:UGRD:500 mb:6 hour fcst:\n"
    "3.2:96515:d=2024010100:VGRD:500 mb:6 hour fcst:\n"
    "4:147274:d=2024010100:TMP:500 mb:6 hour fcst:\n"
)


@pytest.mark.timeout(10)
def test_cfs_index_parser_vector_records(tmp_path):
    # Write a fake .idx file and patch _fetch_remote_file to return it.
    idx_path = tmp_path / "fake.grb2.idx"
    idx_path.write_text(_MOCK_PGBF_IDX)

    ds = CFS_FX(source="aws", cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        import asyncio

        table = asyncio.run(ds._fetch_index("dummy-uri"))

    # Scalar record gets submessage index 1.
    msl_key = next(k for k in table if "PRMSL::mean sea level" in k)
    msl_offset, msl_length, msl_submsg = table[msl_key]
    assert msl_offset == 0
    assert msl_length == 65300
    assert msl_submsg == 1

    # Vector siblings share offset+length but get distinct submessage indices.
    u_key = next(k for k in table if "UGRD::500 mb" in k)
    v_key = next(k for k in table if "VGRD::500 mb" in k)
    u_off, u_len, u_submsg = table[u_key]
    v_off, v_len, v_submsg = table[v_key]
    assert u_off == v_off == 96515
    # Both span the full vector record (offset 96515 -> 147274 = 50759 bytes),
    # not zero length as a naive next-line-minus-this-line would yield.
    assert u_len == v_len == (147274 - 96515)
    assert (u_submsg, v_submsg) == (1, 2)

    # Last record gets the None "read to end of file" sentinel.
    tmp_key = next(k for k in table if "TMP::500 mb" in k)
    tmp_offset, tmp_length, tmp_submsg = table[tmp_key]
    assert tmp_offset == 147274
    assert tmp_length is None
    assert tmp_submsg == 1


# ----------------------------------------------------------------------
# Mock end-to-end (no network, exercises __call__ path)
# ----------------------------------------------------------------------


@pytest.mark.timeout(15)
def test_cfs_call_mock(tmp_path, monkeypatch):
    """Exercise the full __call__ path using mocked index + grib decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::PRMSL::mean sea level": (0, 65300, 1),
        "2::HGT::500 mb": (65300, 31215, 1),
    }
    fake_grid = np.full((181, 360), 101325.0, dtype=np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grb2")

    ds = CFS_FX(member=1, source="aws", cache=True)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch(
            "earth2studio.data.cfs._decode_cfs_grib",
            return_value=fake_grid,
        ),
    ):
        # Bypass real store by stubbing it to a truthy sentinel.
        ds.store = object()  # type: ignore[assignment]
        data = ds(_TEST_CYCLE, timedelta(hours=6), ["msl", "z500"])

    assert data.shape == (1, 1, 2, 181, 360)
    # MSL is identity, z500 multiplies HGT by 9.81 — both records used the
    # same mock grid, so msl matches the grid and z500 = grid * 9.81.
    np.testing.assert_allclose(data.sel(variable="msl").values[0, 0], fake_grid)
    np.testing.assert_allclose(data.sel(variable="z500").values[0, 0], fake_grid * 9.81)


@pytest.mark.timeout(15)
def test_cfs_nomads_call_mock(tmp_path, monkeypatch):
    """Exercise the NOMADS-source __call__ path (obstore HTTP store) with
    mocked index + grib decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {"1::PRMSL::mean sea level": (0, 65300, 1)}
    fake_grid = np.full((181, 360), 101325.0, dtype=np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grb2")

    # Recent 6-hour cycle inside the NOMADS rolling window.
    now = datetime.now()
    recent_cycle = datetime(now.year, now.month, now.day) - timedelta(days=1)

    ds = CFS_FX(member=1, source="nomads", cache=True)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.cfs._decode_cfs_grib", return_value=fake_grid),
    ):
        # Bypass real store by stubbing it to a truthy sentinel.
        ds.store = object()  # type: ignore[assignment]
        data = ds(recent_cycle, timedelta(hours=6), ["msl"])

    assert data.shape == (1, 1, 1, 181, 360)
    np.testing.assert_allclose(data.sel(variable="msl").values[0, 0], fake_grid)


@pytest.mark.timeout(15)
def test_cfs_flux_call_mock(tmp_path, monkeypatch):
    """Exercise the CFS_FX_Flux __call__ path using mocked index + decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::TMP::2 m above ground": (0, 65300, 1),
        "2.1::UGRD::10 m above ground": (65300, 50759, 1),
        "2.2::VGRD::10 m above ground": (65300, 50759, 2),
    }
    # flxf product lives on the T126 Gaussian grid (190 x 384).
    fake_grid = np.full((190, 384), 288.0, dtype=np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grb2")

    ds = CFS_FX_Flux(member=1, source="aws", cache=True)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch(
            "earth2studio.data.cfs._decode_cfs_grib",
            return_value=fake_grid,
        ),
    ):
        # Bypass real store by stubbing it to a truthy sentinel.
        ds.store = object()  # type: ignore[assignment]
        data = ds(
            _TEST_CYCLE,
            [timedelta(hours=0), timedelta(hours=6)],
            ["t2m", "u10m"],
        )

    assert data.shape == (1, 2, 2, 190, 384)
    for j in range(2):
        np.testing.assert_allclose(data.sel(variable="t2m").values[0, j], fake_grid)
        np.testing.assert_allclose(data.sel(variable="u10m").values[0, j], fake_grid)


@pytest.mark.timeout(15)
def test_cfs_missing_variable_returns_nan(tmp_path, monkeypatch):
    # If the requested variable is in the lexicon but absent from the .idx,
    # _create_tasks emits a warning and skips it.  The output slot must be
    # NaN (detectable missing) rather than uninitialised memory.
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Index only has msl; z500 is missing on purpose.
    partial_index = {"1::PRMSL::mean sea level": (0, 65300, 1)}
    fake_grid = np.full((181, 360), 101325.0, dtype=np.float32)

    async def _fake_fetch_index(uri):
        return partial_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grb2")

    ds = CFS_FX(source="aws", cache=True)
    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.cfs._decode_cfs_grib", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(_TEST_CYCLE, timedelta(hours=6), ["msl", "z500"])

    np.testing.assert_allclose(data.sel(variable="msl").values[0, 0], fake_grid)
    # Skipped variable surfaces as all-NaN.
    assert np.isnan(data.sel(variable="z500").values).all()


@pytest.mark.timeout(10)
def test_cfs_fetch_remote_file_obstore_routing(tmp_path, monkeypatch):
    # The obstore path must strip the bucket prefix to a store-relative key
    # while hashing the ORIGINAL bucket-prefixed path (+ byte offset) for the
    # cache file name, so pre-migration warm caches remain valid. Negative
    # byte lengths (last idx record) map to None (read to end of object).
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    ds = CFS_FX(source="aws", cache=True)
    ds.store = object()  # type: ignore[assignment]

    path = (
        "noaa-cfs-pds/cfs.20240601/00/6hrly_grib_01/pgbf2024060106.01.2024060100.grb2"
    )
    expected_key = "cfs.20240601/00/6hrly_grib_01/pgbf2024060106.01.2024060100.grb2"
    expected_cache_key = hashlib.sha256((path + "123").encode()).hexdigest()

    mock_fetch = AsyncMock(return_value="local-file")
    with patch("earth2studio.data.cfs.obstore_fetch_to_cache", mock_fetch):
        out = asyncio.run(ds._fetch_remote_file(path, byte_offset=123, byte_length=456))
        assert out == "local-file"
        args, kwargs = mock_fetch.call_args
        assert args[1] == expected_key
        assert kwargs["byte_offset"] == 123
        assert kwargs["byte_length"] == 456
        assert kwargs["cache_key"] == expected_cache_key

        # Negative sentinel -> read to end of file.
        asyncio.run(ds._fetch_remote_file(path, byte_offset=123, byte_length=-1))
        assert mock_fetch.call_args.kwargs["byte_length"] is None

    # NOMADS: full HTTPS URL is hashed for the cache key while the key is the
    # host-relative path (store is rooted at the NOMADS host).
    ds_nomads = CFS_FX(source="nomads", cache=True)
    ds_nomads.store = object()  # type: ignore[assignment]
    url = (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/"
        "cfs.20240601/00/6hrly_grib_01/pgbf2024060106.01.2024060100.grb2"
    )
    expected_key = (
        "pub/data/nccf/com/cfs/prod/"
        "cfs.20240601/00/6hrly_grib_01/pgbf2024060106.01.2024060100.grb2"
    )
    expected_cache_key = hashlib.sha256((url + "0").encode()).hexdigest()

    mock_fetch = AsyncMock(return_value="local-file")
    with patch("earth2studio.data.cfs.obstore_fetch_to_cache", mock_fetch):
        out = asyncio.run(ds_nomads._fetch_remote_file(url))
    assert out == "local-file"
    args, kwargs = mock_fetch.call_args
    assert args[1] == expected_key
    assert kwargs["cache_key"] == expected_cache_key


# ----------------------------------------------------------------------
# available()
# ----------------------------------------------------------------------


@pytest.mark.timeout(5)
def test_cfs_available_offline_checks():
    # Before AWS archive start: always False (no network call).
    assert not CFS_FX.available(datetime(year=2023, month=4, day=21))
    # Bad cycle hour: False.
    assert not CFS_FX.available(datetime(year=2024, month=1, day=1, hour=3))
    # Bad member: False.
    assert not CFS_FX.available(datetime(year=2024, month=1, day=1), member=9)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_cfs_available_online():
    assert CFS_FX.available(_TEST_CYCLE)
