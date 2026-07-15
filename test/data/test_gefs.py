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

from earth2studio.data import GEFS_FX, GEFS_FX_721x1440


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=3), timedelta(hours=384)],
            "t2m",
        ),
        (
            [
                datetime(year=2021, month=8, day=8, hour=6),
                datetime(year=2022, month=4, day=20, hour=12),
            ],
            timedelta(hours=0),
            ["msl"],
        ),
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=240)],
            np.array(["tcwv", "u10m"]),
        ),
    ],
)
def test_gefs_0p50_fetch(time, lead_time, variable):

    ds = GEFS_FX(cache=False)
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
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=0), timedelta(hours=240)],
            "t2m",
        ),
    ],
)
def test_gefs_0p25_fetch(time, lead_time, variable):

    ds = GEFS_FX_721x1440(cache=False)
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
    assert GEFS_FX_721x1440.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize("product", ["gec00", "gep01", "gep30"])
def test_gefs_products(product):
    time = datetime(year=2022, month=12, day=25)
    lead_time = timedelta(hours=3)
    variable = "u100m"

    ds = GEFS_FX(product, cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array([variable]))


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
def test_gefs_cache(time, variable, cache):

    lead_time = np.array([np.timedelta64(3, "h")])

    ds = GEFS_FX(cache=cache)
    data = ds(time, lead_time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 2
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, lead_time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2020, month=9, day=22),
        datetime.now(),
    ],
)
def test_gefs_available(time):
    variable = ["mpl"]
    lead_time = timedelta(hours=0)
    assert not GEFS_FX.available(time)
    with pytest.raises(ValueError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(243, "h")]),
        np.array([np.timedelta64(390, "h")]),
    ],
)
def test_gefs_invalid_lead(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(241, "h")]),
    ],
)
def test_gefs_0p25_invalid_lead(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GEFS_FX_721x1440(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "variable",
    ["aaa", "t1m"],
)
def test_gefs_invalid_variable(variable):
    time = datetime(year=2022, month=12, day=25)
    lead_time = timedelta(hours=0)
    with pytest.raises(KeyError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "product",
    ["gec0", "gep31", "gep00"],
)
def test_gefs_invalid_product(product):
    with pytest.raises(ValueError):
        GEFS_FX(product, cache=False)


# ----------------------------------------------------------------------
# Index parser (offline, no network)
# ----------------------------------------------------------------------


_MOCK_PGRB2A_IDX = (
    "1:0:d=2024010100:PRMSL:mean sea level:3 hour fcst:ENS=low-res ctl\n"
    "2:65300:d=2024010100:HGT:500 mb:3 hour fcst:ENS=low-res ctl\n"
    "3:96515:d=2024010100:TMP:2 m above ground:3 hour fcst:ENS=low-res ctl\n"
)


@pytest.mark.timeout(10)
def test_gefs_index_parser(tmp_path):
    # Write a fake .idx file and patch _fetch_remote_file to return it.
    idx_path = tmp_path / "fake.pgrb2a.idx"
    idx_path.write_text(_MOCK_PGRB2A_IDX)

    ds = GEFS_FX(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        table = asyncio.run(ds._fetch_index("dummy-uri"))

    # Unlike GFS, a dummy end line is appended so the LAST record is kept,
    # with a negative byte length signalling a read to end-of-file
    assert len(table) == 3
    assert table["1::PRMSL::mean sea level::3 hour fcst"] == (0, 65300)
    assert table["2::HGT::500 mb::3 hour fcst"] == (65300, 96515 - 65300)
    trailing = table["3::TMP::2 m above ground::3 hour fcst"]
    assert trailing[0] == 96515
    assert trailing[1] < 0


@pytest.mark.timeout(10)
def test_gefs_index_parser_max_byte_size(tmp_path):
    # A record spanning more than MAX_BYTE_SIZE must raise.
    idx_path = tmp_path / "fake.pgrb2a.idx"
    idx_path.write_text(
        "1:0:d=2024010100:PRMSL:mean sea level:3 hour fcst:ENS=low-res ctl\n"
        "2:6000000:d=2024010100:HGT:500 mb:3 hour fcst:ENS=low-res ctl\n"
    )

    ds = GEFS_FX(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        with pytest.raises(ValueError):
            asyncio.run(ds._fetch_index("dummy-uri"))


# ----------------------------------------------------------------------
# Mock end-to-end (no network, exercises __call__ path)
# ----------------------------------------------------------------------


@pytest.mark.timeout(15)
def test_gefs_call_mock(tmp_path, monkeypatch):
    """Exercise the full __call__ path using mocked index + grib decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::TMP::2 m above ground::3 hour fcst": (0, 65300),
        "2::HGT::500 mb::3 hour fcst": (65300, 31215),
    }
    fake_grid = np.random.rand(361, 720).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.pgrb2")

    ds = GEFS_FX(cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gefs._decode_gefs_grib", return_value=fake_grid),
    ):
        # Bypass real store by stubbing it to a truthy sentinel.
        ds.store = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1), timedelta(hours=3), ["t2m", "z500"])

    assert data.shape == (1, 1, 2, 361, 720)
    # t2m is identity, z500 multiplies HGT by 9.81 — both records used the
    # same mock grid, so t2m matches the grid and z500 = grid * 9.81.
    np.testing.assert_allclose(data.sel(variable="t2m").values[0, 0], fake_grid)
    np.testing.assert_allclose(
        data.sel(variable="z500").values[0, 0], fake_grid * 9.81, rtol=1e-6
    )


@pytest.mark.timeout(15)
def test_gefs_call_mock_trailing_record(tmp_path, monkeypatch):
    """A negative byte length in the index (last grib message in the file) must
    reach the remote fetch as byte_length=None (read offset -> EOF)."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Negative byte length signals the trailing record, per the dummy index
    # line appended in _fetch_index
    fake_index = {"1::TMP::2 m above ground::3 hour fcst": (96515, -96516)}
    fake_grid = np.random.rand(361, 720).astype(np.float32)
    fetch_calls = []

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(path, byte_offset=0, byte_length=None):
        fetch_calls.append((path, byte_offset, byte_length))
        return str(tmp_path / "ignored.pgrb2")

    ds = GEFS_FX(cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gefs._decode_gefs_grib", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1), timedelta(hours=3), ["t2m"])

    assert data.shape == (1, 1, 1, 361, 720)
    np.testing.assert_allclose(data.values[0, 0, 0], fake_grid)
    assert len(fetch_calls) == 1
    _, byte_offset, byte_length = fetch_calls[0]
    assert byte_offset == 96515
    assert byte_length is None


@pytest.mark.timeout(15)
def test_gefs_0p25_call_mock(tmp_path, monkeypatch):
    """Exercise the GEFS_FX_721x1440 __call__ path on the 0.25 degree grid."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {"1::TMP::2 m above ground::3 hour fcst": (0, 65300)}
    fake_grid = np.random.rand(721, 1440).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.pgrb2")

    ds = GEFS_FX_721x1440(cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gefs._decode_gefs_grib", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(
            datetime(2024, 1, 1),
            [timedelta(hours=0), timedelta(hours=3)],
            ["t2m"],
        )

    assert data.shape == (1, 2, 1, 721, 1440)
    for j in range(2):
        np.testing.assert_allclose(data.values[0, j, 0], fake_grid)


@pytest.mark.timeout(15)
def test_gefs_fetch_remote_file_key_and_cache(tmp_path, monkeypatch):
    """The obstore fetch must receive the store-relative key while the cache
    file name stays hashed from the original bucket-prefixed path."""
    import hashlib

    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    ds = GEFS_FX(cache=True, verbose=False)
    ds.store = object()  # type: ignore[assignment]

    captured = {}

    async def _fake_fetch_to_cache(
        store, key, cache_dir, byte_offset=0, byte_length=None, cache_key=None
    ):
        captured.update(
            key=key,
            byte_offset=byte_offset,
            byte_length=byte_length,
            cache_key=cache_key,
        )
        return str(tmp_path / "cached")

    path = f"{GEFS_FX.GEFS_BUCKET_NAME}/gefs.20240101/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003"
    with patch(
        "earth2studio.data.gefs.obstore_fetch_to_cache",
        side_effect=_fake_fetch_to_cache,
    ):
        result = asyncio.run(
            ds._fetch_remote_file(path, byte_offset=123, byte_length=None)
        )

    assert result == str(tmp_path / "cached")
    # Key is store-relative (bucket prefix stripped)
    assert (
        captured["key"] == "gefs.20240101/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003"
    )
    assert captured["byte_offset"] == 123
    assert captured["byte_length"] is None
    # Cache file name hashes the ORIGINAL bucket-prefixed path for warm-cache
    # compatibility with the pre-obstore implementation
    expected = hashlib.sha256((path + "123").encode()).hexdigest()
    assert captured["cache_key"] == expected
