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
    assert table["1::PRMSL::mean sea level"] == (0, 65300)
    assert table["2::HGT::500 mb"] == (65300, 96515 - 65300)
    assert table["3::TMP::2 m above ground"] == (96515, 147274 - 96515)


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


@pytest.mark.timeout(15)
def test_gfs_call_mock(tmp_path, monkeypatch):
    """Exercise the full __call__ path using mocked index + grib decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::TMP::2 m above ground": (0, 65300),
        "2::HGT::500 mb": (65300, 31215),
    }
    fake_grid = np.random.rand(721, 1440).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.pgrb2")

    ds = GFS(source="aws", cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gfs._decode_gfs_grib", return_value=fake_grid),
    ):
        # Bypass real store by stubbing it to a truthy sentinel.
        ds.store = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1), ["t2m", "z500"])

    assert data.shape == (1, 2, 721, 1440)
    # t2m is identity, z500 multiplies HGT by 9.81 — both records used the
    # same mock grid, so t2m matches the grid and z500 = grid * 9.81.
    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    np.testing.assert_allclose(
        data.sel(variable="z500").values[0], fake_grid * 9.81, rtol=1e-6
    )


@pytest.mark.timeout(15)
def test_gfs_fx_call_mock(tmp_path, monkeypatch):
    """Exercise the GFS_FX __call__ path with multiple lead times."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {"1::TMP::2 m above ground": (0, 65300)}
    fake_grid = np.random.rand(721, 1440).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.pgrb2")

    ds = GFS_FX(source="aws", cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.gfs._decode_gfs_grib", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(
            datetime(2024, 1, 1),
            [timedelta(hours=0), timedelta(hours=6)],
            ["t2m"],
        )

    assert data.shape == (1, 2, 1, 721, 1440)
    for j in range(2):
        np.testing.assert_allclose(data.values[0, j, 0], fake_grid)


@pytest.mark.timeout(15)
def test_gfs_missing_variable_mock(tmp_path, monkeypatch):
    # If the requested variable is absent from the .idx, _create_tasks warns
    # and skips it — the output slot keeps its zero initialization.
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
        patch("earth2studio.data.gfs._decode_gfs_grib", return_value=fake_grid),
    ):
        ds.store = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1), ["t2m", "z500"])

    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    # Skipped variable keeps the zero fill.
    assert (data.sel(variable="z500").values == 0.0).all()
