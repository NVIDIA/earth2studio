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
from unittest.mock import patch

import numpy as np
import pytest
from obstore.store import GCSStore, HTTPStore, S3Store

from earth2studio.data import HRRR, HRRR_FX


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2022, month=12, day=25),
        [
            datetime(year=2022, month=1, day=1, hour=6),
            datetime(year=2023, month=2, day=3, hour=18),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["u10m", "u100"], ["u1hl", "tp"]])
def test_hrrr_fetch(time, variable):

    ds = HRRR(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    if isinstance(variable, str):
        variable = [variable]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
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
            [timedelta(hours=0), timedelta(hours=3)],
        ),
        (
            np.array(
                [np.datetime64("2024-01-01T00:00"), np.datetime64("2024-02-01T00:00")]
            ),
            np.array([np.timedelta64(1, "h")]),
        ),
    ],
)
def test_hrrr_fx_fetch(time, lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "tp"
    ds = HRRR_FX(cache=False)
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
    assert shape[3] == 1059
    assert shape[4] == 1799
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.timeout(15)
def test_hrrr_init():
    """Test HRRR initialization with different sources and parameters"""
    # Test AWS source
    ds = HRRR(source="aws", cache=True, verbose=True, async_timeout=300)
    assert ds.store is None  # Lazy init
    asyncio.run(ds._async_init())

    assert ds.uri_prefix == "noaa-hrrr-bdp-pds"
    assert isinstance(ds.store, S3Store)
    assert ds.async_timeout == 300

    # Test Google source
    ds = HRRR(source="google", cache=False, verbose=False)
    asyncio.run(ds._async_init())
    assert ds.uri_prefix == "high-resolution-rapid-refresh"
    assert isinstance(ds.store, GCSStore)

    # Test Nomads source
    ds = HRRR(source="nomads")
    asyncio.run(ds._async_init())
    assert ds.uri_prefix == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/"
    assert isinstance(ds.store, HTTPStore)

    # Test invalid source
    with pytest.raises(ValueError):
        HRRR(source="invalid_source")

    # Test Azure source (not implemented)
    with pytest.raises(NotImplementedError):
        HRRR(source="azure")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "sp", "t10hl"]])
@pytest.mark.parametrize("cache", [True, False])
def test_hrrr_cache(time, variable, cache):

    ds = HRRR(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
    assert (data.coords["variable"] == np.array(variable)).all()
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
    assert (data.coords["variable"] == np.array(variable[0])).all()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_hrrr_validate_inputs():
    ds = HRRR(cache=False)

    # Test valid time
    valid_time = datetime(year=2022, month=12, day=25, hour=12)
    ds._validate_time([valid_time])  # Should not raise

    # Test invalid hour interval
    invalid_time = datetime(year=2022, month=12, day=25, hour=12, minute=30)
    with pytest.raises(ValueError):
        ds._validate_time([invalid_time])

    # Test time before 2018-07-12 13:00
    old_time = datetime(year=2018, month=7, day=12, hour=12)
    with pytest.raises(ValueError):
        ds._validate_time([old_time])

    # Test invalid variable
    with pytest.raises(KeyError):
        ds(datetime(year=2024, month=12, day=25), "invalid variable")


@pytest.mark.timeout(15)
def test_hrrr_fx_validate_leadtime():
    ds = HRRR_FX(cache=False)
    # Test valid lead times
    times = [datetime(2024, 1, 1)]
    valid_lead_times = [timedelta(hours=1), timedelta(hours=12), timedelta(hours=48)]
    ds._validate_leadtime(times, valid_lead_times)

    # Test invalid lead times
    invalid_lead_times = [
        timedelta(hours=49),  # > 48 hours
        timedelta(hours=-1),  # < 0 hours
        timedelta(hours=1, minutes=30),  # Not hourly
    ]
    for lt in invalid_lead_times:
        with pytest.raises(ValueError):
            ds._validate_leadtime(times, [lt])

    times = [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)]
    for time0 in times:
        with pytest.raises(ValueError):
            ds._validate_leadtime([time0], [timedelta(hours=19)])


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(49, "h")]),
    ],
)
def test_hrrr_fx_available(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = HRRR_FX()
        ds(time, lead_time, variable)


# ----------------------------------------------------------------------
# Index parser (offline, no network)
# ----------------------------------------------------------------------


_MOCK_HRRR_IDX = (
    "1:0:d=2024010100:REFC:entire atmosphere:anl:\n"
    "2:375155:d=2024010100:TMP:2 m above ground:anl:\n"
    "3:475155:d=2024010100:HGT:500 mb:anl:\n"
    "4:575155:d=2024010100:UGRD:10 m above ground:anl:\n"
)


@pytest.mark.timeout(10)
def test_hrrr_index_parser(tmp_path):
    # Write a fake .idx file and patch _fetch_remote_file to return it.
    idx_path = tmp_path / "fake.grib2.idx"
    idx_path.write_text(_MOCK_HRRR_IDX)

    ds = HRRR(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        table = asyncio.run(ds._fetch_index("dummy-uri"))

    # Last record is dropped (no next line to compute its length from)
    assert len(table) == 3
    assert table["1::REFC::entire atmosphere::anl"] == (0, 375155)
    assert table["2::TMP::2 m above ground::anl"] == (375155, 100000)
    assert table["3::HGT::500 mb::anl"] == (475155, 100000)


@pytest.mark.timeout(10)
def test_hrrr_index_parser_max_byte_size(tmp_path):
    # A record spanning more than MAX_BYTE_SIZE must raise.
    idx_path = tmp_path / "fake.grib2.idx"
    idx_path.write_text(
        "1:0:d=2024010100:REFC:entire atmosphere:anl:\n"
        "2:6000000:d=2024010100:TMP:2 m above ground:anl:\n"
    )

    ds = HRRR(cache=False)

    async def _fake_fetch(uri):
        return str(idx_path)

    with patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch):
        with pytest.raises(ValueError):
            asyncio.run(ds._fetch_index("dummy-uri"))


# ----------------------------------------------------------------------
# Mock end-to-end (no network, exercises __call__ path)
# ----------------------------------------------------------------------


@pytest.mark.timeout(15)
@pytest.mark.parametrize("source", ["aws", "google"])
def test_hrrr_call_mock(source, tmp_path, monkeypatch):
    """Exercise the full __call__ path using mocked index + grib decode.

    The real (offline) store construction runs for both the aws (S3) and
    google (GCS) sources; only the remote reads are mocked out.
    """
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::TMP::2 m above ground::anl": (0, 65300),
        "2::HGT::500 mb::anl": (65300, 31215),
    }
    fake_grid = np.random.rand(1059, 1799).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grib2")

    ds = HRRR(source=source, cache=True, verbose=False)

    with (
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.hrrr._decode_hrrr_grib", return_value=fake_grid),
    ):
        data = ds(datetime(2024, 1, 1), ["t2m", "z500"])

    # Store was built lazily for the requested source
    assert isinstance(ds.store, S3Store if source == "aws" else GCSStore)
    assert data.shape == (1, 2, 1059, 1799)
    # t2m is identity, z500 multiplies HGT by 9.81 — both records used the
    # same mock grid, so t2m matches the grid and z500 = grid * 9.81.
    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    np.testing.assert_allclose(
        data.sel(variable="z500").values[0], fake_grid * 9.81, rtol=1e-6
    )


@pytest.mark.timeout(15)
def test_hrrr_fx_call_mock(tmp_path, monkeypatch):
    """Exercise the HRRR_FX __call__ path with multiple lead times."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_index = {
        "1::TMP::2 m above ground::anl": (0, 65300),
        "2::TMP::2 m above ground::6 hour fcst": (65300, 65300),
    }
    fake_grid = np.random.rand(1059, 1799).astype(np.float32)

    async def _fake_fetch_index(uri):
        return fake_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grib2")

    ds = HRRR_FX(source="aws", cache=True, verbose=False)

    with (
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.hrrr._decode_hrrr_grib", return_value=fake_grid),
    ):
        data = ds(
            datetime(2024, 1, 1),
            [timedelta(hours=0), timedelta(hours=6)],
            ["t2m"],
        )

    assert data.shape == (1, 2, 1, 1059, 1799)
    for j in range(2):
        np.testing.assert_allclose(data.values[0, j, 0], fake_grid)


@pytest.mark.timeout(15)
def test_hrrr_missing_variable_mock(tmp_path, monkeypatch):
    # If the requested variable is absent from the .idx, _create_tasks warns
    # and skips it — the output slot keeps its zero initialization.
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Index only has t2m; z500 is missing on purpose.
    partial_index = {"1::TMP::2 m above ground::anl": (0, 65300)}
    fake_grid = np.random.rand(1059, 1799).astype(np.float32) + 1.0

    async def _fake_fetch_index(uri):
        return partial_index

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(tmp_path / "ignored.grib2")

    ds = HRRR(source="aws", cache=True, verbose=False)
    with (
        patch.object(ds, "_fetch_index", side_effect=_fake_fetch_index),
        patch.object(ds, "_fetch_remote_file", side_effect=_fake_fetch_remote_file),
        patch("earth2studio.data.hrrr._decode_hrrr_grib", return_value=fake_grid),
    ):
        data = ds(datetime(2024, 1, 1), ["t2m", "z500"])

    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    # Skipped variable keeps the zero fill.
    assert (data.sel(variable="z500").values == 0.0).all()
