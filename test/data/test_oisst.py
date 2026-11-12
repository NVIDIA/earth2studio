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
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import obstore as obs
import pytest

from earth2studio.data import OISST


def _write_product(path: Path, offset: float) -> None:
    with netCDF4.Dataset(path, "w") as data:
        for name, count in (("time", 1), ("zlev", 1), ("lat", 720), ("lon", 1440)):
            data.createDimension(name, count)
        data.createVariable("lat", "f4", ("lat",))[:] = OISST.LAT[::-1]
        data.createVariable("lon", "f4", ("lon",))[:] = OISST.LON
        sst = data.createVariable("sst", "f4", ("time", "zlev", "lat", "lon"))
        sst.units = "Celsius"
        sst[:] = np.arange(720, dtype=np.float32)[None, None, :, None] / 10 + offset


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1),
        [
            datetime(year=2024, month=1, day=1),
            datetime(year=2024, month=1, day=2, hour=12),
        ],
    ],
)
def test_oisst_fetch(time):
    data = OISST(cache=False)(time, "sst")
    times = [time] if isinstance(time, datetime) else time
    assert data.shape == (len(times), 1, 720, 1440)
    assert data.dims == ("time", "variable", "lat", "lon")
    values = data.values
    assert np.isnan(values).any()  # land
    assert 260 < np.nanmin(values) and np.nanmax(values) < 310
    expected = [np.datetime64(t.date(), "ns") for t in times]
    np.testing.assert_array_equal(data.sst_date.values, expected)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_oisst_cache(cache):
    ds = OISST(cache=cache)
    data = ds(datetime(2024, 1, 1), "sst")
    assert data.shape == (1, 1, 720, 1440)
    assert Path(ds.cache).is_dir() == cache
    data = ds(datetime(2024, 1, 1), ["sst"])
    assert data.shape == (1, 1, 720, 1440)
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_oisst_call_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    final, preliminary = tmp_path / "final.nc", tmp_path / "preliminary.nc"
    _write_product(final, 0.0)
    _write_product(preliminary, 1.0)
    seen = []

    async def fetch(store, key, cache, **kwargs):
        seen.append(key.rsplit(".", 2)[-2])
        # 2024-01-02 is unpublished; 2024-01-01 has both; 2023-12-31 is preliminary
        if "20240102" in key or ("20231231" in key and "preliminary" not in key):
            raise FileNotFoundError(key)
        return str(preliminary if "preliminary" in key else final)

    monkeypatch.setattr("earth2studio.data.oisst.obstore_fetch_to_cache", fetch)
    ds = OISST(lag_days=1, lookback_days=2, verbose=False)

    # Falls back a day and prefers the final product over the preliminary one
    result = ds(datetime(2024, 1, 3, 6), "sst")
    assert result.shape == (1, 1, 720, 1440)
    assert list(result.coords["variable"].values) == ["sst"]
    np.testing.assert_array_equal(
        result.values[0, 0, :, 0],
        np.arange(720, dtype=np.float32)[::-1] / 10 + np.float32(273.15),
    )
    assert result.sst_date.values[0] == np.datetime64("2024-01-01", "ns")
    assert seen == [
        "20240102",
        "20240102_preliminary",
        "20240101",
    ]

    # Uses the preliminary product when the final one is not yet published
    seen.clear()
    result = OISST(verbose=False)(datetime(2023, 12, 31), "sst")
    assert seen == ["20231231", "20231231_preliminary"]
    np.testing.assert_allclose(
        result.values[0, 0, :, 0],
        np.arange(720, dtype=np.float32)[::-1] / 10 + np.float32(274.15),
    )

    # No product within the lookback window
    with pytest.raises(FileNotFoundError):
        OISST(verbose=False)(datetime(2024, 1, 2), "sst")


def test_oisst_overlapping_fetches_share_the_temporary_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    product = tmp_path / "product.nc"
    _write_product(product, 0.0)
    ds = OISST(cache=False, verbose=False)
    first_done = asyncio.Event()

    async def fetch(store, key, cache, **kwargs):
        # The second request reads its file only after the first request finished
        if "20240102" in key:
            await first_done.wait()
        path = Path(cache) / key.replace("/", "_")
        shutil.copy(product, path)
        return str(path)

    monkeypatch.setattr("earth2studio.data.oisst.obstore_fetch_to_cache", fetch)

    async def first():
        try:
            return await ds.fetch(datetime(2024, 1, 1), "sst")
        finally:
            first_done.set()

    async def both():
        return await asyncio.gather(first(), ds.fetch(datetime(2024, 1, 2), "sst"))

    results = asyncio.run(both())
    assert [r.sst_date.values[0] for r in results] == [
        np.datetime64("2024-01-01", "ns"),
        np.datetime64("2024-01-02", "ns"),
    ]
    # The last call to finish removes the temporary cache
    assert not Path(ds.cache).exists()


def test_oisst_retries_obstore_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    product = tmp_path / "product.nc"
    _write_product(product, 0.0)
    attempts = []

    async def fetch(store, key, cache, **kwargs):
        attempts.append(key)
        if len(attempts) == 1:
            raise obs.exceptions.GenericError("connection reset")
        return str(product)

    monkeypatch.setattr("earth2studio.data.oisst.obstore_fetch_to_cache", fetch)
    result = OISST(verbose=False, retries=1)(datetime(2024, 1, 1), "sst")
    assert result.shape == (1, 1, 720, 1440)
    assert len(attempts) == 2


def test_oisst_exceptions():
    with pytest.raises(ValueError):
        OISST(lag_days=-1)
    with pytest.raises(ValueError):
        OISST(lookback_days=0)
    with pytest.raises(KeyError):
        OISST()(datetime(2024, 1, 1), "t2m")
    with pytest.raises(ValueError):
        OISST()(datetime(1980, 1, 1), "sst")


def test_oisst_available():
    assert OISST.available(datetime(2024, 1, 1))
    assert OISST.available(np.datetime64("1981-09-01"))
    assert not OISST.available(datetime(1981, 8, 31))
    tomorrow = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=2)
    assert not OISST.available(tomorrow)
