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
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from earth2studio.data import AIFS_ENS_FX, AIFS_FX, IFS, IFS_ENS, IFS_ENS_FX, IFS_FX
from earth2studio.data import ecmwf as ecmwf_module


@pytest.mark.parametrize(
    "source_class, model",
    [
        (IFS, "ifs"),
        (IFS_FX, "ifs"),
        (IFS_ENS, "ifs"),
        (IFS_ENS_FX, "ifs"),
        (AIFS_FX, "aifs-single"),
        (AIFS_ENS_FX, "aifs-ens"),
    ],
)
@pytest.mark.parametrize("source", ["aws", "google"])
def test_opendata_client_kwargs(source_class, model, source, monkeypatch):
    client_factory = Mock(return_value=Mock())
    monkeypatch.setattr(ecmwf_module.opendata, "Client", client_factory)

    source_class(
        source=source,
        client_kwargs={"preserve_request_order": True},
    )

    client_factory.assert_called_once_with(
        source=source,
        model=model,
        maximum_retries=source_class._MAX_RETRIES,
        retry_after=source_class._RETRY_AFTER,
        preserve_request_order=True,
    )


def test_opendata_client_retry_kwargs_override(monkeypatch):
    client_factory = Mock(return_value=Mock())
    monkeypatch.setattr(ecmwf_module.opendata, "Client", client_factory)

    IFS(client_kwargs={"maximum_retries": 2, "retry_after": 1})

    client_factory.assert_called_once_with(
        source="aws", model="ifs", maximum_retries=2, retry_after=1
    )


def test_accumulation_alias_tasks_use_previous_lead_time():
    ds = object.__new__(AIFS_FX)
    time = [datetime(2025, 12, 31, 0)]
    lead_time = [timedelta(hours=12)]

    tasks = asyncio.run(ds._create_tasks(time, lead_time, ["sf06", "ro06", "t2m"]))

    assert tasks[0].variable == "sf"
    assert tasks[0].lead_time == timedelta(hours=12)
    assert tasks[0].previous_lead_time == timedelta(hours=6)
    assert tasks[1].variable == "rowe"
    assert tasks[1].previous_lead_time == timedelta(hours=6)
    assert tasks[2].variable == "2t"
    assert tasks[2].previous_lead_time is None


def test_accumulation_alias_tasks_reject_short_lead_time():
    ds = object.__new__(IFS_FX)

    with pytest.raises(ValueError, match="too short"):
        asyncio.run(
            ds._create_tasks(
                [datetime(2025, 12, 31, 0)],
                [timedelta(hours=0)],
                ["sf06"],
            )
        )


def test_blocking_download_does_not_stall_event_loop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    started = threading.Event()

    class SlowClient:
        def retrieve(self, request: dict[str, Any]) -> None:
            started.set()
            assert release.wait(5), "Download blocked the event loop"
            pathlib.Path(request["target"]).write_bytes(b"data")

    ds = object.__new__(IFS)
    ds.client = SlowClient()
    ds._executor = ThreadPoolExecutor(1)
    ds._fc_type, ds._members, ds._cache = "fc", [0], True
    ds._model, ds._tmp_cache_hash = "IFS", None
    monkeypatch.setattr(IFS, "cache", property(lambda self: str(tmp_path)))

    async def run() -> str:
        task = asyncio.create_task(
            ds._download_ifs_grib_cached(
                datetime(2026, 7, 1), timedelta(0), "2t", "sfc", "sfc"
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        # Only a responsive event loop can release the worker before its timeout.
        release.set()
        return await task

    try:
        path = asyncio.run(run())
    finally:
        release.set()
        ds._executor.shutdown(wait=True)
    assert pathlib.Path(path).read_bytes() == b"data"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("fail_download", [False, True])
def test_cancelled_download_cleans_up_after_worker_finishes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fail_download: bool
) -> None:
    started = threading.Event()
    release = threading.Event()
    targets: list[pathlib.Path] = []

    class SlowClient:
        def retrieve(self, request: dict[str, Any]) -> None:
            target = pathlib.Path(request["target"])
            targets.append(target)
            started.set()
            assert release.wait(5), "Test did not release the download worker"
            target.write_bytes(b"data")
            if fail_download:
                raise RuntimeError("Download failed")

    ds = object.__new__(IFS)
    ds.client = SlowClient()
    ds._executor = ThreadPoolExecutor(1)
    ds._fc_type, ds._members = "fc", [0]
    monkeypatch.setattr(IFS, "cache", property(lambda self: str(tmp_path)))

    async def run() -> None:
        task = asyncio.create_task(
            ds._download_ifs_grib_cached(
                datetime(2026, 7, 1), timedelta(0), "2t", "sfc", "sfc"
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        assert not targets[0].exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(run())
    finally:
        release.set()
        ds._executor.shutdown(wait=True)

    assert not list(tmp_path.glob("*.part"))
    cached_files = list(tmp_path.iterdir())
    if fail_download:
        assert not cached_files
    else:
        assert len(cached_files) == 1
        assert cached_files[0].read_bytes() == b"data"


def now6h():
    """Get closest 6 hour timestamp"""
    nt = datetime.now()
    delta_hr = nt.hour % 6
    return datetime(nt.year, nt.month, nt.day, nt.hour - delta_hr)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_fetch(time, variable):
    ds = IFS()
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
    assert np.array_equal(data.coords["variable"].values, np.array(variable))

    ds = IFS_FX()
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert np.allclose(data.values, data_fx.values[:, 0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["sp"]])
@pytest.mark.parametrize("member", [2])
def test_ifs_ens_fetch(time, variable, member):
    ds = IFS_ENS(cache=False, member=member)
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
    assert np.array_equal(data.coords["variable"].values, np.array(variable))

    ds = IFS_ENS_FX(cache=False, member=member)
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.allclose(data.values, data_fx.values[:, 0])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=0),
        [
            timedelta(hours=3),
            timedelta(hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_ens_fx_fetch(time, lead_time, variable):
    ds = IFS_ENS_FX(cache=False)
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
    "lead_time",
    [
        timedelta(hours=0),
        [
            timedelta(hours=6),
            timedelta(hours=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["tcc", "tp"]])
@pytest.mark.parametrize("member", [0, 1])
def test_aifs_ens_fx_fetch(lead_time, variable, member):
    time = now6h() - timedelta(hours=12)
    ds = AIFS_ENS_FX(cache=False, member=member)
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
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([now6h() - timedelta(days=2)], dtype="datetime64"),
    ],
)
@pytest.mark.parametrize("variable", [["u10m", "tcwv"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ifs_cache(time, variable, cache):
    ds = IFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [IFS, IFS_ENS])
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_time_available(data_source, time, variable):
    with pytest.raises(ValueError):
        ds = IFS(source="ecmwf")
        ds(time, ["msl"])

        ds = IFS_FX(source="ecmwf")
        ds(time, [timedelta(hours=0)], ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=1),
        timedelta(hours=147),
        timedelta(hours=366),
    ],
)
def test_ifs_leadtime_available(lead_time):
    with pytest.raises(ValueError):
        time = now6h() - timedelta(days=2)
        ds = IFS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])

        ds = IFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", ["msl"])
def test_aifs_time_available(time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = AIFS_FX(source="ecmwf")
        ds(time, lead_time, variable)

    with pytest.raises(ValueError):
        ds = AIFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, variable)
