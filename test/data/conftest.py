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

"""Shared fixtures for data source tests.

This conftest addresses the problem of background threads (fsspec IO loop,
ThreadPoolExecutor workers) persisting after a pytest-timeout fires.  Those
threads block on queues and event loops, preventing the test process from
exiting cleanly within CI time limits.

Strategy:
- Session-end: shut down all ThreadPoolExecutor instances and force process
  exit if non-daemon threads remain stuck.

We intentionally do NOT shut down ThreadPoolExecutor instances or the fsspec
IO loop during individual test teardown because:
- Libraries like dask, zarr, and asyncio lazily create module-level singleton
  executors that are reused across tests.
- fsspec-based filesystem instances (s3fs, etc.) cache a reference to the
  global IO loop at construction time; closing the loop mid-session breaks
  any module-level filesystem objects.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime

import numpy as np
import pytest
from loguru import logger


@pytest.fixture
def mock_statistic_grib(monkeypatch, tmp_path):
    """Tiny GRIB fields whose values encode valid time; retain real record lookup."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    def install(ds, missing=()):
        calls = []
        name = type(ds).__module__.rsplit(".", 1)[-1]
        for axis in (("GFS_LAT", "GFS_LON") if name == "gfs" else ("HRRR_Y", "HRRR_X")):
            monkeypatch.setattr(ds, axis, np.arange(2))
        if name == "hrrr":
            monkeypatch.setattr(
                ds, "grid", lambda: (np.zeros((2, 2)), np.zeros((2, 2)))
            )

        def hours(uri):
            day = re.search(r"(?:gfs|hrrr)\.(\d{8})", uri)[1]
            cycle = int(re.search(r"\.t(\d{2})z", uri)[1])
            lead = int(re.search(r"f(\d+)(?:\.|$)", uri)[1])
            return (
                datetime.strptime(day, "%Y%m%d") - datetime(2024, 1, 1)
            ).total_seconds() / 3600 + cycle + lead, lead

        async def index(uri):
            calls.append(uri)
            value, lead = hours(uri)
            step = "anl" if lead == 0 else f"{lead} hour fcst"
            if value in missing:
                return {}
            records = {
                f"1::TMP::2 m above ground::{step}": (0, 1),
                f"2::HGT::500 mb::{step}": (1, 1),
            }
            if name == "gfs" and lead:
                start = 6 * ((lead - 1) // 6)
                records[f"3::APCP::surface::{start}-{lead} hour acc fcst"] = (2, 1)
            return records

        async def remote(uri, **kwargs):
            return uri, kwargs["byte_offset"]

        def decode(record):
            uri, offset = record
            value, lead = hours(uri)
            if offset == 2:
                start = 6 * ((lead - 1) // 6)
                # Hourly rainfall increases with lead, in kg/m2; buckets reset at 6h.
                return np.full((2, 2), (lead * (lead + 1) - start * (start + 1)) / 2)
            return np.array([[value, np.nan if value % 2 else value], [np.nan, value]])

        monkeypatch.setattr(ds, "_fetch_index", index)
        monkeypatch.setattr(ds, "_fetch_remote_file", remote)
        monkeypatch.setattr(f"earth2studio.data.{name}.decode_grib_message", decode)
        return calls

    return install


@pytest.fixture
def mock_statistic_zarr(monkeypatch, tmp_path):
    """Small surface Zarr fields, shared by ERA5 statistics tests."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    def install(ds):
        reads = []

        class Array:
            shape = (10**6, 2, 2)

            async def getitem(self, index):
                reads.append(int(index))
                return np.full((2, 2), float(index))

        class Group:
            async def get(self, name):
                return Array()

        prefix = "ARCO" if type(ds).__name__ == "ARCO_ERA5" else "WB2_ERA5"
        for axis in ("LAT", "LON"):
            monkeypatch.setattr(ds, f"{prefix}_{axis}", np.arange(2))
        ds.zarr_group = ds.ml_zarr_group = Group()
        return reads

    return install


@pytest.fixture
def caplog(caplog):
    """Extend caplog to also capture loguru log records."""
    handler_id = logger.add(caplog.handler, format="{message}", level=0)
    with caplog.at_level(logging.DEBUG):
        yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True, scope="session")
def _force_exit_on_stuck_threads():
    """Force process exit if non-daemon threads are stuck at session end.

    When data source tests time out, background ThreadPoolExecutor workers
    may remain alive (stuck in blocking network I/O).  These are non-daemon
    threads, so Python's shutdown sequence would wait for them indefinitely.

    This session-scoped fixture:
    1. Shuts down all live ThreadPoolExecutor instances (safe at session end)
    2. Waits briefly for threads to exit
    3. If threads remain stuck, calls os._exit() to force process termination
    """
    import concurrent.futures
    import gc
    import os
    import warnings

    # Record threads that existed before the test session (main, etc.)
    initial_threads = {t.ident for t in threading.enumerate()}

    yield

    # Check for leftover non-daemon threads
    main_thread = threading.main_thread()
    stuck_threads = [
        t
        for t in threading.enumerate()
        if (
            t is not main_thread
            and t.ident not in initial_threads
            and not t.daemon
            and t.is_alive()
        )
    ]

    if not stuck_threads:
        return

    # Shut down all executors — no more tests will run after this
    gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if (
                    isinstance(obj, concurrent.futures.ThreadPoolExecutor)
                    and not obj._shutdown
                ):
                    obj.shutdown(wait=False, cancel_futures=True)
            except (ReferenceError, TypeError):
                pass

    # Wait briefly for threads to respond to shutdown sentinels
    for thread in stuck_threads:
        if thread.is_alive():
            thread.join(timeout=2.0)

    # If any non-daemon threads are still alive, force exit to prevent hang
    remaining = [
        t
        for t in threading.enumerate()
        if (
            t is not main_thread
            and t.ident not in initial_threads
            and not t.daemon
            and t.is_alive()
        )
    ]
    if remaining:
        os._exit(0)
