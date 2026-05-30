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

The ``_cleanup_background_threads`` autouse fixture tears down the global
fsspec IO thread and shuts down orphaned ThreadPoolExecutor instances after
each test, ensuring no leftover non-daemon threads accumulate and block
process exit.
"""

from __future__ import annotations

import concurrent.futures
import gc
import threading

import pytest


def _shutdown_orphaned_executors(
    pre_test_threads: set[int],
    pre_test_executors: set[int],
) -> None:
    """Find and shut down ThreadPoolExecutor instances created during a test.

    Only shuts down executors that were NOT alive before the test started,
    avoiding interference with shared/global executors (e.g. dask global
    thread pool, pytest-asyncio worker pool).

    Uses gc to discover live executor objects and calls ``shutdown(wait=False)``
    on each new one.  This sends None sentinels to idle workers (blocked on
    ``work_queue.get(block=True)``) causing them to exit their main loop.
    """
    import warnings

    gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if (
                    isinstance(obj, concurrent.futures.ThreadPoolExecutor)
                    and not obj._shutdown
                    and id(obj) not in pre_test_executors
                ):
                    obj.shutdown(wait=False, cancel_futures=True)
            except (ReferenceError, TypeError):
                # Object may have been collected or is a weakref proxy
                pass

    # Give idle workers a moment to pick up sentinels and exit
    for thread in threading.enumerate():
        if (
            thread.ident not in pre_test_threads
            and not thread.daemon
            and thread.is_alive()
        ):
            thread.join(timeout=1.0)


@pytest.fixture(autouse=True)
def _cleanup_background_threads():
    """Shut down fsspec's global IO loop and orphaned executor threads after each test.

    Data source tests use fsspec for HTTP access which creates:
    - A global ``fsspecIO`` daemon thread running an asyncio event loop
    - ``ThreadPoolExecutor`` workers (non-daemon) spawned by libraries like
      intake-esm, xarray, or dask for parallel I/O operations

    When a test times out, these threads persist.  The non-daemon executor
    workers are particularly problematic because Python's shutdown sequence
    waits for them indefinitely, preventing the test process from exiting.

    This fixture:
    1. Stops the fsspec IO loop and resets the global state
    2. Finds orphaned ThreadPoolExecutor instances via gc and shuts them down
    """
    import warnings

    # Snapshot threads before the test runs
    pre_test_threads = {t.ident for t in threading.enumerate()}

    # Snapshot existing executors so we don't shut down shared/global ones
    gc.collect()
    pre_test_executors: set[int] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if isinstance(obj, concurrent.futures.ThreadPoolExecutor):
                    pre_test_executors.add(id(obj))
            except (ReferenceError, TypeError):
                pass

    yield

    # --- Tear down fsspec's global IO loop ---
    try:
        import fsspec.asyn
    except ImportError:
        _shutdown_orphaned_executors(pre_test_threads, pre_test_executors)
        return

    loop = fsspec.asyn.loop[0]
    iothread = fsspec.asyn.iothread[0]

    if loop is not None:
        # Stop the event loop (must be scheduled from within its own thread)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            # Loop already closed or not running
            pass

        # Wait for the IO thread to finish
        if iothread is not None and iothread.is_alive():
            iothread.join(timeout=2.0)

        # Close the loop to release resources
        if not loop.is_closed():
            try:
                loop.close()
            except RuntimeError:
                pass

        # Reset global state so the next test gets a fresh loop
        fsspec.asyn.loop[0] = None
        fsspec.asyn.iothread[0] = None

    # --- Shut down orphaned ThreadPoolExecutor workers ---
    _shutdown_orphaned_executors(pre_test_threads, pre_test_executors)
