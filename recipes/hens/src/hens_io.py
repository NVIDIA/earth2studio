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

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

import xarray as xr
from xarray.backends.netCDF4_ import NETCDF4_PYTHON_LOCK

from earth2studio.io import IOBackend

T = TypeVar("T")


def _locked_call(operation: Callable[[], T]) -> T:
    # netCDF-C is not thread safe, even for separate files. Share xarray's
    # lock with auxiliary track/CorrDiff exports on the inference thread.
    with NETCDF4_PYTHON_LOCK:
        return operation()


class BufferedIOBackend:
    """Recipe-local IO adapter with a bounded buffer of owned CPU DataArrays.

    A single worker owns the backend, including construction and teardown.
    ``buffer_size`` bounds outstanding fields (including the active write).
    Transfers to CPU finish before enqueueing; disk writes overlap inference.
    Flush is a barrier, so Pipeline records progress only after durable output.
    """

    def __init__(self, factory: Callable[[], IOBackend], buffer_size: int = 2) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self._capacity = buffer_size
        self._pending: deque[Future] = deque()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hens-io")
        self._closed = False
        try:
            self.backend = self._worker.submit(_locked_call, factory).result()
        except BaseException:
            self._worker.shutdown(wait=True)
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Output backend is closed")

    def _drain(self) -> None:
        # Retain failed futures so subsequent flush/close cannot hide errors.
        while self._pending:
            self._pending[0].result()
            self._pending.popleft()

    def _call(self, operation: Callable[[], T]) -> T:
        self._check_open()
        self._drain()
        return self._worker.submit(_locked_call, operation).result()

    def add_array(self, template: xr.DataArray) -> None:
        """Allocate/validate storage on the IO worker before writes start."""
        self._call(lambda: self.backend.add_array(template))

    def write(self, x: xr.DataArray) -> None:
        """Buffer a detached field, waiting for capacity before copying it."""
        self._check_open()
        while self._pending and (
            len(self._pending) >= self._capacity or self._pending[0].done()
        ):
            self._pending[0].result()
            self._pending.popleft()
        snapshot = x.e2s.to_backend("numpy", device="cpu").copy(deep=True)
        self._pending.append(
            self._worker.submit(_locked_call, lambda: self.backend.write(snapshot))
        )

    def flush(self) -> None:
        """Wait for buffered writes and flush the underlying store."""
        self._call(self.backend.flush)

    def close(self) -> None:
        """Drain writes and release resources, including after a write failure."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._worker.submit(_locked_call, self.backend.close).result()
            finally:
                self._closed = True
                self._worker.shutdown(wait=True)
