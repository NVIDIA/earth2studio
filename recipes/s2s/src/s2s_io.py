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
from typing import Any, TypeVar

import xarray as xr

from earth2studio.io import IOBackend

T = TypeVar("T")


class BufferedIOBackend:
    """IO adapter writing on a background thread, with a bounded buffer.

    Each field is copied to the CPU on the calling thread, then written by a
    single worker thread, so disk writes overlap inference. ``buffer_size``
    bounds the outstanding fields (including the active write); a full buffer
    blocks the caller. ``flush`` waits for every write, so ``Pipeline`` records
    an item as done only once its output is on disk, and writer errors surface
    on the next call.

    Parameters
    ----------
    backend : IOBackend
        Backend that performs the writes.
    buffer_size : int, optional
        Maximum outstanding fields, by default 2
    """

    def __init__(self, backend: IOBackend, buffer_size: int = 2) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.backend = backend
        self._capacity = buffer_size
        self._pending: deque[Future] = deque()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2s-io")
        self._closed = False

    @property
    def write_blocks(self) -> Any:
        """Stored layout of the wrapped backend, for Pipeline's distributed check."""
        return getattr(self.backend, "write_blocks", None)

    @property
    def store(self) -> Any:
        """Zarr store of the wrapped backend, if any."""
        return getattr(self.backend, "store", None)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Output backend is closed")

    def _drain(self) -> None:
        # Keep a failed future queued so later flush/close calls re-raise it
        while self._pending:
            self._pending[0].result()
            self._pending.popleft()

    def _call(self, operation: Callable[[], T]) -> T:
        self._check_open()
        self._drain()
        return self._worker.submit(operation).result()

    def add_array(self, template: xr.DataArray) -> None:
        """Create or validate arrays once earlier writes have finished."""
        self._call(lambda: self.backend.add_array(template))

    def write(self, x: xr.DataArray) -> None:
        """Copy ``x`` to the CPU and queue its write, waiting for buffer space."""
        self._check_open()
        while self._pending and (
            len(self._pending) >= self._capacity or self._pending[0].done()
        ):
            self._pending[0].result()
            self._pending.popleft()
        snapshot = x.e2s.to_backend("numpy", device="cpu").copy(deep=True)
        self._pending.append(self._worker.submit(self.backend.write, snapshot))

    def flush(self) -> None:
        """Wait for queued writes and flush the wrapped backend."""
        self._call(self.backend.flush)

    def close(self) -> None:
        """Finish queued writes and close, also after a write failure."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._worker.submit(self.backend.close).result()
            finally:
                self._closed = True
                self._worker.shutdown(wait=True)
