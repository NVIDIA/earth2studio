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

"""Work construction, distribution, seeding, progress, and rank helpers.

Scaffold only: signatures follow ``dev/spec/EXECUTION_CONTRACT_SPEC.md`` so
other modules can import them while this module is implemented on its own
track. The rank helpers are already correct for a single process; everything
else raises :class:`NotImplementedError`.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from earth2studio.run.runner import WorkItem

T = TypeVar("T")


def work_items(
    times: Sequence[np.datetime64],
    horizon: np.timedelta64,
    members: int | Sequence[int] | None = None,
    member_batch: int = 1,
) -> list[WorkItem]:
    """Build a time-major list of work items."""
    raise NotImplementedError


def member_seed(base: int, time: np.datetime64, member_id: int) -> int:
    """Derive a per-member seed from the base seed, time, and member ID.

    FNV-1a 64 over ``struct.pack(">qqq", base, seconds, member_id)``, with
    ``time`` in integer seconds since the epoch, modulo ``2**63``. Depends only
    on its arguments, never on batching or world size.
    """
    seconds = int(np.datetime64(time, "s").astype("int64"))
    digest = 0xCBF29CE484222325
    for byte in struct.pack(">qqq", int(base), seconds, int(member_id)):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest % 2**63


def distribute_work(items: Sequence[T], rank: int, world_size: int) -> list[T]:
    """Split items into one contiguous, balanced slice per rank."""
    raise NotImplementedError


class ProgressStore:
    """Record finished work items so a run can resume."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        tags: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        raise NotImplementedError

    def open(self, manifest: Mapping[str, Any]) -> None:
        """Open the store, compact logs, and check the manifest. Collective."""
        raise NotImplementedError

    def remaining(self, items: Sequence[WorkItem]) -> list[WorkItem]:
        """Return the items that are not yet done."""
        raise NotImplementedError

    def is_done(self, item: WorkItem) -> bool:
        """Whether every unit of an item is recorded."""
        raise NotImplementedError

    def mark_done(self, item: WorkItem) -> None:
        """Record an item as finished."""
        raise NotImplementedError

    def clear(self) -> None:
        """Delete the recorded progress and manifest."""
        raise NotImplementedError


def rank_info() -> tuple[int, int]:
    """Return ``(rank, world_size)``, or ``(0, 1)`` without a process group."""
    return (0, 1)


def rank0_first(fn: Callable[[], T]) -> T:
    """Run ``fn`` on rank 0, barrier, then run it on the other ranks."""
    return fn()


def broadcast(obj: T | None) -> T:
    """Return rank 0's object on every rank."""
    return obj  # type: ignore[return-value]


def barrier() -> None:
    """Synchronize every rank."""
    return None


def raise_if_any_failed(error: Exception | None) -> None:
    """Raise on every rank if any rank failed. Collective."""
    if error is not None:
        raise error
