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

"""Activation-time schedules.

All schedule algebra lives here: the protocol and every concrete schedule. Both
single-model plans and the component graph use schedules, so they sit below
both; this module imports nothing else from :mod:`earth2studio.run`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, cast

import numpy as np


class Schedule(Protocol):
    """Produce activation times lazily within a bounded interval.

    Plans fold :meth:`fingerprint` into their identity. Supervision supplies only
    the work horizon and never interprets a schedule.
    """

    def iter_between(
        self,
        reference_time: np.datetime64,
        start: np.datetime64,
        stop: np.datetime64,
    ) -> Iterator[np.datetime64]:
        """Yield activations in ascending order within ``[start, stop)``."""
        ...

    def fingerprint(self) -> str:
        """Return stable identity for the unexpanded schedule declaration."""
        ...


class FixedCadence:
    """Activate every ``step`` from the reference time, shifted by ``offset``.

    Parameters
    ----------
    step : np.timedelta64
        Positive interval between activations.
    offset : np.timedelta64, optional
        Shift of the first activation from the reference time, by default zero.
    """

    def __init__(
        self,
        step: np.timedelta64,
        offset: np.timedelta64 = np.timedelta64(0, "s"),
    ) -> None:
        if step <= np.timedelta64(0, "s"):
            raise ValueError(f"FixedCadence step must be positive, got {step}")
        self.step: np.timedelta64 = step.astype("timedelta64[s]")
        self.offset: np.timedelta64 = offset.astype("timedelta64[s]")

    def iter_between(
        self,
        reference_time: np.datetime64,
        start: np.datetime64,
        stop: np.datetime64,
    ) -> Iterator[np.datetime64]:
        """Yield activations in ascending order within ``[start, stop)``."""
        first = reference_time + self.offset
        skipped = 0
        if start > first:
            # numpy stubs type datetime64 differences as datetime64
            elapsed = cast(np.timedelta64, start - first)
            skipped = int(np.ceil(elapsed / self.step))
        current = first + skipped * self.step
        while current < stop:
            yield current
            current = current + self.step

    def fingerprint(self) -> str:
        """Return stable identity for the unexpanded declaration."""
        return f"fixed:{self.step.astype(int)}s+{self.offset.astype(int)}s"
