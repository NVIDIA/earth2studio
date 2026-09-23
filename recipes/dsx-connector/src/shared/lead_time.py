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

"""Convert an Earth2Studio forecast lead time to non-negative whole seconds.

Workflows receive lead times through an Earth2Studio ``lead_time`` coordinate.
This module converts that coordinate to a non-negative number of whole seconds after the forecast's
initial-condition time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def lead_seconds_from_coords(coords: dict[str, Any]) -> int:
    """Return one forecast lead time as a whole number of seconds.

    Each write from the built-in workflows contains one lead time. Multiple lead times are rejected
    because the function could not determine which one to return. Sub-second values are also
    rejected rather than rounded because forecast lead times are stored here as integer seconds.

    Parameters
    ----------
    coords : dict[str, Any]
        Earth2Studio coordinates containing one NumPy timedelta under ``lead_time``.

    Returns
    -------
    int
        Non-negative number of seconds after the forecast's initial-condition time.

    Raises
    ------
    KeyError
        If ``coords`` does not contain ``lead_time``.
    ValueError
        If the coordinate does not contain exactly one NumPy timedelta that is not ``NaT``, is
        non-negative, and represents a whole number of seconds.
    """
    lead_time_values = np.asarray(coords["lead_time"])
    if lead_time_values.size != 1:
        raise ValueError(
            f"expected a single lead_time per write, got size {lead_time_values.size}"
        )
    if not np.issubdtype(lead_time_values.dtype, np.timedelta64):
        raise ValueError("lead_time must be a NumPy timedelta")
    lead_time = np.timedelta64(lead_time_values.reshape(())[()])
    if np.isnat(lead_time):
        raise ValueError("lead_time must not be NaT")
    if lead_time < np.timedelta64(0, "s"):
        raise ValueError(f"lead_time {lead_time} must not be negative")
    if lead_time % np.timedelta64(1, "s") != np.timedelta64(0, "s"):
        raise ValueError(f"lead_time {lead_time} is not a whole number of seconds")
    return int(lead_time / np.timedelta64(1, "s"))
