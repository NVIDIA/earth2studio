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


"""Shared lead-time conversion."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.lead_time import lead_seconds_from_coords


def test_lead_seconds_scalar_and_array() -> None:
    assert lead_seconds_from_coords({"lead_time": np.timedelta64(6, "h")}) == 21600
    assert (
        lead_seconds_from_coords({"lead_time": np.array([np.timedelta64(3600, "s")])})
        == 3600
    )
    assert lead_seconds_from_coords({"lead_time": np.timedelta64(10, "m")}) == 600


@pytest.mark.parametrize(
    "lead_time",
    [
        pytest.param(
            np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
            id="multiple",
        ),
        pytest.param(np.timedelta64(1500, "ms"), id="sub-second"),
        pytest.param(np.timedelta64(-1, "s"), id="negative"),
        pytest.param(np.timedelta64("NaT", "s"), id="nat"),
        pytest.param(1, id="non-timedelta"),
    ],
)
def test_lead_seconds_rejects_invalid_values(lead_time: object) -> None:
    with pytest.raises(ValueError):
        lead_seconds_from_coords({"lead_time": lead_time})
