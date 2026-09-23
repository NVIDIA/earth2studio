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


"""StormCast conditioning coverage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from src.stormcast.coverage import conditioning_covers, conditioning_hours


@pytest.mark.parametrize(("nsteps", "expected_hours"), [(2, 14), (24, 36)])
def test_conditioning_hours(nsteps: int, expected_hours: int) -> None:
    assert conditioning_hours(nsteps) == expected_hours


@pytest.mark.parametrize("nsteps", [2, 24])
def test_conditioning_covers_boundary(nsteps: int) -> None:
    cyc = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

    assert conditioning_covers(cyc, cyc + timedelta(hours=12), nsteps)
    assert not conditioning_covers(cyc, cyc + timedelta(hours=13), nsteps)
