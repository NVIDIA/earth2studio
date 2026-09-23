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

"""Fixed-reference tests for Lambert-Conformal wind rotation."""

from __future__ import annotations

import pytest
from src.shared.wind_rotation import rotate_lcc_grid_to_earth
from src.stormcast.collector import _HRRR_CONE_N, _HRRR_LON0_DEG


def _rot(u: float, v: float, lon: float) -> tuple[float, float]:
    u_e, v_n = rotate_lcc_grid_to_earth(u, v, lon, _HRRR_LON0_DEG, _HRRR_CONE_N)
    return float(u_e), float(v_n)


@pytest.mark.parametrize(
    ("longitude", "expected"),
    [
        pytest.param(_HRRR_LON0_DEG, (1.0, 0.0), id="central"),
        pytest.param(280.0, (0.982, -0.189), id="east"),
        pytest.param(245.0, (0.982, 0.189), id="west"),
    ],
)
def test_rotation_matches_fixed_references(
    longitude: float, expected: tuple[float, float]
) -> None:
    assert _rot(1.0, 0.0, longitude) == pytest.approx(expected, abs=1e-3)
