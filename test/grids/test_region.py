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

import dataclasses

import numpy as np
import pytest

from earth2studio.grids import Region


@pytest.mark.parametrize(
    "region",
    [
        Region.box(-125, 25, -65, 50),
        Region.box(170, -10, -170, 10),
        Region.band(-20, 20),
        Region.rectangle(45, 10, 5e5, 2e5, 30),
        Region.circle(45, 10, 3e5),
    ],
)
def test_region_value_type(region):
    rebuilt = Region(region.kind, region.params)
    assert rebuilt == region and hash(rebuilt) == hash(region)
    assert all(isinstance(value, float) for value in region.params)
    with pytest.raises(dataclasses.FrozenInstanceError):
        region.params = ()


def test_region_box():
    assert Region.box(-125, 25, -65, 50).params == (-125.0, 25.0, -65.0, 50.0)
    # Longitudes normalize; west > east crosses the antimeridian
    assert Region.box(170, -10, 190, 10).params == (170.0, -10.0, -170.0, 10.0)
    assert Region.box(235, 25, 295, 50) == Region.box(-125, 25, -65, 50)
    assert Region.box(180, 0, 180, 1).params == (-180.0, 0.0, -180.0, 1.0)
    # In-range coordinates are never rounded
    assert Region.box(0.1 + 0.2, 0, 1, 1).params[0] == 0.1 + 0.2
    with pytest.raises(ValueError, match="Region.band"):
        Region.box(-180, -90, 180, 90)
    with pytest.raises(ValueError, match="full longitude"):
        Region.box(0, 0, 360, 10)
    with pytest.raises(ValueError, match="south"):
        Region.box(0, 10, 1, 0)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        Region.box(0, -91, 1, 0)
    with pytest.raises(ValueError, match="finite"):
        Region.box(0, 0, np.nan, 1)
    with pytest.raises(ValueError, match="requires"):
        Region("box", (0, 0, 1))


def test_region_band():
    assert Region.band(-20, 20).params == (-20.0, 20.0)
    assert Region.band(60, 90).params == (60.0, 90.0)
    assert Region.band(-90, 90) != Region.box(-180, -90, 179, 90)
    with pytest.raises(ValueError, match="south"):
        Region.band(20, -20)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        Region.band(-20, 91)
    with pytest.raises(ValueError, match="requires"):
        Region("band", (0, 10, 20))


def test_region_rectangle():
    square = Region.rectangle(45, 370, 1e5)
    assert square.params == (45.0, 10.0, 1e5, 1e5, 0.0)
    assert Region.rectangle(45, 10, 1e5, 2e5, -90).params[-1] == 270.0
    assert Region.rectangle(45, 10, 1e5, bearing_deg=360).params[-1] == 0.0
    # The longitude remains the reference meridian at a pole
    assert Region.rectangle(90, 30, 1e5).params[1] == 30.0
    with pytest.raises(ValueError, match="positive"):
        Region.rectangle(0, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        Region.rectangle(0, 0, 1e5, -1)
    with pytest.raises(ValueError, match="lat"):
        Region.rectangle(95, 0, 1e5)
    with pytest.raises(ValueError, match="requires"):
        Region("rectangle", (0, 0, 1e5))


def test_region_circle():
    assert Region.circle(45, 370, 1e5).params == (45.0, 10.0, 1e5)
    # All longitudes name the same pole
    assert Region.circle(90, 30, 1e5) == Region.circle(90, -120, 1e5)
    assert Region.circle(-90, 30, 1e5).params == (-90.0, 0.0, 1e5)
    with pytest.raises(ValueError, match="positive"):
        Region.circle(0, 0, 0)
    with pytest.raises(ValueError, match="lat"):
        Region.circle(-95, 0, 1e5)
    with pytest.raises(ValueError, match="requires"):
        Region("circle", (0, 0, 1e5, 1e5))


def test_region_direct_construction():
    with pytest.raises(ValueError, match="kind"):
        Region("polygon", (0, 0, 10, 0, 10, 10))
    with pytest.raises(TypeError):
        Region("box", (0, 0, 1, 1), "geodesic")
