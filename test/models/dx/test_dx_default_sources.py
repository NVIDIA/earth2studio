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

import pytest

from earth2studio.data import ARCO_ERA5, GFS, NCAR_ERA5, WB2ERA5
from earth2studio.models.dx import (
    CBottleInfill,
    ClimateNet,
    CorrDiffCosmoEra5,
    CorrDiffEra5Hrrr,
    CorrDiffTaiwan,
    OrbitGlobalPrecip,
    PrecipitationAFNO,
    PrecipitationAFNOv2,
    SolarRadiationAFNO1H,
    SolarRadiationAFNO6H,
    TCTrackerVitart,
    TCTrackerWuDuan,
    WindgustAFNO,
)


@pytest.mark.parametrize(
    "model_cls,expected",
    [
        (CBottleInfill, ARCO_ERA5),
        (ClimateNet, ARCO_ERA5),
        (CorrDiffCosmoEra5, ARCO_ERA5),
        (CorrDiffEra5Hrrr, ARCO_ERA5),
        (CorrDiffTaiwan, GFS),
        (OrbitGlobalPrecip, NCAR_ERA5),
        (PrecipitationAFNO, WB2ERA5),
        (PrecipitationAFNOv2, ARCO_ERA5),
        (SolarRadiationAFNO1H, ARCO_ERA5),
        (SolarRadiationAFNO6H, ARCO_ERA5),
        (TCTrackerVitart, GFS),
        (TCTrackerWuDuan, GFS),
        (WindgustAFNO, ARCO_ERA5),
    ],
)
def test_default_sources(model_cls, expected):
    # default_sources does not touch model state, so skip heavy construction.
    # StormScopeDxNSRDB regrids onto its model grid, so its default source is
    # tested with a constructed model in test_stormscope_dx_nsrdb.py.
    model = object.__new__(model_cls)
    source = model.default_sources()
    assert isinstance(source, expected)
