# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.dx.cbottle_infill import CBottleInfill  # noqa
from earth2studio.models.dx.cbottle_sr import CBottleSR  # noqa
from earth2studio.models.dx.climatenet import ClimateNet  # noqa
from earth2studio.models.dx.corrdiff import CorrDiffTaiwan  # noqa
from earth2studio.models.dx.derived import (
    DerivedRH,
    DerivedRHDewpoint,
    DerivedVPD,
    DerivedWS,
)
from earth2studio.models.dx.identity import Identity  # noqa
from earth2studio.models.dx.precipitation_afno import PrecipitationAFNO  # noqa
from earth2studio.models.dx.precipitation_afno_v2 import PrecipitationAFNOv2  # noqa
from earth2studio.models.dx.solarradiation_afno import (
    SolarRadiationAFNO1H,
    SolarRadiationAFNO6H,
)
from earth2studio.models.dx.tc_tracking import (
    TCTrackerVitart,
    TCTrackerWuDuan,
)
from earth2studio.models.dx.wind_gust import WindgustAFNO  # noqa

__all__ = [
    "ClimateNet",
    "CorrDiffTaiwan",
    "PrecipitationAFNO",
    "PrecipitationAFNOv2",
    "SolarRadiationAFNO1H",
    "SolarRadiationAFNO6H",
    "WindgustAFNO",
]
