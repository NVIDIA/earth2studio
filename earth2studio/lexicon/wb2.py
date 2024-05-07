# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Callable

import numpy as np

from .base import LexiconType


class WB2Lexicon(metaclass=LexiconType):
    """WeatherBench Lexicon
    WeatherBench specified <Variable ID>::<Pressure Level>

    Note
    ----
    Variable named based on ERA5 names, see WB2 docs for resources.
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "u100m": "100m_u_component_of_wind::",
        "v100m": "100m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "sp": "surface_pressure::",
        "msl": "mean_sea_level_pressure::",
        "tcwv": "total_column_water_vapour::",
        "tp06": "total_precipitation_6hr::",
        "tp06sdf": "total_precipitation_6hr_seeps_dry_fraction::",
        "tp06st": "total_precipitation_6hr_seeps_threshold::",
        "tp12": "total_precipitation_12hr::",
        "tp24sdf": "total_precipitation_24hr_seeps_dry_fraction::",
        "tp24st": "total_precipitation_24hr_seeps_threshold::",
        "u50": "u_component_of_wind::50",
        "u100": "u_component_of_wind::100",
        "u150": "u_component_of_wind::150",
        "u200": "u_component_of_wind::200",
        "u250": "u_component_of_wind::250",
        "u300": "u_component_of_wind::300",
        "u400": "u_component_of_wind::400",
        "u500": "u_component_of_wind::500",
        "u600": "u_component_of_wind::600",
        "u700": "u_component_of_wind::700",
        "u850": "u_component_of_wind::850",
        "u925": "u_component_of_wind::925",
        "u1000": "u_component_of_wind::1000",
        "v50": "v_component_of_wind::50",
        "v100": "v_component_of_wind::100",
        "v150": "v_component_of_wind::150",
        "v200": "v_component_of_wind::200",
        "v250": "v_component_of_wind::250",
        "v300": "v_component_of_wind::300",
        "v400": "v_component_of_wind::400",
        "v500": "v_component_of_wind::500",
        "v600": "v_component_of_wind::600",
        "v700": "v_component_of_wind::700",
        "v850": "v_component_of_wind::850",
        "v925": "v_component_of_wind::925",
        "v1000": "v_component_of_wind::1000",
        "z50": "geopotential::50",
        "z100": "geopotential::100",
        "z150": "geopotential::150",
        "z200": "geopotential::200",
        "z250": "geopotential::250",
        "z300": "geopotential::300",
        "z400": "geopotential::400",
        "z500": "geopotential::500",
        "z600": "geopotential::600",
        "z700": "geopotential::700",
        "z850": "geopotential::850",
        "z925": "geopotential::925",
        "z1000": "geopotential::1000",
        "t50": "temperature::50",
        "t100": "temperature::100",
        "t150": "temperature::150",
        "t200": "temperature::200",
        "t250": "temperature::250",
        "t300": "temperature::300",
        "t400": "temperature::400",
        "t500": "temperature::500",
        "t600": "temperature::600",
        "t700": "temperature::700",
        "t850": "temperature::850",
        "t925": "temperature::925",
        "t1000": "temperature::1000",
        "q50": "specific_humidity::50",
        "q100": "specific_humidity::100",
        "q150": "specific_humidity::150",
        "q200": "specific_humidity::200",
        "q250": "specific_humidity::250",
        "q300": "specific_humidity::300",
        "q400": "specific_humidity::400",
        "q500": "specific_humidity::500",
        "q600": "specific_humidity::600",
        "q700": "specific_humidity::700",
        "q850": "specific_humidity::850",
        "q925": "specific_humidity::925",
        "q1000": "specific_humidity::1000",
        "r50": "relative_humidity::50",
        "r100": "relative_humidity::100",
        "r150": "relative_humidity::150",
        "r200": "relative_humidity::200",
        "r250": "relative_humidity::250",
        "r300": "relative_humidity::300",
        "r400": "relative_humidity::400",
        "r500": "relative_humidity::500",
        "r600": "relative_humidity::600",
        "r700": "relative_humidity::700",
        "r850": "relative_humidity::850",
        "r925": "relative_humidity::925",
        "r1000": "relative_humidity::1000",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in WeatherBench vocabulary."""
        wb2_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify name (if necessary)."""
            return x

        return wb2_key, mod
