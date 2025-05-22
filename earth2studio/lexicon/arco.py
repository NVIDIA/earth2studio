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

from collections.abc import Callable

import numpy as np

from .base import LexiconType

# ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
# 'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography',
# 'geopotential', 'geopotential_at_surface', 'high_vegetation_cover',
# 'lake_cover', 'lake_depth', 'land_sea_mask', 'low_vegetation_cover',
# 'mean_sea_level_pressure', 'sea_ice_cover', 'sea_surface_temperature',
# 'slope_of_sub_gridscale_orography', 'soil_type', 'specific_humidity',
# 'standard_deviation_of_filtered_subgrid_orography', 'standard_deviation_of_orography',
# 'surface_pressure', 'temperature', 'toa_incident_solar_radiation', 'total_cloud_cover',
# 'total_column_water_vapour', 'total_precipitation', 'type_of_high_vegetation',
# 'type_of_low_vegetation', 'u_component_of_wind', 'v_component_of_wind',
# 'vertical_velocity']

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


class ARCOLexicon(metaclass=LexiconType):
    """ARCO Lexicon
    ARCO specified <Variable ID>::<Pressure Level>

    Note
    ----
    Variable named based on ERA5 names, see CDS docs for resources.
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "u100m": "100m_u_component_of_wind::",
        "v100m": "100m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "d2m": "2m_dewpoint_temperature::",
        "sp": "surface_pressure::",
        "sst": "sea_surface_temperature::",
        "msl": "mean_sea_level_pressure::",
        "tcwv": "total_column_water_vapour::",
        "tp": "total_precipitation::",
        "lsm": "land_sea_mask::",
        "z": "geopotential_at_surface::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ARCO vocabulary."""
        arco_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify name (if necessary)."""
            return x

        return arco_key, mod
