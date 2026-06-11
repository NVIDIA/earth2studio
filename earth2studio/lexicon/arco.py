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

MDL_LEVELS = np.arange(1, 138)

ACCUMULATION_6H_VARIABLES = {
    "cp06",
    "ro",
    "ro06",
    "sf",
    "sf06",
    "ssrd06",
    "strd06",
    "tp06",
}


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
        "mtdwswrf": "mean_top_downward_short_wave_radiation_flux::",
        "skt": "skin_temperature::",
        "sic": "sea_ice_cover::",
        # AIFS/AIFS ENS and AIFS2 aliases backed by ARCO ERA5 single-level fields.
        "cdww": "coefficient_of_drag_with_waves::",
        "cp06": "convective_precipitation::",
        "cos_mwd": "mean_wave_direction::",
        "hcc": "high_cloud_cover::",
        "lcc": "low_cloud_cover::",
        "mcc": "medium_cloud_cover::",
        "mwp": "mean_wave_period::",
        "ro": "runoff::",
        "ro06": "runoff::",
        "sd": "snow_depth::",
        "sdor": "standard_deviation_of_orography::",
        "sf": "snowfall::",
        "sf06": "snowfall::",
        "sin_mwd": "mean_wave_direction::",
        "slor": "slope_of_sub_gridscale_orography::",
        "ssrd06": "surface_solar_radiation_downwards::",
        "stl1": "soil_temperature_level_1::",
        "stl2": "soil_temperature_level_2::",
        "strd06": "surface_thermal_radiation_downwards::",
        "swh": "significant_height_of_combined_wind_waves_and_swell::",
        "swvl1": "volumetric_soil_water_layer_1::",
        "swvl2": "volumetric_soil_water_layer_2::",
        "tcc": "total_cloud_cover::",
        "tcw": "total_column_water::",
        "tp06": "total_precipitation::",
        "wmb": "model_bathymetry::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})
    VOCAB.update(
        {
            f"clwc{level}": f"specific_cloud_liquid_water_content::{level}"
            for level in LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}": f"specific_cloud_ice_water_content::{level}"
            for level in LEVELS
        }
    )

    # Model levels are stored in a separate Zarr group, but use the same base names
    VOCAB.update(
        {f"u{level}k": f"u_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update(
        {f"v{level}k": f"v_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update({f"t{level}k": f"temperature::{level}" for level in MDL_LEVELS})
    VOCAB.update({f"q{level}k": f"specific_humidity::{level}" for level in MDL_LEVELS})
    VOCAB.update(
        {
            f"clwc{level}k": f"specific_cloud_liquid_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}k": f"specific_cloud_ice_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"crwc{level}k": f"specific_rain_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"cswc{level}k": f"specific_snow_water_content::{level}"
            for level in MDL_LEVELS
        }
    )

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ARCO vocabulary."""
        arco_key = cls.VOCAB[val]

        if val == "cos_mwd":

            def mod(x: np.array) -> np.array:
                return np.cos(np.deg2rad(x))

        elif val == "sin_mwd":

            def mod(x: np.array) -> np.array:
                return np.sin(np.deg2rad(x))

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return arco_key, mod
