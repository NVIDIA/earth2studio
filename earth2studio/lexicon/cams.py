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

_GLOBAL = "cams-global-atmospheric-composition-forecasts"


class CAMSGlobalLexicon(metaclass=LexiconType):
    """Copernicus Atmosphere Monitoring Service Global Forecast Lexicon

    CAMS specified ``<dataset>::<api_variable>::<netcdf_key>::<level>``

    The API variable name (used in the cdsapi request) differs from the NetCDF
    key (used to index the downloaded file). Both are stored in the VOCAB.

    Note
    ----
    Additional resources:
    https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
    """

    VOCAB = {
        # Surface meteorological variables
        "u10m": f"{_GLOBAL}::10m_u_component_of_wind::u10::",
        "v10m": f"{_GLOBAL}::10m_v_component_of_wind::v10::",
        "t2m": f"{_GLOBAL}::2m_temperature::t2m::",
        "d2m": f"{_GLOBAL}::2m_dewpoint_temperature::d2m::",
        "sp": f"{_GLOBAL}::surface_pressure::sp::",
        "msl": f"{_GLOBAL}::mean_sea_level_pressure::msl::",
        "tcwv": f"{_GLOBAL}::total_column_water_vapour::tcwv::",
        "tp": f"{_GLOBAL}::total_precipitation::tp::",
        "skt": f"{_GLOBAL}::skin_temperature::skt::",
        "tcc": f"{_GLOBAL}::total_cloud_cover::tcc::",
        "z": f"{_GLOBAL}::surface_geopotential::z::",
        "lsm": f"{_GLOBAL}::land_sea_mask::lsm::",
        # Aerosol optical depth variables (550nm)
        "aod550": f"{_GLOBAL}::total_aerosol_optical_depth_550nm::aod550::",
        "duaod550": f"{_GLOBAL}::dust_aerosol_optical_depth_550nm::duaod550::",
        "omaod550": f"{_GLOBAL}::organic_matter_aerosol_optical_depth_550nm::omaod550::",
        "bcaod550": f"{_GLOBAL}::black_carbon_aerosol_optical_depth_550nm::bcaod550::",
        "ssaod550": f"{_GLOBAL}::sea_salt_aerosol_optical_depth_550nm::ssaod550::",
        "suaod550": f"{_GLOBAL}::sulphate_aerosol_optical_depth_550nm::suaod550::",
        # Total column trace gases
        "tcco": f"{_GLOBAL}::total_column_carbon_monoxide::tcco::",
        "tcno2": f"{_GLOBAL}::total_column_nitrogen_dioxide::tcno2::",
        "tco3": f"{_GLOBAL}::total_column_ozone::gtco3::",
        "tcso2": f"{_GLOBAL}::total_column_sulphur_dioxide::tcso2::",
        # Pressure-level variables (multi-level, available every 3h lead time)
        # u-component of wind
        "u200": f"{_GLOBAL}::u_component_of_wind::u::200",
        "u250": f"{_GLOBAL}::u_component_of_wind::u::250",
        "u300": f"{_GLOBAL}::u_component_of_wind::u::300",
        "u400": f"{_GLOBAL}::u_component_of_wind::u::400",
        "u500": f"{_GLOBAL}::u_component_of_wind::u::500",
        "u600": f"{_GLOBAL}::u_component_of_wind::u::600",
        "u700": f"{_GLOBAL}::u_component_of_wind::u::700",
        "u850": f"{_GLOBAL}::u_component_of_wind::u::850",
        "u925": f"{_GLOBAL}::u_component_of_wind::u::925",
        "u1000": f"{_GLOBAL}::u_component_of_wind::u::1000",
        # v-component of wind
        "v200": f"{_GLOBAL}::v_component_of_wind::v::200",
        "v250": f"{_GLOBAL}::v_component_of_wind::v::250",
        "v300": f"{_GLOBAL}::v_component_of_wind::v::300",
        "v400": f"{_GLOBAL}::v_component_of_wind::v::400",
        "v500": f"{_GLOBAL}::v_component_of_wind::v::500",
        "v600": f"{_GLOBAL}::v_component_of_wind::v::600",
        "v700": f"{_GLOBAL}::v_component_of_wind::v::700",
        "v850": f"{_GLOBAL}::v_component_of_wind::v::850",
        "v925": f"{_GLOBAL}::v_component_of_wind::v::925",
        "v1000": f"{_GLOBAL}::v_component_of_wind::v::1000",
        # Temperature
        "t200": f"{_GLOBAL}::temperature::t::200",
        "t250": f"{_GLOBAL}::temperature::t::250",
        "t300": f"{_GLOBAL}::temperature::t::300",
        "t400": f"{_GLOBAL}::temperature::t::400",
        "t500": f"{_GLOBAL}::temperature::t::500",
        "t600": f"{_GLOBAL}::temperature::t::600",
        "t700": f"{_GLOBAL}::temperature::t::700",
        "t850": f"{_GLOBAL}::temperature::t::850",
        "t925": f"{_GLOBAL}::temperature::t::925",
        "t1000": f"{_GLOBAL}::temperature::t::1000",
        # Geopotential
        "z200": f"{_GLOBAL}::geopotential::z::200",
        "z250": f"{_GLOBAL}::geopotential::z::250",
        "z300": f"{_GLOBAL}::geopotential::z::300",
        "z400": f"{_GLOBAL}::geopotential::z::400",
        "z500": f"{_GLOBAL}::geopotential::z::500",
        "z600": f"{_GLOBAL}::geopotential::z::600",
        "z700": f"{_GLOBAL}::geopotential::z::700",
        "z850": f"{_GLOBAL}::geopotential::z::850",
        "z925": f"{_GLOBAL}::geopotential::z::925",
        "z1000": f"{_GLOBAL}::geopotential::z::1000",
        # Specific humidity
        "q200": f"{_GLOBAL}::specific_humidity::q::200",
        "q250": f"{_GLOBAL}::specific_humidity::q::250",
        "q300": f"{_GLOBAL}::specific_humidity::q::300",
        "q400": f"{_GLOBAL}::specific_humidity::q::400",
        "q500": f"{_GLOBAL}::specific_humidity::q::500",
        "q600": f"{_GLOBAL}::specific_humidity::q::600",
        "q700": f"{_GLOBAL}::specific_humidity::q::700",
        "q850": f"{_GLOBAL}::specific_humidity::q::850",
        "q925": f"{_GLOBAL}::specific_humidity::q::925",
        "q1000": f"{_GLOBAL}::specific_humidity::q::1000",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in CAMS vocabulary."""
        cams_key = cls.VOCAB[val]

        def mod(x: np.ndarray) -> np.ndarray:
            return x

        return cams_key, mod
