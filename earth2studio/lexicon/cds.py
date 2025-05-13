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


class CDSLexicon(metaclass=LexiconType):
    """Climate Data Store Lexicon
    CDS specified <dataset>::<Variable ID>::<Pressure Level>

    Note
    ----
    Additional resources:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview
    https://codes.ecmwf.int/grib/param-db/?filter=grib2
    """

    VOCAB = {
        "u10m": "reanalysis-era5-single-levels::10m_u_component_of_wind::",
        "v10m": "reanalysis-era5-single-levels::10m_v_component_of_wind::",
        "u100m": "reanalysis-era5-single-levels::100m_u_component_of_wind::",
        "v100m": "reanalysis-era5-single-levels::100m_v_component_of_wind::",
        "t2m": "reanalysis-era5-single-levels::2m_temperature::",
        "d2m": "reanalysis-era5-single-levels::2m_dewpoint_temperature::",
        "sp": "reanalysis-era5-single-levels::surface_pressure::",
        "msl": "reanalysis-era5-single-levels::mean_sea_level_pressure::",
        "tcwv": "reanalysis-era5-single-levels::total_column_water_vapour::",
        "tp": "reanalysis-era5-single-levels::total_precipitation::",
        "fg10m": "reanalysis-era5-single-levels::10m_wind_gust_since_previous_post_processing::",
        "sst": "reanalysis-era5-single-levels::sea_surface_temperature::",
        "u50": "reanalysis-era5-pressure-levels::u_component_of_wind::50",
        "u100": "reanalysis-era5-pressure-levels::u_component_of_wind::100",
        "u150": "reanalysis-era5-pressure-levels::u_component_of_wind::150",
        "u200": "reanalysis-era5-pressure-levels::u_component_of_wind::200",
        "u250": "reanalysis-era5-pressure-levels::u_component_of_wind::250",
        "u300": "reanalysis-era5-pressure-levels::u_component_of_wind::300",
        "u400": "reanalysis-era5-pressure-levels::u_component_of_wind::400",
        "u500": "reanalysis-era5-pressure-levels::u_component_of_wind::500",
        "u650": "reanalysis-era5-pressure-levels::u_component_of_wind::650",
        "u600": "reanalysis-era5-pressure-levels::u_component_of_wind::600",
        "u700": "reanalysis-era5-pressure-levels::u_component_of_wind::700",
        "u850": "reanalysis-era5-pressure-levels::u_component_of_wind::850",
        "u925": "reanalysis-era5-pressure-levels::u_component_of_wind::925",
        "u1000": "reanalysis-era5-pressure-levels::u_component_of_wind::1000",
        "v50": "reanalysis-era5-pressure-levels::v_component_of_wind::50",
        "v100": "reanalysis-era5-pressure-levels::v_component_of_wind::100",
        "v150": "reanalysis-era5-pressure-levels::v_component_of_wind::150",
        "v200": "reanalysis-era5-pressure-levels::v_component_of_wind::200",
        "v250": "reanalysis-era5-pressure-levels::v_component_of_wind::250",
        "v300": "reanalysis-era5-pressure-levels::v_component_of_wind::300",
        "v400": "reanalysis-era5-pressure-levels::v_component_of_wind::400",
        "v500": "reanalysis-era5-pressure-levels::v_component_of_wind::500",
        "v600": "reanalysis-era5-pressure-levels::v_component_of_wind::600",
        "v700": "reanalysis-era5-pressure-levels::v_component_of_wind::700",
        "v850": "reanalysis-era5-pressure-levels::v_component_of_wind::850",
        "v925": "reanalysis-era5-pressure-levels::v_component_of_wind::925",
        "v1000": "reanalysis-era5-pressure-levels::v_component_of_wind::1000",
        "z50": "reanalysis-era5-pressure-levels::geopotential::50",
        "z100": "reanalysis-era5-pressure-levels::geopotential::100",
        "z150": "reanalysis-era5-pressure-levels::geopotential::150",
        "z200": "reanalysis-era5-pressure-levels::geopotential::200",
        "z250": "reanalysis-era5-pressure-levels::geopotential::250",
        "z300": "reanalysis-era5-pressure-levels::geopotential::300",
        "z400": "reanalysis-era5-pressure-levels::geopotential::400",
        "z500": "reanalysis-era5-pressure-levels::geopotential::500",
        "z600": "reanalysis-era5-pressure-levels::geopotential::600",
        "z700": "reanalysis-era5-pressure-levels::geopotential::700",
        "z850": "reanalysis-era5-pressure-levels::geopotential::850",
        "z925": "reanalysis-era5-pressure-levels::geopotential::925",
        "z1000": "reanalysis-era5-pressure-levels::geopotential::1000",
        "t50": "reanalysis-era5-pressure-levels::temperature::50",
        "t100": "reanalysis-era5-pressure-levels::temperature::100",
        "t150": "reanalysis-era5-pressure-levels::temperature::150",
        "t200": "reanalysis-era5-pressure-levels::temperature::200",
        "t250": "reanalysis-era5-pressure-levels::temperature::250",
        "t300": "reanalysis-era5-pressure-levels::temperature::300",
        "t400": "reanalysis-era5-pressure-levels::temperature::400",
        "t500": "reanalysis-era5-pressure-levels::temperature::500",
        "t600": "reanalysis-era5-pressure-levels::temperature::600",
        "t700": "reanalysis-era5-pressure-levels::temperature::700",
        "t850": "reanalysis-era5-pressure-levels::temperature::850",
        "t925": "reanalysis-era5-pressure-levels::temperature::925",
        "t1000": "reanalysis-era5-pressure-levels::temperature::1000",
        "r20": "reanalysis-era5-pressure-levels::relative_humidity::20",
        "r50": "reanalysis-era5-pressure-levels::relative_humidity::50",
        "r100": "reanalysis-era5-pressure-levels::relative_humidity::100",
        "r150": "reanalysis-era5-pressure-levels::relative_humidity::150",
        "r200": "reanalysis-era5-pressure-levels::relative_humidity::200",
        "r250": "reanalysis-era5-pressure-levels::relative_humidity::250",
        "r300": "reanalysis-era5-pressure-levels::relative_humidity::300",
        "r400": "reanalysis-era5-pressure-levels::relative_humidity::400",
        "r500": "reanalysis-era5-pressure-levels::relative_humidity::500",
        "r600": "reanalysis-era5-pressure-levels::relative_humidity::600",
        "r700": "reanalysis-era5-pressure-levels::relative_humidity::700",
        "r800": "reanalysis-era5-pressure-levels::relative_humidity::800",
        "r850": "reanalysis-era5-pressure-levels::relative_humidity::850",
        "r925": "reanalysis-era5-pressure-levels::relative_humidity::925",
        "r1000": "reanalysis-era5-pressure-levels::relative_humidity::1000",
        "q50": "reanalysis-era5-pressure-levels::specific_humidity::50",
        "q100": "reanalysis-era5-pressure-levels::specific_humidity::100",
        "q150": "reanalysis-era5-pressure-levels::specific_humidity::150",
        "q200": "reanalysis-era5-pressure-levels::specific_humidity::200",
        "q250": "reanalysis-era5-pressure-levels::specific_humidity::250",
        "q300": "reanalysis-era5-pressure-levels::specific_humidity::300",
        "q400": "reanalysis-era5-pressure-levels::specific_humidity::400",
        "q500": "reanalysis-era5-pressure-levels::specific_humidity::500",
        "q600": "reanalysis-era5-pressure-levels::specific_humidity::600",
        "q700": "reanalysis-era5-pressure-levels::specific_humidity::700",
        "q850": "reanalysis-era5-pressure-levels::specific_humidity::850",
        "q925": "reanalysis-era5-pressure-levels::specific_humidity::925",
        "q1000": "reanalysis-era5-pressure-levels::specific_humidity::1000",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in CDS vocabulary."""
        cds_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify name (if necessary)."""
            return x

        return cds_key, mod
