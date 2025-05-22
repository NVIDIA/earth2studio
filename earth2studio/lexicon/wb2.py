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

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


class WB2Lexicon(metaclass=LexiconType):
    """WeatherBench Lexicon
    WeatherBench specified <Variable ID>::<Pressure Level>

    Note
    ----
    Variable named based on ERA5 names, see WB2 docs for resources.

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html
    - Dew point temperature at 2m seems to be all NaNs
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "sp": "surface_pressure::",
        "lsm": "land_sea_mask::",
        "z": "geopotential_at_surface::",
        "msl": "mean_sea_level_pressure::",
        "sst": "sea_surface_temperature::",
        "tcwv": "total_column_water_vapour::",
        "tp06": "total_precipitation_6hr::",
        "tp12": "total_precipitation_12hr::",
        "tp24": "total_precipitation_24hr::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})
    VOCAB.update({f"r{level}": f"relative_humidity::{level}" for level in LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in WeatherBench vocabulary."""
        wb2_key = cls.VOCAB[val]

        if wb2_key.split("::")[0] == "relative_humidity":

            def mod(x: np.array) -> np.array:
                """Relative humidty in WeatherBench uses older calculation and does
                not scale by 100 natively. Not recommended for use, see IFS method for
                more modern calculation.
                https://github.com/google-research/weatherbench2/blob/main/weatherbench2/derived_variables.py#L468
                """
                return x * 100

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return wb2_key, mod


class WB2ClimatetologyLexicon(metaclass=LexiconType):
    """WeatherBench Climatology Lexicon
    WeatherBench specified <Variable ID>::<Pressure Level>

    Note
    ----
    Variable named based on ERA5 names, see WB2 docs for resources.

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html
    - Dew point temperature at 2m seems to be all NaNs
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "sp": "surface_pressure::",
        "msl": "mean_sea_level_pressure::",
        "sst": "sea_surface_temperature::",
        "tcwv": "total_column_water_vapour::",
        "tp06": "total_precipitation_6hr::",
        "tp06sdf": "total_precipitation_6hr_seeps_dry_fraction::",
        "tp06st": "total_precipitation_6hr_seeps_threshold::",
        "tp12": "total_precipitation_12hr::",
        "tp24": "total_precipitation_24hr::",
        "tp24sdf": "total_precipitation_24hr_seeps_dry_fraction::",
        "tp24st": "total_precipitation_24hr_seeps_threshold::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})
    VOCAB.update({f"r{level}": f"relative_humidity::{level}" for level in LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in WeatherBench vocabulary."""
        wb2_key = cls.VOCAB[val]

        if wb2_key.split("::")[0] == "relative_humidity":

            def mod(x: np.array) -> np.array:
                """Relative humidty in WeatherBench uses older calculation and does
                not scale by 100 natively. Not recommended for use, see IFS method for
                more modern calculation.
                https://github.com/google-research/weatherbench2/blob/main/weatherbench2/derived_variables.py#L468
                """
                return x * 100

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return wb2_key, mod
