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

from typing import Callable


class LexiconType(type):
    """Lexicon."""

    def __getitem__(cls, val: str) -> tuple[str, Callable]:
        """Retrieve variable name."""
        return cls.get_item(val)  # type: ignore[attr-defined]


E2STUDIO_VOCAB = {
    "u10m": "u-component (eastward, zonal) of wind at 10 m",
    "v10m": "v-component (northward, meridional) of wind at 10 m",
    "u100m": "u-component of wind at 100 m",
    "v100m": "v-component of wind at 100 m",
    "t2m": "temperature at 2m",
    "sp": "surface pressure",
    "msl": "mean sea level pressure",
    "tcwv": "total column water vapor",
    "tp": "total precipitation",
    "tp06": "total precipitation accumulated over past 6 hours",
    "fg10m": "maximum 10 m wind gust since previous post-processing",
    "lsm": "land-sea mask",
    "zsl": "geopotential at mean sea-level",
    "uvcossza": "cosine of solar zenith angle",
    "u50": "u-component of wind at 50 hPa",
    "u100": "u-component of wind at 100 hPa",
    "u150": "u-component of wind at 150 hPa",
    "u200": "u-component of wind at 200 hPa",
    "u250": "u-component of wind at 250 hPa",
    "u300": "u-component of wind at 300 hPa",
    "u400": "u-component of wind at 400 hPa",
    "u500": "u-component of wind at 500 hPa",
    "u600": "u-component of wind at 600 hPa",
    "u700": "u-component of wind at 700 hPa",
    "u850": "u-component of wind at 850 hPa",
    "u925": "u-component of wind at 925 hPa",
    "u1000": "u-component of wind at 1000 hPa",
    "v50": "v-component of wind at 50 hPa",
    "v100": "v-component of wind at 100 hPa",
    "v150": "v-component of wind at 150 hPa",
    "v200": "v-component of wind at 200 hPa",
    "v250": "v-component of wind at 250 hPa",
    "v300": "v-component of wind at 300 hPa",
    "v400": "v-component of wind at 400 hPa",
    "v500": "v-component of wind at 500 hPa",
    "v600": "v-component of wind at 600 hPa",
    "v700": "v-component of wind at 700 hPa",
    "v850": "v-component of wind at 850 hPa",
    "v925": "v-component of wind at 925 hPa",
    "v1000": "v-component of wind at 1000 hPa",
    "z50": "geopotential at 50 hPa",
    "z100": "geopotential at 100 hPa",
    "z150": "geopotential at 150 hPa",
    "z200": "geopotential at 200 hPa",
    "z250": "geopotential at 250 hPa",
    "z300": "geopotential at 300 hPa",
    "z400": "geopotential at 400 hPa",
    "z500": "geopotential at 500 hPa",
    "z600": "geopotential at 600 hPa",
    "z700": "geopotential at 700 hPa",
    "z850": "geopotential at 850 hPa",
    "z925": "geopotential at 925 hPa",
    "z1000": "geopotential at 1000 hPa",
    "t50": "temperature at 50 hPa",
    "t100": "temperature at 100 hPa",
    "t150": "temperature at 150 hPa",
    "t200": "temperature at 200 hPa",
    "t250": "temperature at 250 hPa",
    "t300": "temperature at 300 hPa",
    "t400": "temperature at 400 hPa",
    "t500": "temperature at 500 hPa",
    "t600": "temperature at 600 hPa",
    "t700": "temperature at 700 hPa",
    "t850": "temperature at 850 hPa",
    "t925": "temperature at 925 hPa",
    "t1000": "temperature at 1000 hPa",
    "r50": "relative humidity at 50 hPa",
    "r100": "relative humidity at 100 hPa",
    "r150": "relative humidity at 150 hPa",
    "r200": "relative humidity at 200 hPa",
    "r250": "relative humidity at 250 hPa",
    "r300": "relative humidity at 300 hPa",
    "r400": "relative humidity at 400 hPa",
    "r500": "relative humidity at 500 hPa",
    "r600": "relative humidity at 600 hPa",
    "r700": "relative humidity at 700 hPa",
    "r850": "relative humidity at 850 hPa",
    "r925": "relative humidity at 925 hPa",
    "r1000": "relative humidity at 1000 hPa",
    "q50": "specific humidity at 50 hPa",
    "q100": "specific humidity at 100 hPa",
    "q150": "specific humidity at 150 hPa",
    "q200": "specific humidity at 200 hPa",
    "q250": "specific humidity at 250 hPa",
    "q300": "specific humidity at 300 hPa",
    "q400": "specific humidity at 400 hPa",
    "q500": "specific humidity at 500 hPa",
    "q600": "specific humidity at 600 hPa",
    "q700": "specific humidity at 700 hPa",
    "q850": "specific humidity at 850 hPa",
    "q925": "specific humidity at 925 hPa",
    "q1000": "specific humidity at 1000 hPa",
}
