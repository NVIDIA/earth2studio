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


class GHCNLexicon(metaclass=LexiconType):
    """NOAA's Global Historical Climatology Network Daily (GHCN-D) lexicon.

    This lexicon provides variable mappings for GHCN-Daily observation types.
    Metadata fields (station, time, lat, lon, elev) are defined in the
    GHCN data source schema.

    Note
    ----
    GHCN-D raw values use non-SI units (tenths of Celsius for temperatures,
    tenths of mm for precipitation). The modifier functions returned by
    ``get_item`` convert to Earth2Studio standard units (K for temperature,
    m for precipitation/snow depth).

    Additional resources:

    - https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
    - https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
    - https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/doc/GHCND_documentation.pdf
    """

    VOCAB: dict[str, str] = {
        "t2m_max": "TMAX",  # Daily maximum temperature at 2m (K)
        "t2m_min": "TMIN",  # Daily minimum temperature at 2m (K)
        "t2m": "TAVG",  # Daily average temperature at 2m (K)
        "d2m": "ADPT",  # Average dew point temperature at 2m (K)
        "r2m": "RHAV",  # Average relative humidity for the day (%)
        "tp": "PRCP",  # Daily total precipitation (m)
        "sf": "WESF",  # Water equivalent of snowfall (m)
        "sd": "SNWD",  # Snow depth on ground (m)
        "sde": "SNOW",  # Daily snowfall (m)
        "ws10m": "AWND",  # Average daily wind speed at 10m (m/s)
        "fg10m": "WSF2",  # Fastest 2-minute wind speed at 10m (m/s)
        "tcc": "ACMH",  # Average cloudiness midnight-midnight (fraction)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GHCN vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - GHCN element code (e.g. ``"TMAX"``).
            - A modifier function that converts raw GHCN integer values to
              Earth2Studio standard units.
        """
        # Trigger KeyError for variables outside the vocabulary
        element = cls.VOCAB[val]

        if element in ("TMAX", "TMIN", "TAVG", "ADPT"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert tenths of degrees Celsius to Kelvin."""
                return x / 10.0 + 273.15

        elif element in ("PRCP", "WESF"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert tenths of mm to meters."""
                return x / 10000.0

        elif element in ("SNOW", "SNWD"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert mm to meters."""
                return x / 1000.0

        elif element in ("AWND", "WSF2"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert tenths of m/s to m/s."""
                return x / 10.0

        elif element == "ACMH":

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert percent to fraction [0, 1]."""
                return x / 100.0

        elif element == "RHAV":

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity — relative humidity already in percent."""
                return x

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity (no conversion)."""
                return x

        return element, mod
