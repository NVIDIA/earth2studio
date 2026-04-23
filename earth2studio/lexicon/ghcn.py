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

# GHCN element codes mapped from Earth2Studio canonical variable names.
# GHCN raw units:
#   TMAX/TMIN: tenths of degrees Celsius
#   PRCP:      tenths of mm
#   SNOW:      mm
#   SNWD:      mm
GHCN_ELEMENT_MAP: dict[str, str] = {
    "t2m_max": "TMAX",
    "t2m_min": "TMIN",
    "tp": "PRCP",
    "sd": "SNWD",
    "sde": "SNOW",
}


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

    - https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
    - https://registry.opendata.aws/noaa-ghcn/
    - https://docs.opendata.aws/noaa-ghcn-pds/readme.html
    """

    VOCAB: dict[str, str] = {
        "t2m_max": "daily maximum temperature at 2m (K)",
        "t2m_min": "daily minimum temperature at 2m (K)",
        "tp": "daily total precipitation (m)",
        "sd": "snow depth on ground (m)",
        "sde": "daily snowfall (m)",
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
        cls.VOCAB[val]
        element = GHCN_ELEMENT_MAP[val]

        if element in ("TMAX", "TMIN"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert tenths of degrees Celsius to Kelvin."""
                return x / 10.0 + 273.15

        elif element == "PRCP":

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert tenths of mm to meters."""
                return x / 10000.0

        elif element in ("SNOW", "SNWD"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert mm to meters."""
                return x / 1000.0

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity (no conversion)."""
                return x

        return element, mod
