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


class IBTrACSLexicon(metaclass=LexiconType):
    """International Best Track Archive for Climate Stewardship (IBTrACS) lexicon.

    This lexicon provides variable mappings for IBTrACS tropical cyclone track data.
    Variables are mapped from IBTrACS NetCDF variables to Earth2Studio standard names.

    Note
    ----
    IBTrACS uses WMO-harmonized variables where available. Wind speeds are stored in
    knots and converted to m/s. Wind radii are stored in nautical miles and converted
    to kilometers. Pressure is stored in mb (hPa) and converted to Pa.

    Additional resources:

    - https://www.ncei.noaa.gov/products/international-best-track-archive
    - https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/doc/IBTrACS_version4_Technical_Details.pdf
    """

    # Conversion constants
    _KTS_TO_MS = 0.514444  # knots to m/s
    _NMILE_TO_KM = 1.852  # nautical miles to km
    _HPA_TO_PA = 100.0  # hPa (mb) to Pa

    VOCAB: dict[str, str] = {
        # WMO-harmonized intensity variables
        "tcwnd": "wmo_wind",  # Maximum sustained wind speed (m/s)
        "mslp": "wmo_pres",  # Minimum sea level pressure (Pa)
        # Storm translation components (computed from storm_speed and storm_dir)
        "tcustm": "storm_speed::u",  # Storm translation u-component (m/s)
        "tcvstm": "storm_speed::v",  # Storm translation v-component (m/s)
        # Wind radii (average of quadrants, converted to km)
        "tcr34": "usa_r34",  # 34-kt wind radius (km)
        "tcr50": "usa_r50",  # 50-kt wind radius (km)
        "tcr64": "usa_r64",  # 64-kt wind radius (km)
        # Saffir-Simpson Hurricane Scale category
        "tcsshs": "usa_sshs",  # SSHS category (-5 to 5)
        # Distance to land
        "tcd2l": "dist2land",  # Distance to land (km)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from IBTrACS vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - IBTrACS variable key (may include :: for computed variables).
            - A modifier function that converts raw IBTrACS values to
              Earth2Studio standard units.
        """
        # Trigger KeyError for variables outside the vocabulary
        source_key = cls.VOCAB[val]

        # Wind speed: knots to m/s
        if source_key == "wmo_wind":

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert knots to m/s."""
                return x * cls._KTS_TO_MS

        # Pressure: mb (hPa) to Pa
        elif source_key == "wmo_pres":

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert mb (hPa) to Pa."""
                return x * cls._HPA_TO_PA

        # Storm translation u-component: computed from speed and direction
        elif source_key == "storm_speed::u":

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity - u-component computed during data loading."""
                return x

        # Storm translation v-component: computed from speed and direction
        elif source_key == "storm_speed::v":

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity - v-component computed during data loading."""
                return x

        # Wind radii: nautical miles to km (average of quadrants)
        elif source_key in ("usa_r34", "usa_r50", "usa_r64"):

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert nautical miles to km."""
                return x * cls._NMILE_TO_KM

        # SSHS category and distance to land: identity
        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity (no conversion)."""
                return x

        return source_key, mod
