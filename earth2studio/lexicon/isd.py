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


class ISDLexicon(metaclass=LexiconType):
    """NOAA's Integrated Surface Database (ISD) product lexicon.

    This lexicon provides a simple reference for some of the unique variables to ISD,
    vocab translation is manually handled in the data source due to complexity.

    Note
    ----
    Additional resources:

    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    - https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
    """

    VOCAB: dict[str, str] = {
        "station": "FIXED-WEATHER-STATION ID (USAF)(WBAN)",
        "time": "GEOPHYSICAL-POINT-OBSERVATION date time",
        "source": "GEOPHYSICAL-POINT-OBSERVATION data source flag",
        "lat": "GEOPHYSICAL-POINT-OBSERVATION latitude coordinate",
        "lon": "GEOPHYSICAL-POINT-OBSERVATION longitude coordinate",
        "elev": "GEOPHYSICAL-POINT-OBSERVATION elevation dimension relative to msl (m)",
        "ws10m": "wind speed at 10 m (m s-1)",
        "u10m": "u-component (eastward, zonal) of wind at 10 m (m s-1)",
        "v10m": "v-component (northward, meridional) of wind at 10 m (m s-1)",
        "tp": "total precipitation last hour (m)",
        "t2m": "temperature at 2m (K)",
        "fg10m": "maximum 10 m wind gust since previous post-processing (m s-1)",
        "d2m": "dewpoint temperature at 2m (K)",
        "tcc": "total cloud cover (0 - 1)",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from ISD vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - Description of variable (not used).
            - A modifier function to apply to the loaded values (identity).
        """
        isd_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.ndarray:
            """Modify data value (if necessary)."""
            return x

        return isd_key, mod
