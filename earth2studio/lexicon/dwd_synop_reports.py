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


class DWDSynopReportsLexicon(metaclass=LexiconType):
    """DWD SYNOP (``weather_reports``) BUFR product lexicon.

    Maps Earth2Studio variable ids to the WMO BUFR element short names in DWD
    SYNOP reports. ``u10m`` / ``v10m`` are derived from ``windSpeed`` and
    ``windDirection`` by the data source, so they map to ``None`` here.
    Metadata fields (station, time, lat, lon, elev) are defined in the DWD data
    source schema. Precipitation is intentionally omitted for now: the SYNOP
    accumulation period is not disambiguated, so a single ``tp`` variable could
    mix incompatible accumulation windows.

    The height-qualified ids (``t2m``, ``d2m``, ``ws10m``/``u10m``/``v10m``)
    assume the WMO standard land-SYNOP measurement heights (temperature/humidity
    at 2 m, wind at 10 m); the BUFR element itself is not height-qualified, so
    this mapping relies on that convention rather than a decoded sensor height.

    Note
    ----
    Additional resources:

    - https://opendata.dwd.de/weather/weather_reports/synoptic/
    - https://confluence.ecmwf.int/display/ECC (BUFR element names)
    """

    VOCAB: dict[str, str | None] = {
        "t2m": "airTemperature",
        "d2m": "dewpointTemperature",
        "ws10m": "windSpeed",
        "u10m": None,  # derived from windSpeed + windDirection
        "v10m": None,  # derived from windSpeed + windDirection
        "msl": "pressureReducedToMeanSeaLevel",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str | None, Callable]:
        """Get item from the DWD SYNOP vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str | None, Callable]
            - The BUFR element short name, or ``None`` for the derived wind
              components (``u10m`` / ``v10m``).
            - A modifier function applied to the decoded values. Mapped DWD
              SYNOP BUFR observations are already in Earth2Studio target units
              (temperature K, wind speed m/s, pressure Pa; wind direction is
              decoded in degrees for the u/v derivation), so it is the identity.
        """
        key = cls.VOCAB[val]

        def identity(x: np.ndarray) -> np.ndarray:
            """Return values unchanged (already SI)."""
            return x

        return key, identity
