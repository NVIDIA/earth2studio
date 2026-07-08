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

import pandas as pd  # type: ignore[import-untyped]

from .base import LexiconType
from .nnja import get_ncep_conventional_item


class GDASObsConvLexicon(metaclass=LexiconType):
    """NOAA GDAS lexicon for conventional and GPSRO observations.

    Maps Earth2Studio variable names to NCEP observation extraction parameters.
    The vocab value is either a PrepBUFR mnemonic (e.g., ``"TOB"`` for
    temperature) or ``"wind::u"`` / ``"wind::v"`` for wind components.

    Modifier functions convert raw PrepBUFR observation values to
    Earth2Studio standard units:

    - ``t``: TOB (DEG C) → Kelvin (+273.15)
    - ``q``: QOB (mg/kg) → kg kg-1 (÷1e6)
    - ``pres``: source pressure-observation POB rows (hPa / MB) → Pa (×100)
    - ``u``, ``v``: UOB/VOB already in m s-1 (no conversion)
    - ``gps``: combined ionosphere-corrected bending angle (no conversion)

    The GPSRO dump can also contain provider 1D-Var retrieval profiles for
    pressure (descriptor ``10004``), temperature (``12001``), and specific
    humidity (``13001``). They are intentionally not exposed here. UFS
    diagnostic ``gps_t`` and ``gps_q`` are model-background values sampled at
    the bending-angle location, not these BUFR retrieval fields.

    Note
    ----
    Additional resources on PrepBUFR format and observation types:

    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm
    """

    VOCAB: dict[str, str] = {
        "u": "wind::u",
        "v": "wind::v",
        "q": "QOB",
        "t": "TOB",
        "pres": "POB",
        "gps": "gpsro::15037",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[..., pd.DataFrame]]:
        """Get an item from the GDAS conventional vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - PrepBUFR mnemonic/wind key or the GPSRO descriptor route.
            - A modifier function to apply to the loaded DataFrame.  The
              modifier converts the ``observation`` column from raw PrepBUFR
              units to Earth2Studio standard units.
        """
        return get_ncep_conventional_item(val, route_prefix=False)
