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
import pandas as pd

from .base import LexiconType


class GDASObsConvLexicon(metaclass=LexiconType):
    """NOAA GDAS PrepBUFR lexicon for conventional (in-situ) observations.

    Maps Earth2Studio variable names to PrepBUFR extraction parameters.
    The vocab value is either a PrepBUFR mnemonic (e.g., ``"TOB"`` for
    temperature) or ``"wind::u"`` / ``"wind::v"`` for wind components.

    Modifier functions convert raw PrepBUFR observation values to
    Earth2Studio standard units:

    - ``t``: TOB (DEG C) → Kelvin (+273.15)
    - ``q``: QOB (mg/kg) → kg kg-1 (÷1e6)
    - ``pres``: POB (hPa / MB) → Pa (×100)
    - ``u``, ``v``: UOB/VOB already in m s-1 (no conversion)

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
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[..., pd.DataFrame]]:
        """Get item from PrepBUFR vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - PrepBUFR vocab string (mnemonic or ``"wind::u"``/``"wind::v"``)
            - A modifier function to apply to the loaded DataFrame.  The
              modifier converts the ``observation`` column from raw PrepBUFR
              units to Earth2Studio standard units.
        """
        bufr_key = cls.VOCAB[val]

        if val == "t":
            # TOB is in DEG C; convert to Kelvin
            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] + 273.15)
                return df

        elif val == "q":
            # QOB is in mg/kg; convert to kg kg-1
            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 1e-6)
                return df

        elif val == "pres":
            # POB is in hPa (MB); convert to Pa
            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 100.0)
                return df

        else:
            # u, v already in m s-1
            def mod(df: pd.DataFrame) -> pd.DataFrame:
                return df

        return bufr_key, mod
