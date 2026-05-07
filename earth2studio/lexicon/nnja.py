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


class NNJAObsConvLexicon(metaclass=LexiconType):
    """NOAA-NASA Joint Archive (NNJA) lexicon for conventional (in-situ
    and GPS RO) observations.

    Maps Earth2Studio variable names to a route-prefixed source key.
    The route prefix selects which NNJA archive folder (and decoder)
    the data source uses:

    - ``prepbufr::<mnemonic>`` -- read from ``conv/prepbufr/`` PrepBUFR
      cycle files, decoded with pybufrkit using the NCEP-local DX tables
      embedded in the file. Mnemonic is one of ``TOB``/``QOB``/``POB`` for
      pressure/temperature/specific humidity, or ``wind::u``/``wind::v``
      for wind components decomposed from UOB/VOB.
    - ``gpsro::<descriptor_id>`` -- read from ``gps/gpsro/`` GPS RO
      occultation BUFR cycle files. ``descriptor_id`` is the BUFR
      descriptor of the per-level field to emit:

      - ``15037`` -- bending angle (rad), at impact-parameter levels.
      - ``12001`` -- 1D-Var retrieval temperature (K), at retrieval levels.
      - ``13001`` -- 1D-Var retrieval specific humidity (kg/kg).

    Modifier functions convert raw PrepBUFR observation values to
    Earth2Studio standard units:

    - ``t``: TOB (DEG C) -> Kelvin (+273.15)
    - ``q``: QOB (mg/kg) -> kg kg-1 (/1e6)
    - ``pres``: POB (hPa / MB) -> Pa (*100)
    - ``u``, ``v``: UOB/VOB already in m s-1 (no conversion)
    - ``gps``, ``gps_t``, ``gps_q``: already in SI (rad, K, kg/kg)

    Note
    ----
    This lexicon parallels :py:class:`earth2studio.lexicon.GDASObsConvLexicon`
    (PrepBUFR variables) and :py:class:`earth2studio.lexicon.GSIConventionalLexicon`
    (UFS GSI variables, including ``gps``/``gps_t``/``gps_q``) but is kept
    separate to keep the NNJA data source self-contained.

    Additional resources:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
    - NCEP gpsro BUFR descriptor reference (BUFR Table B 0-15-037, 0-12-001, 0-13-001)
    """

    VOCAB: dict[str, str] = {
        "u": "prepbufr::wind::u",
        "v": "prepbufr::wind::v",
        "q": "prepbufr::QOB",
        "t": "prepbufr::TOB",
        "pres": "prepbufr::POB",
        # GPS Radio Occultation, from gps/gpsro/ archive
        # Removing these for now, consistency issues with UFS
        # "gps": "gpsro::15037",
        # "gps_t": "gpsro::12001",
        # "gps_q": "gpsro::13001",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from the NNJA conventional vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - Route-prefixed source key (``"prepbufr::..."`` or ``"gpsro::..."``).
            - A modifier function applied to the loaded DataFrame to convert
              the ``observation`` column from raw BUFR units to Earth2Studio
              standard units.
        """
        source_key = cls.VOCAB[val]

        if val == "t":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] + 273.15)
                return df

        elif val == "q":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 1e-6)
                return df

        elif val == "pres":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 100.0)
                return df

        else:

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                return df

        return source_key, mod
