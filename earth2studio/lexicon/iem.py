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


class IEM_ASOSLexicon(metaclass=LexiconType):
    """Iowa Environmental Mesonet ASOS/AWOS observation lexicon.

    The mappings only reference fields parsed by IEM. The unprocessed ``metar``
    field is intentionally not exposed.

    Note
    ----
    The ``fg10m`` variable has lower coverage than routine fields because IEM
    only populates ``gust`` when a station reports a wind gust.

    Additional resources:

    - https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
    - https://mesonet.agron.iastate.edu/request/download.phtml
    """

    VOCAB: dict[str, str] = {
        "t2m": "tmpf",
        "d2m": "dwpf",
        "r2m": "relh",
        "ws10m": "sknt",
        "u10m": "sknt::drct",
        "v10m": "sknt::drct",
        "fg10m": "gust",
        "tp01": "p01i",
        "msl": "mslp",
        "tcc": "skyc1::skyc2::skyc3::skyc4",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get an IEM field mapping and conversion function.

        Parameters
        ----------
        val : str
            Earth2Studio variable ID.

        Returns
        -------
        tuple[str, Callable]
            IEM parsed field key and a function that converts a parsed frame to
            Earth2Studio units.
        """
        source_key = cls.VOCAB[val]

        def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
            return pd.to_numeric(frame[column], errors="coerce")

        if val in {"t2m", "d2m"}:
            column = source_key

            def mod(frame: pd.DataFrame) -> pd.Series:
                return (numeric(frame, column) - 32.0) * (5.0 / 9.0) + 273.15

        elif val in {"ws10m", "fg10m"}:
            column = source_key

            def mod(frame: pd.DataFrame) -> pd.Series:
                return numeric(frame, column) * 0.514444

        elif val in {"u10m", "v10m"}:

            def mod(frame: pd.DataFrame) -> pd.Series:
                speed = numeric(frame, "sknt") * 0.514444
                direction = numeric(frame, "drct")
                radians = np.deg2rad(direction)
                component = -speed * (
                    np.sin(radians) if val == "u10m" else np.cos(radians)
                )
                component = component.where(speed.notna() & direction.notna())
                return component.mask(speed == 0.0, 0.0)

        elif val == "tp01":

            def mod(frame: pd.DataFrame) -> pd.Series:
                return numeric(frame, "p01i") * 0.0254

        elif val == "msl":

            def mod(frame: pd.DataFrame) -> pd.Series:
                return numeric(frame, "mslp") * 100.0

        elif val == "tcc":

            def mod(frame: pd.DataFrame) -> pd.Series:
                cover = {
                    "CLR": 0.0,
                    "SKC": 0.0,
                    "NSC": 0.0,
                    "NCD": 0.0,
                    "FEW": 0.25,
                    "SCT": 0.5,
                    "BKN": 0.875,
                    "OVC": 1.0,
                    "VV": 1.0,
                }
                layers = [
                    frame[column].astype("string").str.upper().map(cover)
                    for column in source_key.split("::")
                ]
                return pd.concat(layers, axis=1).max(axis=1, skipna=True)

        else:
            column = source_key

            def mod(frame: pd.DataFrame) -> pd.Series:
                return numeric(frame, column)

        return source_key, mod
