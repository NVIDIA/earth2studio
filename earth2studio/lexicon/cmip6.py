# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import numpy as np

from .base import LexiconType


class CMIP6Lexicon(metaclass=LexiconType):
    """CMIP6 Lexicon
    
    Maps between the Earth2Studio vocabulary and CMIP6 variable names.
    TODO: CMIP6 has substantially more variables than in this lexicon.
    
    References
    ----------
    * https://clipc-services.ceda.ac.uk/dreq/mipVars.html
    * https://github.com/PCMDI/cmip6-cmor-tables
    """

    @staticmethod
    def build_vocab() -> dict[str, tuple[str, int]]:
        """Create mapping from Earth2Studio variable names to CMIP6 identifiers.

        Returns
        -------
        dict
            Keys are Earth2Studio variable short-names, values are
            *(cmip6_variable_id, level_hPa)* tuples where *level_hPa = -1*
            for single-level variables.
        """

        surface_params: dict[str, tuple[str, int]] = {
            "u10m": ("uas", -1),  # eastward 10-m wind
            "v10m": ("vas", -1),  # northward 10-m wind
            "t2m": ("tas", -1),   # 2-m air temperature
            "d2m": ("tdps", -1),  # surface dew-point temperature
            "r2m": ("hurs", -1),  # surface relative humidity
            "q2m": ("huss", -1),  # surface specific humidity
            "sic": ("siconc", -1), # sea ice concentration
            "sst": ("tos", -1), # sea surface temperature
            "sp": ("ps", -1),    # surface pressure
            "msl": ("psl", -1),  # mean sea-level pressure
            "tcwv": ("prw", -1), # total column water vapor
            "tp": ("pr", -1),    # accumulated precipitation (kg m-2 s-1)
            "rlut": ("rlut", -1), # outgoing longwave radiation
            "rsut": ("rsut", -1), # outgoing shortwave radiation
            "rsds": ("rsds", -1), # surface downwelling shortwave radiation
            "lsm": ("sftlf", -1), # land-sea mask (for cmip6 this is fraction of land)
            "zsl": ("zg", -1),   # geopotential height at mean sea-level
            "z": ("orog", -1),   # surface geopotential height (orography)
        }

        param_map = {
            "u": "ua",   # eastward wind
            "v": "va",   # northward wind
            "t": "ta",   # air temperature
            "z": "zg",   # geopotential height
            "r": "hur",  # relative humidity
            "q": "hus",  # specific humidity
            "w": "wap",  # vertical velocity (Pa s-1)
        }
        pressure_levels = [
            50,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            850,
            925,
            1000,
        ]

        vocab: dict[str, tuple[str, int]] = {}
        for e2_prefix, cmip6_id in param_map.items():
            for level in pressure_levels:
                vocab[f"{e2_prefix}{level}"] = (cmip6_id, level)

        vocab.update(surface_params)
        return vocab

    VOCAB = build_vocab.__func__()  # type: ignore[misc]

    @classmethod
    def get_item(cls, val: str) -> tuple[tuple[str, int], Callable]:
        """Return CMIP6 mapping and modifier function.

        Parameters
        ----------
        val : str
            Earth2Studio variable name.

        Returns
        -------
        tuple
            *(cmip6_variable_id, level_hPa)* and a modifier callable.
        """
        cmip6_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:  # noqa: D401 – simple modifier
            """Identity modifier (placeholder for future unit conversions)."""
            return x

        return cmip6_key, mod