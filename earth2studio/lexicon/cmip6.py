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
    NOTE: CMIP6 has substantially more variables than in this lexicon.

    References
    ----------
    * https://clipc-services.ceda.ac.uk/dreq/mipVars.html
    * https://github.com/PCMDI/cmip6-cmor-tables
    """

    @staticmethod
    def build_vocab() -> dict[str, tuple[str, int]]:
        """Create mapping from Earth2Studio variable names to CMIP6 identifiers."""

        surface_params: dict[str, tuple[str, int]] = {
            "u10m": ("uas", -1),  # eastward 10-m wind
            "v10m": ("vas", -1),  # northward 10-m wind
            "t2m": ("tas", -1),  # 2-m air temperature
            "q2m": ("huss", -1),  # surface specific humidity
            "msl": ("psl", -1),  # mean sea-level pressure
            "tp": ("pr", -1),  # accumulated precipitation (kg m-2 s-1)
            "rlut": ("rlut", -1),  # outgoing longwave radiation
            "rsds": ("rsds", -1),  # surface downwelling shortwave radiation
            "tcc": ("clt", -1),  # total cloud cover (0-1)
            "hfls": ("hfls", -1),  # surface latent heat flux
            "prc": ("prc", -1),  # convective precipitation rate (kg m-2 s-1)
            "rlds": ("rlds", -1),  # surface downwelling longwave radiation
            "rls": ("rls", -1),  # surface upwelling longwave radiation
            "rss": ("rss", -1),  # shortwave radiation
            "rsus": ("rsus", -1),  # surface upwelling shortwave radiation
            "sfcwind": ("sfcWind", -1),  # surface wind speed
            "snc": ("snc", -1),  # snow area fraction
            "snw": ("snw", -1),  # surface snow water equivalent (kg m-2)
            "sst": ("tos", -1),  # sea surface temperature
            # TODO: The following variables could be in the general CMIP6 lexicon
            # however they are not currently validated. We will leave these commented
            # out for now until they are needed.
            # "d2m": ("tdps", -1),  # surface dew-point temperature
            # "r2m": ("hurs", -1),  # surface relative humidity
            # "sic": ("siconc", -1), # sea ice concentration
            # "sp": ("ps", -1),    # surface pressure
            # "tcwv": ("prw", -1), # total column water vapor
            # "rsut": ("rsut", -1), # outgoing shortwave radiation
            # "ts": ("ts", -1), # skin temperature
            # "siconc": ("siconc", -1), # sea ice concentration (alias of sic)
            # "lsm": ("sftlf", -1), # land-sea mask (for cmip6 this is fraction of land)
            # "zsl": ("zg", -1),   # geopotential height at mean sea-level
            # "z": ("orog", -1),   # surface geopotential height (orography)
        }

        param_map = {
            "u": "ua",  # eastward wind
            "v": "va",  # northward wind
            "t": "ta",  # air temperature
            "z": "zg",  # geopotential height
            # "r": "hur",  # relative humidity # NOTE: Currently we don't include relative humidity
            "q": "hus",  # specific humidity
            "w": "wap",  # vertical velocity (Pa s-1)
        }
        pressure_levels = [
            10,
            50,
            100,
            250,
            500,
            700,
            850,
            1000,
        ]

        vocab: dict[str, tuple[str, int]] = {}
        for e2_prefix, cmip6_id in param_map.items():
            for level in pressure_levels:
                vocab[f"{e2_prefix}{level}"] = (cmip6_id, level)

        vocab.update(surface_params)
        return vocab

    VOCAB = build_vocab.__func__()  # type: ignore[misc,attr-defined]

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
            *((cmip6_variable_id, level_hPa), modifier)*
        """
        cmip6_key = cls.VOCAB[val]

        if val == "sst":  # sea-surface temperature may be in degC → convert

            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Ensure SST is returned in kelvin.

                Heuristic: if the data are mostly below 100 we assume degrees
                Celsius and add 273.15; otherwise we assume already kelvin.
                """
                if np.nanmean(x) < 100:  # likely °C
                    return x + 273.15
                return x

        elif cmip6_key[0] in ["zg", "z", "orog"]:

            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Convert geopotential height from meters to m^2 s^-2 (potential)."""
                return x * 9.80665

        else:

            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Identity modifier (no-op)."""
                return x

        return cmip6_key, mod
