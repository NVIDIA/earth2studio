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
            "u10m": ("uas", -1),  # eastward 10-m wind (m s-1)
            "v10m": ("vas", -1),  # northward 10-m wind (m s-1)
            "uas": ("uas", -1),  # eastward near-surface wind (m s-1)
            "vas": ("vas", -1),  # northward near-surface wind (m s-1)
            "t2m": ("tas", -1),  # near-surface air temperature (K)
            "tas": ("tas", -1),  # near-surface air temperature (K)
            "tasmax": ("tasmax", -1),  # maximum near-surface air temperature (K)
            "tasmin": ("tasmin", -1),  # minimum near-surface air temperature (K)
            "q2m": ("huss", -1),  # near-surface specific humidity
            "huss": ("huss", -1),  # near-surface specific humidity
            "msl": ("psl", -1),  # mean sea-level pressure (Pa)
            "psl": ("psl", -1),  # mean sea-level pressure (Pa)
            "tp": ("pr", -1),  # accumulated precipitation (kg m-2 s-1)
            "pr": ("pr", -1),  # precipitation rate (kg m-2 s-1)
            "rlut": ("rlut", -1),  # TOA Outgoing Longwave Radiation
            "rlus": ("rlus", -1),  # surface upwelling longwave radiation (W/m2)
            "rsds": ("rsds", -1),  # surface downwelling shortwave radiation
            "tcc": ("clt", -1),  # total cloud cover (0-1)
            "clt": ("clt", -1),  # total cloud cover (%)
            "hfls": ("hfls", -1),  # surface upward latent heat flux (W/m2)
            "hfss": ("hfss", -1),  # surface upward sensible heat flux (W/m2)
            "prc": ("prc", -1),  # convective precipitation rate (kg m-2 s-1)
            "rlds": ("rlds", -1),  # surface downwelling longwave radiation (W/m2)
            "rls": ("rls", -1),  # surface upwelling longwave radiation (W/m2)
            "rss": ("rss", -1),  # surface downwelling shortwave radiation (W/m2)
            "rsus": ("rsus", -1),  # surface upwelling shortwave radiation (W/m2)
            "sfcWind": ("sfcWind", -1),  # near-surface wind speed (m s-1)
            "sfcWindmax": ("sfcWindmax", -1),  # maximum near-surface wind speed (m s-1)
            "snc": ("snc", -1),  # snow area fraction (%)
            "snw": ("snw", -1),  # surface snow water equivalent (kg m-2)
            "sst": ("tos", -1),  # sea surface temperature (K)
            "tos": ("tos", -1),  # sea surface temperature (K)
            # TODO: The following variables could be in the general CMIP6 lexicon
            # however they are not currently validated. We will leave these commented
            # out for now until they are needed.
            # "d2m": ("tdps", -1),  # near-surface dew-point temperature
            "r2m": ("hurs", -1),  # near-surface relative humidity (%)
            "hurs": ("hurs", -1),  # near-surface relative humidity (%)
            "hursmax": ("hursmax", -1),  # maximum near-surface relative humidity (%)
            "hursmin": ("hursmin", -1),  # minimum near-surface relative humidity (%)
            # "sic": ("siconc", -1), # sea ice concentration
            # "sp": ("ps", -1),    # surface pressure
            # "tcwv": ("prw", -1), # total column water vapor
            # "rsut": ("rsut", -1), # outgoing shortwave radiation
            "ts": ("ts", -1),  # surface temperature (K)
            "siconc": ("siconc", -1),  # sea ice concentration (alias of sic) [%]
            # "lsm": ("sftlf", -1), # land-sea mask (for cmip6 this is fraction of land)
            # "z": ("orog", -1),   # surface geopotential height (orography)
        }

        param_map = {
            "u": "ua",  # eastward wind (m s-1)
            "ua": "ua",  # eastward wind (m s-1)
            "v": "va",  # northward wind (m s-1)
            "va": "va",  # northward wind (m s-1)
            "t": "ta",  # air temperature (K)
            "ta": "ta",  # air temperature (K)
            "z": "zg",  # geopotential height (m2 s-2)
            "zg": "zg",  # geopotential height (m)
            # "r": "hur",  # relative humidity # NOTE: Currently we don't include relative humidity
            "q": "hus",  # specific humidity
            "hus": "hus",  # specific humidity
            "w": "wap",  # vertical velocity (Pa s-1)
            "wap": "wap",  # vertical velocity (Pa s-1)
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

        if val in ["sst", "tos"]:  # sea-surface temperature may be in degC → convert

            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Ensure SST is returned in kelvin.

                Heuristic: if the data are mostly below 100 we assume degrees
                Celsius and add 273.15; otherwise we assume already kelvin.

                Note: Both E2S "sst" and "tos" map to CMIP6 "tos", so both need this conversion.
                """
                if np.nanmean(x) < 100:  # likely °C
                    return x + 273.15
                return x

        elif val in ["siconc", "clt", "r2m", "hurs", "hursmax", "hursmin", "snc"]:
            # Convert concentration/percentage variables from fraction [0-1] to percentage [0-100]
            # Applies to: cloud cover (clt), relative humidity (r2m/hurs/hursmax/hursmin),
            # sea ice concentration (siconc), snow cover (snc)
            # Note: E2S variable "tcc" is NOT included - it expects fraction [0-1], not percentage
            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Convert fraction [0-1] to percentage [0-100] if data is in fraction format.

                Uses heuristic: if mean value < 1, assumes fraction and multiplies by 100.
                """
                if np.nanmean(x) < 1:
                    return x * 100.0
                else:
                    return x  # already in percentage

        elif val == "orog" or (
            val.startswith("z") and len(val) > 1 and val[1].isdigit()
        ):
            # Earth2Studio "z<level>" (e.g., z50, z500) needs conversion from meters to m^2/s^2
            # Earth2Studio "zg" (geopotential height) does NOT need conversion (already in m)
            # "orog" (orography/surface geopotential) also needs conversion
            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Convert geopotential height from meters to m^2 s^-2 (potential)."""
                return x * 9.80665

        else:

            def mod(x: np.ndarray) -> np.ndarray:  # noqa: D401
                """Identity modifier (no-op)."""
                return x

        return cmip6_key, mod
