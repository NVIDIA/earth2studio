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

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import LexiconType


def _build_variable_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mappings between Earth2Studio and FME variable names
    for the SamudrACE coupled atmosphere-ocean model.

    The SamudrACE model uses CM4 model-level naming for atmosphere variables
    and CMIP6 naming conventions for ocean variables. This mapping covers
    all prognostic, forcing, and diagnostic variables across both components.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Mapping from Earth2Studio variable names to FME variable names and
        vice versa.
    """
    mapping: dict[str, str] = {
        # ---- Atmosphere: near-surface and surface ----
        "u10m": "UGRD10m",
        "v10m": "VGRD10m",
        "t2m": "TMP2m",
        "q2m": "Q2m",
        "sp": "PRESsfc",
        "skt": "surface_temperature",
        "z": "HGTsfc",
        # ---- Atmosphere: forcing ----
        "mtdwswrf": "DSWRFtoa",
        "land_abs": "land_fraction",
        "ocean_abs": "ocean_fraction",
        "sic_abs": "sea_ice_fraction",
        # ---- Atmosphere: diagnostics ----
        "mslhf": "LHTFLsfc",
        "msshf": "SHTFLsfc",
        "tp": "PRATEsfc",
        "msuwlwrf": "ULWRFsfc",
        "mtuwlwrf": "ULWRFtoa",
        "msdwlwrf": "DLWRFsfc",
        "msdwswrf": "DSWRFsfc",
        "msuwswrf": "USWRFsfc",
        "mtuwswrf": "USWRFtoa",
        "mttwp": "tendency_of_total_water_path_due_to_advection",
        "ewss": "eastward_surface_wind_stress",
        "nsss": "northward_surface_wind_stress",
        "t850": "TMP850",
        "z500": "h500",
        # ---- Atmosphere: additional forcing ----
        "lake_abs": "lake_fraction",
        # ---- Ocean: surface prognostic ----
        "sst": "sst",
        "zos": "zos",
        "sithick": "HI",
        "siconc": "ocean_sea_ice_fraction",
    }

    # ---- Atmosphere: CM4 model levels (k=0..7) ----
    for k in range(8):
        mapping[f"u{k}k"] = f"eastward_wind_{k}"
        mapping[f"v{k}k"] = f"northward_wind_{k}"
        mapping[f"t{k}k"] = f"air_temperature_{k}"
        mapping[f"qtot{k}k"] = f"specific_total_water_{k}"

    # ---- Ocean: depth-level variables (19 levels, 0..18) ----
    for d in range(19):
        mapping[f"thetao_{d}"] = f"thetao_{d}"
        mapping[f"so_{d}"] = f"so_{d}"
        mapping[f"uo_{d}"] = f"uo_{d}"
        mapping[f"vo_{d}"] = f"vo_{d}"

    fme_to_e2s = {v: k for k, v in mapping.items()}
    return mapping, fme_to_e2s


# Public constants for reuse across model and data modules
E2S_TO_FME, FME_TO_E2S = _build_variable_mappings()


class SamudrACELexicon(metaclass=LexiconType):
    """SamudrACE Lexicon

    Maps Earth2Studio variable names to the variable labels used by the FME
    CoupledStepper for the SamudrACE coupled atmosphere-ocean model. Covers
    both atmosphere (CM4 model levels) and ocean (CMIP6 naming) variables.

    Examples
    --------
    >>> from earth2studio.lexicon.samudrace import SamudrACELexicon
    >>> SamudrACELexicon["u10m"]
    ("UGRD10m", <function mod at 0x...>)
    >>> SamudrACELexicon.get_e2s_from_fme("sst")
    "sst"
    """

    VOCAB: dict[str, str] = dict(E2S_TO_FME)
    VOCAB_REVERSE: dict[str, str] = dict(FME_TO_E2S)

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in SamudrACE/FME vocabulary.

        Parameters
        ----------
        val : str
            Name in Earth2Studio terminology.

        Returns
        -------
        tuple[str, Callable]
            FME/SamudrACE variable name and modifier function.
        """
        fme_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            return x

        return fme_key, mod

    @classmethod
    def get_e2s_from_fme(cls, val: str) -> str:
        """Return name in Earth2Studio terminology.

        Parameters
        ----------
        val : str
            Name in FME/SamudrACE terminology.

        Returns
        -------
        str
            Earth2Studio variable name.
        """
        return cls.VOCAB_REVERSE[val]
