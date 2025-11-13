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

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import LexiconType


def _build_variable_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mappings between Earth2Studio (E2S) and FME variable names.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Mapping from Earth2Studio variable names to FME variable names and vice versa
    """
    mapping: dict[str, str] = {
        # Near-surface and forcing
        "u10m": "UGRD10m",
        "v10m": "VGRD10m",
        "t2m": "TMP2m",
        "q2m": "Q2m",
        "sp": "PRESsfc",
        "skt": "surface_temperature",
        "z": "HGTsfc",
        "mtdwswrf": "DSWRFtoa",
        "land_abs": "land_fraction",
        "ocean_abs": "ocean_fraction",
        "sic_abs": "sea_ice_fraction",
        "global_mean_co2": "global_mean_co2",
        "t850": "TMP850",
        "z500": "h500",
        # Precip: E2S "tp" is accumulation; map to rate placeholder
        "tp": "PRATEsfc",
        # Diagnostics
        "mtuwlwrf": "ULWRFtoa",
        "msuwlwrf": "ULWRFsfc",
        "msdwswrf": "DSWRFsfc",
        "msdwlwrf": "DLWRFsfc",
        "msuwswrf": "USWRFsfc",
        "mtuwswrf": "USWRFtoa",
        "msshf": "SHTFLsfc",
        "mslhf": "LHTFLsfc",
        "mttwp": "tendency_of_total_water_path_due_to_advection",
    }
    # Model levels (k = 0..7)
    for k in range(8):
        mapping[f"u{k}k"] = f"eastward_wind_{k}"
        mapping[f"v{k}k"] = f"northward_wind_{k}"
        mapping[f"t{k}k"] = f"air_temperature_{k}"
        mapping[f"qtot{k}k"] = f"specific_total_water_{k}"

    fme_to_e2s = {v: k for k, v in mapping.items()}
    return mapping, fme_to_e2s


# Public constants for reuse across model and data modules
E2S_TO_FME, FME_TO_E2S = _build_variable_mappings()


class ACELexicon(metaclass=LexiconType):
    """ACE Lexicon

    Maps Earth2Studio variable names to the variable labels used by the FME ACE stepper.
    Users may also access the ``get_e2s_from_fme`` method to get the reverse mapping.

    Examples
    --------
    >>> from earth2studio.lexicon.ace import ACELexicon
    >>> ACELexicon["u10m"]
    ("UGRD10m", <function mod at 0x...>)
    >>> ACELexicon.get_e2s_from_fme("UGRD10m")
    "u10m"
    """

    VOCAB: dict[str, str] = dict(E2S_TO_FME)
    VOCAB_REVERSE: dict[str, str] = dict(FME_TO_E2S)

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ACE/FME vocabulary.

        Parameters
        ----------
        val : str
            Name in Earth2Studio terminology.

        Returns
        -------
        tuple[str, Callable]
            FME/ACE variable name and modifier function.
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
            Name in FME/ACE terminology.
        """
        return cls.VOCAB_REVERSE[val]
