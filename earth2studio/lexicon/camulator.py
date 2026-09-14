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

CAMULATOR_LEVELS = 32

# CAMulator step length; flux accumulations are converted to mean rates over it.
CAMULATOR_STEP_SECONDS = 21600.0


def _build_variable_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mappings between Earth2Studio and CESM/CAMulator
    variable names.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Earth2Studio -> CESM and CESM -> Earth2Studio name maps
    """
    mapping: dict[str, str] = {}
    # Hybrid sigma-pressure model levels, k = 0 (top, ~3.6 hPa) .. 31 (bottom)
    for k in range(CAMULATOR_LEVELS):
        mapping[f"u{k}k"] = f"U::{k}"
        mapping[f"v{k}k"] = f"V::{k}"
        mapping[f"t{k}k"] = f"T::{k}"
        mapping[f"qtot{k}k"] = f"Qtot::{k}"
    mapping.update(
        {
            # 2D prognostic
            "sp": "PS",
            "t2m": "TREFHT",
            # Diagnostics (output only)
            "tp06": "PRECT",
            "skt": "TS",
            "hcc": "CLDHGH",
            "lcc": "CLDLOW",
            "mcc": "CLDMED",
            "iews": "TAUX",
            "inss": "TAUY",
            "ws10m": "U10",
            "e06": "QFLX",
            "msdwswrf": "FSDS_J",
            "msdwlwrf": "FLDS_J",
            "msuwswrf": "FSUS",
            "msuwlwrf": "FLUS",
            "msshf": "SHFLX",
            "mslhf": "LHFLX",
            "mtuwswrf": "FSUTOA",
            "mtuwlwrf": "FLUT",
            # Forcing (input only)
            "mtdwswrf": "SOLIN",
            "sst": "SST",
            "sic": "ICEFRAC",
            "global_mean_co2": "co2vmr_3d",
        }
    )
    reverse = {v: k for k, v in mapping.items()}
    return mapping, reverse


E2S_TO_CESM, CESM_TO_E2S = _build_variable_mappings()

# Fields CAMulator stores as J m-2 accumulated over its 6 h step; Earth2Studio
# exposes them as mean W m-2 over the step.
ACCUMULATED_FLUX_VARIABLES = (
    "msdwswrf",
    "msdwlwrf",
    "msuwswrf",
    "msuwlwrf",
    "msshf",
    "mslhf",
)


class CAMulatorLexicon(metaclass=LexiconType):
    """CAMulator Lexicon

    Maps Earth2Studio variable names to the CESM/CAM6 variable names used by the
    CAMulator climate emulator. 3D variables use the ``{var}{k}k`` model-level
    convention with ``k`` the CAMulator hybrid sigma-pressure level index
    (0 = top of model, 31 = lowest layer); these are distinct from the ACE
    model-level names. Values are ``NAME`` for 2D fields and ``NAME::k`` for level
    ``k`` of a 3D field. The modifier converts source units to Earth2Studio units
    (CO2 mol mol-1 -> ppm; 6 h accumulated fluxes J m-2 -> mean W m-2).

    Note
    ----
    Additional resources:

    - https://huggingface.co/willychap/camulator
    - https://arxiv.org/abs/2504.06007

    Examples
    --------
    >>> from earth2studio.lexicon.camulator import CAMulatorLexicon
    >>> CAMulatorLexicon["t2m"]
    ("TREFHT", <function mod at 0x...>)
    >>> CAMulatorLexicon.get_e2s_from_cesm("TREFHT")
    "t2m"
    """

    VOCAB: dict[str, str] = dict(E2S_TO_CESM)
    VOCAB_REVERSE: dict[str, str] = dict(CESM_TO_E2S)

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in CESM vocabulary and a modifier converting source units to
        Earth2Studio units.

        Parameters
        ----------
        val : str
            Name in Earth2Studio terminology.

        Returns
        -------
        tuple[str, Callable]
            CESM variable name and modifier function.
        """
        cesm_key = cls.VOCAB[val]

        if val == "global_mean_co2":

            def mod(x: np.ndarray) -> np.ndarray:
                return x * 1.0e6

        elif val in ACCUMULATED_FLUX_VARIABLES:

            def mod(x: np.ndarray) -> np.ndarray:
                return x / CAMULATOR_STEP_SECONDS

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                return x

        return cesm_key, mod

    @classmethod
    def get_e2s_from_cesm(cls, val: str) -> str:
        """Return name in Earth2Studio terminology.

        Parameters
        ----------
        val : str
            Name in CESM terminology (``NAME`` or ``NAME::k``).
        """
        return cls.VOCAB_REVERSE[val]
