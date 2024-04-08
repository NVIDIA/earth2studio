# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Callable

import numpy as np

from .base import LexiconType


class HRRRLexicon(metaclass=LexiconType):
    """High-Resolution Rapid Refresh Lexicon
    HRRR specified <Provider ID>::<Parameter ID>::<Level/ Layer>

    Note
    ----
    Additional resources:
    https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRRR vocab dictionary"""
        sfc_variables = {
            "u10m": "sfc::UGRD::10 m above ground",
            "v10m": "sfc::VGRD::10 m above ground",
            "u80m": "sfc::UGRD::80 m above ground",
            "v80m": "sfc::VGRD::80 m above ground",
            "t2m": "sfc::TMP::2 m above ground",
            "sp": "sfc::PRES::surface",
            "tcwv": "sfc::PWAT::entire atmosphere (considered as a single layer)",
        }
        prs_levels = [
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            525,
            550,
            575,
            600,
            625,
            650,
            675,
            700,
            725,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
        ]
        prs_names = ["UGRD", "VGRD", "HGT", "TMP", "RH", "SPFH"]
        e2s_id = ["u", "v", "z", "t", "r", "q"]
        prs_variables = {}
        for (id, variable) in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"prs::{variable}::{level} mb"

        return {**sfc_variables, **prs_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[1] == "HGT":

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return hrrr_key, mod
