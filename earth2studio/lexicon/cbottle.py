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

from collections.abc import Callable

import numpy as np

from .base import LexiconType


class CBottleLexicon(metaclass=LexiconType):
    """CBottle Lexicon

    Maps between Earth2Studio vocab and the labels used in the original repo.
    The dictionary order of variables produced by the model.


    Note
    ----
    Additional resources:
    https://github.com/NVlabs/cBottle
    """

    @staticmethod
    def build_vocab() -> dict[str, tuple[str, int]]:
        """Create CBottle vocab dictionary"""

        surface_params = {
            "tcwv": ("tcwv", -1),
            "cllvi": ("cllvi", -1),  # liquid water path
            "clivi": ("clivi", -1),  # ice water path
            "t2m": ("tas", -1),  # t2m
            "u2m": ("uas", -1),  # u2m
            "vas": ("vas", -1),  # v2m
            "olr": ("rlut", -1),  # Outgoing long wave radiation
            "osr": ("rsut", -1),  # Outgoing short wave radiation
            "msl": ("pres_msl", -1),  # mean sea level pressure
            "tp": ("pr", -1),  # total precip
            "rsds": ("rsds", -1),  # downwelling SW at surface
            "sst": ("sst", -1),  # sea surface temp
            "sic": ("sic", -1),  # sea ice
        }

        pressure_params = ["u", "v", "t", "z"]
        pressure_levels = [1000, 850, 700, 500, 300, 200, 50, 10]

        vocab = {}
        for param in pressure_params:
            for level in pressure_levels:
                vocab[f"{param}{level}"] = (param.upper(), level)

        vocab.update(surface_params)
        return vocab

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[tuple[str, int], Callable]:
        """Return name in CBottle vocabulary

        Parameters
        ----------
        val : str
            Name in Earth-2 Studio terminology.

        Returns
        -------
        tuple[str, Callable]
            CBottle name and modifier function.
        """
        cbottle_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify value (if necessary)."""
            return x

        return cbottle_key, mod
