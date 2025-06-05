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
    https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/dataset_2d.py#L25
    https://ceres.larc.nasa.gov/documents/cmip5-data/Tech-Note_CERES-EBAF-Surface_L3B_Ed2-8.pdf
    https://ceres.larc.nasa.gov/documents/cmip5-data/Tech-Note_rlut_CERES-EBAF_L3B_Ed2-6r_20121101.pdf
    """

    @staticmethod
    def build_vocab() -> dict[str, tuple[str, int]]:
        """Create CBottle vocab dictionary"""

        surface_params = {
            "tcwv": ("tcwv", -1),
            "tclw": ("cllvi", -1),  # liquid water path
            "tciw": ("clivi", -1),  # ice water path
            "t2m": ("tas", -1),  # t2m
            "u10m": ("uas", -1),  # zonal wind at surface
            "v10m": ("vas", -1),  # meridional at surface
            "rlut": ("rlut", -1),  # Outgoing long wave radiation
            "rsut": ("rsut", -1),  # Outgoing short wave radiation
            "msl": ("pres_msl", -1),  # mean sea level pressure
            "tpf": ("pr", -1),  # precip flux
            "rsds": ("rsds", -1),  # downwelling SW at surface
            "sst": ("sst", -1),  # sea surface temp
            "sic": ("sic", -1),  # sea ice concentration
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
