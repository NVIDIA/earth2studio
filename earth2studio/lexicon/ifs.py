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


class IFSLexicon(metaclass=LexiconType):
    """Integrated Forecast System Lexicon
    IFS specified <Grib Parameter ID>::<Level Type>::<Level>

    Note
    ----
    Additional resources:
    https://codes.ecmwf.int/grib/param-db/?filter=grib2
    https://www.ecmwf.int/en/forecasts/datasets/open-data
    Best bet is to download an index file from the AWS bucket and read it
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRRR vocab dictionary"""
        sfc_variables = {
            "u10m": "10u::sfc::",
            "v10m": "10v::sfc::",
            "u100m": "100u::sfc::",
            "v100m": "100v::sfc::",
            "t2m": "2t::sfc::",
            "d2m": "2d::sfc::",
            "sp": "sp::sfc::",
            "msl": "msl::sfc::",
            "tcw": "tcw::sfc::",  #
            "tcwv": "tcwv::sfc::",
            "tp": "tp::sfc::",
            "skt": "skt::sfc::",
            "slor": "slor::sfc::",
            "sdor": "sdor::sfc::",
            "lsm": "lsm::sfc::",
            "zsl": "z::sfc::",
        }
        soil_variables = {
            "swvl1": "vsw::sl::1",
            "swvl2": "vsw::sl::2",
            "stl1": "sot::sl::1",
            "stl2": "sot::sl::2",
        }
        prs_levels = [
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
        prs_names = ["u", "v", "w", "gh", "t", "r", "q"]
        e2s_id = ["u", "v", "w", "z", "t", "r", "q"]
        prs_variables = {}
        for id, variable in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"{variable}::pl::{level}"

        return {**sfc_variables, **soil_variables, **prs_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Retrieve name from IFS vocabulary."""
        ifs_key = cls.VOCAB[val]
        if ifs_key.split("::")[0] == "gh":

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return ifs_key, mod
