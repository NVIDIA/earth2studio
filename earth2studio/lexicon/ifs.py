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

    VOCAB = {
        "u10m": "10u::sfc::",
        "v10m": "10v::sfc::",
        "t2m": "2t::sfc::",
        "sp": "sp::sfc::",
        "msl": "msl::sfc::",
        "tcwv": "tcwv::sfc::",
        "tp": "tp::sfc::",
        "u50": "u::pl::50",
        "u200": "u::pl::200",
        "u250": "u::pl::250",
        "u300": "u::pl::300",
        "u500": "u::pl::500",
        "u700": "u::pl::700",
        "u850": "u::pl::850",
        "u925": "u::pl::925",
        "u1000": "u::pl::1000",
        "v50": "v::pl::50",
        "v200": "v::pl::200",
        "v250": "v::pl::250",
        "v300": "v::pl::300",
        "v500": "v::pl::500",
        "v700": "v::pl::700",
        "v850": "v::pl::850",
        "v925": "v::pl::925",
        "v1000": "v::pl::1000",
        "z50": "gh::pl::50",
        "z200": "gh::pl::200",
        "z250": "gh::pl::250",
        "z300": "gh::pl::300",
        "z500": "gh::pl::500",
        "z700": "gh::pl::700",
        "z850": "gh::pl::850",
        "z925": "gh::pl::925",
        "z1000": "gh::pl::1000",
        "t50": "t::pl::50",
        "t200": "t::pl::200",
        "t250": "t::pl::250",
        "t300": "t::pl::300",
        "t500": "t::pl::500",
        "t700": "t::pl::700",
        "t850": "t::pl::850",
        "t925": "t::pl::925",
        "t1000": "t::pl::1000",
        "r50": "r::pl::50",
        "r200": "r::pl::200",
        "r250": "r::pl::250",
        "r300": "r::pl::300",
        "r500": "r::pl::500",
        "r700": "r::pl::700",
        "r850": "r::pl::850",
        "r925": "r::pl::925",
        "r1000": "r::pl::1000",
        "q50": "q::pl::50",
        "q200": "q::pl::200",
        "q250": "q::pl::250",
        "q300": "q::pl::300",
        "q500": "q::pl::500",
        "q700": "q::pl::700",
        "q850": "q::pl::850",
        "q925": "q::pl::925",
        "q1000": "q::pl::1000",
    }

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
