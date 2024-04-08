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


class GFSLexicon(metaclass=LexiconType):
    """Global Forecast System Lexicon
    GFS specified <Parameter ID>::<Level/ Layer>

    Note
    ----
    Additional resources:
    https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml
    """

    VOCAB = {
        "u10m": "UGRD::10 m above ground",
        "v10m": "VGRD::10 m above ground",
        "u100m": "UGRD::100 m above ground",
        "v100m": "VGRD::100 m above ground",
        "t2m": "TMP::2 m above ground",
        "sp": "PRES::surface",
        "msl": "PRMSL::mean sea level",
        "tcwv": "PWAT::entire atmosphere (considered as a single layer)",
        "u50": "UGRD::50 mb",
        "u100": "UGRD::100 mb",
        "u150": "UGRD::150 mb",
        "u200": "UGRD::200 mb",
        "u250": "UGRD::250 mb",
        "u300": "UGRD::300 mb",
        "u400": "UGRD::400 mb",
        "u500": "UGRD::500 mb",
        "u600": "UGRD::600 mb",
        "u700": "UGRD::700 mb",
        "u850": "UGRD::850 mb",
        "u925": "UGRD::925 mb",
        "u1000": "UGRD::1000 mb",
        "v50": "VGRD::50 mb",
        "v100": "VGRD::100 mb",
        "v150": "VGRD::150 mb",
        "v200": "VGRD::200 mb",
        "v250": "VGRD::250 mb",
        "v300": "VGRD::300 mb",
        "v400": "VGRD::400 mb",
        "v500": "VGRD::500 mb",
        "v600": "VGRD::600 mb",
        "v700": "VGRD::700 mb",
        "v850": "VGRD::850 mb",
        "v925": "VGRD::925 mb",
        "v1000": "VGRD::1000 mb",
        "z50": "HGT::50 mb",
        "z100": "HGT::100 mb",
        "z150": "HGT::150 mb",
        "z200": "HGT::200 mb",
        "z250": "HGT::250 mb",
        "z300": "HGT::300 mb",
        "z400": "HGT::400 mb",
        "z500": "HGT::500 mb",
        "z600": "HGT::600 mb",
        "z700": "HGT::700 mb",
        "z850": "HGT::850 mb",
        "z925": "HGT::925 mb",
        "z1000": "HGT::1000 mb",
        "t50": "TMP::50 mb",
        "t100": "TMP::100 mb",
        "t150": "TMP::150 mb",
        "t200": "TMP::200 mb",
        "t250": "TMP::250 mb",
        "t300": "TMP::300 mb",
        "t400": "TMP::400 mb",
        "t500": "TMP::500 mb",
        "t600": "TMP::600 mb",
        "t700": "TMP::700 mb",
        "t850": "TMP::850 mb",
        "t925": "TMP::925 mb",
        "t1000": "TMP::1000 mb",
        "r50": "RH::50 mb",
        "r100": "RH::100 mb",
        "r150": "RH::150 mb",
        "r200": "RH::200 mb",
        "r250": "RH::250 mb",
        "r300": "RH::300 mb",
        "r400": "RH::400 mb",
        "r500": "RH::500 mb",
        "r600": "RH::600 mb",
        "r700": "RH::700 mb",
        "r850": "RH::850 mb",
        "r925": "RH::925 mb",
        "r1000": "RH::1000 mb",
        "q50": "SPFH::50 mb",
        "q100": "SPFH::100 mb",
        "q150": "SPFH::150 mb",
        "q200": "SPFH::200 mb",
        "q250": "SPFH::250 mb",
        "q300": "SPFH::300 mb",
        "q400": "SPFH::400 mb",
        "q500": "SPFH::500 mb",
        "q600": "SPFH::600 mb",
        "q700": "SPFH::700 mb",
        "q850": "SPFH::850 mb",
        "q925": "SPFH::925 mb",
        "q1000": "SPFH::1000 mb",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GFS vocabulary."""
        gfs_key = cls.VOCAB[val]
        if gfs_key.split("::")[0] == "HGT":

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return gfs_key, mod
