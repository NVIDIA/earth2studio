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


class GFSLexicon(metaclass=LexiconType):
    """Global Forecast System Lexicon
    GFS specified <Parameter ID>::<Level/ Layer>

    Warning
    -------
    Some variables are only present for lead time greater than 0

    Note
    ----
    Additional resources:
    https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml
    https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml
    """

    VOCAB = {
        "u10m": "UGRD::10 m above ground",
        "v10m": "VGRD::10 m above ground",
        "u100m": "UGRD::100 m above ground",
        "v100m": "VGRD::100 m above ground",
        "t2m": "TMP::2 m above ground",
        "d2m": "DPT::2 m above ground",
        "r2m": "RH::2 m above ground",
        "q2m": "SPFH::2 m above ground",
        "sp": "PRES::surface",
        "msl": "PRMSL::mean sea level",
        "tcwv": "PWAT::entire atmosphere (considered as a single layer)",
        "tp": "596::APCP::surface",  # 3 hour acc
        "2d": "DPT::2 m above ground",
        "fg10m": "GUST::surface",  # Surface
        "u1": "UGRD::1 mb",
        "u2": "UGRD::2 mb",
        "u3": "UGRD::3 mb",
        "u5": "UGRD::5 mb",
        "u7": "UGRD::7 mb",
        "u10": "UGRD::10 mb",
        "u15": "UGRD::15 mb",
        "u20": "UGRD::20 mb",
        "u30": "UGRD::30 mb",
        "u40": "UGRD::40 mb",
        "u50": "UGRD::50 mb",
        "u70": "UGRD::70 mb",
        "u100": "UGRD::100 mb",
        "u150": "UGRD::150 mb",
        "u200": "UGRD::200 mb",
        "u250": "UGRD::250 mb",
        "u300": "UGRD::300 mb",
        "u350": "UGRD::350 mb",
        "u400": "UGRD::400 mb",
        "u450": "UGRD::450 mb",
        "u500": "UGRD::500 mb",
        "u550": "UGRD::550 mb",
        "u600": "UGRD::600 mb",
        "u650": "UGRD::650 mb",
        "u700": "UGRD::700 mb",
        "u750": "UGRD::750 mb",
        "u800": "UGRD::800 mb",
        "u850": "UGRD::850 mb",
        "u900": "UGRD::900 mb",
        "u925": "UGRD::925 mb",
        "u950": "UGRD::950 mb",
        "u975": "UGRD::975 mb",
        "u1000": "UGRD::1000 mb",
        "v1": "VGRD::1 mb",
        "v2": "VGRD::2 mb",
        "v3": "VGRD::3 mb",
        "v5": "VGRD::5 mb",
        "v7": "VGRD::7 mb",
        "v10": "VGRD::10 mb",
        "v15": "VGRD::15 mb",
        "v20": "VGRD::20 mb",
        "v30": "VGRD::30 mb",
        "v40": "VGRD::40 mb",
        "v50": "VGRD::50 mb",
        "v70": "VGRD::70 mb",
        "v100": "VGRD::100 mb",
        "v150": "VGRD::150 mb",
        "v200": "VGRD::200 mb",
        "v250": "VGRD::250 mb",
        "v300": "VGRD::300 mb",
        "v350": "VGRD::350 mb",
        "v400": "VGRD::400 mb",
        "v450": "VGRD::450 mb",
        "v500": "VGRD::500 mb",
        "v550": "VGRD::550 mb",
        "v600": "VGRD::600 mb",
        "v650": "VGRD::650 mb",
        "v700": "VGRD::700 mb",
        "v750": "VGRD::750 mb",
        "v800": "VGRD::800 mb",
        "v850": "VGRD::850 mb",
        "v900": "VGRD::900 mb",
        "v925": "VGRD::925 mb",
        "v950": "VGRD::950 mb",
        "v975": "VGRD::975 mb",
        "v1000": "VGRD::1000 mb",
        "z1": "HGT::1 mb",
        "z2": "HGT::2 mb",
        "z3": "HGT::3 mb",
        "z5": "HGT::5 mb",
        "z7": "HGT::7 mb",
        "z10": "HGT::10 mb",
        "z15": "HGT::15 mb",
        "z20": "HGT::20 mb",
        "z30": "HGT::30 mb",
        "z40": "HGT::40 mb",
        "z50": "HGT::50 mb",
        "z70": "HGT::70 mb",
        "z100": "HGT::100 mb",
        "z150": "HGT::150 mb",
        "z200": "HGT::200 mb",
        "z250": "HGT::250 mb",
        "z300": "HGT::300 mb",
        "z350": "HGT::350 mb",
        "z400": "HGT::400 mb",
        "z450": "HGT::450 mb",
        "z500": "HGT::500 mb",
        "z550": "HGT::550 mb",
        "z600": "HGT::600 mb",
        "z650": "HGT::650 mb",
        "z700": "HGT::700 mb",
        "z750": "HGT::750 mb",
        "z800": "HGT::800 mb",
        "z850": "HGT::850 mb",
        "z900": "HGT::900 mb",
        "z925": "HGT::925 mb",
        "z950": "HGT::950 mb",
        "z975": "HGT::975 mb",
        "z1000": "HGT::1000 mb",
        "t1": "TMP::1 mb",
        "t2": "TMP::2 mb",
        "t3": "TMP::3 mb",
        "t5": "TMP::5 mb",
        "t7": "TMP::7 mb",
        "t10": "TMP::10 mb",
        "t15": "TMP::15 mb",
        "t20": "TMP::20 mb",
        "t30": "TMP::30 mb",
        "t40": "TMP::40 mb",
        "t50": "TMP::50 mb",
        "t70": "TMP::70 mb",
        "t100": "TMP::100 mb",
        "t150": "TMP::150 mb",
        "t200": "TMP::200 mb",
        "t250": "TMP::250 mb",
        "t300": "TMP::300 mb",
        "t350": "TMP::350 mb",
        "t400": "TMP::400 mb",
        "t450": "TMP::450 mb",
        "t500": "TMP::500 mb",
        "t550": "TMP::550 mb",
        "t600": "TMP::600 mb",
        "t650": "TMP::650 mb",
        "t700": "TMP::700 mb",
        "t750": "TMP::750 mb",
        "t800": "TMP::800 mb",
        "t850": "TMP::850 mb",
        "t900": "TMP::900 mb",
        "t925": "TMP::925 mb",
        "t950": "TMP::950 mb",
        "t975": "TMP::975 mb",
        "t1000": "TMP::1000 mb",
        "r1": "RH::1 mb",
        "r2": "RH::2 mb",
        "r3": "RH::3 mb",
        "r5": "RH::5 mb",
        "r7": "RH::7 mb",
        "r10": "RH::10 mb",
        "r15": "RH::15 mb",
        "r20": "RH::20 mb",
        "r30": "RH::30 mb",
        "r40": "RH::40 mb",
        "r50": "RH::50 mb",
        "r70": "RH::70 mb",
        "r100": "RH::100 mb",
        "r150": "RH::150 mb",
        "r200": "RH::200 mb",
        "r250": "RH::250 mb",
        "r300": "RH::300 mb",
        "r350": "RH::350 mb",
        "r400": "RH::400 mb",
        "r450": "RH::450 mb",
        "r500": "RH::500 mb",
        "r550": "RH::550 mb",
        "r600": "RH::600 mb",
        "r650": "RH::650 mb",
        "r700": "RH::700 mb",
        "r750": "RH::750 mb",
        "r800": "RH::800 mb",
        "r850": "RH::850 mb",
        "r900": "RH::900 mb",
        "r925": "RH::925 mb",
        "r950": "RH::950 mb",
        "r975": "RH::975 mb",
        "r1000": "RH::1000 mb",
        "q1": "SPFH::1 mb",
        "q2": "SPFH::2 mb",
        "q3": "SPFH::3 mb",
        "q5": "SPFH::5 mb",
        "q7": "SPFH::7 mb",
        "q10": "SPFH::10 mb",
        "q15": "SPFH::15 mb",
        "q20": "SPFH::20 mb",
        "q30": "SPFH::30 mb",
        "q40": "SPFH::40 mb",
        "q50": "SPFH::50 mb",
        "q70": "SPFH::70 mb",
        "q100": "SPFH::100 mb",
        "q150": "SPFH::150 mb",
        "q200": "SPFH::200 mb",
        "q250": "SPFH::250 mb",
        "q300": "SPFH::300 mb",
        "q350": "SPFH::350 mb",
        "q400": "SPFH::400 mb",
        "q450": "SPFH::450 mb",
        "q500": "SPFH::500 mb",
        "q550": "SPFH::550 mb",
        "q600": "SPFH::600 mb",
        "q650": "SPFH::650 mb",
        "q700": "SPFH::700 mb",
        "q750": "SPFH::750 mb",
        "q800": "SPFH::800 mb",
        "q850": "SPFH::850 mb",
        "q900": "SPFH::900 mb",
        "q925": "SPFH::925 mb",
        "q950": "SPFH::950 mb",
        "q975": "SPFH::975 mb",
        "q1000": "SPFH::1000 mb",
    }

    # sphinx - modifier start
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

    # sphinx - modifier end
