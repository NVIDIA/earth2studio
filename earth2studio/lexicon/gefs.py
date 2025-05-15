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


class GEFSLexicon(metaclass=LexiconType):
    """Global Ensemble Forecast System Lexicon, right now only support isobarric
    GEFS vocab specified <Product ID>::<Parameter ID>::<Level/ Layer>

    Note
    ----
    Additional resources:
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/gec00.t00z.pgrb2a.0p50.f003.shtml
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/gec00.t00z.pgrb2b.0p50.f003.shtml
    """

    VOCAB = {
        "u10m": "pgrb2a::UGRD::10 m above ground",
        "v10m": "pgrb2a::VGRD::10 m above ground",
        "u100m": "pgrb2b::UGRD::100 m above ground",
        "v100m": "pgrb2b::VGRD::100 m above ground",
        "t2m": "pgrb2a::TMP::2 m above ground",
        "d2m": "pgrb2b::DPT::2 m above ground",
        "r2m": "pgrb2a::RH::2 m above ground",
        "q2m": "pgrb2b::SPFH::2 m above ground",
        "t100m": "pgrb2b::TMP::100 m above ground",
        "sp": "pgrb2a::PRES::surface",
        "msl": "pgrb2b::PRES::mean sea level",
        "tcwv": "pgrb2a::PWAT::entire atmosphere (considered as a single layer)",
        "u1": "pgrb2b::UGRD::1 mb",
        "u2": "pgrb2b::UGRD::2 mb",
        "u3": "pgrb2b::UGRD::3 mb",
        "u5": "pgrb2b::UGRD::5 mb",
        "u7": "pgrb2b::UGRD::7 mb",
        "u10": "pgrb2a::UGRD::10 mb",
        "u20": "pgrb2b::UGRD::20 mb",
        "u30": "pgrb2b::UGRD::30 mb",
        "u50": "pgrb2a::UGRD::50 mb",
        "u70": "pgrb2b::UGRD::70 mb",
        "u100": "pgrb2a::UGRD::100 mb",
        "u150": "pgrb2b::UGRD::150 mb",
        "u200": "pgrb2a::UGRD::200 mb",
        "u250": "pgrb2a::UGRD::250 mb",
        "u300": "pgrb2a::UGRD::300 mb",
        "u350": "pgrb2b::UGRD::350 mb",
        "u400": "pgrb2a::UGRD::400 mb",
        "u450": "pgrb2b::UGRD::450 mb",
        "u500": "pgrb2a::UGRD::500 mb",
        "u550": "pgrb2b::UGRD::550 mb",
        "u600": "pgrb2b::UGRD::600 mb",
        "u650": "pgrb2b::UGRD::650 mb",
        "u700": "pgrb2a::UGRD::700 mb",
        "u750": "pgrb2b::UGRD::750 mb",
        "u800": "pgrb2b::UGRD::800 mb",
        "u850": "pgrb2a::UGRD::850 mb",
        "u900": "pgrb2b::UGRD::900 mb",
        "u925": "pgrb2a::UGRD::925 mb",
        "u950": "pgrb2b::UGRD::950 mb",
        "u975": "pgrb2b::UGRD::975 mb",
        "u1000": "pgrb2a::UGRD::1000 mb",
        "v1": "pgrb2b::VGRD::1 mb",
        "v2": "pgrb2b::VGRD::2 mb",
        "v3": "pgrb2b::VGRD::3 mb",
        "v5": "pgrb2b::VGRD::5 mb",
        "v7": "pgrb2b::VGRD::7 mb",
        "v10": "pgrb2a::VGRD::10 mb",
        "v20": "pgrb2b::VGRD::20 mb",
        "v30": "pgrb2b::VGRD::30 mb",
        "v50": "pgrb2a::VGRD::50 mb",
        "v70": "pgrb2b::VGRD::70 mb",
        "v100": "pgrb2a::VGRD::100 mb",
        "v150": "pgrb2b::VGRD::150 mb",
        "v200": "pgrb2a::VGRD::200 mb",
        "v250": "pgrb2a::VGRD::250 mb",
        "v300": "pgrb2a::VGRD::300 mb",
        "v350": "pgrb2b::VGRD::350 mb",
        "v400": "pgrb2a::VGRD::400 mb",
        "v450": "pgrb2b::VGRD::450 mb",
        "v500": "pgrb2a::VGRD::500 mb",
        "v550": "pgrb2b::VGRD::550 mb",
        "v600": "pgrb2b::VGRD::600 mb",
        "v650": "pgrb2b::VGRD::650 mb",
        "v700": "pgrb2a::VGRD::700 mb",
        "v750": "pgrb2b::VGRD::750 mb",
        "v800": "pgrb2b::VGRD::800 mb",
        "v850": "pgrb2a::VGRD::850 mb",
        "v900": "pgrb2b::VGRD::900 mb",
        "v925": "pgrb2a::VGRD::925 mb",
        "v950": "pgrb2b::VGRD::950 mb",
        "v975": "pgrb2b::VGRD::975 mb",
        "v1000": "pgrb2a::VGRD::1000 mb",
        "z1": "pgrb2b::HGT::1 mb",
        "z2": "pgrb2b::HGT::2 mb",
        "z3": "pgrb2b::HGT::3 mb",
        "z5": "pgrb2b::HGT::5 mb",
        "z7": "pgrb2b::HGT::7 mb",
        "z10": "pgrb2a::HGT::10 mb",
        "z20": "pgrb2b::HGT::20 mb",
        "z30": "pgrb2b::HGT::30 mb",
        "z50": "pgrb2a::HGT::50 mb",
        "z70": "pgrb2b::HGT::70 mb",
        "z100": "pgrb2a::HGT::100 mb",
        "z150": "pgrb2b::HGT::150 mb",
        "z200": "pgrb2a::HGT::200 mb",
        "z250": "pgrb2a::HGT::250 mb",
        "z300": "pgrb2a::HGT::300 mb",
        "z350": "pgrb2b::HGT::350 mb",
        "z400": "pgrb2b::HGT::400 mb",
        "z450": "pgrb2b::HGT::450 mb",
        "z500": "pgrb2a::HGT::500 mb",
        "z550": "pgrb2b::HGT::550 mb",
        "z600": "pgrb2b::HGT::600 mb",
        "z650": "pgrb2b::HGT::650 mb",
        "z700": "pgrb2a::HGT::700 mb",
        "z750": "pgrb2b::HGT::750 mb",
        "z800": "pgrb2b::HGT::800 mb",
        "z850": "pgrb2a::HGT::850 mb",
        "z900": "pgrb2b::HGT::900 mb",
        "z925": "pgrb2a::HGT::925 mb",
        "z950": "pgrb2b::HGT::950 mb",
        "z975": "pgrb2b::HGT::975 mb",
        "z1000": "pgrb2a::HGT::1000 mb",
        "t1": "pgrb2b::TMP::1 mb",
        "t2": "pgrb2b::TMP::2 mb",
        "t3": "pgrb2b::TMP::3 mb",
        "t5": "pgrb2b::TMP::5 mb",
        "t7": "pgrb2b::TMP::7 mb",
        "t10": "pgrb2a::TMP::10 mb",
        "t20": "pgrb2b::TMP::20 mb",
        "t30": "pgrb2b::TMP::30 mb",
        "t50": "pgrb2a::TMP::50 mb",
        "t70": "pgrb2b::TMP::70 mb",
        "t100": "pgrb2a::TMP::100 mb",
        "t150": "pgrb2b::TMP::150 mb",
        "t200": "pgrb2a::TMP::200 mb",
        "t250": "pgrb2a::TMP::250 mb",
        "t300": "pgrb2b::TMP::300 mb",
        "t350": "pgrb2b::TMP::350 mb",
        "t400": "pgrb2b::TMP::400 mb",
        "t450": "pgrb2b::TMP::450 mb",
        "t500": "pgrb2a::TMP::500 mb",
        "t550": "pgrb2b::TMP::550 mb",
        "t600": "pgrb2b::TMP::600 mb",
        "t650": "pgrb2b::TMP::650 mb",
        "t700": "pgrb2a::TMP::700 mb",
        "t750": "pgrb2b::TMP::750 mb",
        "t800": "pgrb2b::TMP::800 mb",
        "t850": "pgrb2a::TMP::850 mb",
        "t900": "pgrb2b::TMP::900 mb",
        "t925": "pgrb2a::TMP::925 mb",
        "t950": "pgrb2b::TMP::950 mb",
        "t975": "pgrb2b::TMP::975 mb",
        "t1000": "pgrb2a::TMP::1000 mb",
        "r10": "pgrb2a::RH::10 mb",
        "r20": "pgrb2b::RH::20 mb",
        "r30": "pgrb2b::RH::30 mb",
        "r50": "pgrb2a::RH::50 mb",
        "r70": "pgrb2b::RH::70 mb",
        "r100": "pgrb2a::RH::100 mb",
        "r150": "pgrb2b::RH::150 mb",
        "r200": "pgrb2a::RH::200 mb",
        "r250": "pgrb2a::RH::250 mb",
        "r300": "pgrb2b::RH::300 mb",
        "r350": "pgrb2b::RH::350 mb",
        "r400": "pgrb2b::RH::400 mb",
        "r450": "pgrb2b::RH::450 mb",
        "r500": "pgrb2a::RH::500 mb",
        "r550": "pgrb2b::RH::550 mb",
        "r600": "pgrb2b::RH::600 mb",
        "r650": "pgrb2b::RH::650 mb",
        "r700": "pgrb2a::RH::700 mb",
        "r750": "pgrb2b::RH::750 mb",
        "r800": "pgrb2b::RH::800 mb",
        "r850": "pgrb2a::RH::850 mb",
        "r900": "pgrb2b::RH::900 mb",
        "r925": "pgrb2a::RH::925 mb",
        "r950": "pgrb2b::RH::950 mb",
        "r975": "pgrb2b::RH::975 mb",
        "r1000": "pgrb2a::RH::1000 mb",
        "q10": "pgrb2b::SPFH::10 mb",
        "q20": "pgrb2b::SPFH::20 mb",
        "q30": "pgrb2b::SPFH::30 mb",
        "q50": "pgrb2b::SPFH::50 mb",
        "q70": "pgrb2b::SPFH::70 mb",
        "q100": "pgrb2b::SPFH::100 mb",
        "q150": "pgrb2b::SPFH::150 mb",
        "q200": "pgrb2b::SPFH::200 mb",
        "q250": "pgrb2b::SPFH::250 mb",
        "q300": "pgrb2b::SPFH::300 mb",
        "q350": "pgrb2b::SPFH::350 mb",
        "q400": "pgrb2b::SPFH::400 mb",
        "q450": "pgrb2b::SPFH::450 mb",
        "q500": "pgrb2b::SPFH::500 mb",
        "q550": "pgrb2b::SPFH::550 mb",
        "q600": "pgrb2b::SPFH::600 mb",
        "q650": "pgrb2b::SPFH::650 mb",
        "q700": "pgrb2b::SPFH::700 mb",
        "q750": "pgrb2b::SPFH::750 mb",
        "q800": "pgrb2b::SPFH::800 mb",
        "q850": "pgrb2b::SPFH::850 mb",
        "q900": "pgrb2b::SPFH::900 mb",
        "q925": "pgrb2b::SPFH::925 mb",
        "q950": "pgrb2b::SPFH::950 mb",
        "q975": "pgrb2b::SPFH::975 mb",
        "q1000": "pgrb2b::SPFH::1000 mb",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GFS vocabulary."""
        gfs_key = cls.VOCAB[val]
        if gfs_key.split("::")[1] == "HGT":

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return gfs_key, mod


class GEFSLexiconSel(metaclass=LexiconType):
    """Global Ensemble Forecast System 0.25 Degree Lexicon (Select variables). GEFS
    vocab specified <Product ID>::<Parameter ID>::<Level/ Layer>

    Warning
    -------
    Some variables are only present for lead time greater than 0

    Note
    ----
    Additional resources:
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/gec00.t00z.pgrb2s.0p25.f000.shtml
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/gec00.t00z.pgrb2s.0p25.f003.shtml
    - https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-3.shtml
    """

    VOCAB = {
        "u10m": "pgrb2s::UGRD::10 m above ground",
        "v10m": "pgrb2s::VGRD::10 m above ground",
        "t2m": "pgrb2s::TMP::2 m above ground",
        "d2m": "pgrb2s::DPT::2 m above ground",
        "r2m": "pgrb2s::RH::2 m above ground",
        "sp": "pgrb2s::PRES::surface",
        "msl": "pgrb2s::PRMSL::mean sea level",  # Pressure Reduced to MSL
        "tcwv": "pgrb2s::PWAT::entire atmosphere (considered as a single layer)",
        "tp": "pgrb2s::APCP::surface",  # 3 hour acc
        "csnow": "pgrb2s::CSNOW::surface",  # 3 hour ave
        "cicep": "pgrb2s::CICEP::surface",  # 3 hour ave
        "cfrzr": "pgrb2s::CFRZR::surface",  # 3 hour ave
        "crain": "pgrb2s::CRAIN::surface",  # 3 hour ave
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GFS vocabulary."""
        gfs_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify data value (if necessary)."""
            return x

        return gfs_key, mod
