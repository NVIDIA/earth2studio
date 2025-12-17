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


class ECMWFOpenDataLexicon(metaclass=LexiconType):
    """Base lexicon class for the ECMWF open data client.
    GRIB2 specification: <Grib Parameter ID>::<Level Type>::<Level>

    Note
    ----
    Best bet is to download an index file from the AWS bucket and read it. Additional
    resources:

    - https://codes.ecmwf.int/grib/param-db/?filter=grib2
    - https://www.ecmwf.int/en/forecasts/datasets/open-data
    """

    PRS_NAMES = {
        # Shared mapping from Earth2Studio ID to ECMWF ID
        "d": "d",
        "q": "q",
        "r": "r",
        "t": "t",
        "u": "u",
        "v": "v",
        "vo": "vo",
        "w": "w",
        "z": "gh",
    }
    VOCAB: dict

    @staticmethod
    def build_vocab(
        sfc_variables: dict,
        soil_variables: dict,
        prs_variables: list[str],
        prs_names: dict,
        prs_levels: list[int],
    ) -> dict[str, str]:
        """Create vocabulary dictionary."""
        prs_mapping = {}
        for e2s_id in prs_variables:
            ecmwf_id = prs_names[e2s_id]
            for level in prs_levels:
                prs_mapping[f"{e2s_id}{level:d}"] = f"{ecmwf_id}::pl::{level}"

        return {**sfc_variables, **soil_variables, **prs_mapping}

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Retrieve name from vocabulary."""
        ifs_key = cls.VOCAB[val]
        if ifs_key.split("::")[0] == "gh":

            def mod(x: np.ndarray) -> np.ndarray:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Modify data value (if necessary)."""
                return x

        return ifs_key, mod


class IFSLexicon(ECMWFOpenDataLexicon):
    """Integrated Forecast System (IFS) lexicon."""

    # From a recent GRIB2 index file on AWS S3
    SFC_VARIABLES = {
        "u100m": "100u::sfc::",
        "v100m": "100v::sfc::",
        "fg10m": "10fg::sfc::",
        "u10m": "10u::sfc::",
        "v10m": "10v::sfc::",
        "d2m": "2d::sfc::",
        "t2m": "2t::sfc::",
        "asn": "asn::sfc::",
        "ewss": "ewss::sfc::",
        "lsm": "lsm::sfc::",
        "mn2t3": "mn2t3::sfc::",
        "msl": "msl::sfc::",
        "mucape": "mucape::sfc::",
        "mx2t3": "mx2t3::sfc::",
        "nsss": "nsss::sfc::",
        "ptype": "ptype::sfc::",
        "ro": "ro::sfc::",
        "sdor": "sdor::sfc::",
        "sithick": "sithick::sfc::",
        "skt": "skt::sfc::",
        "slor": "slor::sfc::",
        "sp": "sp::sfc::",
        "ssr": "ssr::sfc::",
        "ssrd": "ssrd::sfc::",
        "str": "str::sfc::",
        "strd": "strd::sfc::",
        "sve": "sve::sfc::",
        "svn": "svn::sfc::",
        "tcw": "tcw::sfc::",
        "tcwv": "tcwv::sfc::",
        "tp": "tp::sfc::",
        "tprate": "tprate::sfc::",
        "ttr": "ttr::sfc::",
        "z": "z::sfc::",
        "zos": "zos::sfc::",
    }
    SOIL_VARIABLES = {
        "stl1": "sot::sl::1",
        "stl2": "sot::sl::2",
        "stl3": "sot::sl::3",
        "stl4": "sot::sl::4",
        "swvl1": "vsw::sl::1",
        "swvl2": "vsw::sl::2",
        "swvl3": "vsw::sl::3",
        "swvl4": "vsw::sl::4",
    }
    PRS_VARIABLES = [
        "d",
        "q",
        "r",
        "t",
        "u",
        "v",
        "vo",
        "w",
        "z",
    ]
    PRS_LEVELS = [
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

    VOCAB = ECMWFOpenDataLexicon.build_vocab(
        sfc_variables=SFC_VARIABLES,
        soil_variables=SOIL_VARIABLES,
        prs_variables=PRS_VARIABLES,
        prs_names=ECMWFOpenDataLexicon.PRS_NAMES,
        prs_levels=PRS_LEVELS,
    )


class AIFSLexicon(ECMWFOpenDataLexicon):
    """Artificial Intelligence Forecast System (AIFS) lexicon.

    Warning
    =======
    The AIFS data store has a lot of inconsistencies with the IFS data store, be very
    careful adding new variables. See the open data documentation for details:

    - https://www.ecmwf.int/en/forecasts/datasets/open-data
    """

    # From a recent GRIB2 index file on AWS S3
    SFC_VARIABLES = {
        "u100m": "100u::sfc::",
        "v100m": "100v::sfc::",
        "u10m": "10u::sfc::",
        "v10m": "10v::sfc::",
        "d2m": "2d::sfc::",
        "t2m": "2t::sfc::",
        "cp": "cp::sfc::",
        "hcc": "hcc::sfc::",
        "lcc": "lcc::sfc::",
        "lsm": "lsm::sfc::",
        "mcc": "mcc::sfc::",
        "msl": "msl::sfc::",
        "rowe": "rowe::sfc::",
        "sdor": "sdor::sfc::",
        "sf": "sf::sfc::",
        "skt": "skt::sfc::",
        "slor": "slor::sfc::",
        "sp": "sp::sfc::",
        "ssrd": "ssrd::sfc::",
        "strd": "strd::sfc::",
        "tcc": "tcc::sfc::",
        "tcw": "tcw::sfc::",
        "tp": "tp::sfc::",
        # "z": "z::sfc::", # Grib error with unique keys
    }
    SOIL_VARIABLES = {
        "stl1": "sot::sl::1",
        "stl2": "sot::sl::2",
        "swvl1": "vsw::sl::1",
        "swvl2": "vsw::sl::2",
    }
    PRS_VARIABLES = [
        "q",
        "t",
        "u",
        "v",
        "w",
        "z",
    ]
    PRS_LEVELS = [
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
    PRS_NAMES = {
        "d": "d",
        "q": "q",
        "r": "r",
        "t": "t",
        "u": "u",
        "v": "v",
        "vo": "vo",
        "w": "w",
        "z": "z",  # z in AIFS store, NOT gh
    }

    VOCAB = ECMWFOpenDataLexicon.build_vocab(
        sfc_variables=SFC_VARIABLES,
        soil_variables=SOIL_VARIABLES,
        prs_variables=PRS_VARIABLES,
        prs_names=PRS_NAMES,
        prs_levels=PRS_LEVELS,
    )

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Retrieve name from vocabulary."""
        aifs_key = cls.VOCAB[val]

        if aifs_key.split("::")[0] in ["tcc", "hcc", "mcc", "lcc"]:
            # TCC in AIFS is precentage param id 228164, convert to 0-1 param id 164
            def mod(x: np.ndarray) -> np.ndarray:
                return x / 100.0

        elif aifs_key.split("::")[0] == "tp":
            # TP in AIFS is (kg m-2) param id 228228, convert to (m) param id 228
            def mod(x: np.ndarray) -> np.ndarray:
                # Assume density of water is 1000 kg m-3
                # x (kg m-2) / 1000 (kg m-3) = (m)
                return x / 1000.0

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Modify data value (if necessary)."""
                return x

        return aifs_key, mod
