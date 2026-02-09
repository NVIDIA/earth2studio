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

from collections.abc import Callable

import pandas as pd

from .base import LexiconType


class GSIConventionalLexicon(metaclass=LexiconType):
    """NOAA maintained Gridpoint Statistical Interpolation (GSI) lexicon for
    conventional (in-situ) observations. GSI is provided as part of the NOAA UFS data
    repository.
    GSI vocab specified <platform>::<sensor/var>::<product>::<gsi column name of obs>

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/
    - https://ral.ucar.edu/solutions/products/gridpoint-statistical-interpolation-gsi
    - https://psl.noaa.gov/data/ufs_replay/202409-PSL-UFSReplay-HistoryFileVariableNames.pdf
    """

    VOCAB: dict[str, str] = {
        "u": "conv::uv::ges::u_Observation",
        "v": "conv::uv::ges::v_Observation",
        "q": "conv::q::ges::Observation",
        "t": "conv::t::ges::Observation",
        "pres": "conv::ps::ges::Observation",
        "u10m": "conv::uv::ges::u_Observation",
        "v10m": "conv::uv::ges::v_Observation",
        "q2m": "conv::q::ges::Observation",
        "t2m": "conv::t::ges::Observation",
        "sp": "conv::ps::ges::Observation",
        "u100m": "conv::uv::ges::u_Observation",
        "v100m": "conv::uv::ges::v_Observation",
        "gps": "conv::gps::ges::Observation",
        # Global Navigation System Radio Occultation is a remote sensing method
        # Could merge these with t and q but unique measurement method so keeping seperate
        "gps_t": "conv::gps::ges::Temperature_at_Obs_Location",
        "gps_q": "conv::gps::ges::Specific_Humidity_at_Obs_Location",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GSI vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - GSI vocab string
            - A modifier function to apply to the loaded values (identity).
        """
        gsi_key = cls.VOCAB[val]

        if val in ["u10m", "v10m", "sp"]:

            def mod(x: pd.DataFrame) -> pd.DataFrame:
                return x[(x["elev"] >= 0) & (x["elev"] <= 15)]

        elif val in ["q2m", "t2m"]:

            def mod(x: pd.DataFrame) -> pd.DataFrame:
                return x[(x["elev"] >= 0) & (x["elev"] <= 4)]

        elif val in ["u100m", "v100m"]:

            def mod(x: pd.DataFrame) -> pd.DataFrame:
                return x[(x["elev"] >= 90) & (x["elev"] <= 110)]

        else:

            def mod(x: pd.DataFrame) -> pd.DataFrame:
                return x

        return gsi_key, mod


class GSISatelliteLexicon(metaclass=LexiconType):
    """NOAA maintained Gridpoint Statistical Interpolation (GSI) lexicon for
    satellite observations. GSI is provided as part of the NOAA UFS data
    repository.
    GSI vocab specified <platforms>::<sensor/var>::<product>::<gsi column name of obs>

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/
    - https://ral.ucar.edu/solutions/products/gridpoint-statistical-interpolation-gsi
    """

    VOCAB: dict[str, str] = {
        "atms": "npp,n20::atms::ges::Observation",
        "mhs": "metop-a,metop-b,metop-c,n18,n19::mhs::ges::Observation",
        "amsua": "metop-a,metop-b,metop-c,n15,n16,n17,n18,n19::amsua::ges::Observation",
        "amsub": "n15,n16,n17::amsub::ges::Observation",
        "iasi": "metop-a,metop-b,metop-c::iasi::ges::Observation",
        "crisfsr": "npp,n20::cris-fsr::ges::Observation",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from GSI vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - GSI vocab string
            - A modifier function to apply to the loaded values (identity).
        """
        gsi_key = cls.VOCAB[val]

        def mod(x: pd.DataFrame) -> pd.DataFrame:
            return x

        return gsi_key, mod
