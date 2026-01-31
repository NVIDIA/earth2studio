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

import pandas as pd
import pyarrow as pa

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
        "u100m": "conv::uv:::ges:u_Observation",
        "v100m": "conv::uv::ges::v_Observation",
    }

    SCHEMA = pa.schema(
        [
            pa.field(
                "observation", pa.float32(), metadata={"gsi_name": "Observation"}
            ),  # Main observation value (required)
            pa.field("variable", pa.string()),  # E2S variable name id (required)
            pa.field("source", pa.string()),  # E2S data source id (required)
            pa.field(
                "time", pa.timestamp("ns"), metadata={"gsi_name": "Time"}
            ),  # Observation time (required)
            pa.field(
                "pres", pa.float32(), nullable=True, metadata={"gsi_name": "Pressure"}
            ),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Height"}
            ),
            pa.field(
                "type",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Observation_Type"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("station", pa.string(), metadata={"gsi_name": "Station_ID"}),
            pa.field(
                "station_elev",
                pa.float32(),
                nullable=True,
                metadata={"gsi_name": "Station_Elevation"},
            ),
            pa.field(
                "station_type",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
        ]
    )

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
                return x[(x["elev"] >= 0) & (x["elev"] <= 20)]

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

    @classmethod
    def column_map(cls) -> dict[str, str]:
        """Data frame translation map from source to e2s column names

        Returns
        -------
        dict[str, str]
            Column map
        """
        column_map = {}
        for field in cls.SCHEMA:
            if field.metadata is None or "gsi_name" not in field.metadata:  # type: ignore
                continue
            column_map[field.metadata["gsi_name"]] = field.name
        return column_map
