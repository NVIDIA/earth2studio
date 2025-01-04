# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


class NCAR_ERA5Lexicon(metaclass=LexiconType):
    """NCAR ERA5 Lexicon
    S3 specified <Prefix>.<N>_<ID>_<Name>.<Postfix>

    Note
    ----
    Additional resources:
    https://registry.opendata.aws/nsf-ncar-era5/
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        # ECMWF ID, ECMWF name, pressure/surface level
        # See https://codes.ecmwf.int/grib/param-db/ for ECMWF ID and name
        params = {
            "z": (129, "z", "pl"),
            "t": (130, "t", "pl"),
            "u": (131, "u", "pl"),
            "v": (132, "v", "pl"),
            "q": (133, "q", "pl"),
            "sp": (134, "sp", "sfc"),
            "tcwv": (137, "tcwv", "sfc"),
            "msl": (151, "msl", "sfc"),
            "r": (157, "r", "pl"),
            "u10m": (165, "10u", "sfc"),
            "v10m": (166, "10v", "sfc"),
            "d2m": (168, "2d", "sfc"),
            "t2m": (167, "2t", "sfc"),
            "u100m": (246, "100u", "sfc"),
            "v100m": (247, "100v", "sfc"),
        }
        pressure_levels = [
            1,
            2,
            3,
            5,
            7,
            10,
            20,
            30,
            50,
            70,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
        ]
        pattern = "e5.oper.an.{lvl}.{n}_{eid}_{ename}.{s}"

        vocab = {}
        for var, (eid, ename, lvl) in params.items():
            # When adding new variables, inspect S3 prefixes for n/s parts
            n = 228 if ename in ("100u", "100v") else 128
            s = "ll025uv" if ename in ("u", "v") else "ll025sc"
            formatted_pattern = pattern.format(lvl=lvl, n=n, eid=eid, ename=ename, s=s)

            if lvl == "sfc":
                vocab[var] = formatted_pattern
            else:
                for pressure_level in pressure_levels:
                    vocab[var + str(pressure_level)] = formatted_pattern

        return vocab

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in NCAR ERA5 vocabulary.

        Parameters
        ----------
        val : str
            Name in Earth-2 terminology.

        Returns
        -------
        tuple[str, Callable]
            NCAR ERA5 name and modifier function.
        """
        cds_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify value (if necessary)."""
            return x

        return cds_key, mod
