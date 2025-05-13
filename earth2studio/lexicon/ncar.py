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


class NCAR_ERA5Lexicon(metaclass=LexiconType):
    """NCAR ERA5 Lexicon
    S3 specified <Prefix>.<N>_<ID>_<Name>.<Postfix>

    <Product ID>::<Variable ID N_ID_Name>::<Grid Type>::<Level Index>

    Note
    ----
    Additional resources:
    https://registry.opendata.aws/nsf-ncar-era5/
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create NCAR ERA5 vocab dictionary"""
        # ECMWF ID, ECMWF name, product ID, Grid Type
        # See https://codes.ecmwf.int/grib/param-db/ for ECMWF ID and name
        params = {
            "z": (129, "z", "e5.oper.an.pl", "ll025sc"),
            "t": (130, "t", "e5.oper.an.pl", "ll025sc"),
            "u": (131, "u", "e5.oper.an.pl", "ll025uv"),
            "v": (132, "v", "e5.oper.an.pl", "ll025uv"),
            "q": (133, "q", "e5.oper.an.pl", "ll025sc"),
            "sp": (134, "sp", "e5.oper.an.sfc", "ll025sc"),
            "tcwv": (137, "tcwv", "e5.oper.an.sfc", "ll025sc"),
            "msl": (151, "msl", "e5.oper.an.sfc", "ll025sc"),
            "r": (157, "r", "e5.oper.an.pl", "ll025sc"),
            "u10m": (165, "10u", "e5.oper.an.sfc", "ll025sc"),
            "v10m": (166, "10v", "e5.oper.an.sfc", "ll025sc"),
            "d2m": (168, "2d", "e5.oper.an.sfc", "ll025sc"),
            "t2m": (167, "2t", "e5.oper.an.sfc", "ll025sc"),
            "u100m": (246, "100u", "e5.oper.an.sfc", "ll025sc"),
            "v100m": (247, "100v", "e5.oper.an.sfc", "ll025sc"),
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
        pattern = "{product}::{n}_{eid}_{ename}::{grid}::{level}"

        vocab = {}
        for var, (eid, ename, product, grid) in params.items():
            # When adding new variables, inspect S3 prefixes for n/s parts
            n = 228 if ename in ("100u", "100v") else 128
            if product == "e5.oper.an.pl":
                for i, pressure_level in enumerate(pressure_levels):
                    formatted_pattern = pattern.format(
                        product=product, n=n, eid=eid, ename=ename, grid=grid, level=i
                    )
                    vocab[var + str(pressure_level)] = formatted_pattern
            else:
                formatted_pattern = pattern.format(
                    product=product, n=n, eid=eid, ename=ename, grid=grid, level=0
                )
                vocab[var] = formatted_pattern

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
