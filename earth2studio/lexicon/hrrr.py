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
from datetime import datetime, timedelta

import numpy as np

from .base import LexiconType


class HRRRLexicon(metaclass=LexiconType):
    """High-Resolution Rapid Refresh Analysis Lexicon
    HRRR specified <Product ID>::<Parameter ID>::<Level/ Layer>::

    Note
    ----
    Additional resources:
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf02.grib2.shtml
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRRR vocab dictionary"""
        sfc_variables = {
            "u10m": "sfc::anl::10 m above ground::UGRD",
            "v10m": "sfc::anl::10 m above ground::VGRD",
            "u80m": "sfc::anl::80 m above ground::UGRD",
            "v80m": "sfc::anl::80 m above ground::VGRD",
            "t2m": "sfc::anl::2 m above ground::TMP",
            "refc": "sfc::anl::entire atmosphere::REFC",
            "sp": "sfc::anl::surface::PRES",
            "mslp": "sfc::anl::mean sea level::MSLMA",
            "tcwv": "sfc::anl::entire atmosphere::PWAT",
            "csnow": "sfc::anl::surface::CSNOW",
            "cicep": "sfc::anl::surface::CICEP",
            "cfrzr": "sfc::anl::surface::CFRZR",
            "crain": "sfc::anl::surface::CRAIN",
        }
        prs_levels = [
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            525,
            550,
            575,
            600,
            625,
            650,
            675,
            700,
            725,
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

        prs_names = ["UGRD", "VGRD", "HGT", "TMP", "RH", "SPFH", "HGT"]
        e2s_id = ["u", "v", "z", "t", "r", "q", "Z"]
        prs_variables = {}
        for id, variable in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"prs::anl::{level} mb::{variable}"

        hybrid_levels = list(range(1, 51))
        hybrid_names = ["UGRD", "VGRD", "HGT", "TMP", "SPFH", "PRES", "HGT"]
        e2s_id = ["u", "v", "z", "t", "q", "p", "Z"]
        hybrid_variables = {}
        for id, variable in zip(e2s_id, hybrid_names):
            for level in hybrid_levels:
                hybrid_variables[f"{id}{level:d}hl"] = (
                    f"nat::anl::{level} hybrid level::{variable}"
                )

        return {**sfc_variables, **prs_variables, **hybrid_variables}

    VOCAB = build_vocab()

    @classmethod
    def index_regex(cls, val: str, time: datetime, lead_time: timedelta) -> str:
        """Gets regex to check index file for"""
        hrrr_key = cls.VOCAB[val]
        variable = hrrr_key.split("::")[3]
        level = hrrr_key.split("::")[2]
        return f"{variable}:{level}"

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[3] == "HGT" and val.startswith("z"):

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return hrrr_key, mod


class HRRRLexiconNew(metaclass=LexiconType):
    """High-Resolution Rapid Refresh Analysis Lexicon
    HRRR specified <Product ID>::<Parameter ID>::<Level/ Layer>::<Forcast Valid Range(optional)>::<ID Number (optional, override)>

    Products include:
        - wrfsfc
        - wrfprs
        - wrfnat
        - wrfsubh

    Note
    ----
    Additional resources:
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf02.grib2.shtml
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRRR vocab dictionary"""
        sfc_variables = {
            "u10m": "wrfsfc::UGRD::10 m above ground::anl::",
            "v10m": "wrfsfc::VGRD::10 m above ground::anl::",
            "u80m": "wrfsfc::UGRD::80 m above ground::anl",
            "v80m": "wrfsfc::VGRD::80 m above ground::anl",
            "t2m": "wrfsfc::TMP::2 m above ground::anl",
            "refc": "wrfsfc::REFC::entire atmosphere::anl",
            "sp": "wrfsfc::PRES::surface::anl",
            "mslp": "wrfsfc::MSLMA::mean sea level::anl",
            "tcwv": "wrfsfc::PWAT::entire atmosphere (considered as a single layer)::anl",
            "csnow": "wrfsfc::CSNOW::surface::anl",
            "cicep": "wrfsfc::CICEP::surface::anl",
            "cfrzr": "wrfsfc::CFRZR::surface::anl",
            "crain": "wrfsfc::CRAIN::surface::anl",
        }
        prs_levels = [
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            525,
            550,
            575,
            600,
            625,
            650,
            675,
            700,
            725,
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

        prs_names = ["UGRD", "VGRD", "HGT", "TMP", "RH", "SPFH", "HGT"]
        e2s_id = ["u", "v", "z", "t", "r", "q", "Z"]
        prs_variables = {}
        for id, variable in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"wrfprs::{variable}::{level} mb::anl"

        hybrid_levels = list(range(1, 51))
        hybrid_names = ["UGRD", "VGRD", "HGT", "TMP", "SPFH", "PRES", "HGT"]
        e2s_id = ["u", "v", "z", "t", "q", "p", "Z"]
        hybrid_variables = {}
        for id, variable in zip(e2s_id, hybrid_names):
            for level in hybrid_levels:
                hybrid_variables[f"{id}{level:d}hl"] = (
                    f"wrfnat::{variable}::{level} hybrid level::anl"
                )

        return {**sfc_variables, **prs_variables, **hybrid_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[1] == "HGT" and val.startswith("z"):

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return hrrr_key, mod


class HRRRFXLexicon(metaclass=LexiconType):
    """High-Resolution Rapid Refresh Forcast Lexicon
    HRRR specified <Provider ID>::<Product ID>::<Level/ Layer>::<Parameter ID>

    Note
    ----
    Additional resources:
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf02.grib2.shtml
    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRRR vocab dictionary"""
        sfc_variables = {
            "u10m": "sfc::fcst::10 m above ground::UGRD",
            "v10m": "sfc::fcst::10 m above ground::VGRD",
            "u80m": "sfc::fcst::80 m above ground::UGRD",
            "v80m": "sfc::fcst::80 m above ground::VGRD",
            "t2m": "sfc::fcst::2 m above ground::TMP",
            "refc": "sfc::fcst::entire atmosphere::REFC",
            "sp": "sfc::fcst::surface::PRES",
            "mslp": "sfc::anl::mean sea level::MSLMA",
            "tp": "sfc::fcst::surface::APCP",  # APCP_1hr_acc_fcst in Zarr store
            "tcwv": "sfc::fcst::entire atmosphere::PWAT",
            "csnow": "sfc::fcst::surface::CSNOW",
            "cicep": "sfc::fcst::surface::CICEP",
            "cfrzr": "sfc::fcst::surface::CFRZR",
            "crain": "sfc::fcst::surface::CRAIN",
        }
        prs_levels = [
            250,
            300,
            500,
            850,
            925,
            1000,
        ]
        prs_names = ["UGRD", "VGRD"]
        e2s_id = ["u", "v"]
        prs_variables = {}
        for id, variable in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"sfc::fcst::{level} mb::{variable}"

        hybrid_levels = list(range(1, 51))
        hybrid_names = ["UGRD", "VGRD", "HGT", "TMP", "SPFH", "PRES", "HGT"]
        e2s_id = ["u", "v", "z", "t", "q", "p", "Z"]
        hybrid_variables = {}
        for id, variable in zip(e2s_id, hybrid_names):
            for level in hybrid_levels:
                hybrid_variables[f"{id}{level:d}hl"] = (
                    f"nat::fcst::{level} hybrid level::{variable}"
                )

        return {**sfc_variables, **prs_variables, **hybrid_variables}

    VOCAB = build_vocab()

    @classmethod
    def index_regex(cls, val: str, time: datetime, lead_time: timedelta) -> str:
        """Gets regex to check index file for"""
        # Deal with ACPC having two different values (thanks NOAA)
        hrrr_key = cls.VOCAB[val]
        if val == "tp":
            lead_index = int(lead_time.total_seconds() // 3600)
            return f"APCP:surface:{lead_index-1}-{lead_index} hour acc"
        else:
            variable = hrrr_key.split("::")[3]
            level = hrrr_key.split("::")[2]
            return f"{variable}:{level}"

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[3] == "HGT" and val.startswith("z"):

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return hrrr_key, mod
