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

from collections.abc import Callable

import numpy as np

from .base import LexiconType


class HRRRLexicon(metaclass=LexiconType):
    """High-Resolution Rapid Refresh Analysis Lexicon
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
            "u10m": "sfc::anl::10m_above_ground::UGRD",
            "v10m": "sfc::anl::10m_above_ground::VGRD",
            "u80m": "sfc::anl::80m_above_ground::UGRD",
            "v80m": "sfc::anl::80m_above_ground::VGRD",
            "t2m": "sfc::anl::2m_above_ground::TMP",
            "refc": "sfc::anl::entire_atmosphere::REFC",
            "sp": "sfc::anl::surface::PRES",
            "tcwv": "sfc::anl::entire_atmosphere_single_layer::PWAT",
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
        prs_names = ["UGRD", "VGRD", "HGT", "TMP", "RH", "SPFH"]
        e2s_id = ["u", "v", "z", "t", "r", "q"]
        prs_variables = {}
        for (id, variable) in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"prs::anl::{level}mb::{variable}"

        return {**sfc_variables, **prs_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[1] == "HGT":

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
            "u10m": "sfc::fcst::10m_above_ground::UGRD",
            "v10m": "sfc::fcst::10m_above_ground::VGRD",
            "u80m": "sfc::fcst::80m_above_ground::UGRD",
            "v80m": "sfc::fcst::80m_above_ground::VGRD",
            "t2m": "sfc::fcst::2m_above_ground::TMP",
            "refc": "sfc::fcst::entire_atmosphere::REFC",
            "sp": "sfc::fcst::surface::PRES",
            "tp": "sfc::fcst::surface::APCP_1hr_acc_fcst",
            "tcwv": "sfc::fcst::entire_atmosphere_single_layer::PWAT",
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
        for (id, variable) in zip(e2s_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"sfc::fcst::{level}mb::{variable}"

        return {**sfc_variables, **prs_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from HRRR vocabulary."""
        hrrr_key = cls.VOCAB[val]
        if hrrr_key.split("::")[1] == "HGT":

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x * 9.81

        else:

            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x

        return hrrr_key, mod
