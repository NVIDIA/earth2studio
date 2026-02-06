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
from typing import Any

import numpy as np

from earth2studio.lexicon.base import LexiconType

Modifier = Callable[[Any], Any]


class OISSTLexicon(metaclass=LexiconType):
    """Lexicon for NOAA OISST collections hosted on the Planetary Computer."""

    VOCAB: dict[str, tuple[str, Modifier]] = {
        "sst": (
            "sst",
            lambda array: array + np.float32(273.15),
        ),  # sea surface temperature (K)
        "ssta": (
            "anom",
            lambda array: array,
        ),  # SST anomaly relative to climatology (K)
        "sstu": ("err", lambda array: array),  # SST analysis uncertainty (K)
        "sic": (
            "ice",
            lambda array: array / np.float32(100.0),
        ),  # sea ice fraction (0-1)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


class Sentinel3AODLexicon(metaclass=LexiconType):
    """Lexicon exposing Sentinel-3 SYNERGY aerosol and reflectance variables."""

    VOCAB: dict[str, tuple[str, Modifier]] = {
        "s3sy01aod": ("AOD_440", lambda array: array),  # AOD band 01 (440 nm)
        "s3sy02aod": ("AOD_550", lambda array: array),  # AOD band 02 (550 nm)
        "s3sy03aod": ("AOD_670", lambda array: array),  # AOD band 03 (670 nm)
        "s3sy04aod": ("AOD_865", lambda array: array),  # AOD band 04 (865 nm)
        "s3sy05aod": ("AOD_1600", lambda array: array),  # AOD band 05 (1600 nm)
        "s3sy06aod": ("AOD_2250", lambda array: array),  # AOD band 06 (2250 nm)
        "s3sy01ssa": (
            "SSA_440",
            lambda array: array,
        ),  # single-scattering albedo band 01 (440 nm)
        "s3sy02ssa": (
            "SSA_550",
            lambda array: array,
        ),  # single-scattering albedo band 02 (550 nm)
        "s3sy03ssa": (
            "SSA_670",
            lambda array: array,
        ),  # single-scattering albedo band 03 (670 nm)
        "s3sy04ssa": (
            "SSA_865",
            lambda array: array,
        ),  # single-scattering albedo band 04 (865 nm)
        "s3sy05ssa": (
            "SSA_1600",
            lambda array: array,
        ),  # single-scattering albedo band 05 (1600 nm)
        "s3sy01sr": (
            "Surface_reflectance_440",
            lambda array: array,
        ),  # surface reflectance band 01 (440 nm)
        "s3sy02sr": (
            "Surface_reflectance_550",
            lambda array: array,
        ),  # surface reflectance band 02 (550 nm)
        "s3sy03sr": (
            "Surface_reflectance_670",
            lambda array: array,
        ),  # surface reflectance band 03 (670 nm)
        "s3sy04sr": (
            "Surface_reflectance_865",
            lambda array: array,
        ),  # surface reflectance band 04 (865 nm)
        "s3sy05sr": (
            "Surface_reflectance_1600",
            lambda array: array,
        ),  # surface reflectance band 05 (1600 nm)
        "s3sysunzen": (
            "sun_zenith_nadir",
            lambda array: array,
        ),  # solar zenith angle (degrees)
        "s3sysatzen": (
            "satellite_zenith_nadir",
            lambda array: array,
        ),  # satellite zenith angle (degrees)
        "s3syrelaz": (
            "relative_azimuth_nadir",
            lambda array: array,
        ),  # relative azimuth angle (degrees)
        "s3sycloudfrac": (
            "cloud_fraction_nadir",
            lambda array: array,
        ),  # cloud fraction (0-1)
        "s3sy_lat": ("latitude", lambda array: array),  # pixel latitude (degrees)
        "s3sy_lon": ("longitude", lambda array: array),  # pixel longitude (degrees)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


class MODISFireLexicon(metaclass=LexiconType):
    """Lexicon exposing MODIS Thermal Anomalies daily fields."""

    VOCAB: dict[str, tuple[str, Modifier]] = {
        "fmask": (
            "fire_mask",
            lambda array: array,
        ),  # fire detection mask (0 = background, 9 = high-confidence fire)
        "mfrp": (
            "max_frp",
            lambda array: array,
        ),  # maximum fire radiative power per pixel (MW)
        "qa": (
            "qa",
            lambda array: array,
        ),  # MODIS thermal anomaly quality assurance bits
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


class ECMWFOpenDataIFSLexicon(metaclass=LexiconType):
    """Lexicon exposing ECMWF Open Data IFS variables.

    For available variables, inspect one of the index files:

    .. highlight:: python
    .. code-block:: python

        import planetary_computer

        url = "ttps://ai4edataeuwest.blob.core.windows.net/ecmwf/20251001/00z/ifs/0p25/oper/20251001000000-0h-oper-fc.index"
        signed_url = planetary_computer.sign(url)  # use this URL to download the index file

    """

    SFC_VARIABLES = {
        # Surface variables
        "u100m": "100u",
        "v100m": "100v",
        "u10m": "10u",
        "v10m": "10v",
        "d2m": "2d",
        "t2m": "2t",
        "asn": "asn",
        "ewss": "ewss",
        "lsm": "lsm",
        "msl": "msl",
        "mucape": "mucape",
        "nsss": "nsss",
        "ptype": "ptype",
        "ro": "ro",
        "sdor": "sdor",
        "sithick": "sithick",
        "skt": "skt",
        "slor": "slor",
        "sp": "sp",
        "ssr": "ssr",
        "ssrd": "ssrd",
        "str": "str",
        "strd": "strd",
        "sve": "sve",
        "svn": "svn",
        "tcw": "tcw",
        "tcwv": "tcwv",
        "tp": "tp",
        "tprate": "tprate",
        "ttr": "ttr",
        "z": "z",
        "zos": "zos",
        # Time maxima
        "max_fg10m": "max_i10fg",
        "max_t2m": "max_2t",
        "min_t2m": "min_2t",
    }
    PRS_VARIABLES = {
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
    SOIL_VARIABLES = {
        "stl": "sot",
        "swvl": "vsw",
    }
    SOIL_LAYERS = [1, 2, 3, 4]

    @staticmethod
    def build_vocab(
        sfc_variables: dict,
        prs_variables: dict,
        prs_levels: list[int],
        soil_variables: dict,
        soil_layers: list[int],
    ) -> dict[str, tuple[str, Modifier]]:
        def nmod(x: np.ndarray) -> np.ndarray:
            """Do not modify."""
            return x

        def zmod(x: np.ndarray) -> np.ndarray:
            """Modify geopotential."""
            return x * 9.81

        sfc_map = {k: (v + "::::", nmod) for k, v in sfc_variables.items()}
        prs_map = {}
        for k, v in prs_variables.items():
            mod = zmod if k == "z" else nmod
            prs_map.update({f"{k}{lvl}": (v + f"::{lvl}::", mod) for lvl in prs_levels})
        soil_map = {}
        for k, v in soil_variables.items():
            soil_map.update(
                {f"{k}{lay}": (v + f"::::{lay}", nmod) for lay in soil_layers}
            )
        return {**sfc_map, **soil_map, **prs_map}

    VOCAB = build_vocab(
        SFC_VARIABLES, PRS_VARIABLES, PRS_LEVELS, SOIL_VARIABLES, SOIL_LAYERS
    )

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]
