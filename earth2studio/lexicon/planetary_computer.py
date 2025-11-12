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
        ),
        "sst_anom": ("anom", lambda array: array),
        "sst_error": ("err", lambda array: array),
        "sst_err": ("err", lambda array: array),
        "sic": ("ice", lambda array: array / np.float32(100.0)),
        "ice_frac": ("ice", lambda array: array / np.float32(100.0)),
        "ice_pct": ("ice", lambda array: array),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in OISST lexicon")
        return cls.VOCAB[val]


class Sentinel3AODLexicon(metaclass=LexiconType):
    """Lexicon exposing Sentinel-3 SYNERGY aerosol and reflectance variables."""

    VOCAB: dict[str, tuple[str, Modifier]] = {
        "aod_440": ("AOD_440", lambda array: array),
        "aod_550": ("AOD_550", lambda array: array),
        "aod_670": ("AOD_670", lambda array: array),
        "aod_865": ("AOD_865", lambda array: array),
        "aod_1600": ("AOD_1600", lambda array: array),
        "aod_2250": ("AOD_2250", lambda array: array),
        "ssa_440": ("SSA_440", lambda array: array),
        "ssa_550": ("SSA_550", lambda array: array),
        "ssa_670": ("SSA_670", lambda array: array),
        "ssa_865": ("SSA_865", lambda array: array),
        "ssa_1600": ("SSA_1600", lambda array: array),
        "surface_reflectance_440": ("Surface_reflectance_440", lambda array: array),
        "surface_reflectance_550": ("Surface_reflectance_550", lambda array: array),
        "surface_reflectance_670": ("Surface_reflectance_670", lambda array: array),
        "surface_reflectance_865": ("Surface_reflectance_865", lambda array: array),
        "surface_reflectance_1600": ("Surface_reflectance_1600", lambda array: array),
        "sun_zenith": ("sun_zenith_nadir", lambda array: array),
        "satellite_zenith": ("satellite_zenith_nadir", lambda array: array),
        "relative_azimuth": ("relative_azimuth_nadir", lambda array: array),
        "cloud_fraction": ("cloud_fraction_nadir", lambda array: array),
        "_lat": ("latitude", lambda array: array),
        "_lon": ("longitude", lambda array: array),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Sentinel-3 AOD lexicon")
        return cls.VOCAB[val]


class MODISFireLexicon(metaclass=LexiconType):
    """Lexicon exposing MODIS Thermal Anomalies daily fields."""

    VOCAB: dict[str, tuple[str, Modifier]] = {
        "fire_mask": ("fire_mask", lambda array: array),
        "max_frp": ("max_frp", lambda array: array),
        "qa": ("qa", lambda array: array),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MODIS Fire lexicon")
        return cls.VOCAB[val]
