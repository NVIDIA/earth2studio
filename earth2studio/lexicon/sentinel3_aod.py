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


def _to_float32(array: Any) -> Any:
    return array.astype(np.float32)


class Sentinel3AODLexicon(metaclass=LexiconType):
    """Lexicon exposing Sentinel-3 SYNERGY aerosol and reflectance variables."""

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        "aod_440": ("AOD_440", _to_float32),
        "aod_550": ("AOD_550", _to_float32),
        "aod_670": ("AOD_670", _to_float32),
        "aod_865": ("AOD_865", _to_float32),
        "aod_1600": ("AOD_1600", _to_float32),
        "aod_2250": ("AOD_2250", _to_float32),
        "ssa_440": ("SSA_440", _to_float32),
        "ssa_550": ("SSA_550", _to_float32),
        "ssa_670": ("SSA_670", _to_float32),
        "ssa_865": ("SSA_865", _to_float32),
        "ssa_1600": ("SSA_1600", _to_float32),
        "surface_reflectance_440": ("Surface_reflectance_440", _to_float32),
        "surface_reflectance_550": ("Surface_reflectance_550", _to_float32),
        "surface_reflectance_670": ("Surface_reflectance_670", _to_float32),
        "surface_reflectance_865": ("Surface_reflectance_865", _to_float32),
        "surface_reflectance_1600": ("Surface_reflectance_1600", _to_float32),
        "sun_zenith": ("sun_zenith_nadir", _to_float32),
        "satellite_zenith": ("satellite_zenith_nadir", _to_float32),
        "relative_azimuth": ("relative_azimuth_nadir", _to_float32),
        "cloud_fraction": ("cloud_fraction_nadir", _to_float32),
        "_lat": ("latitude", _to_float32),
        "_lon": ("longitude", _to_float32),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Sentinel-3 AOD lexicon")
        return cls.VOCAB[val]
