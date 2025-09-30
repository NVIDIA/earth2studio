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
    """Utility to cast xarray/numpy containers to float32."""
    return array.astype(np.float32)


class OISSTLexicon(metaclass=LexiconType):
    """Lexicon for the NOAA OISST (Optimum Interpolation Sea Surface Temperature) dataset.

    The lexicon maps Earth2Studio variable identifiers to variables contained in the
    `NOAA/NCEI 1/4 Degree Daily Optimum Interpolation Sea Surface Temperature`
    collection hosted on the Microsoft Planetary Computer. Scaling factors are applied
    to convert units to the Earth2Studio conventions where appropriate (e.g. Celsius to
    Kelvin, percentage to fractional concentration).
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # Sea surface temperature in Kelvin. The native product is in Celsius.
        "sst": (
            "sst",
            lambda array: _to_float32(array) + np.float32(273.15),
        ),
        # Sea surface temperature anomalies in degrees Celsius (Kelvin difference).
        "sst_anom": ("anom", _to_float32),
        # Estimated standard deviation of the analysed SST in degrees Celsius.
        "sst_error": ("err", _to_float32),
        "sst_err": ("err", _to_float32),
        # Sea ice concentration as a unit fraction (0-1). Native values are %.
        "sic": ("ice", lambda array: _to_float32(array) / np.float32(100.0)),
        "ice_frac": ("ice", lambda array: _to_float32(array) / np.float32(100.0)),
        "ice_pct": ("ice", _to_float32),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Return the dataset variable name and modifier for ``val``."""
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in OISST lexicon")
        return cls.VOCAB[val]
