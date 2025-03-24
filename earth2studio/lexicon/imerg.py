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


class IMERGLexicon(metaclass=LexiconType):
    """IMERG Lexicon"""

    VOCAB = {
        "tp": "precipitation",
        "tpp": "probabilityLiquidPrecipitation",
        "tpi": "precipitationQualityIndex",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in IMERG vocabulary."""
        imerg_key = cls.VOCAB[val]
        if val == "tp":
            # IMERG is mm/hr by default, convert to meters to match ECMWF
            # https://arthurhou.pps.eosdis.nasa.gov/Documents/IMERG_TechnicalDocumentation_final.pdf
            def mod(x: np.array) -> np.array:
                """Modify data value (if necessary)."""
                return x / 1000.0

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return imerg_key, mod
