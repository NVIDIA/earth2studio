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

import numpy as np

from earth2studio.lexicon.base import LexiconType


class NClimGridLexicon(metaclass=LexiconType):
    """Lexicon for NClimGrid daily gridded climate data source.

    Maps Earth2Studio variable names to NClimGrid native variable identifiers
    and provides unit conversion modifiers.

    Note
    ----
    Variable documentation:
    https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00332
    """

    VOCAB = {
        "t2m_max": "tmax",
        "t2m_min": "tmin",
        "tp": "prcp",
        "spi": "spi",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return remote key and modifier function for a variable.

        Parameters
        ----------
        val : str
            Variable name (E2S convention).

        Returns
        -------
        tuple[str, Callable]
            Remote key string and modifier function.
        """
        native = cls.VOCAB[val]

        def modifier(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype="float32")
            if native in ("tmax", "tmin"):
                return x + 273.15  # °C -> K
            if native == "prcp":
                return x / 1000.0  # mm -> m
            return x  # spi is dimensionless

        return native, modifier
