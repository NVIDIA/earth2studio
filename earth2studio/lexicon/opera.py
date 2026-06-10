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

from .base import LexiconType


class OPERALexicon(metaclass=LexiconType):
    """Lexicon for EUMETNET OPERA composite radar data.

    Maps Earth2Studio variable names to ODIM HDF5 quantity codes stored inside
    the composite files.

    Note
    ----
    Variable documentation:
    https://eumetnet.github.io/openradardata-documentation/
    """

    VOCAB: dict[str, str] = {
        "refc": "DBZH",    # composite reflectivity (dBZ)
        "tprate": "RATE",  # instantaneous surface rain rate (mm h-1)
        "tp01": "ACRR",    # 1-hour rainfall accumulation (m)
    }

    # Filename parameter used in the legacy ODYSSEY archive (pre-2024-07).
    # The ODIM quantity code inside the HDF5 is unchanged; only the
    # filename component differs.
    LEGACY_FILENAME_PARAMS: dict[str, str] = {
        "DBZH": "DBZH_QIND",
        "RATE": "QIND_RATE",
        "ACRR": "ACRR_QIND",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return ODIM quantity name and modifier for an Earth2Studio variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable name.

        Returns
        -------
        tuple[str, Callable]
            ODIM quantity code (used to identify the dataset inside the HDF5
            file) and an identity modifier function.

        Raises
        ------
        KeyError
            If ``val`` is not in the lexicon.
        """
        quantity = cls.VOCAB[val]

        if val == "tp01":

            def mod(x: np.ndarray) -> np.ndarray:
                return x / 1000.0  # mm -> m

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                return x

        return quantity, mod
