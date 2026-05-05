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


class MRMSLexicon(metaclass=LexiconType):
    """Multi-Radar/Multi-Sensor (MRMS) product lexicon.

    This lexicon provides a minimal mapping from Earth2Studio variable identifiers
    to MRMS product names. Initially, only composite reflectivity is exposed.

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-mrms-pds/
    - https://www.nssl.noaa.gov/projects/mrms/
    """

    # Minimal vocabulary: map Earth2Studio "refc" to MRMS product name used in keys
    VOCAB: dict[str, str] = {
        "refc": "MergedReflectivityQCComposite_00.50",
        "refc_base": "MergedBaseReflectivityQC_00.50",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from MRMS vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - MRMS product name used in the S3 key path.
            - A modifier function to apply to the loaded values (identity).
        """
        mrms_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify data value (if necessary)."""
            return x

        return mrms_key, mod
