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

from earth2studio.lexicon.base import LexiconType


class JPSSLexicon(metaclass=LexiconType):
    """JPSS VIIRS SDR lexicon mapping standardized band names to product codes.

    This lexicon provides a stable vocabulary for JPSS VIIRS Imagery (I) and
    Moderate (M) bands. Each standardized name maps to the associated VIIRS
    SDR product prefix used in AWS object names and the default dataset to
    extract from the HDF5 file.

    Notes
    -----
    - I-bands (375 m): I1, I2, I3, I4, I5 → products SVI01..SVI05
    - M-bands (750 m): M01..M16 → products SVM01..SVM16
    - Default field is "Radiance". Some products may also include
      "Reflectance"; future extensions can allow selecting specific fields.
    """

    # Mapping of standardized names to (product_code, dataset_name, modifier)
    # product_code corresponds to the prefix of files on S3 (e.g., SVI01, SVM01)
    # dataset_name is the HDF5 dataset to read. We default to "Radiance".
    VOCAB: dict[str, tuple[str, str, Callable[[Any], Any]]] = {
        # I bands
        "i1": ("SVI01", "Radiance", lambda x: x),
        "i2": ("SVI02", "Radiance", lambda x: x),
        "i3": ("SVI03", "Radiance", lambda x: x),
        "i4": ("SVI04", "Radiance", lambda x: x),
        "i5": ("SVI05", "Radiance", lambda x: x),
        # M bands
        "m01": ("SVM01", "Radiance", lambda x: x),
        "m02": ("SVM02", "Radiance", lambda x: x),
        "m03": ("SVM03", "Radiance", lambda x: x),
        "m04": ("SVM04", "Radiance", lambda x: x),
        "m05": ("SVM05", "Radiance", lambda x: x),
        "m06": ("SVM06", "Radiance", lambda x: x),
        "m07": ("SVM07", "Radiance", lambda x: x),
        "m08": ("SVM08", "Radiance", lambda x: x),
        "m09": ("SVM09", "Radiance", lambda x: x),
        "m10": ("SVM10", "Radiance", lambda x: x),
        "m11": ("SVM11", "Radiance", lambda x: x),
        "m12": ("SVM12", "Radiance", lambda x: x),
        "m13": ("SVM13", "Radiance", lambda x: x),
        "m14": ("SVM14", "Radiance", lambda x: x),
        "m15": ("SVM15", "Radiance", lambda x: x),
        "m16": ("SVM16", "Radiance", lambda x: x),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, str, Callable[[Any], Any]]:
        """Get VIIRS SDR product code, dataset, and modifier for a standardized name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'i1', 'm05')

        Returns
        -------
        tuple[str, str, Callable]
            Tuple containing:
            - VIIRS SDR product code (e.g., 'SVI01', 'SVM05')
            - HDF5 dataset name to extract (e.g., 'Radiance')
            - Modifier function for data transformation
        """
        key = val.lower()
        if key not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in VIIRS lexicon")
        return cls.VOCAB[key]


