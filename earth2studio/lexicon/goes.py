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


class GOESLexicon(metaclass=LexiconType):
    """GOES ABI lexicon for mapping standardized variable names to GOES ABI NetCDF variable names.

    This lexicon maps our standardized spectral band names (e.g., 'vis047') to the actual
    variable names used in GOES ABI NetCDF files on AWS. It also includes any necessary
    data transformations or modifiers for each band. For more information on the GOES ABI
    data, see the GOES Beginners Guide or the GOES ABI data documentation:
    https://noaa-goes16.s3.amazonaws.com/Beginners_Guide_to_GOES-R_Series_Data.pdf
    https://www.goes-r.gov/spacesegment/ABI-tech-summary.html

    Parameters
    ----------
    val : str
        Standardized variable name (e.g., 'vis047')

    """

    # Mapping of standardized names to GOES ABI variable names and modifiers
    # Format: "standardized_name": ("goes_variable_name", modifier_function)
    # The modifier function can be used to transform the data if needed
    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # Visible bands (0.5 km resolution)
        "vis047": ("CMI_C01", lambda x: x),  # Blue
        "vis064": ("CMI_C02", lambda x: x),  # Red
        # Near-IR bands (1.0 km resolution)
        "nir086": ("CMI_C03", lambda x: x),  # Vegetation
        "nir137": ("CMI_C04", lambda x: x),  # Cirrus
        "nir161": ("CMI_C05", lambda x: x),  # Snow/Ice
        "nir224": ("CMI_C06", lambda x: x),  # Cloud Particle Size
        # IR bands (2.0 km resolution)
        "ir390": ("CMI_C07", lambda x: x),  # Shortwave Window
        "ir619": ("CMI_C08", lambda x: x),  # Upper-level Water Vapor
        "ir695": ("CMI_C09", lambda x: x),  # Mid-level Water Vapor
        "ir734": ("CMI_C10", lambda x: x),  # Lower-level Water Vapor
        "ir850": ("CMI_C11", lambda x: x),  # Cloud-top Phase
        "ir961": ("CMI_C12", lambda x: x),  # Ozone
        "ir1035": ("CMI_C13", lambda x: x),  # Clean IR Longwave Window
        "ir1120": ("CMI_C14", lambda x: x),  # IR Longwave Window
        "ir1230": ("CMI_C15", lambda x: x),  # Dirty IR Longwave Window
        "ir1330": ("CMI_C16", lambda x: x),  # CO2 Longwave IR
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get GOES ABI variable name and modifier for a standardized variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'vis047')

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - GOES ABI variable name (e.g., 'CMI_C01')
            - Modifier function for data transformation
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in GOES lexicon")
        return cls.VOCAB[val]
