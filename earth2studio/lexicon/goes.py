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

    This lexicon maps our standardized spectral band names (e.g., 'abi01c') to the actual
    variable names used in GOES ABI NetCDF files on AWS. It also includes any necessary
    data transformations or modifiers for each band. For more information on the GOES ABI
    data, see the GOES Beginners Guide or the GOES ABI data documentation:
    https://noaa-goes16.s3.amazonaws.com/Beginners_Guide_to_GOES-R_Series_Data.pdf
    https://www.goes-r.gov/spacesegment/ABI-tech-summary.html

    Parameters
    ----------
    val : str
        Standardized variable name (e.g., 'abi01c')

    """

    # Mapping of standardized names to GOES ABI variable names and modifiers
    # Format: "standardized_name": ("goes_variable_name", modifier_function)
    # The modifier function can be used to transform the data if needed
    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # ABI Channel 1-2: Visible bands (0.5 km resolution)
        "abi01c": ("CMI_C01", lambda x: x),  # Blue (0.47 μm)
        "abi02c": ("CMI_C02", lambda x: x),  # Red (0.64 μm)
        # ABI Channel 3-6: Near-IR bands (1.0 km resolution)
        "abi03c": ("CMI_C03", lambda x: x),  # Vegetation (0.86 μm)
        "abi04c": ("CMI_C04", lambda x: x),  # Cirrus (1.37 μm)
        "abi05c": ("CMI_C05", lambda x: x),  # Snow/Ice (1.61 μm)
        "abi06c": ("CMI_C06", lambda x: x),  # Cloud Particle Size (2.24 μm)
        # ABI Channel 7-16: IR bands (2.0 km resolution)
        "abi07c": ("CMI_C07", lambda x: x),  # Shortwave Window (3.90 μm)
        "abi08c": ("CMI_C08", lambda x: x),  # Upper-level Water Vapor (6.19 μm)
        "abi09c": ("CMI_C09", lambda x: x),  # Mid-level Water Vapor (6.95 μm)
        "abi10c": ("CMI_C10", lambda x: x),  # Lower-level Water Vapor (7.34 μm)
        "abi11c": ("CMI_C11", lambda x: x),  # Cloud-top Phase (8.50 μm)
        "abi12c": ("CMI_C12", lambda x: x),  # Ozone (9.61 μm)
        "abi13c": ("CMI_C13", lambda x: x),  # Clean IR Longwave Window (10.35 μm)
        "abi14c": ("CMI_C14", lambda x: x),  # IR Longwave Window (11.20 μm)
        "abi15c": ("CMI_C15", lambda x: x),  # Dirty IR Longwave Window (12.30 μm)
        "abi16c": ("CMI_C16", lambda x: x),  # CO2 Longwave IR (13.30 μm)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get GOES ABI variable name and modifier for a standardized variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'abi01c')

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
