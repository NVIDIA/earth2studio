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
from typing import Any

from earth2studio.lexicon.base import LexiconType


class HimawariLexicon(metaclass=LexiconType):
    """Himawari AHI lexicon for mapping standardized variable names to AHI channel
    identifiers used in the ISatSS L2 NetCDF files on AWS.

    The Himawari-8/9 Advanced Himawari Imager (AHI) has 16 spectral bands. The ISatSS
    processing produces per-channel tile files with the variable ``Sectorized_CMI``.
    This lexicon maps Earth2Studio variable names (``ahi01``-``ahi16``) to channel
    numbers used in the ISatSS filename convention (``M1C01``-``M1C16``).

    Note
    ----
    AHI band specifications:
    https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html

    NOAA AWS bucket documentation:
    https://aws.amazon.com/marketplace/pp/prodview-eu33kalocbhiw

    Parameters
    ----------
    val : str
        Standardized variable name (e.g., 'ahi01')
    """

    # Mapping of standardized names to AHI channel identifiers and modifiers
    # Format: "standardized_name": ("channel_id", modifier_function)
    # channel_id matches the M1C{nn} portion of the ISatSS filename
    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # AHI Band 1-3: Visible bands
        "ahi01": ("M1C01", lambda x: x),  # Visible Blue (0.47 µm) - 1 km
        "ahi02": ("M1C02", lambda x: x),  # Visible Green (0.51 µm) - 1 km
        "ahi03": ("M1C03", lambda x: x),  # Visible Red (0.64 µm) - 0.5 km
        # AHI Band 4-6: Near-IR bands
        "ahi04": ("M1C04", lambda x: x),  # Near-IR (0.86 µm) - 1 km
        "ahi05": ("M1C05", lambda x: x),  # Near-IR (1.6 µm) - 2 km
        "ahi06": ("M1C06", lambda x: x),  # Near-IR (2.3 µm) - 2 km
        # AHI Band 7-16: IR bands (all 2 km)
        "ahi07": ("M1C07", lambda x: x),  # Shortwave IR (3.9 µm)
        "ahi08": ("M1C08", lambda x: x),  # Water Vapor (6.2 µm)
        "ahi09": ("M1C09", lambda x: x),  # Water Vapor (6.9 µm)
        "ahi10": ("M1C10", lambda x: x),  # Water Vapor (7.3 µm)
        "ahi11": ("M1C11", lambda x: x),  # Cloud-top Phase (8.6 µm)
        "ahi12": ("M1C12", lambda x: x),  # Ozone (9.6 µm)
        "ahi13": ("M1C13", lambda x: x),  # Clean IR Window (10.4 µm)
        "ahi14": ("M1C14", lambda x: x),  # IR Window (11.2 µm)
        "ahi15": ("M1C15", lambda x: x),  # Dirty IR Window (12.4 µm)
        "ahi16": ("M1C16", lambda x: x),  # CO2 Longwave IR (13.3 µm)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get AHI channel identifier and modifier for a standardized variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'ahi01')

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - AHI channel identifier (e.g., 'M1C01')
            - Modifier function for data transformation
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Himawari lexicon")
        return cls.VOCAB[val]
