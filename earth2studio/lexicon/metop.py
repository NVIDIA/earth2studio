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


class MetOpAMSUALexicon(metaclass=LexiconType):
    """Lexicon for MetOp AMSU-A Level 1B brightness temperature data.

    This lexicon maps the ``amsua`` variable to an identity modifier for
    brightness temperature observations in Kelvin.  Individual channels (1-14)
    are distinguished by the ``channel_index`` column of the returned DataFrame,
    following the same convention used by :class:`~earth2studio.data.UFSObsSat`.

    AMSU-A is a 15-channel microwave radiometer on the MetOp satellite series
    providing atmospheric temperature profiles from the surface through the
    stratosphere.

    The data source returns calibrated brightness temperatures (K) derived from
    scene radiances via the Planck function with band correction coefficients.

    Notes
    -----
    AMSU-A Channel Specifications (MetOp-B/C):

    ========  ===========  ==========================================
    Channel   Frequency    Primary Application
    ========  ===========  ==========================================
    1         23.8 GHz     Surface emissivity, precipitation
    2         31.4 GHz     Total column water vapor, cloud liquid water
    3         50.3 GHz     Lower troposphere temperature (~surface)
    4         52.8 GHz     Mid-troposphere temperature (~700 hPa)
    5         53.6 GHz     Mid-troposphere temperature (~500 hPa)
    6         54.4 GHz     Upper troposphere temperature (~400 hPa)
    7         54.9 GHz     Tropopause temperature (~300 hPa)
    8         55.5 GHz     Lower stratosphere (~200 hPa)
    9         57.29 GHz    Stratosphere (~90 hPa)
    10        57.29 GHz    Stratosphere (~50 hPa)
    11        57.29 GHz    Stratosphere (~25 hPa)
    12        57.29 GHz    Stratosphere (~10 hPa)
    13        57.29 GHz    Upper stratosphere (~5 hPa)
    14        57.29 GHz    Upper stratosphere (~2 hPa)
    ========  ===========  ==========================================

    Channel 15 (89.0 GHz) is excluded because the L1B product marks
    ~97% of its measurements as missing due to quality filtering.

    Spatial resolution is approximately 48 km at nadir with 30 field-of-view
    positions per scan line. Each scan line takes 8 seconds.

    Variable documentation:
        https://user.eumetsat.int/s3/eup-strapi-media/pdf_atovsl1b_pg_8bbaa8ba48.pdf

    References
    ----------
    - EUMETSAT. ATOVS Level 1 Products Product Guide (EUM/OPS-EPS/MAN/04/0030).
    - EUMETSAT. AMSU-A Level 1 Product Format Specification (EPS.MIS.SPE.97228).
    """

    # Number of AMSU-A cross-track field-of-view positions per scan line
    AMSUA_NUM_FOVS = 30
    # Number of valid AMSU-A channels (excludes channel 15)
    AMSUA_NUM_CHANNELS = 14
    # AMSU-A channel center frequencies (GHz)
    AMSUA_CHANNEL_FREQS: dict[int, float] = {
        1: 23.8,
        2: 31.4,
        3: 50.3,
        4: 52.8,
        5: 53.596,
        6: 54.4,
        7: 54.94,
        8: 55.5,
        9: 57.29,
        10: 57.29,
        11: 57.29,
        12: 57.29,
        13: 57.29,
        14: 57.29,
    }

    VOCAB: dict[str, str] = {
        "amsua": "brightnessTemperature",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get AMSU-A data key and modifier for a variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (``amsua``)

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - Data key for brightness temperature
            - Modifier function (identity -- brightness temperature in K)
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp AMSU-A lexicon")
        return cls.VOCAB[val], lambda x: x


class MetOpAVHRRLexicon(metaclass=LexiconType):
    """Lexicon for MetOp AVHRR Level 1B data.

    This lexicon maps the ``avhrr`` variable to an identity modifier.
    Individual channels (1, 2, 3a, 3b, 4, 5) are distinguished by the
    ``channel_index`` column of the returned DataFrame.  The ``class``
    column differentiates observation types: ``"refl"`` for visible/NIR
    channels (1, 2, 3A) which return reflectance (%), and ``"rad"`` for
    thermal IR channels (3B, 4, 5) which return brightness temperature (K).

    AVHRR/3 is a 6-channel radiometer covering visible, near-infrared,
    and thermal infrared bands aboard the MetOp satellite series.
    Channels 3A and 3B are mutually exclusive (only one active per scan
    line, switched by telecommand).

    Notes
    -----
    AVHRR/3 Channel Specifications:

    ========  ==============  ==============================================
    Channel   Wavelength      Primary Application
    ========  ==============  ==============================================
    1         0.63 um (VIS)   Cloud/surface mapping, vegetation index (NDVI)
    2         0.865 um (NIR)  Vegetation, water/land boundaries
    3A        1.61 um (SWIR)  Snow/ice discrimination (daytime only)
    3B        3.74 um (MWIR)  Fire detection, night-time cloud/SST
    4         10.8 um (TIR)   Sea/land surface temperature, cloud top temp
    5         12.0 um (TIR)   Surface temperature, water vapor correction
    ========  ==============  ==============================================

    Channels 3A/3B are mutually exclusive: 3A operates during daytime, 3B
    during nighttime (switched by telecommand). Both cannot be active
    simultaneously.

    Spatial resolution is 1.1 km at nadir with 2048 pixels per scan line.

    Variable documentation:
        https://user.eumetsat.int/s3/eup-strapi-media/pdf_ten_97231_eps_avhrr_l1_pfs_8bbab17e80.pdf

    References
    ----------
    - EUMETSAT. AVHRR Level 1b Product Guide (EUM/OPS-EPS/MAN/04/0029).
    - EUMETSAT. AVHRR Level 1 Product Format Specification (EPS.MIS.SPE.97231).
    """

    # Number of AVHRR navigation tie points per scan line
    AVHRR_NAV_NUM_POINTS = 103
    # Number of AVHRR channels
    AVHRR_NUM_CHANNELS = 6
    # AVHRR channel wavelengths (µm)
    AVHRR_CHANNEL_WAVELENGTHS: dict[str, float] = {
        "1": 0.63,
        "2": 0.865,
        "3a": 1.61,
        "3b": 3.74,
        "4": 10.8,
        "5": 12.0,
    }
    # Channels that return reflectance (%) vs brightness temperature (K)
    AVHRR_REFLECTANCE_CHANNELS: frozenset[str] = frozenset({"1", "2", "3a"})
    AVHRR_THERMAL_CHANNELS: frozenset[str] = frozenset({"3b", "4", "5"})

    # Channel index mapping (1-based) for channel_index column
    AVHRR_CHANNEL_INDEX: dict[str, int] = {
        "1": 1,
        "2": 2,
        "3a": 3,
        "3b": 4,
        "4": 5,
        "5": 6,
    }

    VOCAB: dict[str, str] = {
        "avhrr": "calibratedObservation",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get AVHRR data key and modifier for a variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (``avhrr``)

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - Data key for calibrated observation
            - Modifier function (identity)
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp AVHRR lexicon")
        return cls.VOCAB[val], lambda x: x
