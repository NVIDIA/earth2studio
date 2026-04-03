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

from earth2studio.lexicon.base import LexiconType


class MetOpAMSUALexicon(metaclass=LexiconType):
    """Lexicon for MetOp AMSU-A Level 1B brightness temperature data.

    Maps Earth2Studio variable names to AMSU-A channel indices (1-15).
    AMSU-A is a 15-channel microwave radiometer on the MetOp satellite series
    providing atmospheric temperature profiles from the surface through the
    stratosphere. Channels 1-2 are window channels sensitive to surface
    properties and precipitation. Channels 3-8 provide tropospheric temperature
    sounding. Channels 9-14 provide stratospheric temperature sounding.
    Channel 15 is a high-frequency window channel.

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

    VOCAB: dict[str, int] = {
        "amsua01": 1,  # 23.8 GHz - Window: surface emissivity, precipitation
        "amsua02": 2,  # 31.4 GHz - Window: total column water vapor, cloud liquid water
        "amsua03": 3,  # 50.3 GHz - Troposphere: lower troposphere temperature
        "amsua04": 4,  # 52.8 GHz - Troposphere: mid-troposphere (~700 hPa)
        "amsua05": 5,  # 53.6 GHz - Troposphere: mid-troposphere (~500 hPa)
        "amsua06": 6,  # 54.4 GHz - Troposphere: upper troposphere (~400 hPa)
        "amsua07": 7,  # 54.9 GHz - Tropopause: tropopause region (~300 hPa)
        "amsua08": 8,  # 55.5 GHz - Stratosphere: lower stratosphere (~200 hPa)
        "amsua09": 9,  # 57.29 GHz - Stratosphere: ~90 hPa peak weighting
        "amsua10": 10,  # 57.29 GHz - Stratosphere: ~50 hPa peak weighting
        "amsua11": 11,  # 57.29 GHz - Stratosphere: ~25 hPa peak weighting
        "amsua12": 12,  # 57.29 GHz - Stratosphere: ~10 hPa peak weighting
        "amsua13": 13,  # 57.29 GHz - Stratosphere: ~5 hPa peak weighting
        "amsua14": 14,  # 57.29 GHz - Stratosphere: ~2 hPa peak weighting
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[int, Callable]:
        """Get AMSU-A channel index and modifier for a variable name.

        Parameters
        ----------
        val : str
            Variable name (e.g., ``'amsua01'``, ``'amsua14'``)

        Returns
        -------
        tuple[int, Callable]
            Channel index (1-14) and identity modifier function
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp AMSU-A lexicon")
        return cls.VOCAB[val], lambda x: x


class MetOpAVHRRLexicon(metaclass=LexiconType):
    """Lexicon for MetOp AVHRR Level 1B data.

    Maps Earth2Studio variable names to satpy dataset identifiers for the
    AVHRR/3 instrument on the MetOp satellite series. AVHRR/3 is a 6-channel
    radiometer covering visible, near-infrared, and thermal infrared bands.
    Channels 3A and 3B are mutually exclusive (only one active per scan line,
    switched by telecommand).

    For visible/near-IR channels (1, 2, 3A), the data source returns calibrated
    radiances (W m-2 sr-1 um-1). For thermal IR channels (3B, 4, 5), the data
    source returns brightness temperatures (K).

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

    VOCAB: dict[str, str] = {
        "avhrr01": "1",  # 0.63 um - VIS: cloud/surface, NDVI red component
        "avhrr02": "2",  # 0.865 um - NIR: vegetation, water/land, NDVI NIR component
        "avhrr3a": "3a",  # 1.61 um - SWIR: snow/ice discrimination (daytime only)
        "avhrr3b": "3b",  # 3.74 um - MWIR: fire detection, night SST, cloud mapping
        "avhrr04": "4",  # 10.8 um - TIR: sea/land surface temp, cloud top temp
        "avhrr05": "5",  # 12.0 um - TIR: surface temp, water vapor correction
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get satpy dataset name and modifier for a variable name.

        Parameters
        ----------
        val : str
            Variable name (e.g., ``'avhrr01'``, ``'avhrr3b'``)

        Returns
        -------
        tuple[str, Callable]
            Satpy dataset name (e.g. ``'1'``, ``'3b'``) and identity modifier
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp AVHRR lexicon")
        return cls.VOCAB[val], lambda x: x
