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

import numpy as np

from earth2studio.lexicon.base import LexiconType


class MetOpAMSUALexicon(metaclass=LexiconType):
    """Lexicon for MetOp AMSU-A Level 1B brightness temperature data.

    This lexicon maps the ``amsua`` variable to an identity modifier for
    brightness temperature observations in Kelvin.  Individual channels (1-15)
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
    15        89.0 GHz     Surface, precipitation, ice
    ========  ===========  ==========================================

    Channel 15 (89.0 GHz) may have a high fraction of missing values
    in some L1B products due to quality filtering.

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
    # Number of valid AMSU-A channels
    AMSUA_NUM_CHANNELS = 15
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
        15: 89.0,
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


class MetOpMHSLexicon(metaclass=LexiconType):
    """Lexicon for MetOp MHS Level 1B brightness temperature data.

    This lexicon maps the ``mhs`` variable to an identity modifier for
    brightness temperature observations in Kelvin.  Individual channels (1-5)
    are distinguished by the ``channel_index`` column of the returned DataFrame,
    following the same convention used by :class:`~earth2studio.data.UFSObsSat`.

    MHS (Microwave Humidity Sounder) is a 5-channel cross-track scanning
    microwave radiometer aboard the MetOp satellite series, providing
    atmospheric humidity profiles from the surface through the upper
    troposphere.  MHS is the successor to AMSU-B and operates on MetOp-A
    (decommissioned 2021), MetOp-B, and MetOp-C.

    The data source returns calibrated brightness temperatures (K) derived from
    scene radiances via the Planck function with band correction coefficients.

    Notes
    -----
    MHS Channel Specifications (MetOp-B/C):

    ========  ===============  ==========================================
    Channel   Frequency        Primary Application
    ========  ===============  ==========================================
    1         89.0 GHz         Surface emissivity, ice/rain detection
    2         157.0 GHz        Mid-troposphere humidity, precipitation
    3         183.311±1 GHz    Upper troposphere humidity (~300 hPa)
    4         183.311±3 GHz    Mid/upper troposphere humidity (~500 hPa)
    5         190.311 GHz      Lower troposphere humidity, precipitation
    ========  ===============  ==========================================

    Spatial resolution is approximately 16 km at nadir with 90 field-of-view
    positions per scan line.  The scan period is 8/3 seconds (shared antenna
    assembly with AMSU-A), covering a swath of ~2180 km.

    Variable documentation:
        https://user.eumetsat.int/s3/eup-strapi-media/pdf_atovsl1b_pg_8bbaa8ba48.pdf

    References
    ----------
    - EUMETSAT. ATOVS Level 1 Products Product Guide (EUM/OPS-EPS/MAN/04/0030).
    - NOAA KLM User's Guide, Section 3.9 "MHS Instrument Description".
    """

    # Number of MHS cross-track field-of-view positions per scan line
    MHS_NUM_FOVS = 90
    # Number of MHS channels
    MHS_NUM_CHANNELS = 5
    # MHS channel center frequencies (GHz)
    MHS_CHANNEL_FREQS: dict[int, float] = {
        1: 89.0,
        2: 157.0,
        3: 183.311,
        4: 183.311,
        5: 190.311,
    }

    VOCAB: dict[str, str] = {
        "mhs": "brightnessTemperature",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get MHS data key and modifier for a variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (``mhs``)

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - Data key for brightness temperature
            - Modifier function (identity -- brightness temperature in K)
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp MHS lexicon")
        return cls.VOCAB[val], lambda x: x


class MetOpIASILexicon(metaclass=LexiconType):
    """Lexicon for MetOp IASI Level 1C brightness temperature data.

    This lexicon maps the ``iasi`` variable to an identity modifier for
    brightness temperature observations in Kelvin.  Individual channels (1-8461)
    are distinguished by the ``channel_index`` column of the returned DataFrame,
    following the same convention used by :class:`~earth2studio.data.MetOpAMSUA`
    and :class:`~earth2studio.data.UFSObsSat`.

    IASI (Infrared Atmospheric Sounding Interferometer) is a Fourier-transform
    infrared spectrometer on the MetOp satellite series providing 8461 spectral
    channels across the thermal infrared (645--2760 cm-1).  The instrument
    measures calibrated spectral radiances that are converted to brightness
    temperature (K) via the Planck function.

    The data source returns calibrated brightness temperatures (K) for a
    user-selectable subset of the 8461 channels.  Each IASI field of regard
    (EFOV) comprises a 2x2 array of instantaneous fields of view (IFOVs),
    each approximately 12 km diameter at nadir.

    Notes
    -----
    IASI Spectral Band Specifications:

    ======  ===============  ===============  =================
    Band    Channel Range    Wavenumber (cm-1) Primary Application
    ======  ===============  ===============  =================
    1       1 -- 1997        645 -- 1210      CO2 / temperature sounding
    2       1998 -- 5116     1210 -- 1990     Water vapor / ozone
    3       5117 -- 8461     1990 -- 2760     Shortwave IR / trace gases
    ======  ===============  ===============  =================

    Spectral sampling is 0.25 cm-1 with 0.5 cm-1 apodised resolution.

    Each scan line contains 30 EFOV positions x 4 IFOV pixels = 120 spectra.
    Spatial resolution is approximately 12 km at nadir per IFOV.

    Variable documentation:
        https://data.eumetsat.int/product/EO:EUM:DAT:METOP:IASIL1C-ALL

    References
    ----------
    - Clerbaux, C. et al. (2009). Monitoring of atmospheric composition using
      the thermal infrared IASI/MetOp sounder. Atmos. Chem. Phys., 9, 6041-6054.
    - EUMETSAT. IASI Level 1 Product Guide (EUM/OPS-EPS/MAN/04/0032).
    """

    # Number of IASI cross-track extended field-of-view positions per scan line
    IASI_NUM_EFOVS = 30
    # Number of instantaneous fields of view per EFOV (2x2 array)
    IASI_NUM_IFOVS = 4
    # Total number of IASI spectral channels
    IASI_NUM_CHANNELS = 8461
    # Spectral sampling interval (cm-1)
    IASI_SPECTRAL_SAMPLING = 0.25
    # IASI spectral band definitions: (first_channel, last_channel, start_wn, end_wn)
    IASI_BANDS: dict[int, tuple[int, int, float, float]] = {
        1: (1, 1997, 645.0, 1144.0),
        2: (1998, 5116, 1210.0, 1989.75),
        3: (5117, 8461, 2000.0, 2760.0),
    }

    # 174 IASI channels assimilated by NOAA's GSI system (0-based indices).
    # Derived from the ``sensor_chan`` variable in GSI diagnostic NetCDF4 files
    # on S3 (e.g. diag_iasi_metop-c_ges), converted to 0-based
    # (i.e. 1-based channel number minus 1).  These span all three IASI bands
    # and match the channel set available via :class:`~earth2studio.data.UFSObsSat`.
    # fmt: off
    IASI_GSI_CHANNELS: np.ndarray = np.array([
        15,   37,   48,   50,   54,   56,   58,   60,   62,   65,
        69,   71,   73,   78,   80,   82,   84,   86,  103,  105,
       108,  110,  112,  115,  118,  121,  124,  127,  130,  132,
       134,  137,  140,  143,  145,  147,  150,  153,  156,  158,
       160,  162,  166,  169,  172,  175,  179,  184,  186,  192,
       198,  204,  206,  209,  211,  213,  216,  218,  221,  223,
       225,  229,  231,  235,  238,  242,  245,  248,  251,  253,
       259,  261,  264,  266,  274,  281,  293,  295,  298,  302,
       305,  322,  326,  328,  334,  344,  346,  349,  353,  355,
       359,  365,  370,  372,  374,  376,  378,  380,  382,  385,
       388,  397,  400,  403,  406,  409,  413,  415,  425,  427,
       431,  433,  438,  444,  456,  514,  545,  551,  558,  565,
       570,  572,  645,  661,  667,  755,  866,  905,  920, 1026,
      1045, 1120, 1132, 1190, 1193, 1270, 1478, 1508, 1512, 1520,
      1535, 1573, 1578, 1584, 1586, 1625, 1638, 1642, 1651, 1657,
      1670, 1785, 1804, 1883, 1990, 2018, 2093, 2118, 2212, 2238,
      2270, 2320, 2397, 2700, 2888, 2957, 2992, 3001, 3048, 3104,
      3109, 5380, 5398, 5479,
    ], dtype=np.int32)
    # fmt: on

    VOCAB: dict[str, str] = {
        "iasi": "brightnessTemperature",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get IASI data key and modifier for a variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (``iasi``)

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - Data key for brightness temperature
            - Modifier function (identity -- brightness temperature in K)
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MetOp IASI lexicon")
        return cls.VOCAB[val], lambda x: x
