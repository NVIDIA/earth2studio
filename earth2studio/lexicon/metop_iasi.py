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
