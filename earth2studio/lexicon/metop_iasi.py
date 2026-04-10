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
