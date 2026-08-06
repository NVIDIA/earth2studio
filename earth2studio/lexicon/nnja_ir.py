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
"""Lexicon for NNJA IR sounder satellite observations.

Variables map to the sensor name used in the NNJA archive. Channel selection
(which subset of the instrument's full channel axis to read) is controlled by
the ``ir_channels`` constructor parameter of ``NNJAObsIRSat``, not by the
lexicon.

All sensors are returned in brightness temperature (K) regardless of how the
archive stores them:
- AIRS:  directly encoded as BT in the aggregate BUFR (``TMBR``).
- IASI:  scaled integer radiance (``SCRA``); converted via Planck inversion.
- CrIS:  float radiance (``SRAD``, W m⁻² sr⁻¹ cm⁻¹); converted via Planck.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .base import LexiconType


class NNJAObsIRSatLexicon(metaclass=LexiconType):
    """NNJA IR sounder satellite observation lexicon.

    Maps Earth2Studio variable names to the NNJA archive sensor identifier.
    The observation column in the returned DataFrame is always brightness
    temperature in Kelvin.

    Note
    ----
    IR channel selection is a constructor concern, not a lexicon concern. Pass
    ``ir_channels="ir32"`` (or a custom channel dict) to ``NNJAObsIRSat`` to
    narrow the channel axis. The lexicon variable name selects the *sensor*,
    not individual channels.

    Archive documentation:

    - https://registry.opendata.aws/noaa-reanalyses-pds/
    - https://psl.noaa.gov/data/nnja_obs/
    """

    VOCAB: dict[str, str] = {
        "airs": "airs",
        "iasi": "iasi",
        "cris": "cris",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]:
        """Return the sensor route key and an identity modifier.

        Observations are already in Kelvin after BUFR decode + Planck inversion.
        """
        if val not in cls.VOCAB:
            raise KeyError(
                f"NNJAObsIRSatLexicon: unknown variable {val!r}. "
                f"Valid: {sorted(cls.VOCAB)}"
            )

        def modifier(frame: pd.DataFrame) -> pd.DataFrame:
            return frame

        return cls.VOCAB[val], modifier
