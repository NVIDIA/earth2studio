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


class MTGLexicon(metaclass=LexiconType):
    """MTG FCI lexicon for mapping Earth2Studio variable names to MTG channel names.

    Maps standardised Earth2Studio variable IDs (e.g. ``"mtg_vis_04"``) to the
    corresponding EUMETSAT MTG-I FCI Level-1C NetCDF group path names
    (e.g. ``"vis_04"``), plus an optional data modifier.

    The 12 channels supported correspond to the non-HR Full Disk channels stored
    on the uniform 2 km (5568 × 5568) grid.

    Parameters
    ----------
    val : str
        Standardised Earth2Studio variable name (e.g. ``"mtg_ir_87"``).
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        "mtg_vis_04": ("vis_04", lambda x: x),   # Visible 0.4 µm
        "mtg_vis_05": ("vis_05", lambda x: x),   # Visible 0.5 µm
        "mtg_vis_08": ("vis_08", lambda x: x),   # Visible 0.8 µm
        "mtg_vis_09": ("vis_09", lambda x: x),   # Visible 0.9 µm
        "mtg_nir_13": ("nir_13", lambda x: x),   # Near-IR 1.3 µm
        "mtg_nir_16": ("nir_16", lambda x: x),   # Near-IR 1.6 µm
        "mtg_wv_63":  ("wv_63",  lambda x: x),   # Water Vapour 6.3 µm
        "mtg_wv_73":  ("wv_73",  lambda x: x),   # Water Vapour 7.3 µm
        "mtg_ir_87":  ("ir_87",  lambda x: x),   # IR 8.7 µm
        "mtg_ir_97":  ("ir_97",  lambda x: x),   # IR 9.7 µm (Ozone)
        "mtg_ir_123": ("ir_123", lambda x: x),   # IR 12.3 µm
        "mtg_ir_133": ("ir_133", lambda x: x),   # IR 13.3 µm (CO2)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get MTG channel name and modifier for an Earth2Studio variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable name.

        Returns
        -------
        tuple[str, Callable]
            Tuple of (MTG channel name, modifier function).
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable '{val}' not found in MTG lexicon")
        return cls.VOCAB[val]
