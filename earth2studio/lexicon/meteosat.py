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


class MeteosatFCILexicon(metaclass=LexiconType):
    """Lexicon for MTG-I FCI Level-1C Full Disk data source.

    Maps Earth2Studio variable names to MTG FCI channel identifiers.
    The 12 spectral channels cover visible (VIS), near-infrared (NIR),
    water vapour (WV) and infrared (IR) bands. Channels are numbered
    sequentially by increasing wavelength.

    Note
    ----
    Channel documentation:

    - https://www.eumetsat.int/mtg-fci-level-1c-full-disk
    - https://data.eumetsat.int/product/EO:EUM:DAT:0662
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # VIS bands (1 km resolution)
        "fci01": ("vis_04", lambda x: x),  # VIS 0.44 µm
        "fci02": ("vis_05", lambda x: x),  # VIS 0.51 µm
        "fci03": ("vis_08", lambda x: x),  # VIS 0.86 µm
        "fci04": ("vis_09", lambda x: x),  # VIS 0.91 µm
        # NIR bands
        "fci05": ("nir_13", lambda x: x),  # NIR 1.38 µm (1 km)
        "fci06": ("nir_16", lambda x: x),  # NIR 1.61 µm (2 km)
        # WV bands (2 km resolution)
        "fci07": ("wv_63", lambda x: x),  # WV 6.30 µm
        "fci08": ("wv_73", lambda x: x),  # WV 7.35 µm
        # IR bands (2 km resolution)
        "fci09": ("ir_87", lambda x: x),  # IR 8.70 µm
        "fci10": ("ir_97", lambda x: x),  # IR 9.66 µm
        "fci11": ("ir_123", lambda x: x),  # IR 12.30 µm
        "fci12": ("ir_133", lambda x: x),  # IR 13.30 µm
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Return the FCI channel name and modifier for a variable.

        Parameters
        ----------
        val : str
            Variable name (e.g. ``'fci09'``)

        Returns
        -------
        tuple[str, Callable]
            FCI channel key and identity modifier
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Meteosat FCI lexicon")
        return cls.VOCAB[val]
