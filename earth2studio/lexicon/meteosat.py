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
    The 16 spectral channels cover visible (VIS), near-infrared (NIR),
    water vapour (WV) and infrared (IR) bands. Variable names follow
    the pattern ``fci{wavelength}{band}``, e.g. ``fci87ir``.

    Note
    ----
    Channel documentation:

    - https://www.eumetsat.int/mtg-fci-level-1c-full-disk
    - https://data.eumetsat.int/product/EO:EUM:DAT:0662
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # VIS bands
        "fci04vis": ("vis_04", lambda x: x),  # VIS 0.444 µm (1 km)
        "fci05vis": ("vis_05", lambda x: x),  # VIS 0.510 µm (1 km)
        "fci06vis": ("vis_06", lambda x: x),  # VIS 0.640 µm (0.5 km / 1 km)
        "fci08vis": ("vis_08", lambda x: x),  # VIS 0.865 µm (1 km)
        "fci09vis": ("vis_09", lambda x: x),  # VIS 0.914 µm (1 km)
        # NIR bands
        "fci13nir": ("nir_13", lambda x: x),  # NIR 1.380 µm (1 km)
        "fci16nir": ("nir_16", lambda x: x),  # NIR 1.610 µm (1 km)
        "fci22nir": ("nir_22", lambda x: x),  # NIR 2.250 µm (0.5 km / 1 km)
        # IR bands
        "fci38ir": ("ir_38", lambda x: x),  # IR 3.800 µm (1 km / 2 km)
        "fci63wv": ("wv_63", lambda x: x),  # WV 6.300 µm (2 km)
        "fci73wv": ("wv_73", lambda x: x),  # WV 7.350 µm (2 km)
        "fci87ir": ("ir_87", lambda x: x),  # IR 8.700 µm (2 km)
        "fci97ir": ("ir_97", lambda x: x),  # IR 9.660 µm (2 km)
        "fci105ir": ("ir_105", lambda x: x),  # IR 10.500 µm (1 km / 2 km)
        "fci123ir": ("ir_123", lambda x: x),  # IR 12.300 µm (2 km)
        "fci133ir": ("ir_133", lambda x: x),  # IR 13.300 µm (2 km)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Return the FCI channel name and modifier for a variable.

        Parameters
        ----------
        val : str
            Variable name (e.g. ``'fci87ir'``)

        Returns
        -------
        tuple[str, Callable]
            FCI channel key and identity modifier
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Meteosat FCI lexicon")
        return cls.VOCAB[val]
