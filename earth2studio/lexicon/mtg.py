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

import numpy as np

from earth2studio.lexicon.base import LexiconType


class MetOpMTGLexicon(metaclass=LexiconType):
    """Lexicon for MTG-I FCI Level-1C Full Disk data source.

    Maps Earth2Studio variable names to MTG FCI channel identifiers.
    The 12 spectral channels cover visible (VIS), near-infrared (NIR),
    water vapour (WV) and infrared (IR) bands.

    Note
    ----
    Channel documentation:

    - https://www.eumetsat.int/mtg-fci-level-1c-full-disk
    - https://data.eumetsat.int/product/EO:EUM:DAT:0662
    """

    VOCAB: dict[str, tuple[str, Callable]] = {
        "mtg_vis_04": ("vis_04", lambda x: np.array(x)),
        "mtg_vis_05": ("vis_05", lambda x: np.array(x)),
        "mtg_vis_08": ("vis_08", lambda x: np.array(x)),
        "mtg_vis_09": ("vis_09", lambda x: np.array(x)),
        "mtg_nir_13": ("nir_13", lambda x: np.array(x)),
        "mtg_nir_16": ("nir_16", lambda x: np.array(x)),
        "mtg_wv_63": ("wv_63", lambda x: np.array(x)),
        "mtg_wv_73": ("wv_73", lambda x: np.array(x)),
        "mtg_ir_87": ("ir_87", lambda x: np.array(x)),
        "mtg_ir_97": ("ir_97", lambda x: np.array(x)),
        "mtg_ir_123": ("ir_123", lambda x: np.array(x)),
        "mtg_ir_133": ("ir_133", lambda x: np.array(x)),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the FCI channel name and modifier for a variable.

        Parameters
        ----------
        val : str
            Variable name (e.g. ``'mtg_vis_04'``)

        Returns
        -------
        tuple[str, Callable]
            FCI channel key and identity modifier
        """
        return cls.VOCAB[val]
