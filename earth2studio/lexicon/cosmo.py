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

from .base import LexiconType


class CosmoLexicon(metaclass=LexiconType):
    """COSMO-REA Lexicon

    Maps between Earth2Studio vocabulary and the COSMO-REA output variable names
    used by :class:`~earth2studio.models.dx.CosmoDownscaling` (COSMO-REA6 and
    COSMO-REA2). The vocabulary is the union over both resolutions: where a field
    has a canonical Earth2Studio name and matching units it maps to that name
    (e.g. ``t2m``, ``sp``); COSMO-specific fields with no canonical equivalent
    (model-level winds/temperatures/humidity, turbulent fluxes, TKE, PBL height,
    ...) keep a descriptive lowercase name.

    Each entry's value is ``(cosmo_name(s), scale)``: ``cosmo_name`` is the
    interior COSMO output name (a tuple when the two resolutions spell the same
    field differently, e.g. ``U_10M`` in REA6 vs ``10U`` in REA2), and ``scale``
    converts the COSMO value to the Earth2Studio name's units (e.g. ``CLCT`` is a
    percentage and ``tcc`` is a 0-1 fraction, so ``scale = 0.01``).
    """

    @staticmethod
    def build_vocab() -> dict[str, tuple[tuple[str, ...] | str, float]]:
        """Create the COSMO-REA vocab dictionary."""
        vocab: dict[str, tuple[tuple[str, ...] | str, float]] = {
            # Earth2Studio canonical name : (COSMO output name(s), unit scale)
            "u10m": (("U_10M", "10U"), 1.0),  # REA6 / REA2 spellings
            "v10m": (("V_10M", "10V"), 1.0),
            "t2m": (("T_2M", "2MT"), 1.0),
            "d2m": ("TD_2M", 1.0),  # 2 m dewpoint (K) -- same units
            "sp": ("PS", 1.0),  # surface pressure (Pa)
            "tcc": ("CLCT", 0.01),  # cloud cover: COSMO % -> E2S 0-1 fraction
            # COSMO-specific fields (no canonical Earth2Studio name): the name is
            # the lowercased COSMO output name, units unchanged.
            "tot_precip": ("TOT_PRECIP", 1.0),
            "aswdifd_s": ("ASWDIFD_S", 1.0),
            "aswdir_s": ("ASWDIR_S", 1.0),
            "alwu_s": ("ALWU_S", 1.0),
            "athd_s": ("ATHD_S", 1.0),
            "qv_2m": ("QV_2M", 1.0),
            "vmax_10m": ("VMAX_10M", 1.0),
            "lhfl_s": ("LHFL_S", 1.0),
            "shfl_s": ("SHFL_S", 1.0),
            "h_pbl": ("H_PBL", 1.0),
        }
        # Model-level fields. REA6 ships levels 35-40 as ``<P>_L<lvl>``; REA2 ships
        # levels 45-50 as ``<P>3D_L<lvl>``. Distinct names, no overlap.
        for lvl in range(35, 41):
            for p in ("U", "V", "T", "TKE", "Q"):
                vocab[f"{p.lower()}_l{lvl}"] = (f"{p}_L{lvl}", 1.0)
        for lvl in range(45, 51):
            for p in ("U3D", "V3D"):
                vocab[f"{p.lower()}_l{lvl}"] = (f"{p}_L{lvl}", 1.0)
        return vocab

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[tuple[str, ...] | str, Callable]:
        """Return the COSMO name(s) and unit modifier for an Earth2Studio name.

        Parameters
        ----------
        val : str
            Name in Earth2Studio terminology.

        Returns
        -------
        tuple[str | tuple[str, ...], Callable]
            COSMO output name(s) and a modifier converting the COSMO value to the
            Earth2Studio name's units.
        """
        cosmo_name, scale = cls.VOCAB[val]

        def mod(x: np.ndarray) -> np.ndarray:
            """Convert COSMO units to the Earth2Studio name's units."""
            return x * scale

        return cosmo_name, mod

    @classmethod
    def to_e2studio(cls, cosmo_name: str) -> tuple[str, float]:
        """Map a COSMO output variable name to ``(earth2studio_name, scale)``.

        The reverse of :attr:`VOCAB`, accepting either resolution's spelling.
        Unknown names fall back to their lowercased form with unit scale 1.0, so a
        new COSMO field still gets a sensible name rather than raising.
        """
        reverse = getattr(cls, "_REVERSE", None)
        if reverse is None:
            reverse = {}
            for e2s_name, (cosmo, scale) in cls.VOCAB.items():
                names = cosmo if isinstance(cosmo, tuple) else (cosmo,)
                for name in names:
                    reverse[name] = (e2s_name, scale)
            cls._REVERSE = reverse
        return reverse.get(cosmo_name, (cosmo_name.lower(), 1.0))
