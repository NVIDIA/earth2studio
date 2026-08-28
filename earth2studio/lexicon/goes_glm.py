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

import warnings
from collections.abc import Callable

from earth2studio.lexicon.base import LexiconType


class GOESGLMLexicon(metaclass=LexiconType):
    """Lexicon for GOES Geostationary Lightning Mapper (GLM) L2 LCFA data.

    Maps Earth2Studio variable names to a ``{level}::{field}`` key, where
    ``level`` selects one of the three tiers of the GLM detection hierarchy
    stored in each L2 Lightning Cluster-Filter Algorithm NetCDF file and
    ``field`` is the native NetCDF variable name within it:

    - ``event`` — individual illuminated pixels in a single 2 ms frame
    - ``group`` — spatially adjacent events within one frame
    - ``flash`` — groups clustered in space and time

    Variable names follow the instrument-agnostic
    ``lightning_{level}_{quantity}`` convention shared with
    :py:class:`MeteosatLILexicon`.

    - ``*_energy`` is the native GLM optical energy of the detection
      (Joules).
    - ``*_count`` is synthetic: the data source fills it with 1.0 per
      record so users can sum or histogram to obtain per-cell event,
      group or flash density during downstream regridding.

    Note
    ----
    Variable reference:

    - GLM Product Definition and User's Guide:
      https://www.goes-r.gov/products/baseline-lightning-detection.html
    """

    VOCAB = {
        # Events
        "lightning_event_energy": "event::event_energy",
        "lightning_event_count": "event::_count",
        # Groups
        "lightning_group_energy": "group::group_energy",
        "lightning_group_count": "group::_count",
        # Flashes
        "lightning_flash_energy": "flash::flash_energy",
        "lightning_flash_count": "flash::_count",
    }

    # Pre-0.19 variable ids, retained so existing pipelines keep working. These
    # resolve to their canonical replacement and emit a FutureWarning.
    # TODO: Drop the deprecated flashe/flashc aliases in the 0.20.0 release
    DEPRECATED_VOCAB = {
        "flashe": "lightning_event_energy",
        "flashc": "lightning_event_count",
    }

    @classmethod
    def resolve_alias(cls, val: str) -> str:
        """Map a deprecated variable id onto its canonical replacement.

        Non-deprecated ids are returned unchanged, so this is safe to call on
        every requested variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable id, possibly a deprecated alias.

        Returns
        -------
        str
            The canonical variable id.

        Warns
        -----
        FutureWarning
            When *val* is a deprecated alias.
        """
        canonical = cls.DEPRECATED_VOCAB.get(val)
        if canonical is None:
            return val
        # FutureWarning rather than DeprecationWarning so end users actually
        # see it: Python hides DeprecationWarning outside __main__ code
        warnings.warn(
            f"GOES GLM variable id {val!r} is deprecated and will be removed "
            f"in a future release; use {canonical!r} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return canonical

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the GLM field name and modifier function for a variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            ``(key, modifier)`` where ``key`` is ``'{level}::{field}'``.
            ``field`` is the native GLM NetCDF variable name for the
            ``*_energy`` variables and the synthetic sentinel ``"_count"``
            for the ``*_count`` variables. ``modifier`` is the identity
            function; values are returned in their physical units (Joules
            for ``*_energy``, dimensionless 1.0 for ``*_count``).
        """
        return cls.VOCAB[cls.resolve_alias(val)], lambda x: x
