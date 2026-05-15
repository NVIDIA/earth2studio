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

from earth2studio.lexicon.base import LexiconType


class GOESGLMLexicon(metaclass=LexiconType):
    """Lexicon for GOES Geostationary Lightning Mapper (GLM) L2 LCFA events.

    Each Earth2Studio variable maps to a column extracted (or synthesised)
    from the per-event arrays of the GLM L2 Lightning Cluster-Filter
    Algorithm NetCDF files.

    - ``event_energy`` is the native GLM variable for the optical energy
      of a single detected event (Joules).
    - ``event_count`` is synthetic: the data source fills it with 1.0 per
      event so users can sum or histogram to obtain per-cell flash
      density during downstream regridding.

    Note
    ----
    Variable reference:

    - GLM Product Definition and User's Guide:
      https://www.goes-r.gov/products/baseline-LCFA.html
    """

    VOCAB = {
        "flashe": "event_energy",
        "flashc": "event_count",
    }

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
            ``(field_name, modifier)``. ``field_name`` is the native GLM
            NetCDF variable name (for ``flashe``) or the synthetic key
            ``"event_count"`` (for ``flashc``). ``modifier`` is the
            identity function; values are returned in their physical
            units (Joules for ``flashe``, dimensionless 1.0 for ``flashc``).
        """
        return cls.VOCAB[val], lambda x: x
