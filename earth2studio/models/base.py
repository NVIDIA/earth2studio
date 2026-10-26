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

"""Model interfaces shared by prognostic and diagnostic models."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from earth2studio.grids import Region


@runtime_checkable
class DomainAdaptable(Protocol):
    """Model whose spatial domain can be selected.

    Implemented in addition to ``PrognosticModel`` or ``DiagnosticModel``; generic
    callers detect it with ``isinstance``. A loaded model starts on its default
    domain, usually its full native footprint, and ``with_region`` is the only way
    to change it.

    The returned model is a view: a new instance with its own coordinate signatures
    that shares the loaded weights and statics. Views run safely in sequence, and
    concurrently only where the wrapper documents it.
    """

    def with_region(self, region: Region, /) -> Self:
        """Return a view of this model covering a geographic region.

        The model selects the grid of every input, forcing and output slot itself,
        using its grids' own selection logic, so multi-grid models choose each
        slot's grid consistently. Neither this model nor any other view changes.
        Calling it on a view reaches the full supported footprint.

        Parameters
        ----------
        region : Region
            Geographic area the view's outputs must cover. For ``rectangle``
            regions, the center and orientation must also be honored.

        Returns
        -------
        Self
            View whose coordinate signatures describe the selected grids.

        Raises
        ------
        ValueError
            If the region kind is unsupported or cannot be covered.
        """
        ...
