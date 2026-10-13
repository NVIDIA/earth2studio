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

"""Optional domain-selection protocol shared by prognostic and diagnostic models."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from earth2studio.grids import GridDefinition, Region


@runtime_checkable
class DomainAdaptable(Protocol):
    """Model whose spatial domain can be selected.

    Implemented in addition to ``PrognosticModel`` or ``DiagnosticModel``; generic
    callers detect it with ``isinstance``. The bound grid is the prognostic state
    grid or the diagnostic output grid. Input and forcing slots may use grids derived
    from it, and a view's coordinate signatures are authoritative.

    Views may share parameter storage with the model they are made from. They run
    safely in sequence, and concurrently only where the wrapper documents it.
    """

    def domain_for(self, region: Region, /) -> GridDefinition:
        """Select a supported grid covering a geographic request.

        Performs model-specific selection without fetching fields or creating a
        view. A crop model selects and snaps cells from its supported footprint; a
        globe-wide patch model builds a local grid at its native spacing and shape.

        Parameters
        ----------
        region : Region
            Geographic area to cover. For ``around`` regions, the center and
            orientation must also be honored.

        Returns
        -------
        GridDefinition
            Grid whose usable output area covers ``region``, to pass to
            :meth:`with_domain`.

        Raises
        ------
        ValueError
            If the region kind is unsupported or cannot be covered.
        """
        ...

    def with_domain(self, grid: GridDefinition, /) -> Self:
        """Return a model bound to exactly the given grid.

        Never snaps, resizes or rotates the grid. Neither this model nor any other
        view changes. Calling it on a view reaches the full supported footprint.

        Parameters
        ----------
        grid : GridDefinition
            Declared prognostic state or diagnostic output grid.

        Returns
        -------
        Self
            New model whose coordinate signatures describe ``grid`` and its derived
            input and forcing grids.

        Raises
        ------
        ValueError
            If the model cannot run on exactly ``grid``.
        """
        ...
