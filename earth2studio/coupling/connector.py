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

"""Connector: a declaration of which fields flow from one model to another."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import xarray as xr

from earth2studio.grids import Regridder

Transform = Callable[[xr.DataArray], xr.DataArray]


@dataclass(frozen=True)
class Connector:
    """Declare that a destination model's inputs come from a source model.

    A connector holds no runtime state: the :class:`~earth2studio.coupling.Coupler`
    owns every frame buffer and reduction, so one declaration serves every work
    item and two connectors on the same pair of models are distinct.

    Variables match by Earth2Studio label. A destination label such as
    ``"z1000"`` is served by the source's ``"z1000"`` output; a qualified label
    such as ``"z1000:mean:48h"`` (see ``dev/spec/TIME_STATISTICS_SPEC.md``) is
    served by reducing the source's ``"z1000"`` frames over that window. Labels
    are delivered to a prognostic destination's forcing slot, to a diagnostic's
    input slot, or, when the destination declares no such slot, by overwriting
    the matching variable of a prognostic destination's own state before each
    step.

    Parameters
    ----------
    src : str
        Name of the source model in the coupler's ``models`` mapping.
    dst : str
        Name of the destination model.
    variables : Sequence[str], optional
        Destination labels to deliver, by default every destination import the
        source can serve.
    regridder : Regridder, optional
        Regridder from the source's grid to the destination's, by default one
        resolved from the two grids (bilinear where the source is structured,
        nearest otherwise; see :func:`earth2studio.grids.resolve_regridder`).
        Both grids must match its ``source_grid`` and ``target_grid``.
    transform : Callable[[xr.DataArray], xr.DataArray], optional
        Applied to each delivered field after regridding, for unit-like
        conversions. It must keep the field's dimensions and coordinates.
    """

    src: str
    dst: str
    variables: tuple[str, ...] | None = None
    regridder: Regridder | None = None
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError(f"Connector cannot couple {self.src!r} to itself")
        if self.variables is not None:
            variables: Sequence[str] = self.variables
            if isinstance(variables, str):
                variables = (variables,)
            if not variables:
                raise ValueError(
                    f"Connector {self.name}: variables must name at least one label"
                )
            object.__setattr__(self, "variables", tuple(str(v) for v in variables))
        if self.regridder is not None and not isinstance(self.regridder, Regridder):
            raise TypeError(
                f"Connector {self.name}: regridder must implement "
                "earth2studio.grids.Regridder"
            )
        if self.transform is not None and not callable(self.transform):
            raise TypeError(f"Connector {self.name}: transform must be callable")

    @property
    def name(self) -> str:
        """``src->dst``, the connector's display name."""
        return f"{self.src}->{self.dst}"

    def __repr__(self) -> str:
        variables = "auto" if self.variables is None else list(self.variables)
        extras = ""
        if self.regridder is not None:
            extras += f", regridder={type(self.regridder).__name__}"
        if self.transform is not None:
            extras += ", transform=..."
        return f"Connector({self.name}, variables={variables}{extras})"
