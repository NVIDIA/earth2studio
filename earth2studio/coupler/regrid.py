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


"""Coupled runner extension that maps coupling outputs onto the forcing grid."""

from __future__ import annotations

from typing import Any

import xarray as xr
from loguru import logger

from earth2studio.coupler.runner import CoupledRunner
from earth2studio.grids import (
    Regridder,
    infer_grid,
    resolve_regridder,
    same_geometry,
    select_subset,
)
from earth2studio.utils.type import CoordinateSystem


class RegriddingCoupledRunner(CoupledRunner):
    """:class:`CoupledRunner` that accepts coupling outputs on another grid.

    Uses the same utilities as :func:`~earth2studio.data.fetch_data`: a field
    already on the slot grid passes, an exact crop of the source grid is
    selected, and anything else is regridded by nearest neighbor within the
    source resolution, never extrapolating. The regridder is built once per
    model and needs the ``utils`` extra. For another method or a fill, regrid
    inside the coupling.

    Parameters
    ----------
    *args, **kwargs
        As for :class:`CoupledRunner`

    Raises
    ------
    ValueError
        When a coupling output has no determinable grid, or cannot be regridded
        automatically
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._regridders: dict[str, Regridder] = {}

    def _on_slot_grid(
        self, name: str, x: xr.DataArray, signature: CoordinateSystem
    ) -> xr.DataArray:
        """Map ``x`` onto the forcing slot's grid."""
        if self._matches(x, signature):
            return x
        try:
            target, source = infer_grid(signature), infer_grid(x)
        except ValueError as error:
            raise ValueError(
                f"{name!r}: cannot determine the grid of the coupling output or "
                f"the forcing slot ({error}); regrid inside the coupling"
            ) from error
        if same_geometry(source, target):
            return x
        subset = select_subset(x, source, target)
        if subset is not None:
            return subset
        if name not in self._regridders:
            try:
                self._regridders[name] = resolve_regridder(
                    source, target, "nearest", max_dist_km="auto", fill_value=None
                )
            except ValueError as error:
                raise ValueError(
                    f"{name!r}: cannot regrid the coupling output automatically: "
                    f"{error}; regrid inside the coupling to choose a method"
                ) from error
            logger.info(
                "Regridding {!r} coupling output from a {} grid to a {} grid by "
                "nearest neighbor",
                name,
                source.topology,
                target.topology,
            )
        return self._regridders[name](x)
