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

"""Regridder protocol: DataArray field mapping between two grid definitions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.grids.base import GridDefinition


# sphinx - regridder protocol start
@runtime_checkable
class Regridder(Protocol):
    """Map fields from a source grid onto a target grid.

    A regridder is bound to one source and one target grid at construction, where
    it precomputes any indices or weights. Calls then only apply them, so one
    instance serves every variable and time fetched for a model input slot.
    """

    @property
    def source_grid(self) -> GridDefinition:
        """Return the grid the regridder accepts."""
        ...

    @property
    def target_grid(self) -> GridDefinition:
        """Return the grid the regridder produces."""
        ...

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Regrid the spatial dimensions of a field.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-backed CPU or CuPy-backed CUDA field whose spatial dimensions
            match ``source_grid``. Any other dimensions pass through.

        Returns
        -------
        xr.DataArray
            Field on ``target_grid`` with the same array backing and device as
            ``x``: non-spatial dimensions in input order, followed by
            ``target_grid.dims``.

        Raises
        ------
        ValueError
            If the spatial dimensions of ``x`` do not match ``source_grid``.
        """
        ...

    def to(self, device: Any) -> Regridder:
        """Move precomputed indices or weights to a device.

        Parameters
        ----------
        device : Any
            Object representing the device, typically ``torch.device`` or str.

        Returns
        -------
        Regridder
            The regridder on the requested device.
        """
        ...


# sphinx - regridder protocol end
