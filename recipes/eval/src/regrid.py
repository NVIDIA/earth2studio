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

"""Regridding primitives for the eval recipe.

Two concerns are separated here:

* :class:`Regridder` — a tensor-native regridder protocol.  Implementations
  precompute indices/weights as torch buffers and apply them on whatever
  device the buffers live on.  The hot path (inference-time output regrid,
  scoring-time coarsen) uses this protocol so regridding can happen on GPU
  alongside the rest of the model work.
* :class:`RegriddedSource` — a thin CPU/``xr.DataArray`` adapter that wraps
  an Earth2Studio :class:`DataSource` with a :class:`Regridder`.  Used at
  predownload time, where the zarr write-path is CPU-only and the
  ``DataSource`` protocol returns xarray.

The :class:`NearestNeighborRegridder` wraps
:class:`earth2studio.utils.interp.NearestNeighborInterpolator` — a
precomputed KDTree-backed lookup with a haversine distance mask.  The
heavy setup (KDTree build) runs once at construction; the hot path is a
single torch ``gather`` that runs on whatever device the regridder lives
on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from numpy.typing import ArrayLike

from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.interp import NearestNeighborInterpolator
from earth2studio.utils.type import TimeArray, VariableArray


class Regridder(ABC):
    """Tensor-native regridder interface.

    The primary API is :meth:`apply`, which operates on torch tensors and
    preserves the caller's device.  :meth:`apply_dataarray` is provided
    only as a CPU adapter for code paths that must round-trip through
    ``xr.DataArray`` (predownload's zarr write path).

    Subclasses precompute any index/weight buffers in ``__init__`` on CPU
    and expose them via :meth:`to` so that the full regridder can be moved
    to a device in one call.  :meth:`target_coords` returns the spatial
    coordinate system produced by the regridder — callers use it to build
    output zarr schemas without materializing a tensor.
    """

    @abstractmethod
    def to(self, device: str | torch.device) -> "Regridder":
        """Move any internal buffers to *device* and return self."""
        ...

    @abstractmethod
    def target_coords(self) -> CoordSystem:
        """Return the spatial coordinate system produced by this regridder.

        Only spatial dims are included (e.g. ``{"lat": ..., "lon": ...}``
        or ``{"y": ..., "x": ...}`` with optional 2D ``lat``/``lon``
        non-index coordinates carried alongside).
        """
        ...

    @abstractmethod
    def apply(
        self,
        x: torch.Tensor,
        *,
        spatial_dims: tuple[str, ...],
    ) -> torch.Tensor:
        """Regrid *x* along the named trailing spatial dims.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with arbitrary leading dims followed by the
            source spatial dims (in the order given by *spatial_dims*).
        spatial_dims : tuple[str, ...]
            Names of the source spatial dims in tensor-axis order.  Used
            only by implementations that need to know the source layout;
            the dims are always the trailing axes of *x*.

        Returns
        -------
        torch.Tensor
            Regridded tensor with the same leading dims followed by the
            target spatial dims.
        """
        ...

    def apply_with_coords(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply :meth:`apply` and return a tensor + updated coord system.

        Source spatial dims are replaced by the regridder's target dims
        in the returned ``CoordSystem``; leading dims (batch, time,
        lead_time, ensemble, variable) pass through unchanged.
        """
        spatial_dims = tuple(_spatial_dims_of(coords))
        y = self.apply(x, spatial_dims=spatial_dims)

        out_coords: CoordSystem = OrderedDict()
        for dim, vals in coords.items():
            if dim not in spatial_dims:
                out_coords[dim] = vals
        for dim, vals in self.target_coords().items():
            out_coords[dim] = vals
        return y, out_coords

    def apply_dataarray(self, da: xr.DataArray) -> xr.DataArray:
        """CPU/xarray adapter — regrid a DataArray and rename spatial dims.

        Default implementation converts to a torch tensor on CPU, calls
        :meth:`apply`, and rebuilds the DataArray with the target coords.
        Subclasses may override for efficiency (e.g. bilinear on numpy).
        """
        src_spatial = tuple(d for d in da.dims if d not in _STRUCTURAL_DIMS)
        leading = [d for d in da.dims if d not in src_spatial]
        da_t = da.transpose(*leading, *src_spatial)

        x = torch.from_numpy(np.asarray(da_t.values))
        y = self.apply(x, spatial_dims=src_spatial).numpy()

        target = self.target_coords()
        new_coords: dict[str, np.ndarray] = {}
        for d in leading:
            new_coords[d] = np.asarray(da_t.coords[d].values)
        for d, v in target.items():
            new_coords[d] = np.asarray(v)

        return xr.DataArray(
            y,
            dims=(*leading, *target.keys()),
            coords=new_coords,
            name=da.name,
            attrs=dict(da.attrs),
        )


# ---------------------------------------------------------------------------
# Concrete regridders
# ---------------------------------------------------------------------------


class NearestNeighborRegridder(Regridder):
    """Nearest-neighbor regridder backed by earth2studio's interpolator.

    Wraps :class:`earth2studio.utils.interp.NearestNeighborInterpolator`,
    which precomputes a per-target-pixel source index on CPU (via a
    KDTree in 3D unit-vector space) and a haversine-distance mask.
    Target pixels farther than *max_dist_km* from any source pixel
    receive NaN.  The hot path (:meth:`apply`) is a single torch
    ``gather`` and runs on whatever device the regridder has been moved
    to.

    Parameters
    ----------
    source_lats, source_lons : torch.Tensor | ArrayLike
        Source grid coordinates.  2D ``[H_src, W_src]`` or 1D vectors
        (promoted internally via ``meshgrid``).
    target_lats, target_lons : torch.Tensor | ArrayLike
        Target grid coordinates.  Typically 2D for curvilinear grids
        such as HRRR Lambert-conformal.
    target_y, target_x : ArrayLike
        1D index coordinates of the target grid, written to the output
        zarr alongside the regridded data.  For HRRR these are the
        ``HRRR_Y`` / ``HRRR_X`` linear meter coordinates; for
        regular lat/lon grids pass the same arrays as the lat/lon
        vectors.
    max_dist_km : float
        Max great-circle distance (km) to accept a nearest neighbor.
        Targets farther than this are filled with NaN.
    target_dim_names : tuple[str, str]
        Output spatial dimension names.  Default ``("y", "x")`` matches
        StormScope / HRRR; pass ``("lat", "lon")`` for regular grids.
    """

    def __init__(
        self,
        source_lats: torch.Tensor | ArrayLike,
        source_lons: torch.Tensor | ArrayLike,
        target_lats: torch.Tensor | ArrayLike,
        target_lons: torch.Tensor | ArrayLike,
        *,
        target_y: ArrayLike,
        target_x: ArrayLike,
        max_dist_km: float = 12.0,
        target_dim_names: tuple[str, str] = ("y", "x"),
    ) -> None:
        self._interp = NearestNeighborInterpolator(
            source_lats=source_lats,
            source_lons=source_lons,
            target_lats=target_lats,
            target_lons=target_lons,
            max_dist_km=max_dist_km,
        )
        self._target_y = np.asarray(target_y)
        self._target_x = np.asarray(target_x)
        self._target_dim_names = tuple(target_dim_names)
        if len(self._target_dim_names) != 2:
            raise ValueError(
                f"target_dim_names must be a pair, got {target_dim_names}"
            )

    def to(self, device: str | torch.device) -> "NearestNeighborRegridder":
        self._interp = self._interp.to(device)
        return self

    def target_coords(self) -> CoordSystem:
        y_name, x_name = self._target_dim_names
        coords: CoordSystem = OrderedDict()
        coords[y_name] = self._target_y
        coords[x_name] = self._target_x
        return coords

    def apply(
        self,
        x: torch.Tensor,
        *,
        spatial_dims: tuple[str, ...],
    ) -> torch.Tensor:
        if len(spatial_dims) != 2:
            raise ValueError(
                "NearestNeighborRegridder expects exactly two trailing "
                f"spatial dims, got {spatial_dims}"
            )
        return self._interp(x)


# ---------------------------------------------------------------------------
# DataSource adapter
# ---------------------------------------------------------------------------


class RegriddedSource:
    """Wrap a :class:`DataSource` so its output is regridded to a target grid.

    This is the CPU/xarray-side adapter used at predownload time — the
    output of :meth:`__call__` is still an ``xr.DataArray`` (as required by
    the ``DataSource`` protocol), but its spatial coords are the
    regridder's target coords.  GPU-side regridding (inference output
    regrid, scoring coarsening) skips this wrapper and calls
    :meth:`Regridder.apply` directly.

    Parameters
    ----------
    source : DataSource
        Source to wrap.  Called with the original ``(time, variable)``
        arguments; its spatial output is then regridded.
    regridder : Regridder
        Regridder to apply.  Its :meth:`~Regridder.target_coords` defines
        the output spatial dims.
    """

    def __init__(self, source, regridder: Regridder) -> None:
        self._source = source
        self._regridder = regridder

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        da = self._source(time, variable)
        return self._regridder.apply_dataarray(da)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STRUCTURAL_DIMS = frozenset(
    {"batch", "time", "lead_time", "variable", "ensemble"}
)


def _spatial_dims_of(coords: CoordSystem) -> list[str]:
    """Return coord dims that are not structural (i.e. candidate spatial dims)."""
    return [d for d in coords if d not in _STRUCTURAL_DIMS]
