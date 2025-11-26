# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from loguru import logger

from earth2studio.data.base import DataSource
from earth2studio.utils.imports import OptionalDependencyFailure, check_optional_dependencies

try:
    # SciPy is optional; fail gracefully until the class is used
    from scipy.spatial import cKDTree  # type: ignore
except ImportError:
    OptionalDependencyFailure("data")
    cKDTree = None  # type: ignore[assignment]


@check_optional_dependencies("data")
class CurvilinearNNInterp:
    """DataSource wrapper performing nearest-neighbor interpolation to a curvilinear target.

    This wraps a base :py:class:`earth2studio.data.base.DataSource` and maps its last two
    spatial dimensions onto a target curvilinear grid using a precomputed nearest-neighbor
    index map built with :pyclass:`scipy.spatial.cKDTree`.
    The target grid is provided via 2D latitude/longitude arrays along with 1D ``y`` and
    ``x`` core dimension coordinates for the output.

    To improve robustness, NaN values in the source latitude/longitude inputs are
    temporarily filled with dummy values located outside the target domain only for the
    purpose of building the KD-tree. An optional ``max_distance_km`` threshold enables
    filling of target points whose nearest source point lies beyond a given radius.

    Parameters
    ----------
    base : DataSource
        Underlying data source to fetch data from.
    lat_in : ArrayLike
        Source latitude coordinates. Either 2D of shape [H_src, W_src] or 1D of shape [H_src].
    lon_in : ArrayLike
        Source longitude coordinates. Either 2D of shape [H_src, W_src] or 1D of shape [W_src].
    lat_out : ArrayLike
        Target latitude coordinates. 2D array of shape [H_tgt, W_tgt].
    lon_out : ArrayLike
        Target longitude coordinates. 2D array of shape [H_tgt, W_tgt].
    y : ArrayLike
        Output core dimension coordinate for ``y`` of shape [H_tgt].
    x : ArrayLike
        Output core dimension coordinate for ``x`` of shape [W_tgt].
    max_distance_km : float, optional
        If provided, target locations whose nearest neighbor is farther than this geodesic
        distance (kilometers) are filled with ``fill_value``. If None, nearest neighbor
        data continues to be used (possibly extrapolating beyond the source domain). Defaults to None.
    fill_value : float | None, optional
        Value used to fill target locations beyond ``max_distance_km``. If None, NaN fill
        is used, by default None

    Notes
    -----
    - This wrapper assumes the last two dimensions of the base data source output are the
      spatial dimensions and preserves the order of all leading dimensions.
    - The output dims are the base dims with the last two replaced by ``(\"y\", \"x\")``.
    - Curvilinear latitude/longitude 2D arrays are attached to coords as ``\"_lat\"`` and
      ``\"_lon\"`` with dims ``(\"y\", \"x\")`` to align with Earth2Studio conventions.
    """

    # Value used to fill invalid gridpoints in the input data; will not affect interpolation results
    _DUMMY_FILL = -9999.12345 

    def __init__(
        self,
        base: DataSource,
        lat_in: ArrayLike,
        lon_in: ArrayLike,
        lat_out: ArrayLike,
        lon_out: ArrayLike,
        y: ArrayLike,
        x: ArrayLike,
        max_distance_km: float | None = None,
        fill_value: float | None = None,
    ) -> None:
        self._base = base

        # Normalize inputs
        lat_in_arr = np.array(lat_in)
        lon_in_arr = np.array(lon_in)
        lat_out_arr = np.array(lat_out)
        lon_out_arr = np.array(lon_out)
        y_arr = np.array(y)
        x_arr = np.array(x)

        if lat_out_arr.ndim != 2 or lon_out_arr.ndim != 2:
            raise ValueError("lat_out and lon_out must be 2D arrays for curvilinear targets")
        if lat_out_arr.shape != lon_out_arr.shape:
            raise ValueError("lat_out and lon_out must have identical shapes")
        if y_arr.ndim != 1 or x_arr.ndim != 1:
            raise ValueError("y and x must be 1D arrays")
        if (lat_out_arr.shape[0] != y_arr.shape[0]) or (lat_out_arr.shape[1] != x_arr.shape[0]):
            raise ValueError("lat_out/lon_out dimensions must match lengths of y and x")

        # Accept 1D (regular) or 2D (curvilinear) source grids
        if lat_in_arr.ndim == 1 and lon_in_arr.ndim == 1:
            lat_in_arr, lon_in_arr = np.meshgrid(lat_in_arr, lon_in_arr, indexing="ij")
        elif lat_in_arr.ndim != 2 or lon_in_arr.ndim != 2:
            raise ValueError("lat_in and lon_in must both be 1D or both be 2D arrays")
        if lat_in_arr.shape != lon_in_arr.shape:
            raise ValueError("lat_in and lon_in must have identical shapes")

        # Reject if output coords have NaNs
        if np.isnan(lat_out_arr).any() or np.isnan(lon_out_arr).any():
            raise ValueError("NaN values found in CurvilinearNNInterp output coordinates")

        # Standardize coords
        if np.any(lon_in_arr < 0.0):
            logger.warning("Negative longitude values found in CurvilinearNNInterp input coordinates, adjusting to (0, 360) range")
            lon_in_arr = (lon_in_arr + 360.0) % 360.0
        if np.any(lon_out_arr < 0.0):
            logger.warning("Negative longitude values found in CurvilinearNNInterp output coordinates, adjusting to (0, 360) range")
            lon_out_arr = (lon_out_arr + 360.0) % 360.0
        
        # Handle NaNs in input coords which indicate invalid gridpoints:
        # 1) fill coords with dummy values by pushing them outside the target domain
        # 2) cache expected locations of invalid gridpoints for masking in the actual data values
        if np.isnan(lat_in_arr).any() or np.isnan(lon_in_arr).any():
            logger.warning("NaN values found in CurvilinearNNInterp input coordinates, filling with dummy values outside the target domain")
            lat_in_filled = np.where(np.isnan(lat_in_arr), np.nanmax(lat_out_arr) + 10.0, lat_in_arr)
            lon_in_filled = np.where(np.isnan(lon_in_arr), np.nanmin(lon_out_arr) - 10.0, lon_in_arr)
            self._invalid_mask = np.isnan(lat_in_arr) | np.isnan(lon_in_arr)
        else:
            lat_in_filled = lat_in_arr
            lon_in_filled = lon_in_arr
            self._invalid_mask = None

        # Build KDTree mapping from target to nearest source
        src_pts = np.column_stack((lat_in_filled.ravel(), lon_in_filled.ravel()))
        tgt_pts = np.column_stack((lat_out_arr.ravel(), lon_out_arr.ravel()))
        tree = cKDTree(src_pts)  # type: ignore[arg-type]
        nn_dist_deg, nn_idx_flat = tree.query(tgt_pts)  # distances are in degrees space

        # Precompute mapping and optional outside mask using geodesic distance
        self._src_shape = lat_in_arr.shape
        self._tgt_shape = lat_out_arr.shape
        self._idx_flat = nn_idx_flat.astype(np.int64)  # shape [H_tgt*W_tgt]
        self._y = y_arr
        self._x = x_arr
        self._lat_out = lat_out_arr
        self._lon_out = lon_out_arr
        self._fill_value = np.nan if fill_value is None else fill_value
        self._outside_mask: np.ndarray | None = None

        if max_distance_km is not None:
            # Compute true geodesic distances to nearest source and mark outside
            src_lat_nn = lat_in_arr.ravel()[self._idx_flat]
            src_lon_nn = lon_in_arr.ravel()[self._idx_flat]
            d_km = self._haversine_km(
                lat_out_arr.ravel(),
                lon_out_arr.ravel(),
                src_lat_nn,
                src_lon_nn,
            )
            self._outside_mask = (d_km > float(max_distance_km)).reshape(self._tgt_shape)

    def __call__(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Fetch from base source and map to target curvilinear grid.

        Parameters
        ----------
        args : Any
            Positional arguments forwarded to the wrapped data source.
        kwargs : Any
            Keyword arguments forwarded to the wrapped data source.

        Returns
        -------
        xr.DataArray
            Data array with dims equal to the base dims but last two replaced by
            ``(\"y\", \"x\")`` and with curvilinear coords ``\"_lat\"`` and ``\"_lon\"``.
        """
        da_src = self._base(*args, **kwargs)
        if not isinstance(da_src, xr.DataArray):
            raise TypeError("Wrapped data source did not return an xarray.DataArray")
        if da_src.ndim < 2:
            raise ValueError("Expected at least 2 spatial dims in the source data")

        # Enforce last-two dims as spatial; build output dims replacing with y/x
        spatial_in = da_src.shape[-2:]
        if spatial_in != self._src_shape:
            raise ValueError(
                f"Source data spatial shape {spatial_in} does not match provided lat_in/lon_in {self._src_shape}"
            )
        dims_leading = tuple(da_src.dims[:-2])
        dims_out = dims_leading + ("y", "x")

        # Interpolate using precomputed flat indices
        ny, nx = spatial_in
        values = da_src.values
        if self._invalid_mask is not None:
            values = np.where(self._invalid_mask, self._DUMMY_FILL, values)
        if np.isnan(values).any():
            raise ValueError("Unexpected NaN values found in CurvilinearNNInterp input data (after accounting for possible invalid gridpoints)")
        values_flat = values.reshape(*values.shape[:-2], ny * nx)
        mapped = values_flat[..., self._idx_flat].reshape(*values.shape[:-2], *self._tgt_shape)

        # Fill outside-of-radius with requested value if applicable
        if self._outside_mask is not None:
            # Broadcast mask across leading dims
            mask = np.broadcast_to(self._outside_mask, mapped.shape[-2:])
            mapped = np.where(mask, self._fill_value, mapped)

        if np.any(mapped == self._DUMMY_FILL):
            logger.warning("Dummy fill values leaked into CurvilinearNNInterp output data. \
                This can be caused by invalid gridpoints in the input coordinates or by a \
                max_distance_km parameter that is too small. Double-check input/output \
                coordinates and max_distance_km parameter.")

        # Build output coordinates: copy non-spatial, then set y/x and curvilinear lat/lon
        coord_items = {}
        for k, v in da_src.coords.items():
            if k in da_src.dims[-2:] or k in ("_lat", "_lon"):
                continue
            coord_items[k] = v.values
        coord_items["y"] = self._y
        coord_items["x"] = self._x
        coord_items["_lat"] = (("y", "x"), self._lat_out)
        coord_items["_lon"] = (("y", "x"), self._lon_out)

        return xr.DataArray(mapped, coords=coord_items, dims=dims_out)

    @staticmethod
    def _haversine_km(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> np.ndarray:
        """Great-circle distance between two sets of points in kilometers."""
        R = 6371.0  # km
        lat1r = np.radians(lat1)
        lat2r = np.radians(lat2)
        dlat = lat2r - lat1r
        dlon = np.radians(lon2) - np.radians(lon1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c


