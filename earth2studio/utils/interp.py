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

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from scipy.interpolate import LinearNDInterpolator
except ImportError:
    OptionalDependencyFailure("utils", "linear")
    LinearNDInterpolator = None

try:
    from earth2grid.spatial import ang2vec, haversine_distance
    from scipy.spatial import KDTree
except ImportError:
    OptionalDependencyFailure("utils")
    ang2vec = None
    haversine_distance = None
    KDTree = None


def latlon_interpolation_regular(
    values: torch.Tensor,
    lat0: torch.Tensor,
    lon0: torch.Tensor,
    lat1: torch.Tensor,
    lon1: torch.Tensor,
) -> torch.Tensor:
    """Specialized form of bilinear interpolation intended for optimal use on GPU.

    In particular, the mapped values must be defined on a regular rectangular grid,
    (lat0, lon0). Both lat0 and lon0 are vectors with equal spacing.

    lat1, lon1 are assumed to be 2-dimensional meshgrids with possibly unequal spacing.

    Parameters
    ----------
    values : torch.Tensor [..., H_in, W_in]
        Input values defined over (lat0, lon0) that will be interpolated onto
        (lat1, lon1) grid.
    lat0 : torch.Tensor [H_in, ]
        Vector of input latitude coordinates, assumed to be increasing with
        equal spacing.
    lon0 : torch.Tensor [W_in, ]
        Vector of input longitude coordinates, assumed to be increasing with
        equal spacing.
    lat1 : torch.Tensor [H_out, W_out]
        Tensor of output latitude coordinates
    lon1 : torch.Tensor [H_out, W_out]
        Tensor of output longitude coordinates

    Returns
    -------
    result : torch.Tensor [..., H_out, W_out]
        Tensor of interpolated values onto lat1, lon1 grid.
    """

    # Get input grid shape and flatten
    latshape, lonshape = lat1.shape
    lat1 = lat1.flatten()
    lon1 = lon1.flatten()

    # Get indices of nearest points
    latinds = torch.searchsorted(lat0, lat1) - 1
    loninds = torch.searchsorted(lon0, lon1) - 1

    # Get original grid spacing
    dlat = lat0[1] - lat0[0]
    dlon = lon0[1] - lon0[0]

    # Get unit distances
    normed_lat_distance = (lat1 - lat0[latinds]) / dlat
    normed_lon_distance = (lon1 - lon0[loninds]) / dlon

    # Apply bilinear mapping
    result = (
        values[..., latinds, loninds]
        * (1 - normed_lat_distance)
        * (1 - normed_lon_distance)
    )
    result += (
        values[..., latinds, loninds + 1]
        * (1 - normed_lat_distance)
        * (normed_lon_distance)
    )
    result += (
        values[..., latinds + 1, loninds]
        * (normed_lat_distance)
        * (1 - normed_lon_distance)
    )
    result += (
        values[..., latinds + 1, loninds + 1]
        * (normed_lat_distance)
        * (normed_lon_distance)
    )
    return result.reshape(*values.shape[:-2], latshape, lonshape)


@check_optional_dependencies("linear")
class LatLonInterpolation(nn.Module):
    """Bilinear interpolation between arbitrary grids.

    The mapped values can be on arbitrary grid, but the output grid should be
    contained within the input grid.

    Initializing the interpolation object can be somewhat slow, but interpolation is
    fast and can run on the GPU once initialized. Therefore, prefer to reuse the
    interpolation object when possible.

    To run the interpolation on the GPU, use the .to() method of the interpolator
    to move it to the GPU before running the interpolation.

    Parameters
    ----------
    lat_in : torch.Tensor | ArrayLike
        Tensor [H_in, W_in] of input latitude coordinates
    lon_in : torch.Tensor | ArrayLike
        Tensor [H_in, W_in] of input longitude coordinates
    lat_out : torch.Tensor | ArrayLike
        Tensor [H_out, W_out] of output latitude coordinates
    lon_out : torch.Tensor | ArrayLike
        Tensor [H_out, W_out] of output longitude coordinates
    """

    def __init__(
        self,
        lat_in: torch.Tensor | ArrayLike,
        lon_in: torch.Tensor | ArrayLike,
        lat_out: torch.Tensor | ArrayLike,
        lon_out: torch.Tensor | ArrayLike,
    ):
        super().__init__()

        lat_in = (
            lat_in.cpu().numpy() if isinstance(lat_in, Tensor) else np.array(lat_in)
        )
        lon_in = (
            lon_in.cpu().numpy() if isinstance(lon_in, Tensor) else np.array(lon_in)
        )
        lat_out = (
            lat_out.cpu().numpy() if isinstance(lat_out, Tensor) else np.array(lat_out)
        )
        lon_out = (
            lon_out.cpu().numpy() if isinstance(lon_out, Tensor) else np.array(lon_out)
        )

        (i_in, j_in) = np.mgrid[: lat_in.shape[0], : lat_in.shape[1]]

        in_points = np.stack((lat_in.ravel(), lon_in.ravel()), axis=-1)
        i_interp = LinearNDInterpolator(in_points, i_in.ravel())
        j_interp = LinearNDInterpolator(in_points, j_in.ravel())

        out_points = np.stack((lat_out.ravel(), lon_out.ravel()), axis=-1)
        i_map = i_interp(out_points).reshape(lat_out.shape)
        j_map = j_interp(out_points).reshape(lat_out.shape)

        i_map = torch.Tensor(i_map)
        j_map = torch.Tensor(j_map)

        self.register_buffer("i_map", i_map)
        self.register_buffer("j_map", j_map)

    @torch.inference_mode()
    def forward(self, values: Tensor) -> Tensor:
        """Perform bilinear interpolation for values.

        Parameters
        ----------
        values : torch.Tensor
            Input values of shape [..., H_in, W_in] defined over (lat_in, lon_in)
            that will be interpolated onto (lat_out, lon_out) grid.

        Returns
        -------
        result : torch.Tensor
            Tensor of shape [..., H_out, W_out] of interpolated values on lat1, lon1 grid.
        """
        i = self.i_map
        i0 = i.floor().to(dtype=torch.int64).clamp(min=0, max=values.shape[-2] - 2)
        i1 = i0 + 1
        j = self.j_map
        j0 = j.floor().to(dtype=torch.int64).clamp(min=0, max=values.shape[-1] - 2)
        j1 = j0 + 1

        f00 = values[..., i0, j0]
        f01 = values[..., i0, j1]
        f10 = values[..., i1, j0]
        f11 = values[..., i1, j1]

        dj = j - j0
        f0 = torch.lerp(f00, f01, dj)
        f1 = torch.lerp(f10, f11, dj)
        return torch.lerp(f0, f1, i - i0)


@check_optional_dependencies()
class NearestNeighborInterpolator(nn.Module):
    """Nearest-neighbor interpolation between arbitrary lat/lon grids.

    This module precomputes the nearest source-grid index for every target-grid
    location (with an optional maximum great-circle distance threshold) and stores
    the mapping as device-movable buffers. Interpolation then becomes a fast tensor
    gather that can run on GPU via ``.to(device)``.

    Parameters
    ----------
    source_lats : torch.Tensor | ArrayLike
        Latitude of the source grid. Either 2D meshgrid [H_src, W_src] or 1D vector [H_src, ].
    source_lons : torch.Tensor | ArrayLike
        Longitude of the source grid. Either 2D meshgrid [H_src, W_src] or 1D vector [W_src, ].
        Longitudes may be in [-180, 180] or [0, 360] range.
    target_lats : torch.Tensor | ArrayLike
        Latitude of the target grid. Either 2D meshgrid [H_tgt, W_tgt] or 1D vector [H_tgt, ].
    target_lons : torch.Tensor | ArrayLike
        Longitude of the target grid. Either 2D meshgrid [H_tgt, W_tgt] or 1D vector [W_tgt, ].
    max_dist_km : float, optional
        Maximum great-circle distance (km) to accept a nearest neighbor match.
        Target points farther than this threshold are marked invalid and will
        receive NaNs in the output. By default 6.0.

    Notes
    -----
    - Mapping is computed on CPU using a KDTree and stored as buffers
      ``source_flat_index`` (shape [H_tgt*W_tgt], dtype int64) and
      ``valid_mask`` (shape [H_tgt*W_tgt], dtype bool).
    - Forward expects values with shape [..., H_src, W_src] and returns
      [..., H_tgt, W_tgt].
    """

    def __init__(
        self,
        source_lats: torch.Tensor | ArrayLike,
        source_lons: torch.Tensor | ArrayLike,
        target_lats: torch.Tensor | ArrayLike,
        target_lons: torch.Tensor | ArrayLike,
        max_dist_km: float = 6.0,
    ):
        super().__init__()

        # Convert to torch tensors on CPU for processing
        src_lat_t = (
            (
                source_lats
                if isinstance(source_lats, Tensor)
                else torch.tensor(source_lats)
            )
            .detach()
            .cpu()
        )
        src_lon_t = (
            (
                source_lons
                if isinstance(source_lons, Tensor)
                else torch.tensor(source_lons)
            )
            .detach()
            .cpu()
        )
        tgt_lat_t = (
            (
                target_lats
                if isinstance(target_lats, Tensor)
                else torch.tensor(target_lats)
            )
            .detach()
            .cpu()
        )
        tgt_lon_t = (
            (
                target_lons
                if isinstance(target_lons, Tensor)
                else torch.tensor(target_lons)
            )
            .detach()
            .cpu()
        )

        # Normalize to 2D grids if 1D vectors were provided
        if src_lat_t.ndim == 1 and src_lon_t.ndim == 1:
            src_lat_t, src_lon_t = torch.meshgrid(
                src_lat_t.to(torch.float32), src_lon_t.to(torch.float32), indexing="ij"
            )
        elif not (src_lat_t.ndim == 2 and src_lon_t.ndim == 2):
            raise ValueError(
                "source_lats and source_lons must both be 2D or both 1D vectors."
            )
        if tgt_lat_t.ndim == 1 and tgt_lon_t.ndim == 1:
            tgt_lat_t, tgt_lon_t = torch.meshgrid(
                tgt_lat_t.to(torch.float32), tgt_lon_t.to(torch.float32), indexing="ij"
            )
        elif not (tgt_lat_t.ndim == 2 and tgt_lon_t.ndim == 2):
            raise ValueError(
                "target_lats and target_lons must both be 2D or both 1D vectors."
            )

        # Flatten source/target and mask invalid source coordinates
        src_lat_flat = torch.deg2rad(src_lat_t.reshape(-1))
        src_lon_flat = torch.deg2rad(src_lon_t.reshape(-1))
        nan_mask = torch.isnan(src_lat_flat) | torch.isnan(src_lon_flat)
        valid_src_lat = src_lat_flat[~nan_mask]
        valid_src_lon = src_lon_flat[~nan_mask]

        tgt_lat_flat = torch.deg2rad(tgt_lat_t.reshape(-1))
        tgt_lon_flat = torch.deg2rad(tgt_lon_t.reshape(-1))

        # Build KDTree in 3D unit-vector space for numerical stability
        src_vec = torch.stack(ang2vec(valid_src_lon, valid_src_lat), dim=-1).numpy()  # type: ignore[arg-type]
        tree = KDTree(src_vec)  # type: ignore[misc]
        tgt_vec = torch.stack(ang2vec(tgt_lon_flat, tgt_lat_flat), dim=-1).numpy()  # type: ignore[arg-type]
        _, nn_idx_valid_subset = tree.query(tgt_vec, k=1)  # indices into valid subset

        # Map back to indices into the full flattened source grid
        valid_src_indices = (~nan_mask).nonzero(as_tuple=False).reshape(-1)
        src_flat_index_full = valid_src_indices[nn_idx_valid_subset]  # shape [N_tgt]

        # Distance-based validity
        ang_dist = haversine_distance(  # type: ignore[misc]
            tgt_lon_flat,
            tgt_lat_flat,
            valid_src_lon[nn_idx_valid_subset],
            valid_src_lat[nn_idx_valid_subset],
        )
        dist_km = ang_dist * 6371.0
        valid_mask = (dist_km < max_dist_km).to(dtype=torch.bool)

        # Register buffers (moved with .to(device))
        self.register_buffer(
            "source_flat_index", src_flat_index_full.to(dtype=torch.int64)
        )
        self.register_buffer("valid_mask", valid_mask)

        # Save target shape for reshape on output
        self.target_h = int(tgt_lat_t.shape[0])
        self.target_w = int(tgt_lat_t.shape[1])
        self.source_h = int(src_lat_t.shape[0])
        self.source_w = int(src_lat_t.shape[1])

    @torch.inference_mode()
    def forward(self, values: Tensor) -> Tensor:
        """Apply nearest-neighbor interpolation.

        Parameters
        ----------
        values : torch.Tensor
            Input values with shape [..., H_src, W_src] defined on the source grid.

        Returns
        -------
        torch.Tensor
            Interpolated values with shape [..., H_tgt, W_tgt]. Any targets without
            a valid neighbor within the distance threshold are filled with NaN.
        """
        if values.shape[-2] != self.source_h or values.shape[-1] != self.source_w:
            raise ValueError(
                f"Input spatial shape [..., {values.shape[-2]}, {values.shape[-1]}] does not "
                f"match source grid [{self.source_h}, {self.source_w}]."
            )

        # Flatten spatial dims of input and gather with precomputed indices
        leading_shape = values.shape[:-2]
        values_flat = values.reshape(*leading_shape, -1)  # [..., H_src*W_src]

        # Expand indices to match leading dims for gather
        index = self.source_flat_index
        index = index.to(device=values.device)
        index_expanded = index.view(*([1] * len(leading_shape)), -1).expand(
            *leading_shape, index.numel()
        )

        gathered = torch.gather(values_flat, dim=-1, index=index_expanded)

        # Mask out invalid targets with NaN
        valid = self.valid_mask.to(device=values.device)
        valid_expanded = valid.view(*([1] * len(leading_shape)), -1).expand_as(gathered)
        gathered = torch.where(
            valid_expanded, gathered, torch.full_like(gathered, float("nan"))
        )

        return gathered.reshape(*leading_shape, self.target_h, self.target_w)
