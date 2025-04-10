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

from collections import OrderedDict
from typing import Any

try:
    import cupy as cp
    from cucim.skimage.feature import peak_local_max as cucim_peak_local_max
    from cucim.skimage.measure import label, regionprops
    from cucim.skimage.morphology import binary_erosion, remove_small_objects
    from skimage.feature import peak_local_max as skimage_peak_local_max
    from skimage.morphology import convex_hull_image
except ImportError:
    cp = None
    cucim_peak_local_max = None
    label = None
    regionprops = None
    binary_erosion = None
    remove_small_objects = None
    skimage_peak_local_max = None
    convex_hull_image = None

import numpy as np
import torch
from pandas import DataFrame

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import CoordSystem

VARIABLES_TC = [
    "u850",
    "v850",
    "msl",
    "z500",
    "z850",
    "z200",
    "t500",
    "t400",
    "t300",
    "t250",
    "t200",
]
VARIABLES_TCV = ["u850", "v850", "u10m", "v10m", "msl"]
OUT_VARIABLES = ["tc_lat", "tc_lon", "tc_msl", "tc_w10m"]


class _CycloneTrackingBase:

    @classmethod
    def vorticity(
        cls, u: torch.Tensor, v: torch.Tensor, dx: float = 25000.0, dy: float = 25000.0
    ) -> torch.Tensor:
        """Compute Relative Vorticity."""
        dudx = torch.gradient(u, dim=-2)[0] / dx
        dvdy = torch.gradient(v, dim=-1)[0] / dy
        vorticity = torch.add(dvdy, dudx)
        return vorticity

    @classmethod
    def haversine_torch(
        cls,
        lat1: torch.Tensor,
        lon1: torch.Tensor,
        lat2: torch.Tensor,
        lon2: torch.Tensor,
        meters: bool = True,
    ) -> torch.Tensor:
        """Compute haversine distance between two pairs of points."""
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        unit = 6_371_000 if meters else 6371
        return (
            2
            * unit
            * torch.arcsin(
                torch.sqrt(
                    torch.sin(dlat / 2) ** 2
                    + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
                )
            )
        )

    @classmethod
    def get_local_max(
        cls,
        x: torch.Tensor,
        threshold_abs: float | None = None,
        min_distance: int = 1,
        exclude_border: bool | int = True,
    ) -> torch.Tensor:
        """Gets the local maximum of a tensor x, above a given absolute threshold,
        with a minimum distance separating local maximums.

        This is a helper utility that converts a pytorch tensor to a cupy tensor
        to use a CuCIM utility `peak_local_max` to extract the local maxima.

        Note
        ----
        For more details, see:

        - https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [x, y]
        threshold_abs : _type_, optional
            Absolute value to threshold local maximum, by default None
        min_distance : int, optional
            Minimum distance separating local maximums, by default 1
        exclude_border: bool or int
            If positive integer, exclude_border excludes peaks from within
            exclude_border-pixels of the border of the image. If tuple of
            non-negative ints, the length of the tuple must match the input
            array dimensionality. Each element of the tuple will exclude peaks
            from within exclude_border-pixels of the border of the image along
            that dimension. If True, takes the min_distance parameter as value.
            If zero or False, peaks are identified regardless of their distance
            from the border.

        Returns
        -------
        torch.Tensor
            List of coordinates of local maximum [2, N]
        """
        if x.is_cuda:
            x_ = cp.from_dlpack(x)
            local_max = cucim_peak_local_max(
                x_,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                exclude_border=exclude_border,
            )
            local_max = torch.from_dlpack(local_max)
        else:
            x_ = np.from_dlpack(x)
            local_max = skimage_peak_local_max(
                x_,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                exclude_border=exclude_border,
            )
            local_max = torch.as_tensor(local_max, device=x.device)

        return local_max


@check_extra_imports("cyclone", ["cupy", "cucim", "skimage"])
class CycloneTrackingVorticity(torch.nn.Module, _CycloneTrackingBase):
    """Finds a list of tropical cyclone centers using an adaption of the method
    described in the conditions in Wu and Duan 2023. The algorithm converts vorticity
    from reanalysis data into a binary image using a defined critical threshold.
    Subsequent processing with connected component labeling and erosion identifies the
    resulting inner cores as TC seeds.

    Note
    ----
    For more information about this method see:

    - https://doi.org/10.1016/j.wace.2023.100626
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        pass

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES_TCV),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords.pop("lat")
        output_coords.pop("lon")
        output_coords["variable"] = np.array(OUT_VARIABLES)
        output_coords["point"] = np.arange(0)

        return output_coords

    def _find_centers(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        vort850: torch.Tensor,
        w10m: torch.tensor,
        msl: torch.tensor,
        vort850_threshold: torch.Tensor = torch.tensor(1.4e-4),
    ) -> torch.Tensor:
        """Finds a list of tropical cyclone centers

        Parameters
        ----------
        lat : torch.Tensor
            Vector of latitudes for tensors.
        lon : torch.Tensor
            Vector of longitudes for tensors.
        vort850 : torch.Tensor
        850 hPa relative vorticity of dimension [lat, lon].

        Returns
        -------
        centers
            List of TC centers, torch.Tensor of shape [2, N]
        """

        if vort850.is_cuda:
            v_ = cp.from_dlpack(vort850)
        else:
            v_ = cp.array(np.from_dlpack(vort850))
        if w10m.is_cuda:
            w10m_ = cp.from_dlpack(w10m)
        else:
            w10m_ = cp.array(np.from_dlpack(w10m))
        if msl.is_cuda:
            msl_ = cp.from_dlpack(msl)
        else:
            msl_ = cp.array(np.from_dlpack(msl))
        vort850_threshold = float(vort850_threshold)
        x_ = v_ > vort850_threshold

        x_label = label(x_)
        x_label = remove_small_objects(x_label, min_size=18, connectivity=1)

        props = regionprops(x_label, intensity_image=v_)
        centers = []
        for prop in props:
            if prop.axis_minor_length == 0:
                # This detected object is too slim to be a tropical storm
                continue

            object_ratio = prop.axis_major_length / prop.axis_minor_length
            if object_ratio > 2:
                # This object is not round enough
                continue

            # extend prop slice to get on each side of the object an additional line of pixels
            start = max(0, prop.slice[0].start - 1)
            stop = min(x_label.shape[0], prop.slice[0].stop + 1)
            x_slice_ext = slice(start, stop, prop.slice[0].step)
            start = max(0, prop.slice[1].start - 1)
            stop = min(x_label.shape[1], prop.slice[1].stop + 1)
            y_slice_ext = slice(start, stop, prop.slice[1].step)
            prop_slice_ext = (x_slice_ext, y_slice_ext)

            # calculate exact center of storm in longitude and latitude coordinates
            # weighted centroid position of center pixels are used (weighted by vorticity)
            lat_idx, lon_idx = prop.centroid_weighted
            grid_spacing = lon[int(cp.ceil(lon_idx))] - lon[int(cp.floor(lon_idx))]
            residual_step = float(lon_idx - cp.floor(lon_idx))
            lon_ = lon[int(cp.floor(lon_idx))] + residual_step * grid_spacing
            grid_spacing = lat[int(cp.ceil(lat_idx))] - lat[int(cp.floor(lat_idx))]
            residual_step = float(lat_idx - cp.floor(lat_idx))
            lat_ = lat[int(cp.floor(lat_idx))] + residual_step * grid_spacing

            # erode object to get center of storm
            bool_center = binary_erosion(x_label[prop_slice_ext]).astype(bool)

            # skip object if the center is smaller or equal 1 pixel
            if bool_center.sum() <= 1:
                continue

            # check solidity of storm center (eroded object) and storm (uneroded object)
            solidity_center = (
                bool_center.sum() / convex_hull_image(bool_center.get()).sum()
            )
            solidity_storm = prop.solidity
            # objects that are not solid have many holes in the vorticity field are not a tropical storm
            if solidity_center < 0.3:
                continue
            if solidity_storm < 0.3:
                continue

            # a storm needs to be solid (solidity) and round (object ratio)
            if object_ratio / prop.solidity > 4:
                continue

            # skip objects that are not in the tropical storm areas
            # too close to the equator
            if abs(lat_) < 5:
                continue
            if not (lon_ > 20 and lat_ > 5 and lat_ < 50) and not (
                lon_ < 200 and lat_ < -5 and lat_ > -40
            ):
                continue

            # define a mask that describes the pixels that sourrand the center/the pixels that were removed by binary_erosion
            bool_border = x_label[prop_slice_ext].astype(bool) & ~bool_center
            from copy import copy

            # calculate the mean of the voritcity of the border pixels and the center pixels
            v_sub1 = copy(v_[prop_slice_ext])
            v_sub2 = copy(v_sub1)
            cp.putmask(v_sub1, ~bool_center, cp.nan)
            cp.putmask(v_sub2, ~bool_border, cp.nan)
            mean_inner = cp.nanmean(v_sub1)
            mean_outer = cp.nanmean(v_sub2)

            # mean voriticty in the center must be larger than mean vorticity at the border of the storm
            if mean_inner > mean_outer:
                # calculate max wind and lowest msl
                w10m_max = torch.tensor(w10m_[prop_slice_ext].max())
                msl_min = torch.tensor(msl_[prop_slice_ext].min())

                # add storm center to list of centers
                centers.append(
                    torch.as_tensor(
                        [lat_, lon_, msl_min, w10m_max], device=vort850.device
                    )
                )

        if len(centers) > 0:
            return torch.stack(centers, dim=-1)
        else:
            x = torch.empty((4, 1), device=vort850.device)
            x[:] = torch.nan
            return x

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)

        lat = torch.as_tensor(self.input_coords()["lat"], device=x.device)
        lon = torch.as_tensor(self.input_coords()["lon"], device=x.device)

        def get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
            index = VARIABLES_TCV.index(var)
            return x[index]

        outs = []
        for i in range(x.shape[0]):

            # Get wind components at 850 hPa
            u850 = get_variable(x[i], "u850")
            v850 = get_variable(x[i], "v850")
            u10m = get_variable(x[i], "u10m")
            v10m = get_variable(x[i], "v10m")
            msl = get_variable(x[i], "msl")

            # Calculate vorticity at 850 hPa
            vort850 = CycloneTracking.vorticity(u850, v850)
            vort850[361:] *= -1

            # Calculate wind speed at 10m height
            w10m = torch.sqrt(torch.square(u10m) + torch.square(v10m))

            # identify position of tropical storm centers
            centers = self._find_centers(lat, lon, vort850, w10m, msl)
            outs.append(centers)

        if outs:
            # Outs can be of different shapes (different numbers of TCs)
            # Need to create padded tensor
            out_tensor = torch.nested.nested_tensor(
                outs, dtype=torch.float32, device=vort850.device
            )
            out_tensor = torch.nested.to_padded_tensor(out_tensor, torch.nan)
            output_coords["point"] = np.arange(out_tensor.shape[-1])
        else:
            size_dim_batch = len(output_coords["batch"])
            size_dim_variable = len(output_coords["variable"])
            out_tensor = torch.tensor(
                np.ones((size_dim_batch, size_dim_variable, 0)),
                device=x.device,
                dtype=float,
            )
            output_coords = OrderedDict(
                {
                    "batch": output_coords["batch"],
                    "variable": output_coords["variable"],
                    "point": np.array([]),
                }
            )

        return out_tensor, output_coords


@check_extra_imports("cyclone", ["cupy", "cucim", "skimage"])
class CycloneTracking(torch.nn.Module, _CycloneTrackingBase):
    """Finds a list of tropical cyclone centers using the conditions
    in Vitart 1997

    Note
    ----
    For more information about this method see:

    - https://doi.org/10.1175/1520-0442(1997)010%3C0745:SOIVOT%3E2.0.CO;2

    Parameters
    ----------
    vorticity_threshold: float
        The threshold for vorticity at 850, below which a possible
        tropical cyclone center is rejected. By default 3.5e-5 1/s
    mslp_threshold: float
        The threshold for minimum sea level pressure for local minimums
        to be considered tropical cyclones. By default 990 hPa.
    temp_dec_threshold: float
        The value for which average temperature must decrease away from
        the warm core for a possible center to be considered a tropical
        cyclone. By default 0.5 degrees celsius
    lat_threshold: float
        The maximum absolute latitude that a point will be considered to
        be a tropical cyclone. By default 60 degrees (N and S).
    exclude_border: bool or int
        If positive integer, exclude_border excludes peaks from within
        exclude_border-pixels of the border of the image. If tuple of
        non-negative ints, the length of the tuple must match the input
        array dimensionality. Each element of the tuple will exclude peaks
        from within exclude_border-pixels of the border of the image along
        that dimension. If True, takes the min_distance parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    """

    def __init__(
        self,
        vorticity_threshold: float = 3.5e-5,
        mslp_threshold: float = 990.0,
        temp_dec_threshold: float = 0.5,
        lat_threshold: float = 60.0,
        exclude_border: bool | int = True,
    ) -> None:
        super().__init__()

        self.vorticity_threshold = vorticity_threshold
        self.msl_threshold = mslp_threshold
        self.temp_dec_threshold = temp_dec_threshold
        self.lat_threshold = lat_threshold
        self.exclude_border = exclude_border

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES_TC),
                "lat": np.empty(0),
                "lon": np.empty(0),
            }
        )

    def _find_centers(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        msl: torch.Tensor,
        vort850: torch.Tensor,
        t_200_500_mean: torch.Tensor,
        dz_200_850: torch.Tensor,
        vorticity_threshold: float = 3.5e-5,
        mslp_threshold: float = 990.0,
        temp_dec_threshold: float = 0.5,
        exclude_border: bool | int = True,
    ) -> torch.Tensor:
        """Finds a list of tropical cyclone centers

        Parameters
        ----------
        lat : torch.Tensor
            mesh of latitudes tensors. [nlat, nlon]
        lon : torch.Tensor
            mesh of longitudes tensors. [nlat, nlon]
        msl : torch.Tensor
            Mean Sea Level Pressure tensor of dimension
            [nlat, nlon].
        vort850 : torch.Tensor
        850 hPa relative vorticity of dimension [nlat, nlon].
        t_200_500_mean : torch.Tensor
            Average temperature between 200 hPa and 500 hPa of
            dimension [nlat, nlon].
        dz_200_850 : torch.Tensor
            Thickness between 200 hPa and 850 hPa geopotential. [nlat, nlon]
        vorticity_threshold: float
            The threshold for vorticity at 850, below which a possible
            tropical cyclone center is rejected. By default 3.5e-5 1/s
        mslp_threshold: float
            The threshold for minimum sea level pressure for local minimums
            to be considered tropical cyclones. By default 990 hPa.
        temp_dec_threshold: float
            The value for which average temperature must decrease away from
            the warm core for a possible center to be considered a tropical
            cyclone. By default 0.5 degrees celsius
        exclude_border: bool or int
            If positive integer, exclude_border excludes peaks from within
            exclude_border-pixels of the border of the image. If tuple of
            non-negative ints, the length of the tuple must match the input
            array dimensionality. Each element of the tuple will exclude peaks
            from within exclude_border-pixels of the border of the image along
            that dimension. If True, takes the min_distance parameter as value.
            If zero or False, peaks are identified regardless of their distance
            from the border.

        Returns
        -------
        torch.Tensor
            List of TC centers, torch.Tensor of shape [2, N]
        """

        # Get local max vorticity
        vlm = CycloneTracking.get_local_max(
            vort850,
            threshold_abs=vorticity_threshold,
            min_distance=10,
            exclude_border=exclude_border,
        )
        vlm_loc = torch.stack(
            (lat[vlm[:, 0], vlm[:, 1]], lon[vlm[:, 0], vlm[:, 1]]), dim=1
        )

        # Get local maximum of average temperature between 500 and 200mb
        tlm = CycloneTracking.get_local_max(
            t_200_500_mean, min_distance=10, exclude_border=exclude_border
        )
        tlm_loc = torch.stack(
            (lat[tlm[:, 0], tlm[:, 1]], lon[tlm[:, 0], tlm[:, 1]]), dim=1
        )

        # Get local max z200 - z850
        dzlm = CycloneTracking.get_local_max(dz_200_850, exclude_border=exclude_border)
        dzlm_loc = torch.stack(
            (lat[dzlm[:, 0], dzlm[:, 1]], lon[dzlm[:, 0], dzlm[:, 1]]), dim=1
        )

        # Get local min msl
        msllm = CycloneTracking.get_local_max(
            -msl / 100,
            threshold_abs=-mslp_threshold,
            min_distance=10,
            exclude_border=exclude_border,
        )
        mlm_loc = torch.stack(
            (lat[msllm[:, 0], msllm[:, 1]], lon[msllm[:, 0], msllm[:, 1]]), dim=1
        )

        centers = []
        for mins in mlm_loc:
            center0 = torch.as_tensor(mins, device=msl.device)

            # Vorticity filter
            dist = CycloneTracking.haversine_torch(
                mins[0], mins[1], vlm_loc[:, 0], vlm_loc[:, 1], meters=False
            )
            if dist.min() > 8 * 25:  # Distance should be 8 degrees ~= 200 km
                continue

            # Warm Core filter
            dist = CycloneTracking.haversine_torch(
                mins[0], mins[1], tlm_loc[:, 0], tlm_loc[:, 1], meters=False
            )
            if dist.min() > 2 * 25:  # Distance should be less than 2 degrees ~= 50 km
                continue

            ti = torch.argmin(dist)
            ti_loc = tlm_loc[ti]
            ti = tlm[ti]

            lat_inds = (
                CycloneTracking.haversine_torch(
                    ti_loc[0], ti_loc[1], lat[:, ti[1]], lon[:, ti[1]], meters=False
                )
                < 8 * 25
            )
            lon_inds = (
                CycloneTracking.haversine_torch(
                    ti_loc[0], ti_loc[1], lat[ti[0], :], lon[ti[0], :], meters=False
                )
                < 8 * 25
            )
            dec = (
                t_200_500_mean[:, lon_inds][lat_inds]  # approximation to stay regular
                - t_200_500_mean[ti[0], ti[1]]
            ) < -temp_dec_threshold
            new_ti_lat_index = torch.where(lat[lat_inds, ti[1]] == ti_loc[0])[0].item()
            new_ti_lon_index = torch.where(lon[ti[0], lon_inds] == ti_loc[1])[0].item()
            if not all(
                [
                    torch.any(dec[new_ti_lat_index, new_ti_lon_index:]),
                    torch.any(dec[new_ti_lat_index, :new_ti_lon_index]),
                    torch.any(dec[:new_ti_lat_index, new_ti_lon_index]),
                    torch.any(dec[new_ti_lat_index:, new_ti_lon_index]),
                ]
            ):
                continue

            # dZ filter
            dist = CycloneTracking.haversine_torch(
                mins[0], mins[1], dzlm_loc[:, 0], dzlm_loc[:, 1], meters=False
            )
            if dist.min() > 2 * 25:
                continue

            centers.append(center0)
        if len(centers) > 0:
            return torch.stack(centers, dim=-1)
        else:
            x = torch.empty((2, 1), device=vort850.device)
            x[:] = torch.nan
            return x

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords.pop("variable")
        output_coords.pop("lat")
        output_coords.pop("lon")
        output_coords["coord"] = np.array(["lat", "lon"])
        output_coords["point"] = np.arange(0)

        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)

        lat = torch.as_tensor(coords["lat"], device=x.device)
        lon = torch.as_tensor(coords["lon"], device=x.device)

        if lat.ndim != lon.ndim:
            raise ValueError(
                "Error, lat/lon grids must have the same number of dimensions."
            )

        if lat.ndim < 2:
            lon, lat = torch.meshgrid(lon, lat, indexing="xy")

        if lat.shape != lon.shape:
            raise ValueError("Error, lat/lon grids must be the same shape.")

        if self.lat_threshold is not None:
            _, nlon = lat.shape
            indices = lat.abs() < self.lat_threshold

            lat = lat[indices].reshape(-1, nlon)
            lon = lon[indices].reshape(-1, nlon)

        def get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
            index = VARIABLES_TC.index(var)
            return x[index]

        ####
        # First thing to do is to get MSL local minimums
        # extract MSL from x
        # x - [n, 8, nlat, nlon]
        outs = []
        for i in range(x.shape[0]):

            # Get vorticity
            u850 = get_variable(x[i], "u850")
            v850 = get_variable(x[i], "v850")
            vort850 = CycloneTracking.vorticity(u850, v850)
            vort850[361:] *= -1

            # Get MSL
            msl = get_variable(x[i], "msl")

            # Get average temp
            t_200_500_mean = torch.mean(
                torch.stack(
                    [
                        get_variable(x[i], ti)
                        for ti in ["t500", "t400", "t300", "t250", "t200"]
                    ],
                    dim=0,
                ),
                dim=0,
            )

            # Get z200 - z850 width
            dz_200_850 = get_variable(x[i], "z200") - get_variable(x[i], "z850")

            if self.lat_threshold is not None:
                msl = msl[indices].reshape(-1, nlon)
                vort850 = vort850[indices].reshape(-1, nlon)
                t_200_500_mean = t_200_500_mean[indices].reshape(-1, nlon)
                dz_200_850 = dz_200_850[indices].reshape(-1, nlon)

            centers = self._find_centers(
                lat,
                lon,
                msl,
                vort850,
                t_200_500_mean,
                dz_200_850,
                vorticity_threshold=self.vorticity_threshold,
                mslp_threshold=self.msl_threshold,
                temp_dec_threshold=self.temp_dec_threshold,
                exclude_border=self.exclude_border,
            )
            outs.append(centers)

        # Outs can be of different shapes (different numbers of TCs)
        # Need to create padded tensor
        out_tensor = torch.nested.nested_tensor(
            outs, dtype=torch.float32, device=msl.device
        )
        out_tensor = torch.nested.to_padded_tensor(out_tensor, torch.nan)
        output_coords["point"] = np.arange(out_tensor.shape[-1])
        return out_tensor, output_coords


# The rest of this code is a workflow utility functions
def get_tracks_from_positions(
    y: torch.Tensor,
    c: OrderedDict,
    min_length: int = 1,
    search_radius_km: float = 250,
    max_skips: int = 1,
) -> DataFrame:
    """Given a list of possible tropical storms for every timestep this function connects positions to tracks

    Parameters
    ----------
    y : torch.Tensor
        Vector of latitudes for tensors.
    c : torch.Tensor
        Vector of longitudes for tensors.
    min_length: int
        Minimum length to consider a track. If the track is shorter than min_length steps it will be discarded
    search_radius_km: float
        Maximum distance of the successor to the current position
    max_skips: int
        Max allowed timesteps to skip (if no successor is find in the next timestep)

    Returns
    -------
    tracks_df
        A pandas dataframe containing all identified tracks
    """

    if min_length <= 0:
        raise AssertionError("min_length needs to be larger than 0")

    if search_radius_km <= 0:
        raise AssertionError("search_radius_km needs be larger than zero")

    if max_skips <= 0:
        raise AssertionError("max_skips needs to be 0 or larger")

    y = y.cpu().numpy()
    # get list of tc center candidates for each timestep
    y_lists = [
        [
            tuple(y[i, :, j])
            for j in range(0, y[i, :].shape[1])
            if (~np.isnan(y[i, :, j])).all()
        ]
        for i in range(0, y.shape[0])
    ]

    tracks: list = []

    def search_candidate_row(
        y_lists: list,
        t_idx_now: int,
        track_id_now: int,
        min_length: int,
        search_radius_km: float,
        max_skips: int,
    ) -> int:
        """Searches for center candidates that can be connected to a track. Search starts at time index t_idx_now.
        Starting with a position at t1 it searches for a successor at t2 that is within search_radius_km distance of the first position.
        If no direct successor can be found search is extended to the second next time step t3 within 2*search_radius_km.
        If still no successor can be found the track is completed and stored if the track consists at least of min_length elements. Shorter tracks are discarded.

        Parameters
        ----------
        y_list : List of lists of tuples
            Every tuple contains 4 elements that represent TC location and intensity: latitude, longitude, minimum SLP [Pa] and max windspeed at 10m height [m/s]
            First list covers time dimension. Second list represents individual TC centres.
        t_idx_now : int
            index of time
        track_id_now: int
            current track id
        min_length: int
            Minimum length to consider a track. If the track is shorter than min_length steps it will be discarded
        search_radius_km: float
            Maximum distance of the successor to the current position
        max_skips: int
            Max allowed timesteps to skip (if no successor is find in the next timestep)

        Returns
        -------
        track_id_now: int
        """
        while len(y_lists[t_idx_now]) > 0:
            pos_now = y_lists[t_idx_now].pop(0)
            t_idx_now_start = t_idx_now
            track_temp: list[tuple[int, Any, int, Any]] = []
            pos_number = 0
            v = (t_idx_now, pos_now, pos_number, None)
            pos_number = pos_number + 1
            track_temp.append(v)
            while v[0]:
                next_candidates = y_lists[t_idx_now + 1 : t_idx_now + max_skips + 2]
                v = get_next_position(
                    next_candidates=next_candidates,
                    pos_now=pos_now,
                    t_idx_now=t_idx_now,
                    search_radius_km=search_radius_km,
                    max_skips=max_skips,
                )
                if v[0]:
                    # Found next position; add to temporary track
                    t_idx_next, pos_next, ind_next, _ = v
                    track_temp.append((t_idx_next, pos_next, pos_number, ind_next))
                    pos_number = pos_number + 1
                    t_idx_now = t_idx_next
                    pos_now = pos_next
                else:
                    # Could not find next position
                    # decide if temporaray track can be become a final track
                    # minimum duration is
                    if len(track_temp) >= min_length:
                        track = [
                            (track_id_now,) + (c["time"][x[0]],) + x[1:3]
                            for x in track_temp
                        ]

                        to_remove = [(x[0], x[3]) for x in track_temp]
                        for i, j in to_remove:
                            if j is not None:
                                y_lists[i].pop(int(j))
                        tracks.append(track)
                        track_id_now += 1
                    t_idx_now = t_idx_now_start
        return track_id_now

    current_track_id = 0
    for t_idx_now in range(0, len(y_lists)):
        current_track_id = search_candidate_row(
            y_lists,
            t_idx_now,
            current_track_id,
            min_length,
            search_radius_km,
            max_skips,
        )

    # Restructuring data to pandas dataframe
    num_meteo_variables = y.shape[1] - 2
    tracks_df = convert_tracks_to_dataframe(tracks, num_meteo_variables)

    return tracks_df


def get_tracks_df(
    tracks_flattened: list, num_meteo_variables: int, track_columns: list
) -> DataFrame:
    """Convert individual track to data frame

    Parameters
    ----------
    tracks_flattened : list
        list of tracks
    num_meteo_variables : int
        number of variables to add to track
    track_columns : list
        list of column names

    Returns
    -------
    tracks_df: pd.DataFrame
    """
    if num_meteo_variables in [0, 2]:
        tracks_df = DataFrame(
            tracks_flattened, columns=track_columns[: 5 + num_meteo_variables]
        )
    else:
        raise NotImplementedError
    return tracks_df


def convert_tracks_to_dataframe(
    track_list: list, num_meteo_variables: int
) -> DataFrame:
    """Convert tracks from list to pd data frame

    Parameters
    ----------
    track_list : List
        list of tracks
    num_meteo_variables : int
        number of variables to add to track

    Returns
    -------
    tracks_df: pd.DataFrame
    """
    track_columns = [
        "track_id",
        "vt",
        "point_number",
        "tc_lat",
        "tc_lon",
        "tc_msl",
        "tc_speed",
    ]

    tracks_flattened = []
    if track_list:
        for track in track_list:
            for track_element in track:
                tracks_flattened.append(
                    [track_element[0], track_element[1], track_element[3]]
                    + list(track_element[2])
                )
                tracks_df = get_tracks_df(
                    tracks_flattened, num_meteo_variables, track_columns
                )

    else:
        tracks_df = get_tracks_df(tracks_flattened, num_meteo_variables, track_columns)

    return tracks_df


def get_next_position(
    next_candidates: list,
    pos_now: tuple,
    t_idx_now: int,
    search_radius_km: float = 250,
    max_skips: int = 1,
) -> tuple:
    """Parameters
    ----------
    next_candidates : List of tuples
        Every tuple contains 4 elements that represent TC location and intensity: latitude, longitude, minimum SLP [Pa] and max windspeed at 10m height [m/s]
    pos_now: tuple (float, float, float, float)
        4 elements that represent TC location and intensity: latitude, longitude, minimum SLP [Pa] and max windspeed at 10m height [m/s]
    t_idx_now : int
        index of time now
    search_radius_km: float
        Maximum distance of the successor to the current position
    max_skips: int
        Max allowed timesteps to skip (if no successor is find in the next timestep)

    Returns
    -------
    tuple containing 3 elements
        t_idx_next: int
            index of next time step
        pos_next: tuple (float, float, float, float)
            4 elements that represent TC location and intensity: latitude, longitude, minimum SLP [Pa] and max windspeed at 10m height [m/s]
        ind_min: int
            index of the candidate that got selected as the next location
    """
    success = False
    for forward_search_step in range(0, min(len(next_candidates), max_skips + 1)):
        t_idx_next = t_idx_now + forward_search_step + 1
        lat1 = torch.tensor(pos_now[0])
        lon1 = torch.tensor(pos_now[1])
        if len(next_candidates[forward_search_step]) > 0:
            lat2 = torch.tensor(next_candidates[forward_search_step])[:, 0]
            lon2 = torch.tensor(next_candidates[forward_search_step])[:, 1]
            dist = CycloneTracking.haversine_torch(lat1, lon1, lat2, lon2) / 1000
        else:
            continue
        if torch.min(dist) < search_radius_km * (1 + forward_search_step):
            success = True
            ind_min = torch.argmin(dist)
            pos_next = next_candidates[forward_search_step][ind_min]
            break
    if success:
        return (t_idx_next, pos_next, ind_min, None)
    else:
        return (None, None, None, None)


# def run_example() -> None:
#     """Demonstrates the functionality of the CycloneTracking model by running it on a sample dataset and
#     connecting the identified tropical cyclone positions to tracks."""
#     import pandas as pd

#     from earth2studio.data import GFS, prep_data_array

#     # Initialize the CycloneTracking model
#     CT = CycloneTracking()

#     # Define the timestamps to analyze
#     time = np.array(pd.date_range("2024-08-26", "2024-08-29", freq="6H"))

#     # Load the required weather data from GFS analysis
#     gfs = GFS()
#     variable = CT.input_coords()["variable"]
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da)

#     # Run the tropical cyclone tracker
#     y, c = CT(x, coords)

#     # Connect positions to tracks
#     df_tracks = get_tracks_from_positions(
#         y, c, min_length=3, search_radius_km=250, max_skips=1
#     )

#     # Display the tracks DataFrame
#     print(df_tracks)


# if __name__ == "__main__":
#     run_example()
