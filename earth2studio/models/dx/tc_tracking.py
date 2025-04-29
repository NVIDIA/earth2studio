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

try:
    import cupy as cp
    from cucim.skimage.feature import peak_local_max as cucim_peak_local_max
    from cucim.skimage.measure import label, regionprops
    from cucim.skimage.morphology import binary_erosion, remove_small_objects
    from scipy.spatial import KDTree
    from skimage.feature import peak_local_max as skimage_peak_local_max
    from skimage.morphology import convex_hull_image

    # from cupyx.scipy.spatial import KDTree as CuKDTree
except ImportError:
    cp = None
    cucim_peak_local_max = None
    label = None
    regionprops = None
    binary_erosion = None
    remove_small_objects = None
    skimage_peak_local_max = None
    convex_hull_image = None
    KDTree = None
    # CuKDTree = None

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import CoordSystem

VARIABLES_TCV = [
    "u10m",
    "v10m",
    "msl",
    "u850",
    "v850",
    "z500",
    "z850",
    "z200",
    "t500",
    "t400",
    "t300",
    "t250",
    "t200",
]
VARIABLES_TCWD = ["u10m", "v10m", "msl", "u850", "v850"]
OUT_VARIABLES = ["tc_lat", "tc_lon", "tc_msl", "tc_w10m"]


class _TCTrackerBase:

    PATH_FILL_VALUE = -9999  # Should not be in lat/lon range for safety

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
    ) -> torch.Tensor:
        """Compute haversine distance between two pairs of lat/lon points in km.

        Parameters
        ----------
        lat1 : torch.Tensor
            Latitude coordinates of first point [n]
        lon1 : torch.Tensor
            Longitude coordinates of first point [n]
        lat2 : torch.Tensor
            Latitude coordinates of second point [n]
        lon2 : torch.Tensor
            Longitude coordinates of second point [n]

        Returns
        -------
        torch.Tensor
            Distance between two points [n]
        """
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        unit = 6371
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

    @staticmethod
    def latlon_to_equirectangular(
        lat: torch.Tensor, lon: torch.Tensor, R: float = 6371.0
    ) -> torch.Tensor:
        """Convert latitude/longitude to equirectangular projection coordinates.

        Parameters
        ----------
        lat : torch.Tensor
            Latitude in degrees
        lon : torch.Tensor
            Longitude in degrees
        R : float, optional
            Earth radius in km, by default 6371.0

        Returns
        -------
        torch.Tensor
            Stacked x, y coordinates in km
        """
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        # Project to equirectangular coordinates
        x = R * lon_rad * torch.cos(torch.deg2rad(torch.tensor(0.0)))
        y = R * lat_rad

        return torch.stack([x, y], dim=-1)

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

    @classmethod
    def append_paths(
        cls,
        frame: torch.Tensor,
        path_buffer: torch.Tensor,
        path_search_distance: float = 250,
        path_search_window_size: int = 3,
    ) -> torch.Tensor:
        """Appends frame of TC centers into the track path history tensor

        Parameters
        ----------
        frame : torch.Tensor
            Instanteous frame of TC centers to append to existing list of paths of size
            [batch, point_id, variable]
        path_buffer : torch.Tensor
            The current buffer of paths to append to with size
            [batch, path_id, step, variable]
        path_search_distance : float, optional
            Max haversine search distance to connect to in km, by default 250
        path_search_window_size: int, optional
            The historical window size for a path to use when pairing. Namely, the
            path search will use the specified number of historic points to connect a
            new frame to the current set of paths, by default 3

        Returns
        -------
        torch.Tensor
            Updated history buffer of size [batch, path_id(+1), step+1, variable]
        """
        if path_buffer.nelement() == 0:
            return frame.unsqueeze(2).clone()

        if path_search_window_size < 1:
            raise ValueError("Path search window size must be greater than 1")

        if path_search_distance <= 0:
            raise ValueError("Path search distance must be greater than 0")

        if frame.shape[0] != path_buffer.shape[0]:
            raise ValueError(
                f"Error with updating TC tracker history buffer, input and history buffer need the same batch size. Got {frame.shape[0]} and {path_buffer.shape[0]}"
            )

        next_frame = torch.full_like(
            path_buffer[:, :, 0, :], cls.PATH_FILL_VALUE, dtype=frame.dtype
        ).unsqueeze(2)
        # Expand by one path for over flow / new paths
        # I.e. we have [batch, path id + 1, 1, variable]
        next_frame = torch.cat(
            [next_frame, torch.full_like(next_frame[:, :1], cls.PATH_FILL_VALUE)], dim=1
        )

        for i in range(frame.shape[0]):
            # Stack all lat/lon features for window size into a [n,2] feature list
            path_search_window_size = min(
                [path_search_window_size, path_buffer.shape[2]]
            )
            tree_features = (
                path_buffer[i, :, -path_search_window_size:, :2].flip(-2).reshape(-1, 2)
            )

            tree_features = cls.latlon_to_equirectangular(
                tree_features[:, 0], tree_features[:, 1]
            )
            input_features = cls.latlon_to_equirectangular(
                frame[i, :, 0], frame[i, :, 1]
            )

            if tree_features.is_cuda and False:
                # Make need to edit below, not worth it at the moment
                # /.venv/lib/python3.12/site-packages/cupy/_environment.py
                # Liune 350 to min_pypi_version = config[lib]['version']
                # tree = CuKDTree(cp.from_dlpack(tree_features))
                # dist, idx = tree.query(cp.from_dlpack(input_features), k=1)
                pass
            else:
                tree = KDTree(tree_features.cpu())
                dist, idx = tree.query(input_features.cpu(), k=1)

            for p in range(dist.shape[0]):
                # If all variables are 0, its a filler from rnn.pad_sequence
                if torch.all(frame[i, p] == cls.PATH_FILL_VALUE):
                    continue
                # For steps that are further back in the window, increase radius
                # Note, the `.flip(-2)` above make this calculation easier placing most
                # recent step at index 0 instead of last
                past_steps = idx[p] % path_search_window_size + 1
                if dist[p] < past_steps * path_search_distance:
                    # Recall we have a window of points for each path
                    # if any match, append to that path
                    path_id = int(idx[p]) // path_search_window_size
                    next_frame[i, path_id] = frame[i, p]
                # No match so looks like we need a new path
                else:
                    next_frame[i, -1] = frame[i, p]

        # if theres nothing in the extra path row, get rid of it
        if torch.all(next_frame[:, -1] == cls.PATH_FILL_VALUE):
            next_frame = next_frame[:, :-1]
        else:
            # Expand the path_id dim by 1 for concat
            import torch.nn.functional as F

            path_buffer = F.pad(
                path_buffer, (0, 0, 0, 0, 0, 1, 0, 0), "constant", cls.PATH_FILL_VALUE
            )
        return torch.cat([path_buffer, next_frame], axis=2)


@check_extra_imports("cyclone", [cp, KDTree, "cucim", "skimage"])
class TCTrackerWuDuan(torch.nn.Module, _TCTrackerBase):
    """Finds a list of tropical cyclone (TC) centers using an adaption of the method
    described in the conditions in Wu and Duan 2023. The algorithm converts vorticity
    from reanalysis data into a binary image using a defined critical threshold.
    Subsequent processing with connected component labeling and erosion identifies the
    resulting inner cores as TC seeds.

    Note
    ----
    For more information about this method see:

    - https://doi.org/10.1016/j.wace.2023.100626


    Parameters
    ----------
    path_search_distance: int, optional
        The max radial distance two cyclone centers will be considered part of the same
        path in km, by default 300
    path_search_window_size: int, optional
        The historical window size used when creating TC paths, by default 2

    Examples
    --------
    The cyclone tracker will return a tensor of TC paths collected over a series of
    forward passes which are held inside of the models state.
    Namely given a time series of `n` snap shots, the tracker should be called for each
    time-step resulting in a tensor consisting of a set number of paths with n steps.
    Any non-valid / missing data will be `torch.nan` for filtering in post processing
    steps.

    >>> model = TCTrackerWuDuan()
    >>> # Process each timestep
    >>> for time in [datetime(2017, 8, 25) + timedelta(hours=6 * i) for i in range(3)]:
    ...     da = data_source(time, tracker.input_coords()["variable"])
    ...     input, input_coords = prep_data_array(da, device=device)
    ...     output, output_coords = model(input, input_coords)
    >>> # Final path_buffer shape: [batch, path_id, steps, variable]
    >>> output.shape  # torch.Size([1, 6, 3, 4])
    >>> model.path_buffer.shape  # torch.Size([1, 6, 3, 4])
    >>> # Remove current paths from models state
    >>> model.reset_path_buffer()
    >>> model.path_buffer.shape  # torch.Size([0])
    """

    def __init__(
        self, path_search_distance: int = 300, path_search_window_size: int = 2
    ) -> None:
        super().__init__()
        self.register_buffer("path_buffer", torch.empty(0))
        self.path_search_distance = path_search_distance
        self.path_search_window_size = path_search_window_size

    def reset_path_buffer(self) -> None:
        """Resets the internal"""
        self.path_buffer = torch.empty(0)

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
                "variable": np.array(VARIABLES_TCWD),
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

        # [batch, path_id, step, variable]
        output_coords = OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("path_id", np.empty(0)),
                ("step", np.empty(0)),
                ("variable", np.array(OUT_VARIABLES)),
            ]
        )
        return output_coords

    def _find_centers(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        vort850: torch.Tensor,
        w10m: torch.Tensor,
        msl: torch.Tensor,
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
        torch.Tensor
            List of TC centers, torch.Tensor of shape [N, 4]
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
        # Label regions
        x_label = label(x_)

        # Remove labels that are not of the the needed size (less than 18 pixels)
        x_label = remove_small_objects(x_label, min_size=18, connectivity=1)
        # Get region props (bounding box / axis)
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
            return torch.stack(centers, dim=0)
        else:
            x = torch.full((1, 4), self.PATH_FILL_VALUE, device=vort850.device)
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
            index = VARIABLES_TCWD.index(var)
            return x[index]

        outs = []
        for i in range(x.shape[0]):

            # Get wind components at 850 hPa
            # VARIABLES_TCWD = ["u850", "v850", "u10m", "v10m", "msl"]
            u850 = get_variable(x[i], "u850")
            v850 = get_variable(x[i], "v850")
            u10m = get_variable(x[i], "u10m")
            v10m = get_variable(x[i], "v10m")
            msl = get_variable(x[i], "msl")

            # Calculate vorticity at 850 hPa
            vort850 = TCTrackerWuDuan.vorticity(u850, v850)
            vort850[361:] *= -1  # Invert southern hemisphere

            # Calculate wind speed at 10m height
            w10m = torch.sqrt(torch.square(u10m) + torch.square(v10m))

            # identify position of tropical storm centers
            centers = self._find_centers(lat, lon, vort850, w10m, msl)
            # Pack [points, variables]
            outs.append(centers)

        # amazing function!
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        out = torch.nn.utils.rnn.pad_sequence(
            outs, padding_value=self.PATH_FILL_VALUE, batch_first=True
        )
        # [batch, path_id, step, variable]
        self.path_buffer = self.append_paths(
            out,
            self.path_buffer,
            self.path_search_distance,
            self.path_search_window_size,
        )
        out = torch.where(
            self.path_buffer == self.PATH_FILL_VALUE, torch.nan, self.path_buffer
        )

        output_coords = self.output_coords(coords)
        output_coords["path_id"] = np.arange(self.path_buffer.shape[1])
        output_coords["step"] = np.arange(self.path_buffer.shape[2])

        return out, output_coords


@check_extra_imports("cyclone", [cp, KDTree, "cucim", "skimage"])
class TCTrackerVitart(torch.nn.Module, _TCTrackerBase):
    """Finds a list of tropical cyclone centers using the conditions in Vitart 1997

    Note
    ----
    For more information about this method see:

    - https://doi.org/10.1175/1520-0442(1997)010%3C0745:SOIVOT%3E2.0.CO;2


    Parameters
    ----------
    vorticity_threshold : float, optional
        The threshold for vorticity at 850, below which a possible
        tropical cyclone center is rejected, by default 3.5e-5 1/s
    mslp_threshold : float, optional
        The threshold for minimum sea level pressure for local minimums
        to be considered tropical cyclone, by default 99000 Pa
    temp_dec_threshold : float, optional
        The value for which average temperature must decrease away from
        the warm core for a possible center to be considered a tropical
        cyclone, by default 0.5 degrees celsius
    lat_threshold: float, optional
        The maximum absolute latitude that a point will be considered to
        be a tropical cyclone, by default 60 degrees (N and S).
    exclude_border: bool | int, optional
        If positive integer, exclude_border excludes peaks from within
        exclude_border-pixels of the border of the image. If tuple of
        non-negative ints, the length of the tuple must match the input
        array dimensionality. Each element of the tuple will exclude peaks
        from within exclude_border-pixels of the border of the image along
        that dimension. If True, takes the min_distance parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    path_search_distance: int, optional
        The max radial distance two cyclone centers will be considered part of the same
        path in km, by default 300
    path_search_window_size: int, optional
        The historical window size used when creating TC paths, by default 2

    Examples
    --------
    The cyclone tracker will return a tensor of TC paths collected over a series of
    forward passes which are held inside of the models state.
    Namely given a time series of `n` snap shots, the tracker should be called for each
    time-step resulting in a tensor consisting of a set number of paths with n steps.
    Any non-valid / missing data will be `torch.nan` for filtering in post processing
    steps.

    >>> model = TCTrackerVitart()
    >>> # Process each timestep
    >>> for time in [datetime(2017, 8, 25) + timedelta(hours=6 * i) for i in range(3)]:
    ...     da = data_source(time, tracker.input_coords()["variable"])
    ...     input, input_coords = prep_data_array(da, device=device)
    ...     output, output_coords = model(input, input_coords)
    >>> # Final path_buffer shape: [batch, path_id, steps, variable]
    >>> output.shape  # torch.Size([1, 6, 3, 4])
    >>> model.path_buffer.shape  # torch.Size([1, 6, 3, 4])
    >>> # Remove current paths from models state
    >>> model.reset_path_buffer()
    >>> model.path_buffer.shape  # torch.Size([0])
    """

    def __init__(
        self,
        vorticity_threshold: float = 3.5e-5,
        mslp_threshold: float = 99000.0,
        temp_dec_threshold: float = 0.5,
        lat_threshold: float = 60.0,
        exclude_border: bool | int = True,
        path_search_distance: int = 300,
        path_search_window_size: int = 2,
    ) -> None:
        super().__init__()
        # TC Center identification parameters
        self.vorticity_threshold = vorticity_threshold
        self.msl_threshold = mslp_threshold
        self.temp_dec_threshold = temp_dec_threshold
        self.lat_threshold = lat_threshold
        self.exclude_border = exclude_border
        # TC path identification parameters
        self.register_buffer("path_buffer", torch.empty(0))
        self.path_search_distance = path_search_distance
        self.path_search_window_size = path_search_window_size

    def reset_path_buffer(self) -> None:
        """Resets the internal"""
        self.path_buffer = torch.empty(0)

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
        handshake_coords(input_coords, target_input_coords, "variable")

        # [batch, path_id, step, variable]
        output_coords = OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("path_id", np.empty(0)),
                ("step", np.empty(0)),
                ("variable", np.array(OUT_VARIABLES)),
            ]
        )
        return output_coords

    def _find_centers(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        msl: torch.Tensor,
        w10m: torch.Tensor,
        vort850: torch.Tensor,
        t_200_500_mean: torch.Tensor,
        dz_200_850: torch.Tensor,
        vorticity_threshold: float = 3.5e-5,
        mslp_threshold: float = 99000.0,
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
            Mean Sea Level Pressure tensor of dimension [nlat, nlon].
        w10m : torch.Tensor
            Surface wind tensor, just used to return surface wind at core
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
            to be considered tropical cyclones. By default 990000 Pa.
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

        min_distance = 10
        # Get local max vorticity
        vlm = TCTrackerVitart.get_local_max(
            vort850,
            threshold_abs=vorticity_threshold,
            min_distance=min_distance,
            exclude_border=exclude_border,
        )
        vlm_loc = torch.stack(
            (lat[vlm[:, 0], vlm[:, 1]], lon[vlm[:, 0], vlm[:, 1]]), dim=1
        )

        # Get local maximum of average temperature between 500 and 200mb
        tlm = TCTrackerVitart.get_local_max(
            t_200_500_mean, min_distance=min_distance, exclude_border=exclude_border
        )
        tlm_loc = torch.stack(
            (lat[tlm[:, 0], tlm[:, 1]], lon[tlm[:, 0], tlm[:, 1]]), dim=1
        )

        # Get local max z200 - z850
        dzlm = TCTrackerVitart.get_local_max(dz_200_850, exclude_border=exclude_border)
        dzlm_loc = torch.stack(
            (lat[dzlm[:, 0], dzlm[:, 1]], lon[dzlm[:, 0], dzlm[:, 1]]), dim=1
        )

        # Get local min msl
        msllm = TCTrackerVitart.get_local_max(
            -msl,
            threshold_abs=-mslp_threshold,
            min_distance=min_distance,
            exclude_border=exclude_border,
        )
        mlm_loc = torch.stack(
            (lat[msllm[:, 0], msllm[:, 1]], lon[msllm[:, 0], msllm[:, 1]]), dim=1
        )

        dzlm = TCTrackerVitart.get_local_max(dz_200_850, exclude_border=exclude_border)
        dzlm_loc = torch.stack(
            (lat[dzlm[:, 0], dzlm[:, 1]], lon[dzlm[:, 0], dzlm[:, 1]]), dim=1
        )

        centers = []
        for i, mins in enumerate(mlm_loc):
            idx_lat, idx_lon = msllm[i]
            center0 = torch.as_tensor(mins, device=msl.device)

            # Vorticity filter
            dist = TCTrackerVitart.haversine_torch(
                mins[0], mins[1], vlm_loc[:, 0], vlm_loc[:, 1]
            )
            if dist.min() > 8 * 25:  # Distance should be 8 degrees ~= 200 km
                continue

            # Warm Core filter
            dist = TCTrackerVitart.haversine_torch(
                mins[0], mins[1], tlm_loc[:, 0], tlm_loc[:, 1]
            )
            if dist.min() > 2 * 25:  # Distance should be less than 2 degrees ~= 50 km
                continue

            ti = torch.argmin(dist)  # Get closest distance to core
            ti_loc = tlm_loc[ti]
            ti = tlm[ti]

            lat_inds = (
                TCTrackerVitart.haversine_torch(
                    ti_loc[0], ti_loc[1], lat[:, ti[1]], lon[:, ti[1]]
                )
                < 8 * 25
            )
            lon_inds = (
                TCTrackerVitart.haversine_torch(
                    ti_loc[0], ti_loc[1], lat[ti[0], :], lon[ti[0], :]
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
            dist = TCTrackerVitart.haversine_torch(
                mins[0], mins[1], dzlm_loc[:, 0], dzlm_loc[:, 1]
            )
            if dist.min() > 2 * 25:
                continue

            w10m_max = torch.max(
                w10m[
                    idx_lat - min_distance : idx_lat + min_distance,
                    idx_lon - min_distance : idx_lon + min_distance,
                ]
            )
            centers.append(
                torch.tensor([center0[0], center0[1], msl[ti[0], ti[1]], w10m_max]).to(
                    msl.device
                )
            )

        if len(centers) > 0:
            return torch.stack(centers, dim=0)
        else:
            x = torch.full((1, 4), self.PATH_FILL_VALUE, device=vort850.device)
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

        def get_variable(x0: torch.Tensor, var: str) -> torch.Tensor:
            index = VARIABLES_TCV.index(var)
            return x0[:, index]

        ####
        # First thing to do is to get MSL local minimums
        # extract MSL from x
        # x - [n, 8, nlat, nlon]

        u10m = get_variable(x, "u10m")
        v10m = get_variable(x, "v10m")
        w10m = torch.sqrt(torch.pow(u10m, 2) + torch.pow(v10m, 2))

        # Get vorticity
        u850 = get_variable(x, "u850")
        v850 = get_variable(x, "v850")
        vort850 = TCTrackerVitart.vorticity(u850, v850)
        vort850[:, 361:] *= -1
        # Get MSL
        msl = get_variable(x, "msl")
        # Get average temp
        t_200_500_mean = torch.mean(
            torch.stack(
                [
                    get_variable(x, ti)
                    for ti in ["t500", "t400", "t300", "t250", "t200"]
                ],
                dim=1,
            ),
            dim=1,
        )
        # Get z200 - z850 width
        dz_200_850 = get_variable(x, "z200") - get_variable(x, "z850")

        if self.lat_threshold is not None:
            w10m = w10m[:, indices].reshape(x.shape[0], -1, nlon)
            msl = msl[:, indices].reshape(x.shape[0], -1, nlon)
            vort850 = vort850[:, indices].reshape(x.shape[0], -1, nlon)
            t_200_500_mean = t_200_500_mean[:, indices].reshape(x.shape[0], -1, nlon)
            dz_200_850 = dz_200_850[:, indices].reshape(x.shape[0], -1, nlon)

        outs = []
        for i in range(x.shape[0]):
            centers = self._find_centers(
                lat,
                lon,
                msl[i],
                w10m[i],
                vort850[i],
                t_200_500_mean[i],
                dz_200_850[i],
                vorticity_threshold=self.vorticity_threshold,
                mslp_threshold=self.msl_threshold,
                temp_dec_threshold=self.temp_dec_threshold,
                exclude_border=self.exclude_border,
            )
            outs.append(centers)

        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        out = torch.nn.utils.rnn.pad_sequence(
            outs, padding_value=self.PATH_FILL_VALUE, batch_first=True
        )
        # [batch, path_id, step, variable]
        self.path_buffer = self.append_paths(
            out,
            self.path_buffer,
            self.path_search_distance,
            self.path_search_window_size,
        )
        out = torch.where(
            self.path_buffer == self.PATH_FILL_VALUE, torch.nan, self.path_buffer
        )

        output_coords = self.output_coords(coords)
        output_coords["path_id"] = np.arange(self.path_buffer.shape[1])
        output_coords["step"] = np.arange(self.path_buffer.shape[2])

        return out, output_coords
