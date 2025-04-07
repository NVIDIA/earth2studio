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

import zipfile
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import xarray

try:
    import physicsnemo
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    physicsnemo = None
    cos_zenith_angle = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import check_extra_imports, handshake_coords, handshake_dim
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

VARIABLES = ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]


@check_extra_imports("dlwp", [physicsnemo, cos_zenith_angle])
class DLWP(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Deep learning weather prediction (DLWP)  prognostic model. This is a parsimonious
    global forecast model with a time-step size of 6 hours. The core model is a
    convolutional encoder-decoder trained on [64,64] cubed sphere data that has an input
    of 18 fields (2x7 atmos variables + 4 prescriptive) and outputs 14 fields (2x7 atmos
    variables). This implementation provides a wrapper that accepts [721,1440] lat-lon
    equirectangular grid of just the atmospheric varaibles as an input for better
    compatability with common data sources. Prescriptive fields are added inside the
    model wrapper.

    Note
    ----
    For more information about this model see:

    - https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002502
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_dlwp_cubesphere

    Parameters
    ----------
    core_model : torch.nn.Module
        Core cubed-sphere DLWP model.
    landsea_mask : torch.Tensor
        Land sea mask in cubed sphere form [6,64,64]
    orography : torch.Tensor
        Surface geopotential (orography) in cubed sphere form [6,64,64]
    latgrid : torch.Tensor
        Cubed sphere latitude coordinates [6,64,64]
    longrid : torch.Tensor
        Cubed sphere longitude coordinates [6,64,64]
    cubed_sphere_transform : torch.Tensor
        Sparse pytorch tensor to transform equirectangular fields to cubed sphere of
        size [24576, 1038240]
    cubed_sphere_inverse : torch.Tensor
        Sparse pytorch tensor to transform cubed sphere fields to equirectangular of
        size [1038240, 24576]
    center : torch.Tensor
        Model atmospheric variable center normalization tensor of size [1,7,1,1]
    scale : torch.Tensor
        Model atmospheric variable scale normalization tensor of size [1,7,1,1]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        landsea_mask: torch.Tensor,
        orography: torch.Tensor,
        latgrid: torch.Tensor,
        longrid: torch.Tensor,
        cubed_sphere_transform: torch.Tensor,
        cubed_sphere_inverse: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):

        super().__init__()
        self.model = core_model
        self.register_buffer("latgrid", latgrid)
        self.register_buffer("longrid", longrid)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("landsea_mask", landsea_mask.unsqueeze(0))
        self.register_buffer(
            "topographic_height",
            (orography.unsqueeze(0).unsqueeze(0) - 3.724e03) / 8.349e03,
        )
        self.register_buffer("M", cubed_sphere_transform.T)
        self.register_buffer("N", cubed_sphere_inverse)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            handshake_dim(test_coords, key, i)
            if key not in ["batch", "time"]:
                handshake_coords(test_coords, target_input_coords, key)

        # Normal forward pass of DLWP, this method returns two time-steps
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]

        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )

        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLWP model package on NGC"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_dlwp_cubesphere@v0.2",
            cache_options={
                "cache_storage": Package.default_cache("dlwp"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("dlwp", [physicsnemo, cos_zenith_angle])
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        dlwp_zip = Path(package.resolve("dlwp_cubesphere.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(dlwp_zip, "r") as zip_ref:
            zip_ref.extractall(dlwp_zip.parent)

        lsm = torch.Tensor(
            xarray.open_dataset(
                str(dlwp_zip.parent / Path("dlwp/land_sea_mask_rs_cs.nc"))
            )["lsm"].values
        )
        topographic_height = torch.Tensor(
            xarray.open_dataset(
                str(dlwp_zip.parent / Path("dlwp/geopotential_rs_cs.nc"))
            )["z"].values
        )
        latlon_grids = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/latlon_grid_field_rs_cs.nc"))
        )
        latgrid = torch.Tensor(latlon_grids["latgrid"].values)
        longrid = torch.Tensor(latlon_grids["longrid"].values)
        # load maps
        input_map_wts = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/map_LL721x1440_CS64.nc"))
        )
        output_map_wts = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/map_CS64_LL721x1440.nc"))
        )

        i = input_map_wts.row.values - 1
        j = input_map_wts.col.values - 1
        data = input_map_wts.S.values
        cubed_sphere_transform = torch.sparse_coo_tensor(
            np.array((i, j)), data, dtype=torch.float
        )

        i = output_map_wts.row.values - 1
        j = output_map_wts.col.values - 1
        data = output_map_wts.S.values
        cubed_sphere_inverse = torch.sparse_coo_tensor(
            np.array((i, j)), data, dtype=torch.float
        )

        core_model = physicsnemo.Module.from_checkpoint(
            str(dlwp_zip.parent / Path("dlwp/dlwp.mdlus"))
        )

        center = torch.Tensor(
            np.load(str(dlwp_zip.parent / Path("dlwp/global_means.npy")))
        )
        scale = torch.Tensor(
            np.load(str(dlwp_zip.parent / Path("dlwp/global_stds.npy")))
        )

        return cls(
            core_model,
            landsea_mask=lsm,
            orography=topographic_height,
            latgrid=latgrid,
            longrid=longrid,
            cubed_sphere_transform=cubed_sphere_transform,
            cubed_sphere_inverse=cubed_sphere_inverse,
            center=center,
            scale=scale,
        )

    def to_cubedsphere(self, x: torch.Tensor) -> torch.Tensor:
        """[721,1440] eqr to [6,64,64] cs"""
        x = x.reshape(*x.shape[:-2], -1) @ self.M
        x = x.reshape(*x.shape[:-1], 6, 64, 64)
        return x

    def to_equirectangular(self, x: torch.Tensor) -> torch.Tensor:
        """[6,64,64] cs to [721,1440] eqr"""
        input_shape = x.shape[:-3]
        x = (self.N @ x.reshape(-1, 6 * 64 * 64).T).T
        x = x.reshape(*input_shape, 721, 1440)
        return x

    def get_cosine_zenith_fields(
        self, times: np.array, lead_time: timedelta, device: torch.device | str = "cuda"
    ) -> torch.Tensor:
        """Creates cosine zenith fields for input time array"""
        output = []
        for time in timearray_to_datetime(times):
            uvcossza = cos_zenith_angle(
                time + lead_time,
                self.longrid.cpu(),
                self.latgrid.cpu(),
            )
            # Normalize
            uvcossza = torch.Tensor(uvcossza).to(device)
            uvcossza = torch.clamp(uvcossza, min=0) - 1.0 / np.pi
            output.append(uvcossza)
        return torch.stack(output, axis=0)

    def _prepare_input(self, input: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
        """Prepares input cubed sphere tensor by adding land sea mask, uvcossza and
        orography fields to input atmospheric ([14,6,64,64] -> [18,6,64,64])
        """
        # Compress batch dim into time
        time_array = np.tile(coords["time"], input.shape[0])
        input = input.view(-1, *input.shape[2:])

        uvcossza_6 = self.get_cosine_zenith_fields(
            time_array, timedelta(hours=-6), input.device
        ).unsqueeze(1)
        uvcossza_0 = self.get_cosine_zenith_fields(
            time_array, timedelta(hours=0), input.device
        ).unsqueeze(1)
        x = torch.cat([input[:, 0], uvcossza_6, input[:, 1], uvcossza_0], dim=1)

        input = torch.cat(
            (
                x,
                self.landsea_mask.repeat(x.shape[0], 1, 1, 1, 1),
                self.topographic_height.repeat(x.shape[0], 1, 1, 1, 1),
            ),
            dim=1,
        )
        return input

    def _prepare_output(
        self, output: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        output = torch.split(output, output.shape[1] // 2, dim=1)
        # Add lead time dimension back in
        output = torch.stack(output, dim=1)
        # Add batch dimension back in
        output = output.view(-1, coords["time"].shape[0], *output.shape[1:])
        return output

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:

        center = self.center.unsqueeze(-1)
        scale = self.scale.unsqueeze(-1)

        x = (x - center) / scale
        x = self._prepare_input(x, coords)
        x = self.model(x)
        x = self._prepare_output(x, coords)
        x = scale * x + center
        return x

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """

        output_coords = self.output_coords(coords)

        x = self.to_cubedsphere(x)
        x = self._forward(x, coords)
        x = self.to_equirectangular(x)

        return x[:, :, :1], output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()
        self.output_coords(coords)

        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield x[:, :, 1:], coords_out

        x = self.to_cubedsphere(x)
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward pass
            x = self._forward(x, coords)
            coords["lead_time"] = (
                coords["lead_time"]
                + 2 * self.output_coords(self.input_coords())["lead_time"]
            )
            x = x.clone()

            # Rear hook for first predicted step
            coords_out = coords.copy()
            coords_out["lead_time"] = coords["lead_time"][0]
            x[:, :, :1], coords_out = self.rear_hook(x[:, :, :1], coords_out)

            # Output first predicted step
            out = self.to_equirectangular(x[:, :, :1])
            yield out, coords_out

            # Rear hook for second predicted step
            coords_out["lead_time"] = coords["lead_time"][-1]
            x[:, :, 1:], coords_out = self.rear_hook(x[:, :, 1:], coords_out)
            out = self.to_equirectangular(x[:, :, 1:])
            yield out, coords_out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system


        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
