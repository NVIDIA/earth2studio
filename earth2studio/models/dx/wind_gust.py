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
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.imports import check_extra_imports

try:
    from physicsnemo import Module
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    Module = None
    cos_zenith_angle = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "u100m",
    "v100m",
    "msl",
    "u300",
    "u850",
    "u925",
    "v300",
    "v850",
    "v925",
    "z300",
    "z850",
    "z925",
    "t850",
    "t925",
    "q500",
    "q850",
    "q925",
]


@check_extra_imports("windgust-afno", [Module, cos_zenith_angle])
class WindgustAFNO(torch.nn.Module, AutoModelMixin):
    """Wind gust AFNO diagnsotic model. Predicts the maximum wind gust during the
    preceding hour with the units m/s. This model uses an 17 atmospheric inputs and
    outputs one on a 0.25 degree lat-lon grid (south-pole excluding) [720 x 1440].

    Note
    ----
    For more information on the model, please refer to:

    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/afno_dx_wg-v1-era5

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    landsea_mask : torch.Tensor
        Land sea mask tensor of size [720,1440]
    orography : torch.Tensor
        Surface geopotential (orography) tensor of size [720,1440]
    center : torch.Tensor
        Model center normalization tensor of size [17,1,1]
    scale : torch.Tensor
        Model scale normalization tensor of size [17,1,1]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        landsea_mask: torch.Tensor,
        orography: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("landsea_mask", landsea_mask)
        self.register_buffer(
            "topographic_height",
            (orography),
        )

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
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
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
        handshake_dim(input_coords, "lon", 5)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["fg10m"])
        return output_coords

    def __str__(self) -> str:
        return "windgustnet"

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained windgustnet model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/earth-2/afno_dx_wg-v1-era5@v0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("windgustnet"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        model = Module.from_checkpoint(package.resolve("afno_windgust_1h.mdlus"))
        model.eval()

        input_center = torch.Tensor(np.load(package.resolve("global_means.npy")))

        input_scale = torch.Tensor(np.load(package.resolve("global_stds.npy")))
        lsm = torch.Tensor(
            xr.open_dataset(package.resolve("land_sea_mask.nc"))["LSM"].values
        )[None, :, :-1]

        orography = torch.Tensor(
            xr.open_dataset(package.resolve("orography.nc"))["Z"].values
        )[None, :, :-1]
        orography = (orography - orography.mean()) / orography.std()

        return cls(model, lsm, orography, input_center, input_scale)

    def _compute_sza(
        self,
        lon: np.array,
        lat: np.array,
        time: np.datetime64,
        lead_time: np.timedelta64,
    ) -> torch.Tensor:
        """Compute solar zenith angle"""
        _unix = np.datetime64(0, "s")
        _ds = np.timedelta64(1, "s")
        t = time + lead_time
        t = datetime.fromtimestamp((t - _unix) / _ds, tz=timezone.utc)
        return torch.Tensor(cos_zenith_angle(t, lon, lat))

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)
        out = torch.zeros_like(x[..., :1, :, :])
        x = (x - self.center) / self.scale

        grid_x, grid_y = torch.meshgrid(
            torch.tensor(coords["lat"]), torch.tensor(coords["lon"])
        )

        # compute solar zenith angle and concatenate
        for j, _ in enumerate(coords["batch"]):
            for k, t in enumerate(coords["time"]):
                for lt, dt in enumerate(coords["lead_time"]):
                    sza = (
                        self._compute_sza(grid_x, grid_y, t, dt)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(x.device)
                    )
                    tran = torch.cat(
                        [sza, self.topographic_height, self.landsea_mask], dim=1
                    )
                    in_ = torch.cat((x[j, k, lt : lt + 1], tran), dim=1)
                    out[j, k, lt : lt + 1] = self.core_model(in_)

        out = torch.clamp(out, min=0)
        return out, output_coords
