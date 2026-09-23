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

import warnings
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_dim,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from physicsnemo.utils.zenith_angle import cos_zenith_angle

    from earth2studio.models.nn.afno_precip_v2 import PrecipNet_v2
except ImportError:
    OptionalDependencyFailure("precip-afno-v2")
    PrecipNet_v2 = None
    cos_zenith_angle = None

VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u500",
    "u850",
    "u1000",
    "v500",
    "v850",
    "v1000",
    "z50",
    "z500",
    "z850",
    "z1000",
    "t500",
    "t850",
    "q500",
    "q850",
]


@check_optional_dependencies()
class PrecipitationAFNOv2(torch.nn.Module, AutoModelMixin):
    """Improved Precipitation AFNO diagnostic model. Predicts the total precipitation
    for the next 6 hours [t, t+6h] with the units m. This model uses 20 atmospheric
    inputs and outputs one on a 0.25 degree lat-lon grid (south-pole excluding)
    [720 x 1440].

    Warning
    -------
    PrecipitationAFNOv2 performs worse than PrecipitationAFNO v1 and will be
    deprecated in upcoming releases.

    Note
    ----
    For more information on the model, please refer to:

    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/afno_dx_tp-v1-era5

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    landsea_mask : torch.Tensor
        Land sea mask tensor of size [720,1440]
    orography : torch.Tensor
        Surface geopotential (orography) tensor of size [720,1440]
    center : torch.Tensor
        Model center normalization tensor of size [20,1,1]
    scale : torch.Tensor
        Model scale normalization tensor of size [20,1,1]

    Badges
    ------
    region:global class:medium-range product:precip year:2024 gpu:40gb
    provider:nvidia backend:pytorch
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
        warnings.warn(
            "PrecipitationAFNOv2 performs worse than PrecipitationAFNO v1 and "
            "will be deprecated in upcoming releases.",
            DeprecationWarning,
            stacklevel=3,
        )
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("landsea_mask", landsea_mask)
        self.register_buffer(
            "topographic_height",
            (orography),
        )

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free atmospheric signature on the FCN grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {"variable": np.array(VARIABLES)},
            dynamic=("batch", "time", "lead_time"),
            grid="latlon-0.25deg-south-pole-excluded",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Plan precipitation accumulated over the next six hours.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input signature or field with datetime time and timedelta lead labels.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature preserving input metadata and grid.
        """
        handshake_dataarray(input_coords, self.input_coords())
        handshake_dim(input_coords, "time", -5)
        handshake_dim(input_coords, "lead_time", -4)
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time", allow_dynamic=True)
        output = coord_array_like(input_coords, {"variable": ["tp:sum:0h:6h"]})
        output.encoding = input_coords.encoding.copy()
        return output

    def __str__(self) -> str:
        return "PrecipNet"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/earth-2/afno_dx_tp-v1-era5@v0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("precipitation_afno_v2"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""

        # Hack because old checkpoint
        model = PrecipNet_v2(
            inp_shape=[720, 1440],
            patch_size=[8, 8],
            in_channels=23,
            out_channels=1,
            embed_dim=768,
            depth=12,
            num_blocks=8,
            mlp_ratio=8,
        )
        model.load(package.resolve("afno_precip.mdlus"))
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
        lon: np.ndarray,
        lat: np.ndarray,
        time: np.datetime64,
        lead_time: np.timedelta64,
    ) -> torch.Tensor:
        _unix = np.datetime64(0, "s")
        _ds = np.timedelta64(1, "s")
        t = time + lead_time
        t = datetime.fromtimestamp((t - _unix) / _ds, tz=timezone.utc)
        return torch.Tensor(cos_zenith_angle(t, lon, lat))

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(x)
        coords = x.coords
        encoding = x.encoding.copy()
        x, _ = x.e2s.to_torch()
        out = torch.zeros_like(x[..., :1, :, :])
        x = (x - self.center) / self.scale

        lat_grid, lon_grid = torch.meshgrid(
            torch.tensor(coords["lat"].values),
            torch.tensor(coords["lon"].values),
            indexing="ij",
        )

        for j in range(x.shape[0]):
            for k, t in enumerate(coords["time"].values):
                for lt, dt in enumerate(coords["lead_time"].values):
                    sza = (
                        self._compute_sza(lon_grid, lat_grid, t, dt)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(x.device)
                    )
                    tran = torch.cat(
                        [sza, self.topographic_height, self.landsea_mask], dim=1
                    )
                    in_ = torch.cat((x[j, k, lt : lt + 1], tran), dim=1)
                    out[j, k, lt : lt + 1] = self.core_model(in_)

        out = 1e-5 * (torch.exp(out) - 1)
        # convert from mm to m
        out = out / 1000.0
        out[out < 0] = 0
        output = from_torch(out, output_coords)
        output.encoding = encoding
        return output
