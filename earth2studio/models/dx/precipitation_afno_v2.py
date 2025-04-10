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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

try:
    from physicsnemo import Module
    from physicsnemo.models.afno import AFNO
    from physicsnemo.models.meta import ModelMetaData
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    Module = None
    AFNO = None
    ModelMetaData = None
    cos_zenith_angle = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem


class PeriodicPad2d(nn.Module):
    """
    pad longitudinal (left-right) circular
    and pad latitude (top-bottom) with zeros
    """

    def __init__(self, pad_width: int):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


@dataclass
class MetaData(ModelMetaData):
    name: str = "AFNO"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class PrecipNet(Module):
    def __init__(
        self,
        inp_shape: tuple,
        patch_size: tuple,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        depth: int,
        num_blocks: int,
        *args,
        **kwargs
    ):
        super().__init__(meta=MetaData())
        backbone = AFNO(
            inp_shape,
            in_channels,
            out_channels,
            patch_size,
            embed_dim,
            depth,
            num_blocks,
        )

        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x


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


@check_extra_imports(
    "precipitation_afno_v2", [Module, AFNO, ModelMetaData, cos_zenith_angle]
)
class PrecipitationAFNOV2(torch.nn.Module, AutoModelMixin):
    """Improved Precipitation AFNO diagnostic model. Predicts the average hourly precipitation for the next 6 hour
    with the units mm/h. This model uses an 20 atmospheric inputs and outputs one on a 0.25 degree
    lat-lon grid (south-pole excluding) [720 x 1440].

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
        output_coords["variable"] = np.array(["tp06"])
        return output_coords

    def __str__(self) -> str:
        return "precipnet"

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
    @check_extra_imports(
        "precipitation_afno_v2", [Module, AFNO, ModelMetaData, cos_zenith_angle]
    )
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        if (
            Module is None
            or AFNO is None
            or ModelMetaData is None
            or cos_zenith_angle is None
        ):
            raise ImportError(
                "Additional PrecipitationAFNOV2 model dependencies are not installed. See install documentation for details."
            )

        p = package.resolve("afno_precip.mdlus")
        model = PrecipNet.from_checkpoint(str(Path(p)))
        model.eval()

        input_center = torch.Tensor(
            np.load(str(package.resolve(Path("global_means.npy"))))
        )
        input_scale = torch.Tensor(
            np.load(str(package.resolve(Path("global_stds.npy"))))
        )
        lsm = torch.Tensor(
            xr.open_dataset(str(package.resolve(Path("land_sea_mask.nc"))))[
                "LSM"
            ].values
        )[None, :, :-1]

        orography = torch.Tensor(
            xr.open_dataset(str(package.resolve(Path("orography.nc"))))["Z"].values
        )[None, :, :-1]
        orography = (orography - orography.mean()) / orography.std()

        return cls(model, lsm, orography, input_center, input_scale)

    def compute_sza(
        self,
        lon: np.array,
        lat: np.array,
        time: np.datetime64,
        lead_time: np.timedelta64,
    ):
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

        for j, _ in enumerate(coords["batch"]):
            for k, t in enumerate(coords["time"]):
                for lt, dt in enumerate(coords["lead_time"]):
                    sza = (
                        self.compute_sza(grid_x, grid_y, t, dt)
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
        out[out < 0] = 0
        return out, output_coords
