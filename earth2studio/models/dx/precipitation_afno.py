# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.nn.afno_precip import PrecipNet
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
]


class PrecipitationAFNO(torch.nn.Module, AutoModelMixin):
    """Precipitation AFNO diagnsotic model. Predicts the total precipation parameter
    which is the accumulated amount of liquid and frozen water (rain or snow) with
    units m. This model uses an 20 atmospheric inputs and outputs one on a 0.25 degree
    lat-lon grid (south-pole excluding) [720 x 1440].

    Note:
        This checkpoint is from Parthik et al. 2022:

        - https://arxiv.org/abs/2202.11214
        - https://github.com/NVlabs/FourCastNet

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    center : torch.Tensor
        Model center normalization tensor of size [20,1,1]
    scale : torch.Tensor
        Model scale normalization tensor of size [20,1,1]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.eps = 1e-5

    input_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(["tp"]),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    def __str__(self) -> str:
        return "precipnet"

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained precipation model package from Nvidia model registry"""
        return Package("ngc://models/nvidia/modulus/modulus_diagnostics@v0.1")

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        cached_path = package.get("precipitation_afno.zip")
        model = PrecipNet.from_checkpoint(
            cached_path + "/precipitation_afno/precipitation_afno.mdlus"
        )
        model.eval()

        input_center = torch.Tensor(
            np.load(cached_path + "/precipitation_afno/global_means.npy")
        )
        input_scale = torch.Tensor(
            np.load(cached_path + "/precipitation_afno/global_stds.npy")
        )
        return cls(model, input_center, input_scale)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        handshake_dim(coords, "lon", 3)
        handshake_dim(coords, "lat", 2)
        handshake_dim(coords, "variable", 1)
        handshake_coords(coords, self.input_coords, "lon")
        handshake_coords(coords, self.input_coords, "lat")
        handshake_coords(coords, self.input_coords, "variable")

        x = (x - self.center) / self.scale
        out = self.core_model(x)
        # Unlog output
        # https://github.com/NVlabs/FourCastNet/blob/master/utils/weighted_acc_rmse.py#L66
        out = self.eps * (torch.exp(out) - 1)

        output_coords = coords.copy()
        output_coords["variable"] = self.output_coords["variable"]
        return out, output_coords  # Softmax channels
