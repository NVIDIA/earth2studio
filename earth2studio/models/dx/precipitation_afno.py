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
from pathlib import Path

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel

try:
    from earth2studio.models.nn.afno_precip import PrecipNet
except ImportError:
    PrecipNet = None

from earth2studio.utils import (
    check_extra_imports,
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


@check_extra_imports("precip-afno", [PrecipNet])
class PrecipitationAFNO(torch.nn.Module, AutoModelMixin):
    """Precipitation AFNO diagnsotic model. Predicts the total precipation parameter
    which is the accumulated amount of liquid and frozen water (rain or snow) with
    units m. This model was trained on ERA5 data thus the accumulation period is over
    the 1 hour ending at the validity date and time.This model uses an 20 atmospheric
    inputs and outputs one on a 0.25 degree lat-lon grid (south-pole excluding)
    [720 x 1440].

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
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["tp"])
        return output_coords

    def __str__(self) -> str:
        return "precipnet"

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained precipation model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_diagnostics@v0.1",
            cache_options={
                "cache_storage": Package.default_cache("precip_afno"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("precip-afno", [PrecipNet])
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        checkpoint_zip = Path(package.resolve("precipitation_afno.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_zip.parent)

        # Hack because old checkpoint
        model = PrecipNet(
            inp_shape=[720, 1440],
            in_channels=20,
            out_channels=1,
            patch_size=[8, 8],
            embed_dim=768,
        )
        model.load(
            str(
                checkpoint_zip.parent
                / Path("precipitation_afno/precipitation_afno.mdlus")
            )
        )
        model.eval()

        input_center = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/global_means.npy"))
            )
        )
        input_scale = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/global_stds.npy"))
            )
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
        output_coords = self.output_coords(coords)

        x = (x - self.center) / self.scale
        out = self.core_model(x)
        # Unlog output
        # https://github.com/NVlabs/FourCastNet/blob/master/utils/weighted_acc_rmse.py#L66
        out = self.eps * (torch.exp(out) - 1)

        return out, output_coords  # Softmax channels
