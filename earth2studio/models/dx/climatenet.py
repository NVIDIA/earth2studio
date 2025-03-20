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
from earth2studio.models.nn.climatenet_conv import CGNetModule
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "tcwv",
    "u850",
    "v850",
    "msl",
]

OUT_VARIABLES = [
    "climnet_bg",  # Background
    "climnet_tc",  # Tropical Cyclone
    "climnet_ar",  # Atmospheric River
]


class ClimateNet(torch.nn.Module, AutoModelMixin):
    """Climate Net diagnostic model, built into Earth2Studio. This model can be used to
    create prediction labels for tropical cyclones and atmospheric rivers from a set of
    three atmospheric variables on a quater degree resolution equirectangular grid. It
    produces three non-standard output channels climnet_bg, climnet_tc and climnet_ar
    representing background label, tropical cyclone and atmospheric river labels.

    Note
    ----
    This model and checkpoint are from Prabhat et al. 2021. For more information see the
    following references:

    - https://doi.org/10.5194/gmd-14-107-2021
    - https://github.com/andregraubner/ClimateNet

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
        output_coords["variable"] = np.array(OUT_VARIABLES)
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained climatenet model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_diagnostics@v0.1",
            cache_options={
                "cache_storage": Package.default_cache("climatenet"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        checkpoint_zip = Path(package.resolve("climatenet.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_zip.parent)

        model = CGNetModule(channels=len(VARIABLES), classes=len(OUT_VARIABLES))
        model.load_state_dict(
            torch.load(str(checkpoint_zip.parent / Path("climatenet/weights.tar")))
        )
        model.eval()

        input_center = torch.Tensor(
            np.load(str(checkpoint_zip.parent / Path("climatenet/global_means.npy")))
        ).reshape(-1, 1, 1)
        input_scale = torch.Tensor(
            np.load(str(checkpoint_zip.parent / Path("climatenet/global_stds.npy")))
        ).reshape(-1, 1, 1)
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
        out = torch.softmax(self.core_model(x), 1)

        return out, output_coords
