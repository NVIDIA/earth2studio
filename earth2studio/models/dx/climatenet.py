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

    input_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(OUT_VARIABLES),
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained climatenet model package from Nvidia model registry"""
        return Package("ngc://models/nvidia/modulus/modulus_diagnostics@v0.1")

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        cached_path = package.get("climatenet.zip")
        model = CGNetModule(channels=len(VARIABLES), classes=len(OUT_VARIABLES))
        model.load_state_dict(torch.load(cached_path + "/climatenet/weights.tar"))
        model.eval()

        input_center = torch.Tensor(
            np.load(cached_path + "/climatenet/global_means.npy")
        ).reshape(-1, 1, 1)
        input_scale = torch.Tensor(
            np.load(cached_path + "/climatenet/global_stds.npy")
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
        handshake_dim(coords, "lon", 3)
        handshake_dim(coords, "lat", 2)
        handshake_dim(coords, "variable", 1)
        handshake_coords(coords, self.input_coords, "lon")
        handshake_coords(coords, self.input_coords, "lat")
        handshake_coords(coords, self.input_coords, "variable")

        x = (x - self.center) / self.scale
        out = torch.softmax(self.core_model(x), 1)

        output_coords = coords.copy()
        output_coords["variable"] = self.output_coords["variable"]
        return out, output_coords
