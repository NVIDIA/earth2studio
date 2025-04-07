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
from pathlib import Path

import numpy as np
import torch

try:
    from physicsnemo.models.afno import AFNO
except ImportError:
    AFNO = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import check_extra_imports, handshake_coords, handshake_dim
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
    "u100m",
    "v100m",
    "u250",
    "v250",
    "z250",
    "t250",
]


@check_extra_imports("fcn", [AFNO])
class FCN(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """FourCastNet global prognostic model. Consists of a single model with a time-step
    size of 6 hours. FourCastNet operates on 0.25 degree lat-lon grid (south-pole
    excluding) equirectangular grid with 26 variables.

    Note
    ----
    This model is a retrained version on more atmospgeric variables from the FourCastNet
    paper. For additional information see the following resources:

    - https://arxiv.org/abs/2202.11214
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcn

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor of size [26]
    scale : torch.Tensor
        Model scale normalization tensor of size [26]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    # sphinx - coords start
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
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
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
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key != "batch":
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords = output_coords.copy()
        output_coords["batch"] = input_coords["batch"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )

        return output_coords

    # sphinx - coords end

    def __str__(
        self,
    ) -> str:
        return "fcn"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_fcn@v0.2",
            cache_options={
                "cache_storage": Package.default_cache("fcn"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("fcn", [AFNO])
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        fcn_zip = Path(package.resolve("fcn.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(fcn_zip, "r") as zip_ref:
            zip_ref.extractall(fcn_zip.parent)

        model = AFNO.from_checkpoint(str(fcn_zip.parent / Path("fcn/fcn.mdlus")))
        model.eval()

        local_center = torch.Tensor(
            np.load(str(fcn_zip.parent / Path("fcn/global_means.npy")))
        )
        local_std = torch.Tensor(
            np.load(str(fcn_zip.parent / Path("fcn/global_stds.npy")))
        )
        return cls(model, center=local_center, scale=local_std)

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.squeeze(1)
        x = (x - self.center) / self.scale
        x = self.model(x)
        x = self.scale * x + self.center
        x = x.unsqueeze(1)
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

        x = self._forward(x)

        return x, output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        self.output_coords(coords)

        yield x, coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            coords = self.output_coords(coords)
            x = self._forward(x)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield x, coords.copy()

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
