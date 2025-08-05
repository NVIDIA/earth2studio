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
import fnmatch
import os
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    from makani.models import model_registry
    from makani.models.model_package import (
        LocalPackage,
        ModelWrapper,
        load_model_package,
    )
    from makani.utils.driver import Driver
    from makani.utils.YParams import ParamsBase
except ImportError:
    OptionalDependencyFailure("sfno")
    load_model_package = None
    Driver = None
    ParamsBase = None
    LocalPackage = None
    model_registry = None
    ModelWrapper = None

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]


@check_optional_dependencies()
class SFNO(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Spherical Fourier Operator Network global prognostic model.
    Consists of a single model with a time-step size of 6 hours.
    FourCastNet operates on 0.25 degree lat-lon grid (south-pole excluding)
    equirectangular grid with 73 variables.

    Note
    ----
    This model and checkpoint are trained using Modulus-Makani. For more information
    see the following references:

    - https://arxiv.org/abs/2306.03838
    - https://github.com/NVIDIA/modulus-makani
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor
    scale : torch.Tensor
        Model scale normalization tensor
    variables : np.array, optional
        Variables associated with model, by default 73 variable model.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        variables: np.array = np.array(VARIABLES),
    ):
        super().__init__()
        self.model = core_model
        self.variables = variables
        if "2d" in self.variables:
            self.variables[self.variables == "2d"] = "d2m"

    def __str__(self) -> str:
        return "sfno_73ch_small"

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
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
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
            by default None, will use self.input_coords.
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
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        if input_coords is None:
            return output_coords
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("sfno"),
                "same_names": True,
            },
        )
        package.root = os.path.join(package.root, "sfno_73ch_small")
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, variables: list = VARIABLES, device: str = "cpu"
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        variables : list, optional
            Model variable override, by default VARIABLES for SFNO 73 channel

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        # Makani load_model_package
        path = package.resolve("config.json")
        params = ParamsBase.from_json(path)

        # Set global_means_path and global_stds_path
        params.global_means_path = package.resolve("global_means.npy")
        params.global_stds_path = package.resolve("global_stds.npy")
        # Need to manually set min and max paths to none.
        params.min_path = None
        params.max_path = None

        # Need to manually set in and out channels to all variables.
        if params.channel_names is None:
            params.channel_names = variables
        else:
            variables = params.channel_names
        params.in_channels = np.arange(len(variables))
        params.out_channels = np.arange(len(variables))

        LocalPackage._load_static_data(package, params)

        # assume we are not distributed
        # distributed checkpoints might be saved with different params values
        params.img_local_offset_x = 0
        params.img_local_offset_y = 0
        params.img_local_shape_x = params.img_shape_x
        params.img_local_shape_y = params.img_shape_y

        # set grid type to sinusoidal without cosine features added in makani 0.2.0
        if params.get("add_cos_to_grid", None) is None:
            params.add_cos_to_grid = False

        # get the model
        model = model_registry.get_model(params, multistep=False).to(device)

        # Load checkpoint
        best_checkpoint_path = package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        checkpoint = torch.load(
            best_checkpoint_path, weights_only=False, map_location=device
        )
        state_dict = checkpoint["model_state"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )

        # Resize model.blocks filters for some reason
        keys_to_resize = fnmatch.filter(
            state_dict.keys(), "model.blocks.*.filter.filter.weight"
        )
        for key in keys_to_resize:
            state_dict[key] = state_dict[key].unsqueeze(0)

        model.load_state_dict(state_dict)

        # Wrap model
        model = ModelWrapper(model, params=params)

        # Set model to eval mode
        model.eval()

        # Load variables
        variables = np.array(model.params.channel_names)

        return cls(model, variables=variables)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        x = x.squeeze(2)
        for j, _ in enumerate(coords["batch"]):
            for i, t in enumerate(coords["time"]):
                # https://github.com/NVIDIA/modulus-makani/blob/933b17d5a1ebfdb0e16e2ebbd7ee78cfccfda9e1/makani/third_party/climt/zenith_angle.py#L197
                # Requires time zone data
                t = [
                    datetime.fromisoformat(dt.isoformat() + "+00:00")
                    for dt in timearray_to_datetime(t + coords["lead_time"])
                ]
                x[j, i : i + 1] = self.model(x[j, i : i + 1], t, normalized_data=False)
        x = x.unsqueeze(2)
        return x, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system
        """
        return self._forward(x, coords)

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
            x, coords = self._forward(x, coords)
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
