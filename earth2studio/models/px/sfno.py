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
import fnmatch
import os
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordinateSystem

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
    FourCastNet operates on 0.25 degree lat-lon grid (south-pole including)
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
    variables : np.array, optional
        Variables associated with model, by default 73 variable model.

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2023 gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        variables: np.array = np.array(VARIABLES),
    ):
        super().__init__()
        self.model = core_model
        self.variables = np.array(variables, copy=True)
        if "2d" in self.variables:
            self.variables[self.variables == "2d"] = "d2m"
        self.register_buffer("device_buffer", torch.empty(0))

    def __str__(self) -> str:
        return "sfno_73ch_small"

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free input signature on the checkpoint grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate input coordinates and advance the final lead by six hours."""
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(6, "h")}
        )

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
        coords: CoordinateSystem,
    ) -> torch.Tensor:
        x = x.clone().squeeze(2)
        for j in range(x.shape[0]):
            for i, t in enumerate(coords["time"].values):
                # https://github.com/NVIDIA/modulus-makani/blob/933b17d5a1ebfdb0e16e2ebbd7ee78cfccfda9e1/makani/third_party/climt/zenith_angle.py#L197
                # Requires time zone data
                t = [
                    datetime.fromisoformat(dt.isoformat() + "+00:00")
                    for dt in timearray_to_datetime(t + coords["lead_time"].values)
                ]
                x[j, i : i + 1] = self.model(x[j, i : i + 1], t, normalized_data=False)
        x = x.unsqueeze(2)
        return x

    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict a six-hour DataArray on the model device, without iterator hooks."""
        signature = self.output_coords(x)
        handshake_time(x)
        tensor, _ = x.e2s.to_torch()
        out = from_torch(
            self._forward(tensor.to(self.device_buffer.device), x),
            signature,
            name=x.name,
        )
        out.encoding = x.encoding.copy()
        return out

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        handshake_nonempty(x)
        handshake_time(x)
        self.output_coords(x)
        yield x.copy(deep=False)
        while True:
            x = self.rear_hook(self(self.front_hook(x.copy(deep=True))))
            yield x.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial field then forecasts at six-hour intervals."""
        yield from self._default_generator(x)
