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

from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
)
from earth2studio.utils.checkpoint import bind_checkpoint_state
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from physicsnemo.models.afno import AFNO
except ImportError:
    OptionalDependencyFailure("fcn")
    AFNO = None

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


@dataclass
class _FCNCheckpointState:
    x: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FCN(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """FourCastNet global prognostic model. Consists of a single model with a time-step
    size of 6 hours. FourCastNet operates on 0.25 degree lat-lon grid (south-pole
    excluding) equirectangular grid with 26 variables.

    Note
    ----
    This model is a retrained version on more atmospgeric variables from the FourCastNet
    paper. For additional information see the following resources:

    - https://arxiv.org/abs/2202.11214
    - https://huggingface.co/nvidia/fourcastnet1

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor of size [26]
    scale : torch.Tensor
        Model scale normalization tensor of size [26]

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2022 gpu:40gb
    provider:nvidia backend:pytorch
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
        self.checkpoint = bind_checkpoint_state(_FCNCheckpointState())

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free FCN input coordinate signature."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
            },
            dynamic=("batch",),
            grid="fcn1",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Return the FCN coordinate signature after one forecast step."""
        if "lead_time" not in input_coords.coords:
            raise ValueError("Input lead_time coordinate is required")
        lead = np.asarray(input_coords["lead_time"])
        if (
            input_coords["lead_time"].dims != ("lead_time",)
            or lead.size != 1
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("Input lead_time must contain one finite timedelta")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(6, "h")}
        )

    def __str__(
        self,
    ) -> str:
        return "fcn"

    def _restore_checkpoint_state(self, x: xr.DataArray) -> tuple[xr.DataArray, bool]:
        if (
            self.checkpoint.checkpoint_level == 2
            and self.checkpoint.checkpoint_state_loaded
            and self.checkpoint.x is not None
            and self.checkpoint.metadata
        ):
            tensor, _ = x.e2s.to_torch()
            metadata = deepcopy(self.checkpoint.metadata)
            state = self.checkpoint.x.to(tensor.device)
            signature = coord_array(
                metadata["dims"],
                metadata["coords"],
                sizes=metadata["sizes"],
                attrs=metadata["attrs"],
            )
            restored = from_torch(
                state, signature, name=metadata["name"], attrs=metadata["attrs"]
            )
            restored.encoding = metadata["encoding"]
            return restored, True
        return x, False

    def _save_checkpoint_state(self, x: xr.DataArray) -> None:
        if self.checkpoint.checkpoint_enabled and self.checkpoint.checkpoint_level == 2:
            tensor, _ = x.e2s.to_torch()
            self.checkpoint.x = tensor.detach().clone().to(self.checkpoint.device)
            self.checkpoint.metadata = deepcopy(
                {
                    "dims": tuple(x.dims),
                    "sizes": dict(x.sizes),
                    "name": x.name,
                    "coords": {
                        name: (
                            tuple(value.dims),
                            value.values.copy(),
                            dict(value.attrs),
                        )
                        for name, value in x.coords.items()
                    },
                    "attrs": dict(x.attrs),
                    "encoding": dict(x.encoding),
                }
            )
        else:
            self.checkpoint.x = None
            self.checkpoint.metadata = {}

    # --8<-- [start:fcn-default-package]
    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://nvidia/fourcastnet1@c67a63995f6c8e0e557eb3d791f32f437e9b02d5",
            cache_options={
                "cache_storage": Package.default_cache("fcn"),
                "same_names": True,
            },
        )

    # --8<-- [end:fcn-default-package]

    # --8<-- [start:fcn-load-model]
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        model = AFNO.from_checkpoint(package.resolve("fcn.mdlus"))
        model.eval()

        local_center = torch.Tensor(np.load(package.resolve("global_means.npy")))
        local_std = torch.Tensor(np.load(package.resolve("global_stds.npy")))
        return cls(model, center=local_center, scale=local_std)

    # --8<-- [end:fcn-load-model]

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.squeeze(1)
        x = (x - self.center) / self.scale
        x = self.model(x)
        x = self.scale * x + self.center
        x = x.unsqueeze(1)
        return x

    @batch_func()
    def _step(self, x: xr.DataArray) -> xr.DataArray:
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        output = from_torch(self._forward(tensor), signature, name=x.name)
        output.encoding = x.encoding.copy()
        return output

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : xr.DataArray
            NumPy- or CuPy-backed input with coordinates and grid metadata.

        Returns
        -------
        xr.DataArray
            Forecast six hours in the future on the same device.
        """
        x, _ = self._restore_checkpoint_state(x)
        x = self._step(x)
        self._save_checkpoint_state(x)
        return x

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        x, restored = self._restore_checkpoint_state(x)
        self.output_coords(x)

        if not restored:
            self._save_checkpoint_state(x)
            yield x.copy(deep=False)

        while True:
            x = self.rear_hook(self._step(self.front_hook(x)))
            self._save_checkpoint_state(x)
            yield x.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : xr.DataArray
            Initial state with coordinates and grid metadata.


        Yields
        ------
        xr.DataArray
            Initial condition followed by six-hour forecast steps. A restored
            level-two checkpoint resumes at the next step.
        """
        yield from self._default_generator(x)
