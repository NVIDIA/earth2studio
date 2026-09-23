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

from collections.abc import Iterator
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    from aurora import Aurora as Aurora_model
    from aurora import Batch, Metadata
except ImportError:
    OptionalDependencyFailure("aurora")
    Aurora_model = None
    Batch = None
    Metadata = None

VARIABLES = [
    "z1000",
    "z925",
    "z850",
    "z700",
    "z600",
    "z500",
    "z400",
    "z300",
    "z250",
    "z200",
    "z150",
    "z100",
    "z50",
    "q1000",
    "q925",
    "q850",
    "q700",
    "q600",
    "q500",
    "q400",
    "q300",
    "q250",
    "q200",
    "q150",
    "q100",
    "q50",
    "t1000",
    "t925",
    "t850",
    "t700",
    "t600",
    "t500",
    "t400",
    "t300",
    "t250",
    "t200",
    "t150",
    "t100",
    "t50",
    "u1000",
    "u925",
    "u850",
    "u700",
    "u600",
    "u500",
    "u400",
    "u300",
    "u250",
    "u200",
    "u150",
    "u100",
    "u50",
    "v1000",
    "v925",
    "v850",
    "v700",
    "v600",
    "v500",
    "v400",
    "v300",
    "v250",
    "v200",
    "v150",
    "v100",
    "v50",
    "msl",
    "u10m",
    "v10m",
    "t2m",
]

ATMOS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def _aurora_history(previous: xr.DataArray, prediction: xr.DataArray) -> xr.DataArray:
    previous = previous.isel(lead_time=slice(-1, None))
    signature = coord_array_like(
        prediction,
        {
            "lead_time": np.concatenate(
                (previous.lead_time.values, prediction.lead_time.values)
            )
        },
    )
    a, _ = previous.e2s.to_torch()
    b, _ = prediction.e2s.to_torch()
    result = from_torch(
        torch.cat((a.to(b.device), b), dim=previous.get_axis_num("lead_time")),
        signature,
        name=prediction.name,
    )
    result.encoding = prediction.encoding.copy()
    return result


# Adapted from https://microsoft.github.io/aurora/example_era5.html
@check_optional_dependencies()
class Aurora(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """Aurora 0.25 degree global forecast model. This model consists of single
    auto-regressive model with a time-step size of 6 hours. This model operates on
    0.25 degree lat-lon grid (720, 1440) equirectangular grid with 4 surface-level
    variables, 5 atmospheric variables with 13 pressure levels and 3 static variables.

    Note
    ----
    This model uses the checkpoints from the original publication.
    For additional information see the following resources:

    - https://arxiv.org/abs/2405.13063
    - https://github.com/microsoft/aurora
    - https://huggingface.co/microsoft/aurora

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora model
    z : torch.Tensor
        Geopotential
    slt : torch.Tensor
        Soil type
    lsm : torch.Tensor
        Land sea mask

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2024 gpu:48gb
    provider:microsoft backend:pytorch
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        z: torch.Tensor,
        slt: torch.Tensor,
        lsm: torch.Tensor,
    ) -> None:
        super().__init__()

        self.model = core_model
        self.register_buffer("z", z)
        self.register_buffer("slt", slt)
        self.register_buffer("lsm", lsm)

        self.preds_idx = 0

    def input_coords(self) -> CoordinateSystem:
        """Declare the two-frame history on Aurora's native grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
                "variable": VARIABLES,
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg-south-pole-excluded",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate history and plan the next six-hour forecast without field data."""
        handshake_dataarray(input_coords, self.input_coords(), relative_lead_time=True)
        return coord_array_like(
            input_coords,
            {"lead_time": input_coords.lead_time.values[-1:] + np.timedelta64(6, "h")},
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://microsoft/aurora@refs%2Fpr%2F1",
            cache_options={
                "cache_storage": Package.default_cache("aurora"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""

        # Import the static variables: z, slt, lsm
        z = torch.from_numpy(
            np.load(package.resolve("static-npy/aurora-0.25-static-z.npy"))[:-1]
        )
        slt = torch.from_numpy(
            np.load(package.resolve("static-npy/aurora-0.25-static-slt.npy"))[:-1]
        )
        lsm = torch.from_numpy(
            np.load(package.resolve("static-npy/aurora-0.25-static-lsm.npy"))[:-1]
        )

        # Load 0.25 degrees resolution Aurora pretrained model
        aurora_model = package.resolve("aurora-0.25-pretrained.ckpt")
        # The pretrained version does not use LoRA.
        model = Aurora_model(use_lora=False)
        model.load_checkpoint_local(aurora_model)
        model.eval()

        return cls(model, z, slt, lsm)

    def _prepare_input(self, input: torch.Tensor, coords: CoordSystem) -> Batch:
        """Prepares input Batch"""
        len_atmos_levels = len(ATMOS_LEVELS)
        # Only len 1 time array is supported by Auroral model
        ts = (
            (coords["time"][0] + coords["lead_time"][-1]).astype("datetime64[s]")
            - np.datetime64("1970-01-01T00:00:00Z")
        ) / np.timedelta64(1, "s")
        time = datetime.fromtimestamp(ts, tz=timezone.utc)

        # input variable order: z, q, t, u, v, msl, u10m, v10m, t2m
        batch = Batch(
            surf_vars={
                # select time points `i` and `i - 1`
                "2t": input[:, 0, -2:, len_atmos_levels * 5 + 3],
                "10u": input[:, 0, -2:, len_atmos_levels * 5 + 1],
                "10v": input[:, 0, -2:, len_atmos_levels * 5 + 2],
                "msl": input[:, 0, -2:, len_atmos_levels * 5],
            },
            static_vars={
                "z": self.z,
                "slt": self.slt,
                "lsm": self.lsm,
            },
            atmos_vars={
                "t": input[:, 0, -2:, len_atmos_levels * 2 : len_atmos_levels * 3],
                "u": input[:, 0, -2:, len_atmos_levels * 3 : len_atmos_levels * 4],
                "v": input[:, 0, -2:, len_atmos_levels * 4 : len_atmos_levels * 5],
                "q": input[:, 0, -2:, len_atmos_levels : len_atmos_levels * 2],
                "z": input[:, 0, -2:, :len_atmos_levels],
            },
            metadata=Metadata(
                lat=torch.from_numpy(coords["lat"]),
                lon=torch.from_numpy(coords["lon"]),
                time=(time,),
                atmos_levels=tuple(int(level) for level in ATMOS_LEVELS),
                rollout_step=self.preds_idx,
            ),
        )

        return batch

    def _prepare_output(self, output: Batch, coords: CoordSystem) -> torch.Tensor:
        # retrieving tensor output from Batch format
        x = torch.cat(
            [
                output.atmos_vars["z"],
                output.atmos_vars["q"],
                output.atmos_vars["t"],
                output.atmos_vars["u"],
                output.atmos_vars["v"],
                output.surf_vars["msl"].unsqueeze(2),
                output.surf_vars["10u"].unsqueeze(2),
                output.surf_vars["10v"].unsqueeze(2),
                output.surf_vars["2t"].unsqueeze(2),
            ],
            dim=2,
        )
        x = x.view(-1, 1, *x.shape[1:])
        # According to https://github.com/microsoft/aurora/blob/main/aurora/rollout.py, the static variables of next step are from the previous step.
        self.z = output.static_vars["z"]
        self.slt = output.static_vars["slt"]
        self.lsm = output.static_vars["lsm"]

        return x

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:

        # output shape: [b,t,1,69,720,1440], batch, time, lead_time, variables, lat, lon
        out = torch.empty_like(x[:, :, :1])
        # The aurora model can only process one time per iteration, so loop if multiple
        for t in range(coords["time"].shape[0]):
            batch_coords = coords.copy()
            batch_coords["time"] = batch_coords["time"][t : t + 1]
            # x shape: [b,1,2,69,720,1440], batch, lead_time, variables, lat, lon
            input_batch = self._prepare_input(x[:, t : t + 1], batch_coords)
            # Convert tensor to Batch, atmos_vars is the first 65 variables, surf_vars is the last 4 variables of x
            output_batch = self.model(input_batch)
            # Convert Batch to tensor
            out[:, t : t + 1] = self._prepare_output(output_batch, coords)

        return out

    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict six hours ahead, without iterator hooks."""
        signature = self.output_coords(x)
        handshake_time(x)
        tensor, coords = x.e2s.to_torch()
        out = self._forward(tensor.to(self.z.device).clone(), coords)
        result = from_torch(out, signature, name=x.name)
        result.encoding = x.encoding.copy()
        return result

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the final input, then forecasts with hooks in original dimensions."""
        handshake_dataarray(x, runtime=True)
        handshake_time(x)
        self.output_coords(x)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=True)
        while True:
            history = self.front_hook(x.copy(deep=True))
            out = self.rear_hook(self(history))
            self.preds_idx += 1
            x = _aurora_history(history, out)
            yield out
