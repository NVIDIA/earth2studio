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
from collections.abc import Generator, Iterator
from datetime import datetime
from typing import TypeVar

import numpy as np
import xarray as xr

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore
import torch
from aurora import Aurora, Batch, Metadata

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

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


# adapted from https://microsoft.github.io/aurora/example_era5.html
class Aurora_cls(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Aurora class"""

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

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        self.device = torch.ones(1).device  # Hack to get default device
        self.preds_idx = 0

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(
        self,
        input_coords: CoordSystem,
    ) -> CoordSystem:
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
        output_coords = self._output_coords.copy()

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
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )

        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "/ivan/earth2studio/earth_ai_aurora",
            cache_options={
                "cache_storage": Package.default_cache("aurora"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""

        # Import the static variables: z, slt, lsm
        static_path = package.resolve("static.nc")
        static_vars_ds = xr.open_dataset(static_path, engine="netcdf4")
        z = torch.from_numpy(static_vars_ds["z"].values[0])[:-1]
        slt = torch.from_numpy(static_vars_ds["slt"].values[0])[:-1]
        lsm = torch.from_numpy(static_vars_ds["lsm"].values[0])[:-1]

        # Load 0.25 degrees resolution Aurora pretrained model
        aurora_model = package.resolve("aurora-0.25-pretrained.ckpt")
        # The pretrained version does not use LoRA.
        model = Aurora(use_lora=False)
        model.load_checkpoint_local(aurora_model)
        model.eval()

        return cls(model, z, slt, lsm)

    def _prepare_input(self, input: torch.Tensor, coords: CoordSystem) -> Batch:
        """Prepares input Batch"""
        len_atmos_levels = len(ATMOS_LEVELS)
        ts = (
            (coords["time"][-1] + coords["lead_time"][-1]).astype("datetime64[s]")
            - np.datetime64("1970-01-01T00:00:00Z")
        ) / np.timedelta64(1, "s")
        time = datetime.utcfromtimestamp(ts)

        # input variable order: z, q, t, u, v, msl, u10m, v10m, t2m
        batch = Batch(
            surf_vars={
                # select time points `i` and `i - 1`
                "2t": input[:, -2:, len_atmos_levels * 5 + 3],
                "10u": input[:, -2:, len_atmos_levels * 5 + 1],
                "10v": input[:, -2:, len_atmos_levels * 5 + 2],
                "msl": input[:, -2:, len_atmos_levels * 5],
            },
            static_vars={
                "z": self.z,
                "slt": self.slt,
                "lsm": self.lsm,
            },
            atmos_vars={
                "t": input[:, -2:, len_atmos_levels * 2 : len_atmos_levels * 3],
                "u": input[:, -2:, len_atmos_levels * 3 : len_atmos_levels * 4],
                "v": input[:, -2:, len_atmos_levels * 4 : len_atmos_levels * 5],
                "q": input[:, -2:, len_atmos_levels : len_atmos_levels * 2],
                "z": input[:, -2:, :len_atmos_levels],
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
        # x shape: [1,2,69,720,1440], batch, time, channels, lat, lon
        x = self._prepare_input(x, coords)
        # Convert tensor to Batch, atmos_vars is the first 65 channels, surf_vars is the last 4 channels of x
        x = self.model(x)
        # Batch output
        x = self._prepare_output(x, coords)
        # Convert Batch to tensor
        # x shape: [1,1,69,720,1440], batch, time, channels, lat, lon
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

        x = self._forward(x, coords)

        return x, output_coords

    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        self.output_coords(coords)

        yield x[:, 1:], coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            init_x = x[:, 1:].clone()

            # Forward pass
            x = self._forward(x, coords)

            self.preds_idx = self.preds_idx + 1

            coords["lead_time"] = (
                coords["lead_time"]
                + self.output_coords(self.input_coords())["lead_time"]
            )
            # Concat the step now and prediction for next step
            x = torch.cat([init_x, x], dim=1)
            x = x.clone()

            # Rear hook for first predicted step
            coords_out = coords.copy()
            coords_out["lead_time"] = coords["lead_time"][-1]
            x[:, 1:], coords_out = self.rear_hook(x[:, 1:], coords_out)

            yield x[:, 1:], coords_out

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
