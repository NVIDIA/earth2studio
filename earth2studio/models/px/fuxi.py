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

from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import TypeVar

import numpy as np
import pandas as pd
from loguru import logger

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import create_ort_session
from earth2studio.utils import check_extra_imports, handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem, TimeArray

VARIABLES = [
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
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
    "t2m",
    "u10m",
    "v10m",
    "msl",
    "tp06",
]


@check_extra_imports("fuxi", [ort])
class FuXi(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """FuXi weather model consists of three auto-regressive U-net transfomer models with
    a time-step size of 6 hours. The three models are trained to predict short (5days),
    medium (10 days) and longer (15 days) forecasts respectively. FuXi operates on
    0.25 degree lat-lon grid (south-pole including) equirectangular grid with 70
    atmospheric/surface variables. This model uses two time-steps as an input.

    Note
    ----
    This model uses the ONNX checkpoint from the original publication repository. For
    additional information see the following resources:

    - https://arxiv.org/abs/2306.12873
    - https://github.com/tpys/FuXi

    Note
    ----
    To avoid ONNX init session overhead of this model we recommend setting the default
    Pytorch device to the correct target prior to model construction.

    Parameters
    ----------
    ort_short : str
        Path to FuXi short model onnx file
    ort_medium : str
        Path to FuXi medium model onnx file
    ort_long : str
        Path to FuXi long model onnx file
    """

    def __init__(
        self,
        ort_short: str,
        ort_medium: str,
        ort_long: str,
    ) -> None:
        super().__init__()

        self.device = torch.ones(1).device  # Hack to get default device

        self.ort_short_path = ort_short
        self.ort_medium_path = ort_medium
        self.ort_long_path = ort_long
        # Load short model into memory
        self.ort = create_ort_session(ort_short, self.device)

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
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
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
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            handshake_dim(test_coords, key, i)
            if key != "batch" and key != "time":
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            input_coords["lead_time"] + output_coords["lead_time"]
        )
        output_coords["lead_time"] = output_coords["lead_time"][1:]
        return output_coords

    def to(self, device: str | torch.device | int) -> PrognosticModel:
        """Move model (and default ORT session) to device"""
        device = torch.device(device)
        if device.index is None:
            if device.type == "cuda":
                device = torch.device(device.type, torch.cuda.current_device())
            else:
                device = torch.device(device.type, 0)

        super().to(device)

        if device != self.device:
            self.device = device
            # Move base ort session
            if self.ort is not None:
                model_path = self.ort._model_path
                del self.ort
                self.ort = create_ort_session(model_path, device)

        return self

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://NickGeneva/earth_ai/fuxi",
            cache_options={
                "cache_storage": Package.default_cache("fuxi"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("fuxi", [ort])
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""

        # Short model
        onnx_short = package.resolve("short.onnx")
        package.open("short")
        # Medium model
        onnx_medium = package.resolve("medium.onnx")
        package.open("medium")
        # Long model
        onnx_long = package.resolve("long.onnx")
        package.open("long")

        return cls(onnx_short, onnx_medium, onnx_long)

    def _time_encoding(self, time_array: TimeArray) -> torch.Tensor:
        """FuXi Generating time embedding

        Parameters
        ----------
        time_array : TimeArray
            Time numpy array, from input coordinate system, of size [t]

        Returns
        -------
        torch.Tensor
            Time embedding array of size [t, 12]
        """
        time_deltas = np.array(
            [np.timedelta64(-6, "h"), np.timedelta64(0, "h"), np.timedelta64(6, "h")]
        )
        time_array = np.array(time_array[:, None] + time_deltas[None])

        pd_array = [pd.Period(date, "h") for date in time_array.reshape(-1)]
        hour_array = np.array([dt.hour / 24 for dt in pd_array]).reshape(-1, 3)
        day_array = np.array([dt.day_of_year / 366 for dt in pd_array]).reshape(-1, 3)

        temb = np.stack([day_array, hour_array], axis=-1)
        embedding = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1).reshape(
            -1, 12
        )

        return torch.FloatTensor(embedding).to(self.device)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        ort_session: InferenceSession,
    ) -> tuple[torch.Tensor, CoordSystem]:

        output_coords = self.output_coords(coords)

        # Ref https://onnxruntime.ai/docs/api/python/api_summary.html
        binding = ort_session.io_binding()

        def bind_input(name: str, input: torch.Tensor) -> None:
            input = input.contiguous()
            binding.bind_input(
                name=name,
                device_type=self.device.type,
                device_id=self.device.index,
                element_type=np.float32,
                shape=tuple(input.shape),
                buffer_ptr=input.data_ptr(),
            )

        def bind_output(name: str, like: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(like).contiguous()
            binding.bind_output(
                name=name,
                device_type=self.device.type,
                device_id=self.device.index,
                element_type=np.float32,
                shape=tuple(out.shape),
                buffer_ptr=out.data_ptr(),
            )
            return out

        # FuXi ONNX Input
        # name: input
        # tensor: float32[1,2,70,721,1440]
        # name: temb
        # tensor: float32[1,12]

        # FuXi ONNX Output
        # name: output (for short model its 15379)
        # tensor: float32[1,ScatterNDoutput_dim_1,70,721,1440]

        # Convert tp06 to mm
        # https://github.com/tpys/FuXi/blob/9292fe0692156a01cd3d62bcb427cc3798cf8add/make_era5_input.py#L19-L22
        tp06_index = np.isin(coords["variable"], "tp06")
        x[..., tp06_index, :, :] = torch.nan_to_num(x[..., tp06_index, :, :], nan=0)
        x[..., tp06_index, :, :] = torch.clip(
            x[..., tp06_index, :, :] * 1000, min=0, max=1000
        )

        # Flatten batch and time dim
        time_array = self._time_encoding(
            np.tile(coords["time"] + coords["lead_time"][-1], x.shape[0])
        )
        x = x.view(-1, *x.shape[2:])

        # Not sure if FuXi supports batching atm
        output = torch.empty_like(x)
        for b in range(x.shape[0]):
            bind_input("input", x[b : b + 1])
            bind_input("temb", time_array[b : b + 1])

            output_bind = ort_session.get_outputs()[0].name
            out = bind_output(output_bind, like=output[b : b + 1])

            ort_session.run_with_iobinding(binding)
            output[b : b + 1] = out

        # Reshape to batch and time dimension
        output = output.view(-1, coords["time"].shape[0], *output.shape[1:])

        # Convert tp06 back to m
        output[..., tp06_index, :, :] = output[..., tp06_index, :, :] / 1000

        return output, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs short prognostic model 1 step.

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

        if self.ort._model_path != self.ort_short_path:
            logger.warning("Loading short range model")
            self.ort = create_ort_session(self.ort_short_path, self.device)

        output, out_coords = self._forward(x, coords, self.ort)
        output = output[:, :, 1:]

        return output, out_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        self.output_coords(coords)

        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield x[:, :, 1:], coords_out

        step = 0
        while True:
            # Cascade models for longer roll outs
            if step == 0 and self.ort._model_path != self.ort_short_path:
                logger.warning(f"Time-step {step}, loading short range model")
                self.ort = create_ort_session(self.ort_short_path, self.device)
            elif step == 20:
                logger.warning(f"Time-step {step}, loading medium range model")
                self.ort = create_ort_session(self.ort_medium_path, self.device)
            elif step == 40:
                logger.warning(f"Time-step {step}, loading long range model")
                self.ort = create_ort_session(self.ort_long_path, self.device)
            step += 1

            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            out, out_coords = self._forward(x, coords, self.ort)

            # Rear hook
            out, out_coords = self.rear_hook(out, out_coords)

            # Yield current output
            output = out[:, :, 1:]
            yield output, out_coords.copy()

            # Use output as next input ([t-1, t] -> [t, t+1])
            x = out
            coords["lead_time"] = (
                coords["lead_time"]
                + self.output_coords(self.input_coords())["lead_time"]
            )

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
