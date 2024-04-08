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

import os
from collections import OrderedDict
from typing import Generator, Iterator, TypeVar

import numpy as np

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "msl",
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
]


class FengWu(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """FengWu (operational) weather model consists of single auto-regressive model with
    a time-step size of 6 hours. FengWu operates on 0.25 degree lat-lon grid (south-pole
    including) equirectangular grid with 69 atmospheric/surface variables. This model
    uses two time-steps as an input.

    Note
    ----
    This model uses the ONNX checkpoint from the original publication repository. This
    checkpoint is a operational version to the one used in the paper which requires less
    variables. For additional information see the following resources:

    - https://arxiv.org/abs/2304.02948
    - https://github.com/OpenEarthLab/FengWu

    Note
    ----
    To avoid ONNX init session overhead of this model we recommend setting the default
    Pytorch device to the correct target prior to model construction.

    Parameters
    ----------
    ort : str
        Path to FengWu 6 hour onnx file
    center : torch.Tensor
        Model variable center normalization tensor of size [69]
    scale : torch.Tensor
        Model variable scale normalization tensor of size [69]
    """

    def __init__(
        self,
        ort: str,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()

        self.device = torch.ones(1).device  # Hack to get default device
        self.ort = self.create_ort_session(ort, self.device)

        self.register_buffer("center", center.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("scale", scale.unsqueeze(-1).unsqueeze(-1))

    input_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "lead_time": np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "lead_time": np.array([np.timedelta64(6, "h")]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

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
                self.ort = self.create_ort_session(model_path, device)

        return self

    @staticmethod
    def create_ort_session(
        onnx_file: str,
        device: torch.device = torch.device("cpu", 0),
    ) -> InferenceSession:
        """Create ORT session on specified device

        Parameters
        ----------
        onnx_file : str
            ONNX file
        device : torch.device, optional
            Device for session to run on, by default "cpu"

        Returns
        -------
        ort.InferenceSession
            ORT inference session
        """
        if ort is None:
            raise ImportError(
                "onnxruntime (onnxruntime-gpu) is required for FengWu. "
                + "Install the [fengwu] optional dependencies"
            )
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1

        # That will trigger a FileNotFoundError
        os.stat(onnx_file)
        if device.type == "cuda":
            if device.index is None:
                device_index = torch.cuda.current_device()
            else:
                device_index = device.index

            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_index,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]

        ort_session = ort.InferenceSession(
            onnx_file,
            sess_options=options,
            providers=providers,
        )

        return ort_session

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package("hf://NickGeneva/earth_ai/fengwu")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file = package.get("fengwu_v1.onnx")
        global_center = torch.Tensor(np.load(package.get("global_means.npy")))
        global_std = torch.Tensor(np.load(package.get("global_stds.npy")))
        return cls(onnx_file, global_center, global_std)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        ort_session: InferenceSession,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords.copy()
        output_coords["batch"] = coords["batch"]
        output_coords["lead_time"] = (
            coords["lead_time"][1:] + output_coords["lead_time"]
        )

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

        x = (x - self.center) / self.scale  # Normalize
        x = x.view(x.shape[0], -1, 721, 1440)  # Concat time-steps
        # Forward pass, fengwu onnx supports batched
        bind_input("input", x)
        output = bind_output("output", like=x)
        ort_session.run_with_iobinding(binding)

        # ONNX model outputs two time-steps, take the first
        output_tensor = output[:].contiguous()
        x = self.scale * output_tensor[:, :69].unsqueeze(1) + self.center  # UnNormalize
        return x, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs 6 hour prognostic model 1 step.

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
        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        return self._forward(x, coords, self.ort)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        out = x[:, 1:]
        out_coords = coords.copy()
        out_coords["lead_time"] = out_coords["lead_time"][1:]
        yield out, out_coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            out, out_coords = self._forward(x, coords, self.ort)

            # Rear hook
            out, out_coords = self.rear_hook(out, out_coords)

            # Update inputs for next time-step
            x = torch.cat([x[:, 1:], out], dim=1)
            coords["lead_time"] = np.array(
                [coords["lead_time"][-1], out_coords["lead_time"][-1]]
            )

            yield out, out_coords.copy()

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
