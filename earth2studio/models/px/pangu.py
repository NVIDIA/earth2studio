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


# Pangu Weather License
# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import TypeVar

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


# adapted from https://raw.githubusercontent.com/ecmwf-lab/ai-models-panguweather/main/ai_models_panguweather/model.py
class PanguBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Pangu base class"""

    def __init__(self) -> None:
        super().__init__()
        # Shape of pressure fields (var, level, lat, lon)
        self.pressure_shape = (5, 13, 721, 1440)
        self.n_pres = 65
        # Shape of surface variable fields
        self.surface_shape = (4, 721, 1440)

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        self.device = torch.ones(1).device  # Hack to get default device
        self.ort = None

    @property
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of prognostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords

    @property
    def output_coords(self) -> CoordSystem:
        """Ouput coordinate system of prognostic model, time dimension should contain
        time-delta objects representing time-step of the model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package("hf://NickGeneva/earth_ai/pangu")

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
                "onnxruntime (onnxruntime-gpu) is required for Pangu. See model install notes for details.\n"
                + "https://nvidia.github.io/earth2studio/userguide/about/install.html#model-dependencies"
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

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        ort_session: InferenceSession,
        lead_time: np.array,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords.copy()
        output_coords["batch"] = coords["batch"]
        output_coords["lead_time"] = lead_time + coords["lead_time"]

        # Ref: https://onnxruntime.ai/docs/api/python/api_summary.html
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

        batch_output = torch.zeros_like(x)
        x = x.squeeze(1)
        # Process batches (model is single batch)
        for i in range(x.shape[0]):
            # Forward pass
            fields_pl = x[i, : self.n_pres].resize(*self.pressure_shape)
            fields_sfc = x[i, self.n_pres :]

            bind_input("input", fields_pl)
            bind_input("input_surface", fields_sfc)
            output = bind_output("output", like=fields_pl)
            output_sfc = bind_output("output_surface", like=fields_sfc)
            ort_session.run_with_iobinding(binding)
            output_tensor = torch.cat(
                [
                    output.view(-1, self.pressure_shape[-2], self.pressure_shape[-1]),
                    output_sfc,
                ],
                dim=0,
            ).contiguous()
            batch_output[i, 0] = output_tensor

        return batch_output, output_coords

    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        raise NotImplementedError

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


class Pangu24(PanguBase):
    """Pangu Weather 24 hour model. This model consists of single auto-regressive
    model with a time-step size of 24 hours. Pangu Weather operates on 0.25 degree
    lat-lon grid (south-pole including) equirectangular grid with 69
    atmospheric/surface variables.

    Note
    ----
    This model uses the ONNX checkpoints from the original publication.
    For additional information see the following resources:

    - https://doi.org/10.1038/s41586-023-06185-3
    - https://github.com/198808xc/Pangu-Weather

    Note
    ----
    To avoid ONNX init session overhead of this model we recommend setting the default
    Pytorch device to the correct target prior to model construction.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ort_24hr : str
        Path to Pangu 24 hour onnx file
    """

    def __init__(
        self,
        ort_24hr: str,
    ):
        super().__init__()

        self.ort: ort.InferenceSession = PanguBase.create_ort_session(
            ort_24hr, self.device
        )
        self._output_coords["lead_time"] = np.array([np.timedelta64(24, "h")])

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file = package.get("pangu_weather_24.onnx")
        return cls(onnx_file)

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs 24 hour prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 24 hours in the future
        """
        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        return self._forward(x, coords, self.ort, self.output_coords["lead_time"])

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        yield x, coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            x, coords = self._forward(
                x, coords, self.ort, np.array([np.timedelta64(24, "h")])
            )

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield x, coords.copy()


class Pangu6(PanguBase):
    """Pangu Weather 6 hour model. This model consists of two underlying auto-regressive
    models with a time-step size of 24 hours and 6 hours. These two models are
    interweaved during prediction. Pangu Weather operates on 0.25 degree lat-lon grid
    (south-pole including) equirectangular grid with 69 atmospheric/surface variables.

    Note
    ----
    This model uses the ONNX checkpoints from the original publication.
    For additional information see the following resources:

    - https://doi.org/10.1038/s41586-023-06185-3
    - https://github.com/198808xc/Pangu-Weather

    Note
    ----
    To avoid ONNX init session overhead of this model we recommend setting the default
    Pytorch device to the correct target prior to model construction.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ort_24hr : str
        Path to Pangu 24 hour onnx file
    ort_6hr : str
        Path to Pangu 6 hour onnx file
    """

    def __init__(
        self,
        ort_24hr: str,
        ort_6hr: str,
    ):
        super().__init__()
        # Only require 6 hour to load session on construction
        self.ort: ort.InferenceSession = PanguBase.create_ort_session(
            ort_6hr, self.device
        )
        self.ort24 = ort_24hr
        self._output_coords["lead_time"] = np.array([np.timedelta64(6, "h")])

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file_24 = package.get("pangu_weather_24.onnx")
        onnx_file_6 = package.get("pangu_weather_6.onnx")
        return cls(onnx_file_24, onnx_file_6)

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

        return self._forward(x, coords, self.ort, self.output_coords["lead_time"])

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        # Load other sessions (note .to() does not impact these)
        ort24 = PanguBase.create_ort_session(self.ort24, self.device)

        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        yield x, coords

        while True:
            x0 = x.clone()
            coords0 = coords.copy()
            # Three 6-hour steps
            for i in range(3):
                x, coords = self.front_hook(x, coords)
                x, coords = self._forward(
                    x, coords, self.ort, np.array([np.timedelta64(6, "h")])
                )
                x, coords = self.rear_hook(x, coords)
                yield x, coords.copy()
            # 24 hour step
            x, coords = self.front_hook(x0, coords0)
            x, coords = self._forward(
                x0, coords, ort24, np.array([np.timedelta64(24, "h")])
            )
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()


class Pangu3(PanguBase):
    """Pangu Weather 3 hour model. This model consists of three underlying
    auto-regressive models with a time-step size of 24, 6 and 3 hours. These three
    models are interweaved during prediction. Pangu Weather operates on 0.25 degree
    lat-lon grid (south-pole including) equirectangular grid with 69 atmospheric/surface
    variables.

    Note
    ----
    This model uses the ONNX checkpoints from the original publication.
    For additional information see the following resources:

    - https://doi.org/10.1038/s41586-023-06185-3
    - https://github.com/198808xc/Pangu-Weather

    Note
    ----
    To avoid ONNX init session overhead of this model we recommend setting the default
    Pytorch device to the correct target prior to model construction.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ort_24hr : str
        Path to Pangu 24 hour onnx file
    ort_6hr : str
        Path to Pangu 6 hour onnx file
    ort_3hr : str
        Path to Pangu 3 hour onnx file
    """

    def __init__(
        self,
        ort_24hr: str,
        ort_6hr: str,
        ort_3hr: str,
    ):
        super().__init__()
        # Only require 3 hour to load session on construction
        self.ort: ort.InferenceSession = PanguBase.create_ort_session(
            ort_3hr, self.device
        )
        self.ort24 = ort_24hr
        self.ort6 = ort_6hr
        self._output_coords["lead_time"] = np.array([np.timedelta64(3, "h")])

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file_24 = package.get("pangu_weather_24.onnx")
        onnx_file_6 = package.get("pangu_weather_6.onnx")
        onnx_file = package.get("pangu_weather_3.onnx")
        return cls(onnx_file_24, onnx_file_6, onnx_file)

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs 3 hour prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 3 hours in the future
        """
        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        return self._forward(x, coords, self.ort, self.output_coords["lead_time"])

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        # Load other sessions (note that .to() does not impact these)
        ort24 = PanguBase.create_ort_session(self.ort24, self.device)
        ort6 = PanguBase.create_ort_session(self.ort6, self.device)

        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        yield x, coords

        while True:
            x0 = x.clone()  # Used with 24 hour model
            coord0 = coords.copy()

            x1 = x.clone()  # Used with 6 hour model
            coords1 = coords.copy()

            # Single 3-hour step
            x, coords = self.front_hook(x, coords)
            x, coords = self._forward(
                x, coords, self.ort, np.array([np.timedelta64(3, "h")])
            )
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

            # Three 6-hour steps
            for i in range(3):
                x, coords = self.front_hook(x1, coords1)
                x, coords = self._forward(
                    x, coords, ort6, np.array([np.timedelta64(6, "h")])
                )
                x, coords = self.rear_hook(x, coords)
                yield x, coords.copy()

                x1 = x.clone()
                coords1 = coords.copy()

                # Single 3-hour step
                x, coords = self.front_hook(x, coords)
                x, coords = self._forward(
                    x, coords, self.ort, np.array([np.timedelta64(3, "h")])
                )
                x, coords = self.rear_hook(x, coords)
                yield x, coords.copy()

            # 24 hour step
            x, coords = self.front_hook(x0, coord0)
            x, coords = self._forward(
                x0, coords, ort24, np.array([np.timedelta64(24, "h")])
            )
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()
