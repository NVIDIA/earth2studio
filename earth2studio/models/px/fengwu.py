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
from typing import TypeVar

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.models.utils import create_ort_session
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    OptionalDependencyFailure("fengwu")
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore

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


@check_optional_dependencies()
class FengWu(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
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
    - https://huggingface.co/NickGeneva/earth_ai

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

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2023 gpu:40gb
    backend:onnx
    """

    def __init__(
        self,
        ort: str,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()

        self.device = torch.ones(1).device  # Hack to get default device
        self.ort = create_ort_session(ort, self.device)

        self.register_buffer("center", center.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("scale", scale.unsqueeze(-1).unsqueeze(-1))

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free two-step input signature."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
            },
            dynamic=("batch",),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate relative history and advance the final lead by six hours."""
        handshake_dataarray(input_coords, self.input_coords(), relative_lead_time=True)
        lead = np.asarray(input_coords.lead_time)
        return coord_array_like(
            input_coords, {"lead_time": lead[-1:] + np.timedelta64(6, "h")}
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
                self.ort = create_ort_session(model_path, device)

        return self

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://NickGeneva/earth_ai/fengwu",
            cache_options={
                "cache_storage": Package.default_cache("fengwu"),
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
        onnx_file = package.resolve("fengwu_v1.onnx")
        global_center = torch.Tensor(np.load(package.open("global_means.npy")))
        global_std = torch.Tensor(np.load(package.open("global_stds.npy")))
        return cls(onnx_file, global_center, global_std)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        ort_session: InferenceSession,
    ) -> torch.Tensor:

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
        # reshape, not view: x is the caller's tensor and may be non-contiguous
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])  # Concat time-steps
        # Forward pass, fengwu onnx supports batched
        bind_input("input", x)
        output = bind_output("output", like=x)
        binding.synchronize_inputs()
        ort_session.run_with_iobinding(binding)
        binding.synchronize_outputs()

        # ONNX model outputs two time-steps, take the first
        output_tensor = output[:].contiguous()
        x = self.scale * output_tensor[:, :69].unsqueeze(1) + self.center  # UnNormalize
        return x

    @batch_func()
    def _step(self, x: xr.DataArray) -> xr.DataArray:
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        out = from_torch(
            self._forward(tensor.to(self.device), self.ort), signature, name=x.name
        )
        out.encoding = x.encoding.copy()
        return out

    def _advance_history(self, x: xr.DataArray, out: xr.DataArray) -> xr.DataArray:
        previous, _ = x.isel(lead_time=slice(-1, None)).e2s.to_torch()
        future, _ = out.e2s.to_torch()
        signature = coord_array_like(
            out,
            {
                "lead_time": np.concatenate(
                    (x.lead_time.values[-1:], out.lead_time.values)
                )
            },
        )
        state = from_torch(
            torch.cat(
                (previous.to(future.device), future), dim=x.get_axis_num("lead_time")
            ),
            signature,
        )
        state.encoding = out.encoding.copy()
        return state

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict one six-hour DataArray from two input fields."""
        return self._step(x)

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        handshake_dataarray(x, runtime=True)
        self.output_coords(x)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=False)
        while True:
            x = self.front_hook(x.copy(deep=True))
            out = self.rear_hook(self._step(x))
            x = self._advance_history(x, out)
            yield out.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the latest input followed by six-hour forecasts."""
        yield from self._default_generator(x)
