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
import pandas as pd
import torch
import xarray as xr
from loguru import logger

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
from earth2studio.utils.type import CoordinateSystem, TimeArray

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    OptionalDependencyFailure("fuxi")
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore

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
    "tp:sum:6h",
]


@check_optional_dependencies()
class FuXi(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
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
    - https://huggingface.co/NickGeneva/earth_ai

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

    Badges
    ------
    region:global class:medium-range product:wind product:precip product:temp product:atmos year:2023
    gpu:40gb backend:onnx
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

    def input_coords(self) -> CoordinateSystem:
        """Return two six-hour inputs; precipitation is labelled ``tp:sum:6h``."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate relative history and advance the final lead by six hours."""
        if "lead_time" not in input_coords.coords:
            raise ValueError("Input lead_time coordinate is required")
        lead = np.asarray(input_coords.lead_time)
        if (
            input_coords.lead_time.dims != ("lead_time",)
            or lead.size != 2
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain two finite timedeltas")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
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
            "hf://NickGeneva/earth_ai/fuxi",
            cache_options={
                "cache_storage": Package.default_cache("fuxi"),
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
        coords: CoordinateSystem,
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
        x = x.clone()
        tp06_index = np.isin(coords["variable"].values, "tp:sum:6h")
        x[..., tp06_index, :, :] = torch.nan_to_num(x[..., tp06_index, :, :], nan=0)
        x[..., tp06_index, :, :] = torch.clip(
            x[..., tp06_index, :, :] * 1000, min=0, max=1000
        )

        # Flatten batch and time dim
        time_array = self._time_encoding(
            np.tile(coords["time"].values + coords["lead_time"].values[-1], x.shape[0])
        )
        # reshape, not view: x is the caller's tensor and may be non-contiguous
        x = x.reshape(-1, *x.shape[2:]).contiguous()

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
        output = output.reshape(-1, coords["time"].shape[0], *output.shape[1:])

        # Convert tp06 back to m
        output[..., tp06_index, :, :] = output[..., tp06_index, :, :] / 1000

        return output

    @batch_func()
    def _step(self, x: xr.DataArray, step: int = 0) -> xr.DataArray:
        self.output_coords(x)
        if "time" not in x.coords or x.time.dims != ("time",):
            raise ValueError("A one-dimensional time coordinate is required")
        if (
            not np.issubdtype(x.time.dtype, np.datetime64)
            or np.isnat(x.time.values).any()
        ):
            raise ValueError("time must contain finite datetimes")
        path = (
            self.ort_short_path
            if step < 20
            else self.ort_medium_path if step < 40 else self.ort_long_path
        )
        if self.ort._model_path != path:
            logger.warning(f"Time-step {step}, loading {path}")
            self.ort = create_ort_session(path, self.device)
        tensor, _ = x.e2s.to_torch()
        signature = coord_array_like(
            x, {"lead_time": x.lead_time.values + np.timedelta64(6, "h")}
        )
        out = from_torch(
            self._forward(tensor.to(self.device), x, self.ort), signature, name=x.name
        )
        out.encoding = x.encoding.copy()
        return out

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict one six-hour field with the short-range model, without hooks."""
        state = self._step(x)
        return state.isel(lead_time=slice(-1, None))

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        step = 0
        self.output_coords(x)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=False)
        while True:
            # The rear hook sees both returned history fields, with matching labels.
            x = self.rear_hook(self._step(self.front_hook(x.copy(deep=True)), step))
            step += 1
            yield x.isel(lead_time=slice(-1, None)).copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the latest input then cascaded six-hour forecasts."""
        yield from self._default_generator(x)
