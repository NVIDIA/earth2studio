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


# Pangu Weather License
# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import create_ort_session
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.checkpoint import bind_checkpoint_state
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
    OptionalDependencyFailure("pangu")
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore

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


@dataclass
class _PanguCheckpointState:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    step: int = 0


# Adapted from https://raw.githubusercontent.com/ecmwf-lab/ai-models-panguweather/main/ai_models_panguweather/model.py
class PanguBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Pangu base class"""

    def __init__(self) -> None:
        super().__init__()
        # Shape of pressure fields (var, level, lat, lon)
        self.pressure_shape = (5, 13, 721, 1440)
        self.n_pres = 65
        # Shape of surface variable fields
        self.surface_shape = (4, 721, 1440)

        self._time_step = np.timedelta64(6, "h")
        self.checkpoint = bind_checkpoint_state(_PanguCheckpointState())
        self.device = torch.ones(1).device  # Hack to get default device
        self.ort = None
        self._ort24_session: InferenceSession | None = None
        self._ort6_session: InferenceSession | None = None

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free Pangu input signature."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
            },
            dynamic=("batch",),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input signature and advance by this variant's time step."""
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(input_coords, {"lead_time": lead + self._time_step})

    def _restore_checkpoint_state(
        self, x: xr.DataArray
    ) -> tuple[dict[str, xr.DataArray], int, bool]:
        if (
            self.checkpoint.checkpoint_level == 2
            and self.checkpoint.checkpoint_state_loaded
            and self.checkpoint.tensors
        ):
            states = {}
            for key, tensor in self.checkpoint.tensors.items():
                metadata = deepcopy(self.checkpoint.metadata[key])
                signature = coord_array(
                    metadata["dims"],
                    metadata["coords"],
                    sizes=metadata["sizes"],
                    attrs=metadata["attrs"],
                )
                restored = from_torch(
                    tensor.to(self.device),
                    signature,
                    name=metadata["name"],
                    attrs=metadata["attrs"],
                )
                restored.encoding = metadata["encoding"]
                states[key] = restored
            return states, self.checkpoint.step, True
        return {"current": x}, 0, False

    def _save_checkpoint_state(
        self, states: dict[str, xr.DataArray], step: int
    ) -> None:
        self.checkpoint.tensors = {}
        self.checkpoint.metadata = {}
        if self.checkpoint.checkpoint_enabled and self.checkpoint.checkpoint_level == 2:
            for key, x in states.items():
                tensor, _ = x.e2s.to_torch()
                self.checkpoint.tensors[key] = (
                    tensor.detach().clone().to(self.checkpoint.device)
                )
                self.checkpoint.metadata[key] = deepcopy(
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
            self.checkpoint.step = step

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://NickGeneva/earth_ai/pangu",
            cache_options={
                "cache_storage": Package.default_cache("pangu"),
                "same_names": True,
            },
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
            for attr, path_attr in (
                ("_ort24_session", "ort24"),
                ("_ort6_session", "ort6"),
            ):
                if getattr(self, attr, None) is not None:
                    setattr(self, attr, None)
                    if getattr(self, "_eager_sessions", False):
                        setattr(
                            self,
                            attr,
                            create_ort_session(getattr(self, path_attr), device),
                        )

        return self

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        ort_session: InferenceSession,
    ) -> torch.Tensor:

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
            fields_pl = x[i, : self.n_pres].reshape(*self.pressure_shape).contiguous()
            fields_sfc = x[i, self.n_pres :].contiguous()

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

        return batch_output

    @batch_func()
    def _step(
        self, x: xr.DataArray, session: InferenceSession, hours: int | None = None
    ) -> xr.DataArray:
        signature = self.output_coords(x)
        if hours is not None:
            signature = coord_array_like(
                x, {"lead_time": x.lead_time.values + np.timedelta64(hours, "h")}
            )
        tensor, _ = x.e2s.to_torch()
        out = from_torch(
            self._forward(tensor.to(self.device), session), signature, name=x.name
        )
        out.encoding = x.encoding.copy()
        return out

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Advance one DataArray using this variant's shortest-step model."""
        handshake_nonempty(x)
        states, _, _ = self._restore_checkpoint_state(x)
        out = self._step(states["current"], self.ort)
        self._save_checkpoint_state({"current": out}, 0)
        return out

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        handshake_nonempty(x)
        states, step, restored = self._restore_checkpoint_state(x)
        handshake_nonempty(states["current"])
        self.output_coords(states["current"])
        hours = int(self._time_step / np.timedelta64(1, "h"))
        if hours < 24 and "day" not in states:
            states["day"] = states["current"].copy(deep=True)
        if hours == 3 and "six" not in states:
            states["six"] = states["current"].copy(deep=True)
        if not restored:
            self._save_checkpoint_state(states, step)
            yield states["current"].copy(deep=False)
        while True:
            step += 1
            elapsed = step * hours
            source = states["current"]
            session = self.ort
            stride = hours
            if hours < 24 and elapsed % 24 == 0:
                if self._ort24_session is None:
                    self._ort24_session = create_ort_session(self.ort24, self.device)
                source, session, stride = states["day"], self._ort24_session, 24
            elif hours == 3 and elapsed % 6 == 0:
                if self._ort6_session is None:
                    self._ort6_session = create_ort_session(self.ort6, self.device)
                source, session, stride = states["six"], self._ort6_session, 6
            out = self.rear_hook(
                self._step(self.front_hook(source.copy(deep=True)), session, stride)
            )
            states["current"] = out
            if hours == 3 and elapsed % 6 == 0:
                states["six"] = out.copy(deep=True)
            if hours < 24 and elapsed % 24 == 0:
                states["day"] = out.copy(deep=True)
            self._save_checkpoint_state(states, step)
            yield out.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial field and interleaved forecasts, resuming checkpoints next step."""
        yield from self._default_generator(x)


@check_optional_dependencies()
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
    - https://huggingface.co/NickGeneva/earth_ai

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

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2023 gpu:40gb
    backend:onnx
    """

    def __init__(
        self,
        ort_24hr: str,
    ):
        super().__init__()

        self.ort: ort.InferenceSession = create_ort_session(ort_24hr, self.device)
        self._time_step = np.timedelta64(24, "h")

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file = package.resolve("pangu_weather_24.onnx")
        return cls(onnx_file)


@check_optional_dependencies()
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
    - https://huggingface.co/NickGeneva/earth_ai

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
    eager_sessions : bool, optional
        Build the 24 hour session at construction instead of on first use in a
        rollout. Either way the session is built once and cached; eager keeps
        the build cost out of the first rollout step, by default False

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2023 gpu:40gb
    backend:onnx
    """

    def __init__(
        self,
        ort_24hr: str,
        ort_6hr: str,
        eager_sessions: bool = False,
    ):
        super().__init__()
        # Only require 6 hour to load session on construction
        self.ort: ort.InferenceSession = create_ort_session(ort_6hr, self.device)
        self.ort24 = ort_24hr
        self._eager_sessions = eager_sessions
        self._ort24_session: ort.InferenceSession | None = None
        if eager_sessions:
            self._ort24_session = create_ort_session(ort_24hr, self.device)
        self._time_step = np.timedelta64(6, "h")

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        eager_sessions: bool = False,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file_24 = package.resolve("pangu_weather_24.onnx")
        onnx_file_6 = package.resolve("pangu_weather_6.onnx")
        return cls(onnx_file_24, onnx_file_6, eager_sessions=eager_sessions)


@check_optional_dependencies()
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
    - https://huggingface.co/NickGeneva/earth_ai

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
    eager_sessions : bool, optional
        Build the 24 and 6 hour sessions at construction instead of on first
        use in a rollout. Either way each session is built once and cached;
        eager keeps the build cost out of the first rollout steps, by default
        False

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2023 gpu:40gb
    backend:onnx
    """

    def __init__(
        self,
        ort_24hr: str,
        ort_6hr: str,
        ort_3hr: str,
        eager_sessions: bool = False,
    ):
        super().__init__()
        # Only require 3 hour to load session on construction
        self.ort: ort.InferenceSession = create_ort_session(ort_3hr, self.device)
        self.ort24 = ort_24hr
        self.ort6 = ort_6hr
        self._eager_sessions = eager_sessions
        self._ort24_session: ort.InferenceSession | None = None
        self._ort6_session: ort.InferenceSession | None = None
        if eager_sessions:
            self._ort24_session = create_ort_session(ort_24hr, self.device)
            self._ort6_session = create_ort_session(ort_6hr, self.device)
        self._time_step = np.timedelta64(3, "h")

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        eager_sessions: bool = False,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        onnx_file_24 = package.resolve("pangu_weather_24.onnx")
        onnx_file_6 = package.resolve("pangu_weather_6.onnx")
        onnx_file = package.resolve("pangu_weather_3.onnx")
        return cls(onnx_file_24, onnx_file_6, onnx_file, eager_sessions=eager_sessions)
