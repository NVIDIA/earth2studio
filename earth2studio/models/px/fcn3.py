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
import json
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import (
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
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    import torch_harmonics
    from makani.models.model_package import load_model_package
    from packaging.version import Version

    if Version(torch_harmonics.__version__) >= Version("0.8.1"):
        from torch_harmonics.disco import cuda_kernels_is_available

        _cuda_extension_available = cuda_kernels_is_available()
    else:
        from importlib.util import find_spec

        _cuda_extension_available = find_spec("disco_cuda_extension") is not None
except ImportError:
    OptionalDependencyFailure("fcn3")
    load_model_package = None
    _cuda_extension_available = False


VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
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
class FCN3(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """FourCastNet 3 advances global weather modeling by implementing a scalable,
    geometric machine learning (ML) approach to probabilistic ensemble forecasting.
    The approach is designed to respect spherical geometry and to accurately model the
    spatially correlated probabilistic nature of the problem, resulting in stable
    spectra and realistic dynamics across multiple scales.

    FourCastNet 3 is a global probabilistic prognostic model.
    It operates on a 0.25 degree lat-lon grid (both poles included)
    equirectangular grid with 72 variables.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2507.12144v2
    - https://arxiv.org/abs/2402.16845
    - https://huggingface.co/nvidia/fourcastnet3
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/fourcastnet3

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    variables : np.array, optional
        Variables associated with model, by default 72 variable model.
    seed : int, optional
        Seed of the underlying FCN3 model's random generators, by default 333

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2025 gpu:80gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        variables: np.array = np.array(VARIABLES),
        seed: int = 333,
    ):
        super().__init__()
        self.model = core_model
        self.variables = np.array(variables, copy=True)
        self.register_buffer("device_buffer", torch.empty(0))
        if "2d" in self.variables:
            self.variables[self.variables == "2d"] = "d2m"

        self.set_rng(reset=True, seed=seed)

    def __str__(self) -> str:
        return "fcn3"

    stochastic = True

    def set_rng(self, seed: int = 333, reset: bool = True) -> None:
        """Set the underlying FCN3 model's RNG

        Parameters
        ----------
        seed : int, optional
            Seed for the RNG, by default 333
        reset : bool, optional
            Whether to reset the state of the RNG, by default True
        """
        self.seed = seed
        self.model.set_rng(reset=reset, seed=seed)

    def input_coords(self) -> CoordinateSystem:
        """Declare one input frame on the registered 0.25 degree grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": self.variables,
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input signature and declare the next six-hour forecast."""
        handshake_dataarray(input_coords, self.input_coords(), relative_lead_time=True)
        lead = input_coords.lead_time.values
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(6, "h")}
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://nvidia/fourcastnet3@76ef0c60237e458b33196ba027134e27f3fc4538",
            cache_options={
                "cache_storage": Package.default_cache("fcn3"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, variables: list = VARIABLES
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        variables : list, optional
            Model variable override, by default VARIABLES for FCN3 72 channel

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        if not _cuda_extension_available:
            logger.warning(
                "torch-harmonics disco CUDA extension is not available.\n"
                "FCN3 run on GPU/CUDA will be slower.\n"
                "Please install torch-harmonics in the following way:\n"
                "export FORCE_CUDA_EXTENSION=1\n"
                "pip install --no-build-isolation torch-harmonics"
            )
        model = load_model_package(package)
        model.eval()

        # Load variables
        config_path = package.get("config.json")
        with open(config_path) as f:
            config = json.load(f)
            variables = config["channel_names"]

        variables = np.array(variables)

        return cls(model, variables=variables)

    def _get_internal_state(self, ensemble_index: int, time_index: int) -> torch.Tensor:
        """Get the internal RNG state for the given ensemble and time index

        Parameters
        ----------
        ensemble_index : int
            Ensemble index
        time_index : int
            Time index
        """
        return self._internal_noise_states[ensemble_index][time_index]

    def _set_internal_state(self, ensemble_index: int, time_index: int) -> None:
        """Set the internal RNG state for the given ensemble and time index

        Parameters
        ----------
        ensemble_index : int
            Ensemble index
        time_index : int
            Time index
        """
        self._internal_noise_states[ensemble_index][time_index] = (
            self.model.model.preprocessor.get_internal_state(tensor=True)
        )
        return

    def _reset_internal_state(self, num_ensemble: int, num_time: int) -> None:
        """Reset the internal RNG state for the given number of ensembles and time steps

        Parameters
        ----------
        num_ensemble : int
            Number of ensembles
        num_time : int
            Number of time steps
        """
        _internal_noise_states = [
            [None for _ in range(num_time)] for _ in range(num_ensemble)
        ]
        for i in range(num_ensemble):
            for j in range(num_time):
                self.model.model.preprocessor.update_internal_state(replace_state=True)
                _internal_noise_states[i][j] = (
                    self.model.model.preprocessor.get_internal_state(tensor=True)
                )
        self._internal_noise_states = _internal_noise_states

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        x = x.clone().squeeze(2)

        # For normalization, we will use both z-normalization and minmax normalization
        # The center/scale and min/max should be constructed to only apply to the correct variables, respectively.
        # See `load_model` for more details.
        for j, _ in enumerate(coords["batch"]):
            for i, t in enumerate(coords["time"]):
                # Get the noise state for the batch index
                noise_state = self._get_internal_state(j, i)
                self.model.model.preprocessor.set_internal_state(noise_state)

                # https://github.com/NVIDIA/modulus-makani/blob/933b17d5a1ebfdb0e16e2ebbd7ee78cfccfda9e1/makani/third_party/climt/zenith_angle.py#L197
                # Requires time zone data
                t = [
                    datetime.fromisoformat(dt.isoformat() + "+00:00")
                    for dt in timearray_to_datetime(t + coords["lead_time"])
                ]
                with torch.autocast(
                    device_type=x.device.type,
                    dtype=(
                        torch.bfloat16 if _cuda_extension_available else torch.float32
                    ),
                ):
                    x[j, i : i + 1] = self.model(
                        x[j, i : i + 1], t, normalized_data=False, replace_state=False
                    )
                self._set_internal_state(j, i)

        x = x.unsqueeze(2)
        return x

    @batch_func()
    def _step(self, x: xr.DataArray, reset: bool = False) -> xr.DataArray:
        signature = self.output_coords(x)
        handshake_time(x)
        tensor, coords = x.e2s.to_torch()
        tensor = tensor.to(self.device_buffer.device)
        if reset:
            self._reset_internal_state(x.sizes["batch"], x.sizes["time"])
        result = from_torch(self._forward(tensor, coords), signature, name=x.name)
        result.encoding = x.encoding.copy()
        return result

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict one six-hour field with freshly initialized core noise states."""
        return self._step(x, reset=True)

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        handshake_dataarray(x, runtime=True)
        handshake_time(x)
        self.output_coords(x)
        yield x.copy(deep=True)
        reset = True
        while True:
            if self.front_hook is not self._default_hook:
                x = self.front_hook(x.copy(deep=True))
            x = self.rear_hook(self._step(x, reset=reset))
            reset = False
            yield x

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the input then six-hour forecasts, retaining per-sample noise state."""
        yield from self._default_generator(x)
