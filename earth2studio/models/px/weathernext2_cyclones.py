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

import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.weathernext2_cyclones_mini import (
    WeatherNext2CyclonesMini,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from weathernext.utils import checkpoint
    from weathernext.weathernext2 import fgn
except ImportError:
    OptionalDependencyFailure("weathernext")
    checkpoint = None
    fgn = None


@check_optional_dependencies()
class WeatherNext2Cyclones(WeatherNext2CyclonesMini):
    """WeatherNext 2 Cyclones operational medium-range forecast model.

    This wrapper uses Google DeepMind's operational 0.25 degree
    ``WeatherNextCyclones_<2025`` checkpoint family. These are the models that
    ran during the 2025 Atlantic hurricane season. Four trained checkpoint
    members are available; ``load_model`` selects member 1 by default.

    The model requires two input states, valid at ``-6h`` and ``0h`` lead time,
    and predicts 6 hours forward per model call. Cyclone tracking can be enabled
    with ``track_cyclones=True`` to accumulate WeatherNext's tropical cyclone
    diagnostics in the ``cyclone_tracks`` property.

    Note
    ----
    The operational model requires an NVIDIA H100 GPU. To avoid JAX
    preallocating GPU memory and use the CUDA virtual memory management
    allocator, set these variables before importing JAX or Earth2Studio:

    .. code-block:: console

        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_PYTHON_CLIENT_ALLOCATOR=vmm

    For more information see the following references:

    - https://github.com/google-deepmind/weathernext#provided-pretrained-models
    - https://docs.jax.dev/en/latest/gpu_memory_allocation.html

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ckpt : fgn.CheckPoint
        Model checkpoint containing weights.
    land_sea_mask : np.ndarray
        Land-sea mask on the WeatherNext grid.
    geopotential_at_surface : np.ndarray
        Surface geopotential on the WeatherNext grid.
    seed : int, optional
        Initial random seed for the stochastic FGN noise generator, by default 0.
    jit_compile : bool, optional
        JIT-compile the model forward pass, by default True.
    track_cyclones : bool, optional
        Accumulate tropical cyclone tracks in the ``cyclone_tracks`` property,
        by default False.

    Badges
    ------
    region:global class:medium-range product:wind product:precip product:temp product:atmos
    product:ocean year:2025 gpu:80gb provider:google backend:jax
    """

    MODEL_NAME = "WeatherNextCyclones"
    PARAMS_PATH = "params/WeatherNextCyclones_<2025_model{checkpoint_member}.npz"
    SAMPLE_PATH = (
        "dataset/source-hres_forecast_init-2024-10-07 00:00:00_"
        "res-0.25_levels-13_steps-01.nc"
    )

    @classmethod
    def _params_path(cls, checkpoint_member: int) -> str:
        if checkpoint_member not in range(1, 5):
            raise ValueError("checkpoint_member must be an integer from 1 through 4")
        return cls.PARAMS_PATH.format(checkpoint_member=checkpoint_member)

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        seed: int = 0,
        jit_compile: bool = True,
        track_cyclones: bool = False,
        checkpoint_member: int = 1,
    ) -> PrognosticModel:
        """Load the operational prognostic model from a package.

        Parameters
        ----------
        package : Package
            Package to load model from.
        seed : int, optional
            Initial random seed for the stochastic FGN noise generator, by default 0.
        jit_compile : bool, optional
            JIT-compile the model forward pass, by default True.
        track_cyclones : bool, optional
            Accumulate tropical cyclone tracks in the ``cyclone_tracks`` property,
            by default False.
        checkpoint_member : int, optional
            Operational checkpoint member from 1 through 4, by default 1.

        Returns
        -------
        PrognosticModel
            Prognostic model.
        """
        params_path = package.resolve(cls._params_path(checkpoint_member))
        with open(params_path, "rb") as f:
            ckpt = checkpoint.load(f, fgn.CheckPoint)

        sample_input = xr.load_dataset(package.resolve(cls.SAMPLE_PATH))
        return cls(
            ckpt,
            sample_input["land_sea_mask"].values,
            sample_input["geopotential_at_surface"].values,
            seed=seed,
            jit_compile=jit_compile,
            track_cyclones=track_cyclones,
        )
