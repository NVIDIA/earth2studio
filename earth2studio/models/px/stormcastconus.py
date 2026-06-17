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

from collections.abc import Generator

import numpy as np
import torch

from earth2studio.data import DataSource, ForecastSource
from earth2studio.models.batch import batch_func
from earth2studio.models.nn.stormcastconus import (
    CONDITIONING_VARIABLES,
    VARIABLES,
    StormCastCONUSBase,
)
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem


@check_optional_dependencies()
class StormCastCONUS(StormCastCONUSBase, PrognosticMixin):
    """StormCast-CONUS generative convection-allowing model for the full CONUS domain.

    - High-resolution (3km) HRRR state over the Continental United States (99 vars)
    - High-resolution land-sea mask and orography invariants
    - Coarse resolution (25km) global state (26 vars)

    The high-resolution grid is the HRRR Lambert conformal projection.
    Coarse-resolution inputs are regridded to the HRRR grid internally.

    This class provides standard (non-SDA) inference. For score-based data
    assimilation (observation-guided diffusion), use
    :class:`earth2studio.models.da.StormCastCONUSSDA` instead.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2408.10958
    - https://huggingface.co/nvidia/stormcast-v1-era5-hrrr

    Parameters
    ----------
    diffusion_model : torch.nn.Module
        Configured diffusion model (e.g. a ``_SplitModelWrapper`` instance
        created by :meth:`load_model`).
    means : torch.Tensor
        Per-channel mean for normalising the high-resolution state.
    stds : torch.Tensor
        Per-channel standard deviation for normalising the high-resolution state.
    invariants : torch.Tensor
        Static invariant fields (e.g. land-sea mask, orography).
    conditioning_means : torch.Tensor
        Per-channel mean for normalising the low-resolution conditioning.
    conditioning_stds : torch.Tensor
        Per-channel standard deviation for normalising the low-resolution conditioning.
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, by default (17, 1041)
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, by default (3, 1795)
    variables : np.ndarray, optional
        High-resolution variable names, by default ``np.array(VARIABLES)``.
    conditioning_variables : np.ndarray, optional
        Low-resolution conditioning variable names, by default
        ``np.array(CONDITIONING_VARIABLES)``.
    conditioning_data_source : DataSource or ForecastSource or None, optional
        Data source for global conditioning. Required for inference, by default None.
    sampler_args : dict, optional
        Overrides for the EDM sampler/scheduler. Recognised keys:
        ``sigma_min``, ``sigma_max``, ``rho`` (scheduler), and
        ``S_churn``, ``S_min``, ``S_max``, ``S_noise`` (solver).
        Unspecified keys use sensible defaults.
    num_diffusion_steps : int, optional
        Number of diffusion sampling steps, by default 18.
    batch_size : int, optional
        Maximum batch size processed in one forward pass, by default 1.
    use_amp : bool, optional
        Whether to run the diffusion forward pass under ``torch.autocast`` with
        bfloat16, by default True.
    clamp_values : bool, optional
        Whether to apply reflectivity clipping in ``_forward``. When the model is
        loaded via :meth:`load_model`, this flag is also forwarded to the internal
        ``_SplitModelWrapper`` to enable per-variable physical-minimum clamping,
        by default True.
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        conditioning_means: torch.Tensor,
        conditioning_stds: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (17, 1041),
        hrrr_lon_lim: tuple[int, int] = (3, 1795),
        variables: np.ndarray = np.array(VARIABLES),
        conditioning_variables: np.ndarray = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | ForecastSource | None = None,
        sampler_args: dict[str, float | int] | None = None,
        num_diffusion_steps: int = 18,
        batch_size: int = 1,
        use_amp: bool = True,
        clamp_values: bool = True,
    ):
        super().__init__(
            diffusion_model=diffusion_model,
            means=means,
            stds=stds,
            invariants=invariants,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            hrrr_lat_lim=hrrr_lat_lim,
            hrrr_lon_lim=hrrr_lon_lim,
            variables=variables,
            conditioning_variables=conditioning_variables,
            conditioning_data_source=conditioning_data_source,
            sampler_args=sampler_args,
            num_diffusion_steps=num_diffusion_steps,
            batch_size=batch_size,
            use_amp=use_amp,
            clamp_values=clamp_values,
        )

    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run the prognostic model one step forward.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system.

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized.
        """
        return super().__call__(x, coords, obs=None)

    @batch_func()
    def create_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Create a generator for autoregressive rollout.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system after each time step.

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized.
        """
        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCastCONUS has been called without initializing the model's "
                "conditioning_data_source"
            )

        yield x, coords

        try:
            while True:
                x, coords = self.front_hook(x, coords)
                x, coords = self(x, coords)
                x, coords = self.rear_hook(x, coords)
                yield x, coords
        except GeneratorExit:
            pass
