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
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.utils import validate_observation_fields
from earth2studio.models.px.stormcastconus import StormCastCONUS
from earth2studio.utils import handshake_nonempty
from earth2studio.utils.type import CoordinateSystem, FrameSchema


class StormCastConusSDA(torch.nn.Module, AutoModelMixin):
    """StormCast-CONUS with score-based observation guidance and explicit state.

    Wraps the CONUS diffusion model, retaining its cropped projected grid, sampler,
    normalization, and observation mapping. Conditioning and observations both
    describe the output valid time, one hour after the input. Observations use
    physical units and the wrapped model's ``time_tolerance`` window (unfiltered
    when unset). Pass ``None`` for an unguided forecast.

    Install the ``stormcast-conus`` extra. Configure domain and SDA settings through
    :meth:`load_model` or the wrapped :class:`StormCastCONUS` constructor.

    Parameters
    ----------
    model : StormCastCONUS
        Configured CONUS model, including its observation noise and sampler settings.

    Badges
    ------
    region:na class:data-assimilation class:nowcasting product:atmos product:insitu
    year:2026 provider:nvidia backend:pytorch
    """

    stochastic = True

    def __init__(self, model: StormCastCONUS) -> None:
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        """Device of the wrapped model's normalization buffers."""
        return self.model.means.device

    def input_coords(self) -> CoordinateSystem:
        """Return the initial HRRR field signature on the configured CONUS grid."""
        return self.model.input_coords()

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate input coordinates and return the one-hour-ahead signature."""
        return self.model.output_coords(input_coords)

    def forcing_coords(self) -> tuple[CoordinateSystem, FrameSchema]:
        """Return output-time conditioning and optional observation declarations.

        Observations require time, latitude, longitude, variable, and physical-unit
        observation columns. The wrapped model filters its configured time window.
        """
        return self.model.forcing_coords(), FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=float),
                "lon": np.empty(0, dtype=float),
                "variable": self.model.variables.copy(),
                "observation": np.empty(0, dtype=float),
            }
        )

    def default_sources(
        self,
    ) -> tuple[DataSource, DataSource | ForecastSource, None]:
        """Recommend HRRR and global conditioning; leave observations caller-chosen."""
        return (*self.model.default_sources(), None)

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed new analyses without changing explicit continuation state.

        Parameters
        ----------
        seed : int
            Seed for latent and sampler noise; global RNG is unaffected.
        reset : bool, optional
            Restart the stream, or initialize only if absent when False.
        """
        self.model.set_rng(seed, reset=reset)

    def __call__(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Compute one analysis without hooks; see :meth:`initialize` for slots.

        Parameters
        ----------
        x : xr.DataArray
            Initial fields matching ``input_coords()``.
        conditioning : xr.DataArray
            Gridded conditioning at the output valid time (+1 hour).
        obs : pd.DataFrame or None, optional
            Physical-unit observations; None disables guidance.

        Returns
        -------
        xr.DataArray
            One-hour-ahead analysis.
        """
        return self.initialize(x, conditioning, obs)[0]

    def initialize(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> tuple[xr.DataArray, tuple[int, dict[str, torch.Tensor]]]:
        """Compute the first analysis and return serializable seed/RNG state.

        Inputs follow the declared background, conditioning, observation order.
        Required conditioning is already on the model grid at output valid time.
        Neither inputs nor observations are modified; hooks are iterator-only.

        Parameters
        ----------
        x : xr.DataArray
            Initial fields matching ``input_coords()``.
        conditioning : xr.DataArray
            Conditioning on the model grid at the next output valid time.
        obs : pd.DataFrame or None, optional
            Observations matching ``forcing_coords()[1]``, or no guidance.

        Returns
        -------
        tuple[xr.DataArray, tuple[int, dict[str, torch.Tensor]]]
            First analysis and serializable sampler seed/RNG snapshots.
        """
        handshake_nonempty(x)
        if not isinstance(conditioning, xr.DataArray):
            raise ValueError("conditioning must be a DataArray")
        if obs is not None:
            validate_observation_fields(obs, list(self.forcing_coords()[1]))
        if self.model._rng_seed is None:
            self.set_rng(int(torch.randint(2**31, ())))
        y = self.model._predict(x, conditioning, obs)
        if self.model._rng_seed is None or self.model._rng_states is None:
            raise RuntimeError("Sampling stream was not initialized")
        return y, (self.model._rng_seed, deepcopy(self.model._rng_states))

    def step(
        self,
        y: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None,
        state: tuple[int, dict[str, torch.Tensor]],
    ) -> tuple[xr.DataArray, tuple[int, dict[str, torch.Tensor]]]:
        """Advance the prior analysis with fresh output-time forcing and RNG state.

        State is the seed and RNG snapshots returned by initialization or stepping.
        Replaying the same arguments reproduces the output without changing inputs,
        state, or the stream used to initialize other trajectories. No hooks run.

        Parameters
        ----------
        y : xr.DataArray
            Previous analysis.
        conditioning : xr.DataArray
            Conditioning at the next output valid time on the model grid.
        obs : pd.DataFrame or None
            Fresh observations, or no guidance.
        state : tuple[int, dict[str, torch.Tensor]]
            Sampler seed and RNG snapshots from the preceding computation.

        Returns
        -------
        tuple[xr.DataArray, tuple[int, dict[str, torch.Tensor]]]
            Next analysis and updated state; arguments remain unchanged.
        """
        previous = self.model._rng_seed, self.model._rng_states
        self.model._rng_seed, self.model._rng_states = deepcopy(state)
        try:
            return self.initialize(y, conditioning, obs)
        finally:
            self.model._rng_seed, self.model._rng_states = previous

    def create_iterator(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> Generator[xr.DataArray, tuple[xr.DataArray, pd.DataFrame | None], None]:
        """Yield the first computed analysis, then send ``(conditioning, obs)``.

        Each forcing pair describes the next output time. The wrapped model's rear
        hook runs before every yield, and its front hook before subsequent steps.
        Hook results feed continuation; required conditioning is never reused.

        Parameters
        ----------
        x : xr.DataArray
            Initial fields matching ``input_coords()``.
        conditioning : xr.DataArray
            Conditioning for the first output valid time.
        obs : pd.DataFrame or None, optional
            Observations for the first output, or no guidance.

        Yields
        ------
        xr.DataArray
            Complete hourly analyses, starting at +1 hour.
        """
        y, state = self.initialize(x, conditioning, obs)
        while True:
            y = self.model.rear_hook(y)
            forcing = yield y
            if not isinstance(forcing, tuple) or len(forcing) != 2:
                raise ValueError("Send (conditioning, obs) in forcing_coords() order")
            y, state = self.step(self.model.front_hook(y), *forcing, state=state)

    @classmethod
    def load_default_package(cls) -> Package:
        """Return the pinned StormCast-CONUS checkpoint package."""
        return StormCastCONUS.load_default_package()

    @classmethod
    def load_model(cls, package: Package, **model_kwargs: Any) -> "StormCastConusSDA":
        """Load CONUS weights and configure observation-guided sampling.

        Parameters
        ----------
        package : Package
            StormCast-CONUS checkpoint package.
        **model_kwargs : Any
            Domain, sampler, and SDA settings forwarded to ``StormCastCONUS``.
            Use ``solver`` to select a numerical integration method and
            ``solver_options`` to customize its settings. Set forecast steps with
            ``num_diffusion_steps`` and guided analysis steps with
            ``num_sda_diffusion_steps``; do not pass ``num_steps`` in solver options.

        Returns
        -------
        StormCastConusSDA
            Configured assimilation model.
        """
        return cls(StormCastCONUS.load_model(package, **model_kwargs))
