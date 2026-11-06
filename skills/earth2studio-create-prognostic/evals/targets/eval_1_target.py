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

"""Identity prognostic model that returns input unchanged.

This is a minimal prognostic model for testing purposes. It simply returns
the input data unchanged while incrementing the lead_time by 6 hours.
"""

from collections.abc import Generator
from typing import cast

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

VARIABLES = ["t2m", "u10m", "v10m", "msl"]


class IdentityModel(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Identity prognostic model that returns input unchanged.

    A minimal prognostic model for testing that passes through the input
    field unchanged while correctly incrementing lead_time.

    Parameters
    ----------
    core_model : torch.nn.Module, optional
        Core model (not used, for interface compatibility).

    Note
    ----
    This model is primarily useful for testing Earth2Studio infrastructure.

    Badges
    ------
    region:global provider:nvidia backend:pytorch
    """

    def __init__(self, core_model: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.model = core_model
        self.register_buffer("device_buffer", torch.empty(0))
        self._time_step = np.timedelta64(6, "h")

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature for one frame on the 1 degree global grid,
            with a dynamic leading batch dimension.
        """
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": VARIABLES,
            },
            dynamic=("batch",),
            grid="latlon-1deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input and plan the next absolute lead time.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free output signature six hours after the input lead time.
        """
        if "lead_time" not in input_coords.coords:
            raise ValueError("lead_time is required")
        lead = input_coords.lead_time.values
        if (
            input_coords.lead_time.dims != ("lead_time",)
            or lead.size != 1
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain one finite timedelta")
        # Validate against the signature using the relative (zero-based) lead time
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": lead[-1:] + self._time_step}
        )

    def default_sources(self) -> None:
        """Recommend no input data source.

        Returns
        -------
        None
            No source recommendation; the caller supplies the initial condition.
        """
        return None

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package (empty for identity model).
        """
        return Package(".")

    @classmethod
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Model package (not used for identity model).

        Returns
        -------
        PrognosticModel
            Loaded model instance.
        """
        return cls(core_model=None)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute the first forecast without iterator hooks.

        Parameters
        ----------
        x : xr.DataArray
            Initial field on the 1 degree grid.

        Returns
        -------
        xr.DataArray
            Input values with lead time advanced six hours.
        """
        return cast(xr.DataArray, self._default_call(x))

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
        """Compute the first forecast and empty continuation state without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Initial field, moved to the model device at the core boundary.

        Returns
        -------
        tuple[xr.DataArray, None]
            Forecast and state; this model needs only its preceding output.
        """
        return self._forward(x), None

    def step(self, y: xr.DataArray, state: None) -> tuple[xr.DataArray, None]:
        """Advance a preceding forecast without modifying it or applying hooks.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast matching the output signature.
        state : None
            Empty state returned alongside that forecast.

        Returns
        -------
        tuple[xr.DataArray, None]
            Next forecast and empty state.
        """
        if state is not None:
            raise ValueError("This single-frame model expects state=None")
        return self._forward(y), None

    @torch.inference_mode()
    @batch_func()
    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        # Batch only the array-to-array numerical boundary, not (y, state).
        handshake_nonempty(x)
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        tensor = tensor.to(self.device_buffer.device).clone()
        return from_torch(tensor, signature)

    def create_iterator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        """Yield complete forecasts with hooks in original leading dimensions.

        Parameters
        ----------
        x : xr.DataArray
            Initial field. No forcing slots are declared for this model.

        Yields
        ------
        xr.DataArray
            Forecasts at +6h, +12h, and so on. The rear hook runs before every
            yield and the front hook before each subsequent step.
        """
        yield from self._default_create_iterator(x)
