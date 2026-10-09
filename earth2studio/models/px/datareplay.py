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

from collections.abc import Generator, Hashable
from copy import deepcopy

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource
from earth2studio.grids import CurvilinearGrid, GridDefinition, LatLonGrid, resolve_grid
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.type import CoordinateSystem, CoordSystem


class DataReplay(torch.nn.Module, PrognosticMixin):
    """Replay a data source through the prognostic model interface.

    The caller supplies successive source frames as forcing. The source is a
    recommendation for the driver; executing the model never fetches data.

    Parameters
    ----------
    source : DataSource | ForecastSource
        Source to replay.
    variable : str | list[str]
        Variables to fetch.
    domain_coords : CoordSystem | GridDefinition | str
        Spatial coordinates, grid definition, or registered grid expected from the source.
    step : np.timedelta64, optional
        Time between frames, by default np.timedelta64(6, "h")

    Badges
    ------
    region:global provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        source: DataSource | ForecastSource,
        variable: str | list[str],
        domain_coords: CoordSystem | GridDefinition | str,
        step: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> None:
        super().__init__()
        if not isinstance(step, np.timedelta64):
            raise TypeError("step must be an np.timedelta64")
        if np.isnat(step) or step <= np.timedelta64(0, "ns"):
            raise ValueError("step must be a positive duration")

        if isinstance(variable, str):
            variable = [variable]

        self.source = source
        self._step = step
        self._variable = np.asarray(variable).copy()
        grid: str | GridDefinition | None = None
        coordinates: dict[Hashable, np.ndarray] = {}
        if isinstance(domain_coords, (str, GridDefinition)):
            grid = domain_coords
        elif tuple(domain_coords) == ("lat", "lon"):
            lat, lon = domain_coords["lat"], domain_coords["lon"]
            grid = LatLonGrid(lat, lon) if lat.ndim == 1 else CurvilinearGrid(lat, lon)
        else:
            coordinates = {key: value for key, value in domain_coords.items()}
        definition = resolve_grid(grid) if isinstance(grid, str) else grid
        dims = definition.dims if definition is not None else tuple(coordinates)
        self._input_coords = coord_array(
            ("batch", "time", "lead_time", "variable", *dims),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": self._variable,
                **coordinates,
            },
            dynamic=("batch", "time"),
            grid=grid,
        )

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature with dynamic batch and
            time dimensions.
        """
        return self._input_coords.copy(deep=True)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature one source step ahead.
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.lead_time
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead.values - lead.values[-1]),
            self.input_coords(),
        )
        return coord_array_like(input_coords, {"lead_time": lead.values + self._step})

    def forcing_coords(self) -> CoordinateSystem:
        """Declare the next source frame relative to the current input.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature for the frame one configured time step
            ahead, which the caller supplies as forcing.
        """
        return self.output_coords(self.input_coords())

    def default_sources(self) -> tuple[None, DataSource | ForecastSource]:
        """Recommend the configured source for the caller-supplied replay slot.

        Returns
        -------
        tuple[None, DataSource | ForecastSource]
            No input recommendation, followed by the configured replay source.
        """
        return None, self.source

    @torch.inference_mode()
    def _forward(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        handshake_nonempty(x)
        handshake_time(x)
        signature = self.output_coords(x)
        handshake_nonempty(forcing)
        required_coords = set(self.forcing_coords().coords) | set(signature.dims)
        handshake_dataarray(
            forcing,
            signature.drop_vars(
                [name for name in signature.coords if name not in required_coords]
            ),
        )
        if not torch.isfinite(forcing.e2s.to_torch()[0]).all():
            raise ValueError("DataReplay forcing contains non-finite values")
        output = forcing.astype(x.dtype).copy(deep=True)
        output = output.assign_coords(signature.coords)
        output.name = x.name
        output.attrs = deepcopy({**forcing.attrs, **x.attrs})
        output.encoding = deepcopy(x.encoding)
        return output

    def __call__(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        """Publish the supplied next source frame without iterator hooks.

        Parameters
        ----------
        x : xr.DataArray
            Current fields matching ``input_coords()``.
        forcing : xr.DataArray
            Caller-supplied next frame matching ``output_coords(x)``.

        Returns
        -------
        xr.DataArray
            Owned copy of the supplied frame with the input dtype and metadata.
        """
        return self.initialize(x, forcing)[0]

    def initialize(
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        """Publish the first supplied forecast; replay has no private state.

        Parameters
        ----------
        x : xr.DataArray
            Initial fields matching ``input_coords()``.
        forcing : xr.DataArray
            Caller-supplied next frame matching ``forcing_coords()`` relative to
            the final input lead time. No data are fetched internally.

        Returns
        -------
        tuple[xr.DataArray, None]
            First forecast in the input dtype and ``None`` continuation state.
            Iterator hooks are not applied.
        """
        return self._forward(x, forcing), None

    def step(
        self, y: xr.DataArray, forcing: xr.DataArray, state: None
    ) -> tuple[xr.DataArray, None]:
        """Publish the next supplied forecast without fetching from the source.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast supplying the dtype and metadata for the next frame.
        forcing : xr.DataArray
            Next source frame, one model time step after ``y``.
        state : None
            Empty continuation state returned by ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, None]
            Next forecast and ``None``, without iterator hooks.
        """
        if state is not None:
            raise ValueError("DataReplay state must be None")
        return self.initialize(y, forcing)

    def create_iterator(
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        """Yield supplied forecasts, receiving each new source frame via ``send``.

        Parameters
        ----------
        x : xr.DataArray
            Initial fields matching ``input_coords()``.
        forcing : xr.DataArray
            First forecast frame, supplied by the caller.

        Yields
        ------
        xr.DataArray
            Forecasts beginning with the supplied frame. The rear hook runs
            before every yield and the front hook before each subsequent step.
        """
        yield from self._default_create_iterator(x, forcing)
