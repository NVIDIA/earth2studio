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

"""Reference baselines for the scorecard, run as ordinary campaigns.

Two classic skill baselines, both expressed as prognostic models on the
Earth2Studio 1.0 protocol so the eval recipe's forecast runner, online
scoring and exports apply unchanged:

* :class:`PersistenceBaseline` -- the forecast is the initial condition at
  every lead time.  A thin loader around
  :class:`earth2studio.models.px.Persistence` on the registered ERA5 grid,
  so a campaign can select it with ``model.architecture`` and pass the
  scored variable list through ``model.load_args``.
* :class:`ClimatologyForecast` -- the forecast at each lead is the
  climatology at the valid time (day of year x hour of day).  The
  climatology is a *forcing slot* of the model: the forecast runner fetches
  it at every step from ``forcing_sources[0]`` (for example
  :class:`earth2studio.data.WB2Climatology`) and ``predownload.py`` caches
  it into ``forcing0.zarr`` through the runner's ``data_requests``, so no
  remote fetch happens inside the inference loop.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.grids import GridDefinition
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.type import CoordinateSystem

#: The shared verification grid: ERA5 0.25 degree, 721 x 1440, latitude 90 -> -90.
ERA5_GRID = "latlon-0.25deg"


def _variables(variables: list[str] | None, name: str) -> list[str]:
    if not variables:
        raise ValueError(
            f"{name} needs model.load_args.variables "
            "(e.g. variables: ${output.variables})."
        )
    return [str(v) for v in variables]


class PersistenceBaseline:
    """Loader for :class:`earth2studio.models.px.Persistence` on the ERA5 grid.

    Provides the ``load_default_package`` / ``load_model`` classmethods the
    recipe's ``load_prognostic`` expects.
    """

    @classmethod
    def load_default_package(cls) -> None:
        """No package: persistence has no weights."""
        return None

    @classmethod
    def load_model(
        cls,
        package: object = None,
        variables: list[str] | None = None,
        grid: str | GridDefinition = ERA5_GRID,
    ) -> Any:
        """Build the persistence model for ``variables`` on ``grid``.

        Parameters
        ----------
        package : object, optional
            Ignored; persistence has no weights.
        variables : list[str], optional
            Variables the baseline carries (``model.load_args.variables``).
        grid : str | GridDefinition, optional
            Registered grid name or definition, by default the ERA5 grid.

        Returns
        -------
        Persistence
            The identity prognostic model.
        """
        from earth2studio.models.px import Persistence

        return Persistence(
            variable=_variables(variables, "PersistenceBaseline"), domain_coords=grid
        )


class ClimatologyForecast(torch.nn.Module, PrognosticMixin):
    """Prognostic-protocol baseline that forecasts the climatology.

    The input slot is the analysis at lead zero on the grid; the forcing slot
    (:meth:`forcing_coords`) is the climatology of the same variables at
    lead ``dt`` on the same grid.  The forecast runner fetches the forcing at
    every step's valid time, and :meth:`initialize` / :meth:`step` return it
    as the forecast, so the model itself holds no state and never touches a
    data source.

    Parameters
    ----------
    variable : list[str]
        Variables the baseline emits.
    dt : np.timedelta64, optional
        Forecast step, by default 6 hours.
    grid : str | GridDefinition, optional
        Registered grid name or definition, by default the ERA5 grid.
    """

    def __init__(
        self,
        variable: list[str],
        dt: np.timedelta64 = np.timedelta64(6, "h"),
        grid: str | GridDefinition = ERA5_GRID,
    ) -> None:
        super().__init__()
        if np.isnat(dt) or dt <= np.timedelta64(0, "s"):
            raise ValueError("ClimatologyForecast requires a positive time step")
        self._variable = [str(v) for v in variable]
        self._dt = np.timedelta64(dt)
        self._grid = grid
        self._input_coords = coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self._variable),
            },
            dynamic=("batch",),
            grid=grid,
        )

    def __str__(self) -> str:
        return "climatology"

    @classmethod
    def load_default_package(cls) -> None:
        """No package: the climatology comes from the forcing source."""
        return None

    @classmethod
    def load_model(
        cls,
        package: object = None,
        variables: list[str] | None = None,
        step_hours: int = 6,
        grid: str | GridDefinition = ERA5_GRID,
    ) -> ClimatologyForecast:
        """Build the baseline for ``variables``.

        Parameters
        ----------
        package : object, optional
            Ignored; the baseline has no weights.
        variables : list[str], optional
            Variables the baseline emits (``model.load_args.variables``).
        step_hours : int, optional
            Forecast step in hours, by default 6.
        grid : str | GridDefinition, optional
            Registered grid name or definition, by default the ERA5 grid.

        Returns
        -------
        ClimatologyForecast
            The baseline model.
        """
        return cls(
            variable=_variables(variables, "ClimatologyForecast"),
            dt=np.timedelta64(int(step_hours), "h"),
            grid=grid,
        )

    @property
    def dt(self) -> np.timedelta64:
        """Forecast step."""
        return self._dt

    def input_coords(self) -> CoordinateSystem:
        """Input signature: the analysis at lead zero on the grid.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray signature.
        """
        return self._input_coords.copy()

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output signature: the final input lead time advanced by ``dt``.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input signature or DataArray to check and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature.
        """
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        final = input_coords.isel(lead_time=slice(-1, None))
        return coord_array_like(
            final.assign_coords(
                lead_time=final.lead_time.variable.copy(data=lead[-1:] + self._dt)
            )
        )

    def forcing_coords(self) -> CoordinateSystem:
        """Declare the climatology at the next valid time as a forcing slot.

        Returns
        -------
        CoordinateSystem
            The input signature at lead time ``dt``.  The runner fetches the
            slot at ``dt`` past the latest forecast, that is at every step's
            valid time.
        """
        return coord_array_like(self._input_coords, {"lead_time": np.array([self._dt])})

    def default_sources(self) -> tuple[DataSource | ForecastSource | None, ...]:
        """Recommend no initial-condition source and WB2 climatology forcing.

        Returns
        -------
        tuple[None, WB2Climatology]
            Nothing for the analysis slot (the campaign supplies it) and the
            WeatherBench2 ERA5 hourly climatology for the forcing slot.
        """
        from earth2studio.data import WB2Climatology

        return None, WB2Climatology(cache=False, verbose=False)

    def _forecast(
        self, template: xr.DataArray, climatology: xr.DataArray
    ) -> xr.DataArray:
        """Place the climatology field into ``template``'s layout and coordinates."""
        handshake_nonempty(climatology)
        field = climatology.isel(lead_time=slice(-1, None)).assign_coords(
            lead_time=template.lead_time.values
        )
        field, _ = xr.broadcast(field, template)
        if dict(field.sizes) != dict(template.sizes):
            raise ValueError(
                "Climatology forcing does not match the forecast layout: got "
                f"{dict(climatology.sizes)}, expected {dict(template.sizes)}"
            )
        return template.copy(data=field.transpose(*template.dims).data.copy())

    def __call__(self, x: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
        """Forecast one step: the climatology at the next valid time.

        Parameters
        ----------
        x : xr.DataArray
            Analysis matching ``input_coords()``.
        climatology : xr.DataArray
            Climatology matching ``forcing_coords()``.

        Returns
        -------
        xr.DataArray
            The climatology at lead ``dt`` in the analysis's layout.
        """
        return self._default_call(x, climatology)  # type: ignore[return-value]

    @torch.inference_mode()
    def initialize(
        self, x: xr.DataArray, climatology: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        """Return the first forecast and no continuation state.

        Parameters
        ----------
        x : xr.DataArray
            Analysis matching ``input_coords()``.
        climatology : xr.DataArray
            Climatology at lead ``dt`` matching ``forcing_coords()``.

        Returns
        -------
        tuple[xr.DataArray, None]
            First forecast and empty state.
        """
        handshake_nonempty(x)
        signature = self.output_coords(x)
        template = x.isel(lead_time=slice(-1, None)).assign_coords(signature.coords)
        template.attrs = dict(signature.attrs)
        for key in (
            "earth2studio_kind",
            "earth2studio_schema_version",
            "earth2studio_dynamic_dims",
        ):
            template.attrs.pop(key, None)
        return self._forecast(template, climatology), None

    @torch.inference_mode()
    def step(
        self, y: xr.DataArray, climatology: xr.DataArray, state: None
    ) -> tuple[xr.DataArray, None]:
        """Advance one step: the climatology at the next valid time.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast.
        climatology : xr.DataArray
            Climatology at ``dt`` past ``y``'s lead time.
        state : None
            Empty continuation state returned by initialization.

        Returns
        -------
        tuple[xr.DataArray, None]
            Next forecast and empty state.
        """
        if state is not None:
            raise ValueError("ClimatologyForecast state must be None")
        handshake_nonempty(y)
        handshake_time(y, "lead_time")
        template = y.assign_coords(lead_time=y.lead_time + self._dt)
        return self._forecast(template, climatology), None

    def create_iterator(
        self, x: xr.DataArray, climatology: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | None, None]:
        """Yield the climatology at every forecast valid time.

        Parameters
        ----------
        x : xr.DataArray
            Analysis matching ``input_coords()``.
        climatology : xr.DataArray
            Climatology at lead ``dt`` matching ``forcing_coords()``.

        Yields
        ------
        xr.DataArray
            Forecast at each step; the caller sends the next step's
            climatology through ``send``.
        """
        return self._default_create_iterator(x, climatology)  # type: ignore[return-value]
