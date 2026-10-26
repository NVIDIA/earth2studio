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

"""Toy models for coupler tests and examples.

Ordinary :class:`~earth2studio.models.px.PrognosticModel` and
:class:`~earth2studio.models.dx.DiagnosticModel` implementations with trivial
arithmetic in place of networks, so every coupled value is hand-computable.
:class:`ToyPrognostic` is the configurable base; the two named models are the
atmosphere/ocean pair the examples use:

- :class:`ToyAtmosphere`: 6 h steps on a 32 x 64 grid, state ``z1000``, forcing
  ``sst`` at the step's input time::

      z1000 <- z1000 + 1 + gain * 0.1 * sst

- :class:`ToyOcean`: 48 h steps on a 16 x 32 grid, state ``sst``, forcing
  ``z1000:mean:48h`` over the step being taken::

      sst <- sst + gain * 0.01 * mean(z1000)

- :class:`ToyImpact`: a diagnostic turning ``z1000`` into ``ws10m`` as
  ``0.01 * z1000``.

No weights, downloads or GPU are needed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import xarray as xr

from earth2studio.data import Constant
from earth2studio.grids import LatLonGrid
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

ATMOS_SHAPE = (32, 64)
OCEAN_SHAPE = (16, 32)

Advance = Callable[[xr.DataArray, xr.DataArray | None], xr.DataArray]


def toy_grid(nlat: int, nlon: int) -> LatLonGrid:
    """Regular global latitude-longitude grid, poles included."""
    return LatLonGrid(
        latitude=np.linspace(90.0, -90.0, nlat),
        longitude=np.linspace(0.0, 360.0, nlon, endpoint=False),
    )


def toy_source(grid: LatLonGrid, value: float) -> Constant:
    """Constant source on ``grid``, for initial conditions and forcing."""
    coords = grid.coords(only_index=True)
    return Constant(
        OrderedDict(lat=np.asarray(coords["lat"]), lon=np.asarray(coords["lon"])),
        value,  # type: ignore[arg-type]
    )


def _plus_one(x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
    return x + 1.0


class ToyPrognostic(PrognosticMixin):
    """Configurable toy prognostic model with one input and at most one forcing slot.

    Parameters
    ----------
    variables : Sequence[str]
        State variables.
    forcing : Sequence[str], optional
        Forcing variables, by default none (``forcing_coords()`` is ``None``).
    forcing_lead : str | np.timedelta64, optional
        Lead time of the forcing frame each step needs, relative to the step's
        input time, by default ``"0h"``.
    step : str | np.timedelta64, optional
        Lead-time advance of one step, by default ``"6h"``.
    grid : LatLonGrid, optional
        Model grid, by default a 32 x 64 global grid.
    advance : Callable[[xr.DataArray, xr.DataArray | None], xr.DataArray], optional
        ``(x, forcing) -> y`` on field values; the lead time is relabelled
        afterwards. By default ``x + 1``.
    """

    def __init__(
        self,
        variables: Sequence[str],
        forcing: Sequence[str] = (),
        forcing_lead: str | np.timedelta64 = "0h",
        step: str | np.timedelta64 = "6h",
        grid: LatLonGrid | None = None,
        advance: Advance = _plus_one,
    ) -> None:
        self.variables = list(variables)
        self.forcing_variables = list(forcing)
        self.forcing_lead = _timedelta(forcing_lead)
        self.step_size = _timedelta(step)
        self.grid = grid or toy_grid(*ATMOS_SHAPE)
        self.advance = advance
        self.calls: list[tuple[str, list[int]]] = []

    def input_coords(self) -> CoordinateSystem:
        """One input slot: the state variables at lead ``0h`` on the model grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": np.array(self.variables),
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def forcing_coords(self) -> CoordinateSystem | None:
        """One forcing slot at ``forcing_lead``, or ``None`` without forcing."""
        if not self.forcing_variables:
            return None
        return coord_array_like(
            self.input_coords(),
            {
                "lead_time": np.array([self.forcing_lead]),
                "variable": np.array(self.forcing_variables),
            },
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """The input signature advanced by one step."""
        lead = np.asarray(input_coords["lead_time"].values)
        return coord_array_like(input_coords, {"lead_time": lead + self.step_size})

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
        if (forcing is None) != (not self.forcing_variables):
            raise ValueError("Forcing must be passed exactly when declared")
        if forcing is not None:
            self.calls.append(
                (
                    ",".join(str(v) for v in forcing["variable"].values),
                    [
                        int(v)
                        for v in np.asarray(forcing["lead_time"].values)
                        .astype("timedelta64[h]")
                        .astype(int)
                    ],
                )
            )
        y = self.advance(x, forcing)
        y = y.assign_coords(lead_time=x["lead_time"] + self.step_size)
        return y.transpose(*x.dims).assign_attrs(x.attrs)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> tuple[xr.DataArray, int]:
        """Return the first forecast and a step counter as the state."""
        return self._advance(x, forcing), 1

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray | None = None, state: int = 0
    ) -> tuple[xr.DataArray, int]:
        """Advance one step from the previous forecast."""
        return self._advance(y, forcing), state + 1

    def __call__(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> xr.DataArray:
        """Return the first forecast."""
        args = (x,) if forcing is None else (x, forcing)
        return self._default_call(*args)  # type: ignore[return-value]

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> Iterator[xr.DataArray]:
        """Iterate forecasts, taking new forcing through ``send``."""
        args = (x,) if forcing is None else (x, forcing)
        return self._default_create_iterator(*args)  # type: ignore[return-value]

    def to(self, device: object) -> ToyPrognostic:
        """Return the model; it holds no device state."""
        return self


def _timedelta(value: str | np.timedelta64) -> np.timedelta64:
    from earth2studio.nvcoupler._time import as_timedelta

    return as_timedelta(value)


def _forcing_value(forcing: xr.DataArray | None, label: str) -> xr.DataArray:
    assert forcing is not None  # noqa: S101 - declared forcing is always passed
    return forcing.isel(lead_time=-1, drop=True).sel(variable=label, drop=True)


class ToyAtmosphere(ToyPrognostic):
    """Fast toy prognostic: state ``z1000``, forced by ``sst`` at its input time."""

    def __init__(self, gain: float = 1.0, grid: LatLonGrid | None = None) -> None:
        self.gain = gain
        super().__init__(
            ["z1000"],
            forcing=["sst"],
            forcing_lead="0h",
            step="6h",
            grid=grid or toy_grid(*ATMOS_SHAPE),
            advance=self._rule,
        )

    def _rule(self, x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
        return x + 1.0 + self.gain * 0.1 * _forcing_value(forcing, "sst")


class ToyOcean(ToyPrognostic):
    """Slow toy prognostic: state ``sst``, forced by the 48 h mean of ``z1000``.

    The forcing is declared at lead ``+48h``: the step from ``T`` to ``T + 48h``
    uses the atmosphere's mean over ``[T, T + 48h)``, so the atmosphere advances
    through that window first.
    """

    def __init__(self, gain: float = 1.0, grid: LatLonGrid | None = None) -> None:
        self.gain = gain
        super().__init__(
            ["sst"],
            forcing=["z1000:mean:48h"],
            forcing_lead="48h",
            step="48h",
            grid=grid or toy_grid(*OCEAN_SHAPE),
            advance=self._rule,
        )

    def _rule(self, x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
        return x + self.gain * 0.01 * _forcing_value(forcing, "z1000:mean:48h")


class ToyImpact:
    """Toy diagnostic: ``ws10m = 0.01 * z1000`` on the atmosphere grid."""

    stochastic = False

    def __init__(self, grid: LatLonGrid | None = None) -> None:
        self.grid = grid or toy_grid(*ATMOS_SHAPE)
        self.calls = 0

    def input_coords(self) -> CoordinateSystem:
        """One input slot: ``z1000`` on the model grid."""
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": np.array(["z1000"])},
            dynamic=("batch",),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """The input signature with ``ws10m`` in place of ``z1000``."""
        return coord_array_like(input_coords, {"variable": np.array(["ws10m"])})

    def default_sources(self) -> None:
        """Recommend no source."""
        return None

    def to(self, device: object) -> ToyImpact:
        """Return the model; it holds no device state."""
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Scale ``z1000`` into ``ws10m``."""
        self.calls += 1
        return (0.01 * x).assign_coords(variable=np.array(["ws10m"]))
