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

import numpy as np
import torch
import xarray as xr
from earth2studio.models._array_utils import _resolve_domain

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.grids import GridDefinition
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.type import CoordinateSystem, CoordSystem


class DataReplay(torch.nn.Module, DataArrayPrognosticMixin):
    """Replay a data source through the prognostic model interface.

    Data sources are queried at successive valid times. Forecast sources are queried
    at successive lead times from the initial time.

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
        self.step = step
        self._variable = np.asarray(variable).copy()
        dims, coordinates, grid = _resolve_domain(domain_coords)
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
        """Return the allocation-free signature with dynamic batch and time axes."""
        return self._input_coords.copy(deep=True)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the domain and return the signature one source step ahead."""
        self._require_time(input_coords)
        if "lead_time" not in input_coords.coords:
            raise ValueError("Input lead_time coordinate is required")
        lead = input_coords.lead_time
        if (
            lead.dims != ("lead_time",)
            or lead.size != 1
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("Input lead_time must contain one finite timedelta")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(input_coords, {"lead_time": lead.values + self.step})

    @staticmethod
    def _require_time(coords: CoordinateSystem) -> None:
        if "time" not in coords.coords or coords.sizes.get("time", 0) == 0:
            raise ValueError("DataReplay requires a non-empty time coordinate")
        if (
            coords.time.dims != ("time",)
            or not np.issubdtype(coords.time.dtype, np.datetime64)
            or np.isnat(coords.time).any()
        ):
            raise ValueError("DataReplay requires finite datetime time coordinates")

    @torch.inference_mode()
    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        fetched = fetch_data(
            self.source,
            time=signature.time,
            variable=self._variable,
            lead_time=signature.lead_time,
            device=tensor.device,
        )
        # Validate the source domain before broadcasting over caller-owned axes.
        source_signature = coord_array_like(
            self.input_coords(),
            {"time": signature.time, "lead_time": signature.lead_time},
        )
        handshake_dataarray(fetched, source_signature)
        leading = signature.dims[: signature.dims.index("time")]
        for dim in leading:
            if dim not in fetched.dims:
                fetched = fetched.expand_dims({dim: signature.coords[dim]})
        fetched = fetched.transpose(*signature.dims)
        if not torch.isfinite(fetched.e2s.to_torch()[0]).all():
            raise ValueError("DataReplay source returned non-finite values")
        output = fetched.astype(x.dtype)
        output = output.assign_coords(signature.coords)
        output.name = x.name
        output.attrs = {**x.attrs, **fetched.attrs}
        output.encoding = x.encoding.copy()
        return output

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Fetch the next source frame on the input device and with its dtype."""
        return self._forward(x)

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        self.output_coords(x)
        yield x.copy(deep=True)

        while True:
            # Hooks may mutate fields and nested metadata in place.
            x = self.front_hook(x.copy(deep=True))
            x = self.rear_hook(self._forward(x))
            yield x.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial field followed by successive source frames."""
        yield from self._default_generator(x)
