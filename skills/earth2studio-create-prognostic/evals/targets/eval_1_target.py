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

from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class CosineZenith(torch.nn.Module, PrognosticMixin):
    """Prognostic model that computes the cosine of the solar zenith angle at each
    grid point for the current forecast valid time. Primarily useful for testing.

    The model takes any input variable on a 1-degree global grid and outputs the
    cosine zenith angle field at each 6-hour step.

    Parameters
    ----------
    base_time : np.datetime64, optional
        Reference base time for computing solar position, by default
        np.datetime64("2024-01-01T00:00")
    """

    def __init__(
        self,
        base_time: np.datetime64 = np.datetime64("2024-01-01T00:00"),
    ):
        super().__init__()
        self._dt = np.timedelta64(6, "h")
        self._base_time = base_time

        lat = np.linspace(90.0, -90.0, 181)
        lon = np.linspace(0.0, 360.0, 360, endpoint=False)

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["t2m"]),
                "lat": lat,
                "lon": lon,
            }
        )
        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([self._dt]),
                "variable": np.array(["cos_zenith"]),
                "lat": lat,
                "lon": lon,
            }
        )

    def __str__(self) -> str:
        return "cosine_zenith"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        output_coords = self._output_coords.copy()

        if input_coords is None:
            return output_coords

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key != "batch":
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"][-1]
        )
        return output_coords

    def _compute_cos_zenith(
        self, lead_time: np.ndarray, lat: np.ndarray, lon: np.ndarray
    ) -> torch.Tensor:
        """Compute cosine of solar zenith angle for given time offset and grid.

        Parameters
        ----------
        lead_time : np.ndarray
            Lead time array (timedelta64)
        lat : np.ndarray
            Latitude array in degrees
        lon : np.ndarray
            Longitude array in degrees

        Returns
        -------
        torch.Tensor
            Cosine zenith angle field [1, lat, lon]
        """
        # Valid time from base_time + lead_time
        valid_time = self._base_time + lead_time[-1]
        # Day of year (approximate solar declination)
        day_of_year = (
            valid_time - np.datetime64(str(valid_time)[:4], "Y")
        ) / np.timedelta64(1, "D")
        # Solar declination (radians)
        declination = (
            -23.45 * np.pi / 180.0 * np.cos(2.0 * np.pi * (day_of_year + 10.0) / 365.0)
        )
        # Hour of day
        hour = (valid_time - valid_time.astype("datetime64[D]")) / np.timedelta64(
            1, "h"
        )
        # Hour angle for each longitude
        hour_angle = (hour / 24.0 * 360.0 + lon - 180.0) * np.pi / 180.0  # [lon]
        # Latitude in radians
        lat_rad = lat * np.pi / 180.0  # [lat]

        # cos(zenith) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(hour_angle)
        cos_z = np.sin(lat_rad[:, None]) * np.sin(declination) + np.cos(
            lat_rad[:, None]
        ) * np.cos(declination) * np.cos(hour_angle[None, :])
        # Clamp to [0, 1] (negative = night)
        cos_z = np.clip(cos_z, 0.0, 1.0)

        return torch.tensor(cos_z, dtype=torch.float32).unsqueeze(0)  # [1, lat, lon]

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)

        # Compute cosine zenith for the output lead time
        cos_z = self._compute_cos_zenith(
            output_coords["lead_time"],
            output_coords["lat"],
            output_coords["lon"],
        )
        # Shape: [batch, 1 (lead_time), 1 (variable), lat, lon]
        batch_size = x.shape[0]
        out = cos_z.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1, -1, -1)
        out = out.to(device=x.device)

        return out, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system
        """
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        self.output_coords(coords.copy())
        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][-1:]
        yield x[:, -1:], coords_out

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward pass — compute cosine zenith
            x_out, coords_out = self._forward(x, coords)

            # Rear hook
            x_out, coords_out = self.rear_hook(x_out, coords_out)

            coords["lead_time"] = np.concatenate(
                [coords["lead_time"][1:], coords_out["lead_time"]]
            )
            x = torch.cat([x[:, 1:], x_out], dim=1)

            yield x_out, coords_out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
