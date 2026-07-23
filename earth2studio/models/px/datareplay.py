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

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class DataReplay(torch.nn.Module, PrognosticMixin):
    """Replay a data source through the prognostic model interface.

    Data sources are queried at successive valid times. Forecast sources are queried
    at successive lead times from the initial time.

    Parameters
    ----------
    source : DataSource | ForecastSource
        Source to replay.
    variable : str | list[str]
        Variables to fetch.
    domain_coords : CoordSystem
        Spatial coordinates expected from the source.
    step : np.timedelta64, optional
        Time between frames, by default np.timedelta64(6, "h")
    """

    def __init__(
        self,
        source: DataSource | ForecastSource,
        variable: str | list[str],
        domain_coords: CoordSystem,
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
        self._domain_coords = OrderedDict(
            (key, np.asarray(value).copy()) for key, value in domain_coords.items()
        )
        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": self._variable,
                **self._domain_coords,
            }
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Input coordinate system.
        """
        return OrderedDict(
            (key, np.asarray(value).copy()) for key, value in self._input_coords.items()
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system one step ahead.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        CoordSystem
            Output coordinate system.
        """
        target_coords = self.input_coords()
        handshake_dim(input_coords, "lead_time", 2)
        for index, key in enumerate(target_coords):
            if key not in ("batch", "time", "lead_time"):
                handshake_dim(input_coords, key, index)
                handshake_coords(input_coords, target_coords, key)

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"][-1:] + self.step
        return output_coords

    @staticmethod
    def _require_time(coords: CoordSystem) -> None:
        if "time" not in coords or len(coords["time"]) == 0:
            raise ValueError("DataReplay requires a non-empty time coordinate")

    @torch.inference_mode()
    def _fetch(
        self,
        coords: CoordSystem,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        fetched, fetched_coords = fetch_data(
            self.source,
            time=coords["time"],
            variable=self._variable,
            lead_time=coords["lead_time"],
            device=device,
        )

        for key in ("time", "lead_time", "variable", *self._domain_coords):
            handshake_coords(fetched_coords, coords, key)
        if not torch.isfinite(fetched).all():
            raise ValueError("DataReplay source returned non-finite values")

        return fetched.unsqueeze(0).expand(batch_size, *fetched.shape).contiguous()

    def _forward(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        self._require_time(coords)
        output_coords = self.output_coords(coords)
        output = self._fetch(output_coords, x.shape[0], x.device)
        return output.to(x.dtype), output_coords

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Advance the source by one step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Source data and coordinates one step ahead.
        """
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        self._require_time(coords)
        self.output_coords(coords)

        coords = coords.copy()
        coords["lead_time"] = coords["lead_time"][-1:]
        x = x[:, :, -1:]
        yield x, coords

        while True:
            x, coords = self.front_hook(x, coords)
            x, coords = self._forward(x, coords)
            x, coords = self.rear_hook(x, coords)
            yield x, coords

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create an iterator over source frames.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial condition coordinates.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Initial condition followed by source frames.
        """
        yield from self._default_generator(x, coords)
