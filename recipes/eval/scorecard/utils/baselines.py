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

Two classic skill baselines, both expressed as prognostic "models" so the
eval recipe's pipelines, online scoring, and exports apply unchanged:

* :class:`PersistenceBaseline` — the forecast is the initial condition at
  every lead time.  A thin AutoModel-style loader around
  ``earth2studio.models.px.Persistence`` on the ERA5 0.25° grid (its
  ``domain_coords`` argument is a coordinate mapping, which a campaign
  yaml cannot express directly).
* :class:`ClimatologyForecast` — the forecast at each lead is the
  climatology at the valid time (day of year x hour of day), read from a
  predownloaded store (see
  :class:`scorecard.utils.pipelines.ClimatologyPipeline`) so no remote
  fetch ever happens inside the inference loop.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from earth2studio.utils.coords import CoordSystem

# The shared verification grid: ERA5 0.25°, latitude 90 -> -90.
ERA5_LAT = np.linspace(90.0, -90.0, 721)
ERA5_LON = np.arange(0.0, 360.0, 0.25)


class PersistenceBaseline:
    """Loader for ``earth2studio.models.px.Persistence`` on the ERA5 grid.

    Provides the ``load_default_package`` / ``load_model`` classmethods the
    recipe's ``load_prognostic`` expects, so a campaign can select the
    baseline with ``model.architecture`` and pass the scored variable list
    through ``model.load_args``.
    """

    @classmethod
    def load_default_package(cls) -> None:
        """No package: persistence has no weights."""
        return None

    @classmethod
    def load_model(
        cls, package: object = None, variables: list[str] | None = None
    ) -> Any:
        """Build the persistence model for *variables* on the ERA5 grid."""
        from earth2studio.models.px import Persistence

        if not variables:
            raise ValueError(
                "PersistenceBaseline needs model.load_args.variables "
                "(e.g. variables: ${output.variables})."
            )
        return Persistence(
            variable=[str(v) for v in variables],
            domain_coords=OrderedDict({"lat": ERA5_LAT, "lon": ERA5_LON}),
        )


class ClimatologyForecast(torch.nn.Module):
    """Prognostic-protocol baseline that forecasts the climatology.

    At every lead time the "forecast" is the climatological field for the
    valid time, read from a local :class:`src.data.PredownloadedSource`
    that :class:`~scorecard.utils.pipelines.ClimatologyPipeline` populates
    at predownload time.  The lead-0 yield is the analysis itself, like
    every other model (scoring drops lead 0 anyway).

    Parameters
    ----------
    variable : list[str]
        Variables the baseline emits, in store order.
    dt : np.timedelta64, optional
        Forecast step, by default 6 hours.
    """

    def __init__(
        self,
        variable: list[str],
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> None:
        super().__init__()
        self._variable = [str(v) for v in variable]
        self._dt = dt
        # Injected by ClimatologyPipeline.setup.
        self._source: Any = None

    @classmethod
    def load_default_package(cls) -> None:
        """No package: the climatology comes from a predownloaded store."""
        return None

    @classmethod
    def load_model(
        cls, package: object = None, variables: list[str] | None = None
    ) -> ClimatologyForecast:
        """Build the baseline for *variables* (pipeline injects the source)."""
        if not variables:
            raise ValueError(
                "ClimatologyForecast needs model.load_args.variables "
                "(e.g. variables: ${output.variables})."
            )
        return cls(variable=list(variables))

    def set_source(self, source: object) -> None:
        """Attach the (local) climatology ``DataSource`` to read from."""
        self._source = source

    def input_coords(self) -> CoordSystem:
        """Initial-condition coordinate system (single analysis frame)."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self._variable),
                "lat": ERA5_LAT.copy(),
                "lon": ERA5_LON.copy(),
            }
        )

    def output_coords(self, input_coords: CoordSystem | None = None) -> CoordSystem:
        """Output coordinate system: one step of ``dt`` past the input."""
        out: CoordSystem = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([self._dt]),
                "variable": np.array(self._variable),
                "lat": ERA5_LAT.copy(),
                "lon": ERA5_LON.copy(),
            }
        )
        if input_coords is not None and "lead_time" in input_coords:
            out["lead_time"] = out["lead_time"] + input_coords["lead_time"][-1]
            if "batch" in input_coords:
                out["batch"] = input_coords["batch"]
        return out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Yield the analysis at lead 0, then climatology at each lead.

        The climatology field is re-indexed onto the model's own lat/lon
        axes before use, so a source with a different latitude orientation
        cannot silently flip the fields (the scorer only checks shapes).
        """
        if self._source is None:
            raise RuntimeError(
                "ClimatologyForecast has no source attached — run it through "
                "scorecard.utils.pipelines.ClimatologyPipeline, which "
                "predownloads the climatology store and injects it."
            )
        base: CoordSystem = OrderedDict(
            (k, v.copy() if isinstance(v, np.ndarray) else v) for k, v in coords.items()
        )
        time0 = np.datetime64(np.asarray(coords["time"]).ravel()[0], "ns")
        yield x, base

        step = 0
        while True:
            step += 1
            lead = self._dt * step
            da = self._source([time0 + lead], self._variable)
            da = da.reindex(lat=ERA5_LAT, lon=ERA5_LON)
            field = torch.from_numpy(
                np.ascontiguousarray(
                    da.transpose("time", "variable", "lat", "lon").values[0],
                    dtype=np.float32,
                )
            ).to(x.device)
            # Broadcast the [variable, lat, lon] field into x's layout
            # (leading singleton axes, e.g. time and lead_time).
            out = torch.broadcast_to(
                field.reshape((1,) * (x.ndim - 3) + field.shape), x.shape
            )
            out_coords = base.copy()
            out_coords["lead_time"] = np.array([lead])
            yield out, out_coords
