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

"""Collect StormCast output and convert it into per-site forecasts.

This collector is specific to StormCast. It extracts the required source
fields, rotates HRRR winds to earth-relative directions, and calculates
relative humidity and wet-bulb temperature. It supports deterministic,
single-member output.

Earth2Studio sends fields through ``write()``. The collector stores values by
site and lead time, then ``collect()`` groups them into ``ForecastSeries``
objects. DSX components handle validation, serialization, and publishing.

``collect()`` does not clear the buffer. The workflow clears it only after
staging succeeds, so a staging error leaves the collected values intact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..shared.derived_variables import rh_from_specific_humidity, wet_bulb_K
from ..shared.forecast import ForecastSeries
from ..shared.lead_time import lead_seconds_from_coords
from ..shared.site_extraction import SiteExtractor
from ..shared.wind_rotation import rotate_lcc_grid_to_earth

# StormCast source variables requested via ``output_coords``; the raw fields collect() consumes.
SOURCE_VARS = ["t2m", "u10m", "v10m", "q1hl", "p1hl"]

# HRRR Lambert projection values used to rotate grid-relative winds:
# central meridian 262.5° E and tangent parallel 38.5° N.
_HRRR_LON0_DEG = 262.5
_HRRR_CONE_N = float(np.sin(np.deg2rad(38.5)))


class StormCastCollector:
    """Collect streamed StormCast fields and group them into per-site forecasts.

    Earth2Studio can use this class as an IO backend because it provides
    ``add_array()`` and ``write()`` without importing torch.

    Parameters
    ----------
    extractor : SiteExtractor
        Extracts values for each site and provides site longitudes for wind
        rotation.
    model_id : str
        Model name stored in each forecast.
    init_ms : int, optional
        Forecast initialization time in Unix milliseconds. ``begin_cycle()``
        updates it for each cycle.
    """

    def __init__(
        self, extractor: SiteExtractor, model_id: str, init_ms: int = 0
    ) -> None:
        self.extractor = extractor
        self.model_id = model_id
        self.init_ms = init_ms
        self._buffer: dict[tuple[str, int], dict[str, float | None]] = defaultdict(dict)

    # --- IOBackend protocol ---------------------------------------------------
    def add_array(
        self, coords: dict[str, np.ndarray], array_name: str | np.ndarray, **kwargs: Any
    ) -> None:
        """Accept Earth2Studio's setup call without creating storage."""
        pass

    def write(
        self, x: Any, coords: dict[str, np.ndarray], array_name: str | np.ndarray
    ) -> None:
        """Buffer each source variable's per-site value, keyed by (site, leadSeconds).

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Per-variable field tensor(s), as produced by ``split_coords``.
        coords : CoordSystem
            Reduced coordinate system carrying ``lead_time``.
        array_name : np.ndarray | str
            Variable name(s) parallel to ``x``.
        """
        # split_coords hands us a list of per-variable tensors + an ndarray of their names.
        tensors = x if isinstance(x, list) else [x]
        variable_names = (
            [array_name] if isinstance(array_name, str) else list(array_name)
        )
        if len(tensors) != len(variable_names):
            raise ValueError(
                f"received {len(tensors)} field(s) but "
                f"{len(variable_names)} variable name(s)"
            )

        lead_seconds = lead_seconds_from_coords(coords)
        for tensor, name in zip(tensors, variable_names, strict=True):
            name = str(name)
            if name not in SOURCE_VARS:
                continue
            field = self._to_yx(tensor)
            for site, value in self.extractor.extract(field).items():
                buffered_values = self._buffer[(site, lead_seconds)]
                if name in buffered_values:
                    raise ValueError(
                        f"duplicate write for site {site} lead {lead_seconds} "
                        f"variable {name}"
                    )
                buffered_values[name] = value

    @staticmethod
    def _to_yx(tensor: Any) -> np.ndarray:
        """Convert a model tensor to a two-dimensional HRRR field.

        Extra batch, time, and lead dimensions must contain one item each.
        Raise an error instead of silently dropping additional fields.
        """
        array = (
            tensor.detach().to("cpu").numpy()
            if hasattr(tensor, "detach")
            else np.asarray(tensor)
        )
        if array.ndim < 2:
            raise ValueError(
                f"expected a field with at least two dimensions, got shape {array.shape}"
            )
        if array.ndim > 2 and int(np.prod(array.shape[:-2])) != 1:
            raise ValueError(
                f"expected singleton batch/time/lead dims, got shape {array.shape}"
            )
        return array.reshape(array.shape[-2:])

    # --- cycle / collect ------------------------------------------------------
    def begin_cycle(self, init_ms: int) -> None:
        """Set the new initialization time and clear values from the previous cycle."""
        self.init_ms = init_ms
        self._buffer.clear()

    def clear_buffer(self) -> None:
        """Clear values for the current cycle.

        The workflow calls this only after staging succeeds.
        """
        self._buffer.clear()

    def collect(self) -> list[ForecastSeries]:
        """Build one forecast series for each site and output variable.

        Calculate relative humidity and wet-bulb temperature, and rotate winds
        to earth-relative directions. Each forecast has one deterministic
        member. The collected input values remain in the buffer.
        """
        values_by_site: dict[str, dict[str, dict[int, float | None]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (site, lead), src in self._buffer.items():
            if "t2m" in src:
                # The model already provides temperature in Kelvin.
                values_by_site[site]["Temperature"][lead] = (
                    None if src["t2m"] is None else float(src["t2m"])
                )
            if src.get("u10m") is not None and src.get("v10m") is not None:
                # Convert grid-relative winds to true east and north directions.
                u_e, v_n = rotate_lcc_grid_to_earth(
                    src["u10m"],
                    src["v10m"],
                    self.extractor.site_lon[site],
                    _HRRR_LON0_DEG,
                    _HRRR_CONE_N,
                )
                values_by_site[site]["WindU"][lead] = float(u_e)
                values_by_site[site]["WindV"][lead] = float(v_n)
            if all(src.get(k) is not None for k in ("t2m", "q1hl", "p1hl")):
                rh = float(
                    rh_from_specific_humidity(src["t2m"], src["q1hl"], src["p1hl"])
                )
                values_by_site[site]["RelativeHumidity"][lead] = rh
                # The pressure-aware calculation handles elevation and returns Kelvin.
                values_by_site[site]["WetBulb"][lead] = float(
                    wet_bulb_K(src["t2m"], src["q1hl"], src["p1hl"])
                )

        forecasts: list[ForecastSeries] = []
        for site, values_by_variable in values_by_site.items():
            for variable, values_by_lead in values_by_variable.items():
                leads = sorted(values_by_lead)
                forecasts.append(
                    ForecastSeries(
                        site_id=site,
                        variable=variable,
                        init_ms=self.init_ms,
                        lead_seconds=leads,
                        # StormCast produces one deterministic member.
                        member_values=[[values_by_lead[lead] for lead in leads]],
                        model=self.model_id,
                    )
                )
        return forecasts
