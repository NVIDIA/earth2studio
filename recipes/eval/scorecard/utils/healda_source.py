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

"""HealDA v2 analysis as an Earth2Studio initial-condition data source.

A HealDA v2 inference run writes one zarr holding the full analysis period:
104 physical channels plus the ``mae``/``rmse`` DA diagnostics, six-hourly, on
the 721 x 1440 equiangular grid (lat 90 -> -90, lon 0 -> 359.75).  That is the
same grid and the same units ERA5 uses, so the analysis drops into any campaign
that would otherwise initialise from ``ARCO_ERA5``.

Channel names follow the HealDA convention (``T850``, ``pres_msl``, ``tas``);
:func:`earth2studio.models.da.healda_v2.channel_to_e2s` is the repo's existing
map to Earth2Studio names, and this source inverts it so callers ask for
``t850``/``msl``/``t2m`` as usual.

Use it as a BYO IC source, leaving verification on ERA5::

    ic_source:
        _target_: scorecard.utils.healda_source.HealDAAnalysis
        path: /path/to/healda-v2-r14_...zarr
"""

from datetime import datetime

import numpy as np
import xarray as xr
import zarr

from earth2studio.models.da.healda_v2 import channel_to_e2s
from earth2studio.utils.type import TimeArray, VariableArray

# Coordinates HealDA v2 writes; see models/da/healda_v2.py (NLAT, NLON).
HEALDA_LAT = np.linspace(90.0, -90.0, 721, endpoint=True)
HEALDA_LON = np.linspace(0.0, 360.0, 1440, endpoint=False)

# Non-physical channels: the DA's own error diagnostics, never initial conditions.
DIAGNOSTIC_CHANNELS = frozenset({"mae", "rmse"})


class HealDAAnalysis:
    """HealDA v2 analysis zarr as a :class:`~earth2studio.data.base.DataSource`.

    Parameters
    ----------
    path : str
        Path to the HealDA v2 analysis zarr (zarr v3, one array per channel
        with dims ``(time, lat, lon)``).
    lat_trim : int, optional
        Drop this many rows from the south end of the grid.  Models whose
        input grid excludes the south pole (Aurora: 720 rows,
        ``linspace(90, -90, 720, endpoint=False)``) need ``lat_trim=1``;
        models on the full 721-row grid (WeatherNext 2) leave it at 0.
    """

    def __init__(self, path: str, lat_trim: int = 0):
        self.path = path
        self.lat_trim = int(lat_trim)
        self._group = zarr.open(path, mode="r")

        self._time = np.asarray(self._group["time"][:], dtype="int64").astype(
            "datetime64[s]"
        ).astype("datetime64[ns]")
        self._time_index = {t: i for i, t in enumerate(self._time)}

        # e2s name -> HealDA channel name, for the physical channels only.
        self._channels: dict[str, str] = {}
        for name in self._group.array_keys():
            if name in ("lat", "lon", "time", "run_id"):
                continue
            if name in DIAGNOSTIC_CHANNELS:
                continue
            self._channels[channel_to_e2s(name)] = name

        self.lat = HEALDA_LAT[: 721 - self.lat_trim] if self.lat_trim else HEALDA_LAT
        self.lon = HEALDA_LON

    @property
    def variables(self) -> list[str]:
        """Earth2Studio variable names this analysis can serve."""
        return sorted(self._channels)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return the analysis for *time* and *variable*.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis times; each must be a six-hourly step inside the run.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names (``t850``, ``msl``, ``t2m``, ...).

        Returns
        -------
        xr.DataArray
            Dims ``(time, variable, lat, lon)``.
        """
        if isinstance(time, (datetime, np.datetime64, str)):
            time = [time]
        if isinstance(variable, str):
            variable = [variable]
        times = np.asarray(time, dtype="datetime64[ns]")
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]

        missing_t = [str(t) for t in times if t not in self._time_index]
        if missing_t:
            raise KeyError(
                f"HealDAAnalysis: times outside the analysis "
                f"({self._time[0]} .. {self._time[-1]}, 6-hourly): {missing_t}"
            )
        missing_v = [v for v in variables if v not in self._channels]
        if missing_v:
            raise KeyError(
                f"HealDAAnalysis: channels not in {self.path}: {missing_v}"
            )

        out = np.empty(
            (len(times), len(variables), len(self.lat), len(self.lon)),
            dtype=np.float32,
        )
        lat_stop = len(self.lat)
        for ti, t in enumerate(times):
            idx = self._time_index[t]
            for vi, v in enumerate(variables):
                out[ti, vi] = self._group[self._channels[v]][idx, :lat_stop, :]

        return xr.DataArray(
            data=out,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": np.asarray(variables),
                "lat": self.lat,
                "lon": self.lon,
            },
        )
