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

"""Data source wrappers for predownloaded zarr stores."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from earth2studio.utils.type import TimeArray, VariableArray


class PredownloadedSource:
    """DataSource backed by a predownloaded zarr store.

    The zarr store is expected to have dimensions
    ``(time, variable, <spatial...>)`` with no ``lead_time`` dimension.
    Because ``lead_time`` is absent from :meth:`__call__`, ``fetch_data``
    automatically handles lead-time expansion by calling
    ``source(ic_time + lead, variable)`` for each requested lead time.

    Parameters
    ----------
    store_path : str
        Path to the zarr store on disk.
    """

    def __init__(self, store_path: str) -> None:
        # to_array puts "variable" first; transpose so time leads per DataSource protocol.
        self._da = (
            xr.open_zarr(store_path)
            .to_array("variable")
            .transpose("time", "variable", ...)
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return data for the requested times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Variable names to return.

        Returns
        -------
        xr.DataArray
            Data with dimensions ``[time, variable, <spatial...>]``.
        """
        if not isinstance(time, (list, np.ndarray)):
            time = [time]
        if not isinstance(variable, (list, np.ndarray)):
            variable = [variable]
        return self._da.sel(time=time, variable=variable)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, variable)
