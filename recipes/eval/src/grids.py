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

"""Hydra-instantiable grid resolvers for Earth2Studio data sources.

Some pipelines (e.g. StormScope's coupled GOES+MRMS setup) need to
supply source lat/lon grids at setup time.  Each data source exposes
its grid slightly differently — some as class-level constants (GFS,
ARCO), some via a classmethod (GOES), some only via a sample fetch
(MRMS).  This module centralises that per-source logic behind small
resolvers each campaign config wires in via ``_target_``::

    ic_grid:
        _target_: src.grids.goes_grid
        satellite: goes16
        scan_mode: C

New pipelines that need grid information should either reuse these
resolvers or add new ones here.

This module intentionally imports only from ``earth2studio`` so that
any recipe module can depend on it without risk of circular imports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import torch
from loguru import logger


def goes_grid(
    satellite: str = "goes16",
    scan_mode: str = "C",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lats, lons)`` for a GOES data source.

    Delegates to ``earth2studio.data.GOES.grid``, which is the
    canonical way to retrieve the GOES pixel grid without fetching any
    actual data.  Meant to be called via Hydra ``_target_`` from a
    StormScope model config::

        ic_grid:
            _target_: src.grids.goes_grid
            satellite: goes16
            scan_mode: C
    """
    from earth2studio.data import GOES

    lats, lons = GOES.grid(satellite=satellite, scan_mode=scan_mode)
    return _as_tensor(lats), _as_tensor(lons)


def gfs_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lats, lons)`` for the GFS data source.

    Reads the class-level ``GFS_LAT`` / ``GFS_LON`` constants exposed
    by :class:`earth2studio.data.GFS_FX`.
    """
    from earth2studio.data import GFS_FX

    return _as_tensor(GFS_FX.GFS_LAT), _as_tensor(GFS_FX.GFS_LON)


def arco_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lats, lons)`` for the ARCO ERA5 data source.

    Reads the class-level ``ARCO_LAT`` / ``ARCO_LON`` constants exposed
    by :class:`earth2studio.data.ARCO` — 0.25° regular grid, 721×1440.

    Useful as a ``conditioning_grid`` when running an ablation that
    substitutes ERA5 analysis for GFS forecast conditioning.  ARCO is a
    :class:`~earth2studio.data.base.DataSource` (not a
    :class:`~earth2studio.data.base.ForecastSource`), so the predownload
    path uses it directly without the
    :class:`~src.data.ValidTimeForecastAdapter` wrapper.
    """
    from earth2studio.data import ARCO

    return _as_tensor(ARCO.ARCO_LAT), _as_tensor(ARCO.ARCO_LON)


def mrms_grid(
    sample_time: str | datetime | None = None,
    variable: str = "refc",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lats, lons)`` for MRMS by probing a sample fetch.

    MRMS does not expose its grid as a class constant, so we fetch a
    small DataArray at *sample_time* to read the lat/lon coords.  This
    runs once at pipeline setup; users needing to avoid the network
    call at setup time should replace this resolver with a custom one
    that loads a cached grid file.

    Parameters
    ----------
    sample_time : str | datetime | None
        Time to fetch for grid extraction.  Defaults to a benign
        historical timestamp (2023-01-01 00:00 UTC) if unspecified.
    variable : str
        Variable to request for the probe fetch.  Defaults to
        composite reflectivity, which is always available.
    """
    from earth2studio.data import MRMS

    if sample_time is None:
        probe = datetime(2023, 1, 1)
    elif isinstance(sample_time, str):
        probe = datetime.fromisoformat(sample_time)
    else:
        probe = sample_time

    logger.info(f"Probing MRMS grid via sample fetch at {probe}")
    mrms = MRMS()
    da = mrms(probe, [variable])
    return _as_tensor(da["lat"].values), _as_tensor(da["lon"].values)


def _as_tensor(arr: Any) -> torch.Tensor:
    """Coerce an array-like into a ``torch.Tensor``.

    The StormScope interpolator accepts torch tensors or array-likes
    but downstream ``.to(device)`` calls assume tensors, so we
    normalize at the helper boundary.
    """
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.as_tensor(np.asarray(arr))
