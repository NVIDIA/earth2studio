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

"""Lightweight data-loading utilities shared between scoring and comparison scripts.

This module contains functions for loading prediction/verification data from
zarr stores and partitioning lead times. It depends only on numpy, torch, and
xarray — no Hydra, no physicsnemo, no loguru.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import xarray as xr

_NON_SPATIAL = frozenset({"ensemble", "time", "lead_time", "batch"})

CoordSystem = OrderedDict


def spatial_coords_from_dataset(ds: xr.Dataset) -> CoordSystem:
    """Extract spatial coordinate arrays from a prediction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Opened prediction zarr store.

    Returns
    -------
    CoordSystem
        Spatial coordinate arrays (e.g. ``{lat: [...], lon: [...]}``,
        or ``{x: [...], y: [...]}`` for non-lat/lon grids).
    """
    coords: CoordSystem = OrderedDict()
    for dim in ds.dims:
        if dim not in _NON_SPATIAL:
            coords[dim] = ds.coords[dim].values
    return coords


def build_lead_time_chunks(
    lead_times: np.ndarray,
    chunk_size: int | None,
) -> list[np.ndarray]:
    """Partition lead times into chunks for memory-bounded processing.

    Parameters
    ----------
    lead_times : np.ndarray
        Full array of lead-time values.
    chunk_size : int | None
        Maximum number of lead times per chunk.  ``None`` or ``<= 0``
        means no chunking (all lead times in one chunk).

    Returns
    -------
    list[np.ndarray]
        List of lead-time sub-arrays.
    """
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(lead_times):
        return [lead_times]
    return [
        lead_times[i : i + chunk_size] for i in range(0, len(lead_times), chunk_size)
    ]


def load_prediction_chunk(
    prediction_ds: xr.Dataset,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load a chunk of prediction data for one IC time and lead-time range.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    time : np.datetime64
        Initial-condition time to select.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : list[str]
        Variable names to load.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tensor and coords with dimensions
        ``(ensemble?, lead_time, variable, <spatial...>)``.
    """
    subset = prediction_ds[variables].sel(time=time, lead_time=lead_times)
    da = subset.to_array(dim="variable")

    has_ensemble = "ensemble" in da.dims
    spatial_dims = [
        d for d in da.dims if d not in {"variable", "ensemble", "lead_time"}
    ]

    if has_ensemble:
        dim_order = ["ensemble", "lead_time", "variable"] + spatial_dims
    else:
        dim_order = ["lead_time", "variable"] + spatial_dims

    da = da.transpose(*dim_order)
    tensor = torch.from_numpy(da.values.copy()).to(device=device, dtype=torch.float32)

    coords: CoordSystem = OrderedDict()
    for dim in dim_order:
        coords[dim] = np.array(da.coords[dim].values)
    return tensor, coords


def load_verification_chunk(
    source: Any,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: list[str],
    spatial_coords: CoordSystem,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load verification data aligned to a prediction chunk.

    For each lead time, the valid time ``time + lead_time`` is computed and
    fetched from the verification source.  The returned tensor uses
    ``lead_time`` (not valid time) as its first dimension so that it aligns
    with the prediction chunk.

    Parameters
    ----------
    source
        Verification data source (callable accepting times and variables).
    time : np.datetime64
        Initial-condition time.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : list[str]
        Variable names to load.
    spatial_coords : CoordSystem
        Spatial coordinate arrays (for building the output CoordSystem).
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tensor and coords with dimensions
        ``(lead_time, variable, <spatial...>)``.
    """
    valid_times = time + lead_times
    da = source(list(valid_times), list(variables))

    spatial_dims = [d for d in da.dims if d not in {"time", "variable"}]
    dim_order = ["time", "variable"] + spatial_dims
    da = da.transpose(*dim_order)

    tensor = torch.from_numpy(da.values.copy()).to(device=device, dtype=torch.float32)

    coords: CoordSystem = OrderedDict()
    coords["lead_time"] = lead_times
    coords["variable"] = np.array(variables)
    for dim in spatial_dims:
        coords[dim] = spatial_coords[dim]
    return tensor, coords
