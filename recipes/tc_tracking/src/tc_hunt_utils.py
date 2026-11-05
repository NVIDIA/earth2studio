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

from collections.abc import Callable
from math import ceil
from typing import Any

import cupy
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.utils.time import to_time_array

EARTH_RADIUS_M = 6371000


def set_initial_times(cfg: DictConfig) -> np.ndarray:
    """Build array of initial conditions.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.

    Returns
    -------
    np.ndarray
        Array of initial conditions (dtype ``datetime64``).
    """
    # list of ICs
    if "ics" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError(
                "either provide a list of start times or define a block, not both"
            )
        ics = to_time_array(sorted(cfg.ics))

    # block of ICs
    else:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        ics = np.arange(
            ics[0],
            ics[1] + np.timedelta64(cfg.ic_block_step, "h"),
            np.timedelta64(cfg.ic_block_step, "h"),
        )

    return ics


def share_gpu_memory_with_cupy() -> None:
    """Let CuPy allocate GPU memory from PyTorch's caching allocator.

    Earth2Studio models compute with PyTorch but exchange fields as CuPy-backed
    DataArrays. With separate allocators, memory cached by PyTorch is not
    available to CuPy, which runs out of memory for large models. Call after
    all imports, as importing cuDF installs a different CuPy allocator.
    """

    def malloc(size: int) -> cupy.cuda.MemoryPointer:
        # the tensor owns the memory and returns it to PyTorch's cache once
        # CuPy releases the pointer
        buffer = torch.empty(
            size, dtype=torch.uint8, device=f"cuda:{cupy.cuda.Device().id}"
        )
        memory = cupy.cuda.UnownedMemory(buffer.data_ptr(), size, buffer)
        return cupy.cuda.MemoryPointer(memory, 0)

    cupy.cuda.set_allocator(malloc)


def get_set_of_random_seeds(
    n_ics: int, ensemble_size: int, batch_size: int, seed: int | None
) -> np.ndarray:
    """Generate a unique set of random seeds for ensemble batches.

    Parameters
    ----------
    n_ics : int
        Number of initial conditions.
    ensemble_size : int
        Total number of ensemble members per IC.
    batch_size : int
        Number of ensemble members per batch.
    seed : int | None
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of ``uint32`` random seeds with length ``n_batches * n_ics``.

    Raises
    ------
    RecursionError
        If a unique set cannot be found within 1000 iterations.
    """
    n_batches = ceil(ensemble_size / batch_size)
    rng = np.random.default_rng(seed=seed)

    seeds = np.array([])
    ii = 0
    while len(np.unique(seeds)) != n_batches * n_ics:
        ii += 1
        if ii > 1000:
            raise RecursionError(
                f"failed to generate unique set of {n_batches*n_ics} random seeds after 1000 iterations. giving up :("
            )
        seeds = rng.integers(low=0, high=2**32, size=n_batches * n_ics, dtype=np.uint32)

    return seeds


def assign_runs_to_rank(items: list[Any]) -> list[Any] | None:
    """Partition work items across distributed ranks.

    Splits *items* evenly across all ranks via contiguous slicing, which is
    deterministic for a deterministic input. Returns ``None`` for ranks that
    receive no work.

    Parameters
    ----------
    items : list[Any]
        List of work items (e.g. ``(initial_condition, member_indices, seed)``
        tuples or storm names).

    Returns
    -------
    list[Any] | None
        Subset of *items* assigned to this rank, or ``None`` if idle.
    """
    dist = DistributedManager()

    items_per_rank = len(items) // dist.world_size
    if len(items) % dist.world_size != 0:
        items_per_rank += 1

    items = items[dist.rank * items_per_rank : (dist.rank + 1) * items_per_rank]

    if len(items) == 0:
        logger.info(f"nothing to do for rank {dist.rank}, exiting")
        return None

    return items


def run_with_rank_ordered_execution(
    func: Callable, *args: Any, first_rank: int = 0, **kwargs: Any
) -> Any:
    """Execute a function safely in a distributed setting.

    The function runs first on ``first_rank``, then after a barrier
    synchronisation on all remaining ranks.

    Parameters
    ----------
    func : Callable
        Function to execute.
    *args : Any
        Positional arguments forwarded to *func*.
    first_rank : int, optional
        Rank that runs the function first.  Defaults to ``0``.
    **kwargs : Any
        Keyword arguments forwarded to *func*.

    Returns
    -------
    Any
        Return value of ``func(*args, **kwargs)``.
    """
    dist = DistributedManager()
    current_rank = dist.rank

    if current_rank == first_rank:
        result = func(*args, **kwargs)
    else:
        result = None

    # Synchronize all processes after the first rank runs the function
    # Skip the barrier if single-process (no distributed process group)
    if dist.distributed:
        torch.distributed.barrier()

    if current_rank != first_rank:
        result = func(*args, **kwargs)

    if dist.distributed:
        torch.distributed.barrier()

    return result


class InstabilityDetection:
    """Monitor field-mean deviations to detect blow ups.

    Computes spatially averaged values per variable and ensemble member and
    checks whether they remain within the given thresholds relative to a
    baseline established on the first call after :meth:`reset`.

    Parameters
    ----------
    vars : list[str]
        Variable names to monitor.
    thresholds : list[float]
        Absolute deviation threshold for each variable.

    Raises
    ------
    ValueError
        If the number of thresholds does not match the number of variables.
    """

    def __init__(self, vars: list[str], thresholds: list[float]) -> None:
        if not len(vars) == len(thresholds):
            raise ValueError("please provide exactly one threshold per variable")
        self.vars = list(vars)
        self.thresholds = np.asarray(thresholds)
        self.reset()

    def reset(self) -> None:
        """Forget the baseline, the next call establishes a new one."""
        self.baseline: np.ndarray | None = None

    def __call__(self, x: xr.DataArray) -> np.ndarray:
        """Check whether field means remain within threshold of the baseline.

        Parameters
        ----------
        x : xr.DataArray
            Fields with dimensions ``ensemble`` and ``variable`` plus any
            others, which are averaged over.

        Returns
        -------
        np.ndarray
            Boolean array (``True`` = stable) per ensemble member.
        """
        x = x.sel(variable=self.vars)
        means = x.mean(dim=[d for d in x.dims if d not in ("ensemble", "variable")])
        means = means.transpose("ensemble", "variable").e2s.as_numpy().values

        if self.baseline is None:
            self.baseline = means

        return (np.abs(means - self.baseline) < self.thresholds).all(axis=-1)


def great_circle_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> float | np.ndarray:
    """Compute the great-circle distance between two points on a sphere.

    Uses the Haversine formula on the sphere, the radius of which defaults
    to Earth's mean radius of 6371 km.

    Parameters
    ----------
    lat1 : float or np.ndarray
        Latitude(s) of the first point in degrees.
    lon1 : float or np.ndarray
        Longitude(s) of the first point in degrees.
    lat2 : float or np.ndarray
        Latitude(s) of the second point in degrees.
    lon2 : float or np.ndarray
        Longitude(s) of the second point in degrees.
    radius : float, optional
        Sphere radius in metres, by default ``EARTH_RADIUS_M`` (6 371 km).

    Returns
    -------
    float or np.ndarray
        Distance(s) in metres.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    cc = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1 - aa))

    return radius * cc
