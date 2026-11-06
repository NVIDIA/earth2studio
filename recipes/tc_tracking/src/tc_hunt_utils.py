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

import pickle
import time
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
    n_ics: int,
    ensemble_size: int,
    batch_size: int,
    seed: int | None,
    n_retries: int = 0,
) -> np.ndarray:
    """Generate unique random seeds for ensemble batches and their retries.

    The original seeds come first, then one round of retry seeds per retry,
    all from the same generator and all unique. A round is drawn again until
    it does not collide with any earlier seed, so a larger ``n_retries`` only
    adds rounds and leaves the earlier seeds unchanged.

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
    n_retries : int, optional
        Number of retry seeds per batch, by default 0

    Returns
    -------
    np.ndarray
        ``uint32`` seeds of shape ``(n_batches * n_ics, 1 + n_retries)``,
        column 0 holding the original seeds.

    Raises
    ------
    RecursionError
        If a unique set cannot be found within 1000 iterations.
    """
    n_runs = ceil(ensemble_size / batch_size) * n_ics
    rng = np.random.default_rng(seed=seed)

    rounds: list[np.ndarray] = []
    for _ in range(1 + n_retries):
        drawn = np.concatenate(rounds) if rounds else np.array([], dtype=np.uint32)
        for _ in range(1000):
            seeds = rng.integers(low=0, high=2**32, size=n_runs, dtype=np.uint32)
            if len(np.unique(np.concatenate([drawn, seeds]))) == len(drawn) + n_runs:
                break
        else:
            raise RecursionError(
                f"failed to generate unique set of {n_runs} random seeds after 1000 iterations. giving up :("
            )
        rounds.append(seeds)

    return np.stack(rounds, axis=1)


def assign_runs_to_rank(items: list[Any]) -> list[Any]:
    """Partition work items across distributed ranks.

    Splits *items* into contiguous blocks whose sizes differ by at most one,
    the first ranks taking the extra items. Contiguous blocks keep the batches
    of an initial condition on few ranks. Deterministic for a deterministic
    input.

    Parameters
    ----------
    items : list[Any]
        List of work items, e.g. ``(initial_condition, member_indices, seeds)``
        tuples.

    Returns
    -------
    list[Any]
        Subset of *items* assigned to this rank, empty if the rank is idle.
    """
    dist = DistributedManager()

    base, remainder = divmod(len(items), dist.world_size)
    start = dist.rank * base + min(dist.rank, remainder)
    end = start + base + (dist.rank < remainder)
    items = items[start:end]

    if items:
        logger.info(f"rank {dist.rank}: {len(items)} runs")
    else:
        logger.info(f"rank {dist.rank}: no runs")

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
    # no distributed setup (e.g. worker processes of baseline extraction)
    if not DistributedManager.is_initialized():
        return func(*args, **kwargs)

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


def shared_store() -> torch.distributed.Store:
    """Key-value store shared by all ranks.

    The TCP store of the default process group, hosted by the root process,
    or an in-process store without a distributed setup. Store calls are TCP
    requests, not collectives, so they can be made by single ranks at any time
    and do not run into NCCL timeouts.

    Returns
    -------
    torch.distributed.Store
        Store with keys prefixed by ``tc_hunt/``
    """
    if DistributedManager.is_initialized() and DistributedManager().distributed:
        # private, but the only handle on the store torch.distributed already runs
        from torch.distributed.distributed_c10d import _get_default_store

        store = _get_default_store()
    else:
        store = torch.distributed.HashStore()
    return torch.distributed.PrefixStore("tc_hunt/", store)


class RetryQueue:
    """Batches to retry, shared by all ranks through the root's key-value store.

    A rank puts a batch that became unstable into the queue. Once a rank has
    finished its initial runs, it takes batches from the queue until all ranks
    have finished their initial runs, no retry is running any more and every
    queued batch has been taken. A rank therefore helps with the retries of
    other ranks instead of idling.

    Every item is taken by exactly one rank: items are numbered by an atomic
    counter, and a rank owns item ``i`` if it is the first to increment
    ``claim/i``. Items are only put by ranks that are still running, before they
    report to be done, so no item is left behind.

    Parameters
    ----------
    store : torch.distributed.Store
        Store shared by all ranks, see :func:`shared_store`
    world_size : int
        Number of ranks
    poll_seconds : float, optional
        Waiting time between two looks at an empty queue, by default 5
    """

    def __init__(
        self, store: torch.distributed.Store, world_size: int, poll_seconds: float = 5
    ) -> None:
        self.store = torch.distributed.PrefixStore("retries/", store)
        self.world_size = world_size
        self.poll_seconds = poll_seconds
        self._cursor = 0

    def _count(self, key: str) -> int:
        return self.store.add(key, 0)

    def put(self, item: Any) -> None:
        """Queue an item for any rank to take."""
        index = self.store.add("queued", 1) - 1
        self.store.set(f"item/{index}", pickle.dumps(item))

    def finish_initial(self) -> None:
        """Report that this rank has finished its initial runs."""
        self.store.add("done", 1)

    def take(self) -> Any | None:
        """Wait for an item, ``None`` once there is no work left for any rank.

        Every item taken has to be reported with :meth:`task_done` once it ran.
        """
        while True:
            queued = self._count("queued")
            while self._cursor < queued:
                index = self._cursor
                self._cursor += 1
                if self.store.add(f"claim/{index}", 1) == 1:
                    self.store.add("running", 1)
                    # waits if the item is claimed before it is set
                    return pickle.loads(self.store.get(f"item/{index}"))  # noqa: S301

            if self._count("done") == self.world_size and self._count("running") == 0:
                # a rank may have queued an item before it finished
                if self._count("queued") == queued:
                    return None
                continue
            time.sleep(self.poll_seconds)

    def task_done(self) -> None:
        """Report that a taken item has been run."""
        self.store.add("running", -1)

    def add_failed(self, count: int) -> None:
        """Count members that stayed unstable after their last retry."""
        self.store.add("failed", count)

    def failed(self) -> int:
        """Members of all ranks that stayed unstable, final once take returned None."""
        return self._count("failed")


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
        Raised by a call if the reference values (first call after
        :meth:`reset`) contain NaN, which would fail every check.
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
        # skipna=False: NaNs make the mean NaN, which fails the check below
        means = x.mean(
            dim=[d for d in x.dims if d not in ("ensemble", "variable")],
            skipna=False,
        )
        means = means.transpose("ensemble", "variable").e2s.as_numpy().values

        if self.baseline is None:
            nan_vars = np.array(self.vars)[np.isnan(means).any(axis=0)]
            if nan_vars.size:
                raise ValueError(
                    f"stability check variables {nan_vars.tolist()} contain NaN in "
                    "the initial condition, choose other variables"
                )
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
