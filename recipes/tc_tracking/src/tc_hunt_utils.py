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
from collections.abc import Callable
from math import ceil
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.utils.time import to_time_array


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


def remove_duplicates(
    data_list: list[list],
) -> list[list]:
    """Remove duplicate sub-lists while preserving numpy dtype distinctions.

    Two arrays with the same values but different dtypes are considered
    different.

    Parameters
    ----------
    data_list : list[list]
        List of sub-lists, each potentially containing numpy arrays or
        datetime64 objects.

    Returns
    -------
    list[list]
        De-duplicated list preserving original order.
    """

    def to_hashable(item: object) -> object:
        if isinstance(item, np.ndarray):
            return (tuple(item.tolist()), str(item.dtype), item.shape)
        elif isinstance(item, np.datetime64):
            return str(item)
        return item

    seen: set[tuple] = set()
    result: list[list] = []

    for sublist in data_list:
        hashable_key = tuple(to_hashable(item) for item in sublist)
        if hashable_key not in seen:
            seen.add(hashable_key)
            result.append(sublist)

    return result


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
    baseline established on the first call.

    Parameters
    ----------
    vars : list[str]
        Variable names to monitor.
    thresholds : list[float]
        Absolute deviation threshold for each variable.
    input_coords : OrderedDict | None, optional
        Initial coordinate mapping.  Defaults to ``None``.

    Raises
    ------
    ValueError
        If the number of thresholds does not match the number of variables.
    """

    def __init__(
        self,
        vars: list[str],
        thresholds: list[float],
        input_coords: OrderedDict | None = None,
    ) -> None:
        self.vars = vars
        self.thresh = torch.Tensor(thresholds)
        self.reset(input_coords)

        if not len(vars) == len(thresholds):
            raise ValueError("please provide exactly one threshold per variable")

    def reset(self, coords: OrderedDict | None = None) -> None:
        """Reset the baseline and coordinate state.

        Parameters
        ----------
        coords : OrderedDict | None, optional
            New coordinate mapping.  If provided, ``update_coords`` is
            called immediately.  Defaults to ``None``.
        """
        self.baseline: torch.Tensor | None = None
        self._input_coords: OrderedDict | None = None
        self._output_coords: OrderedDict | None = None

        if coords:
            self.update_coords(coords)

    def update_coords(self, coords: OrderedDict) -> None:
        """Derive input/output coordinate mappings from coords.

        ``time``, ``lead_time`` and ``ensemble`` dimensions are stripped from
        the input mapping.

        Parameters
        ----------
        coords : OrderedDict
            Coordinate mapping (modified in place).
        """
        coords.pop("time", None)
        coords.pop("lead_time", None)
        self._output_coords = OrderedDict({"ensemble": coords.pop("ensemble", None)})

        self._input_coords = coords
        self._input_coords["variable"] = self.vars

    @property
    def input_coords(self) -> OrderedDict | None:
        """Expected input coordinate mapping, or ``None`` before first call."""
        return self._input_coords

    @property
    def output_coords(self) -> OrderedDict | None:
        """Output coordinate mapping (ensemble only), or ``None`` before first call."""
        return self._output_coords

    def __call__(
        self, xx: torch.Tensor, coords: OrderedDict
    ) -> tuple[torch.Tensor, OrderedDict | None]:
        """Check whether field means remain within threshold of the baseline.

        On the first invocation the baseline is established from xx.
        Subsequent calls compare against it.

        Parameters
        ----------
        xx : torch.Tensor
            Data tensor matching the coordinate mapping.
        coords : OrderedDict
            Coordinate mapping for *xx*.

        Returns
        -------
        tuple[torch.Tensor, OrderedDict]
            Boolean tensor (``True`` = stable) per ensemble member, and the
            output coordinate mapping.
        """
        var_dim = list(coords).index("variable")
        batch_dim = list(coords).index("ensemble") if "ensemble" in coords else None

        if self.baseline is None:
            self.baseline = xx.mean(
                dim=tuple(
                    [ii for ii in range(len(coords)) if ii not in (var_dim, batch_dim)]
                )
            )
            comp = self.baseline
            self.update_coords(coords)
        else:
            comp = xx.mean(
                dim=tuple(
                    [ii for ii in range(len(coords)) if ii not in (var_dim, batch_dim)]
                )
            )

        return (torch.abs(comp - self.baseline) < self.thresh.to(xx.device)).all(
            dim=-1
        ), self._output_coords


def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points on a sphere.

    Uses the Haversine formula with Earth's mean radius of 6371 km.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    cc = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1 - aa))

    return 6371000 * cc
