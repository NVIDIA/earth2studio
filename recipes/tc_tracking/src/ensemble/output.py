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

"""Output stores of the ensemble: fields and the seed of every member.

Fields go to one Zarr store shared by all ranks, or to one NetCDF file per
batch with :class:`BatchNetCDFBackend`. :class:`SeedRecorder` adds the seed
each member was run with as array ``random_seed``.
"""

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import xarray as xr

from earth2studio.io import AsyncZarrBackend, NetCDF4Backend
from earth2studio.io.base import IOBackend
from earth2studio.run import WorkItem
from earth2studio.utils.coords import coord_array, coord_array_like

# seed of the members of a batch that stayed unstable in every attempt
FAILED_SEED = -1


def batch_name(time: np.datetime64, members: Sequence[int]) -> str:
    """Name of a batch's output, e.g. ``2024-09-24T00.00.00_mems0000-0001``.

    Parameters
    ----------
    time : np.datetime64
        Initial condition of the batch
    members : Sequence[int]
        Members of the batch

    Returns
    -------
    str
        IC and member range, without colons, which some file systems reject
    """
    ic = np.datetime_as_string(np.datetime64(time, "s"), unit="s").replace(":", ".")
    return f"{ic}_mems{members[0]:04d}-{members[-1]:04d}"


def run_coords(items: Sequence[WorkItem]) -> OrderedDict[str, np.ndarray]:
    """Members and ICs of all batches of a run, the extent of its stores.

    Parameters
    ----------
    items : Sequence[WorkItem]
        All batches of the run

    Returns
    -------
    OrderedDict[str, np.ndarray]
        Sorted labels of ``ensemble`` and ``time``
    """
    members = sorted({member for item in items for member in item.member_ids})
    times = sorted({item.time for item in items})
    return OrderedDict(
        ensemble=np.array(members), time=np.array(times, dtype="datetime64[ns]")
    )


def zarr_backend(
    path: str,
    items: Sequence[WorkItem],
    lead_times: np.ndarray,
    shard_per_member: bool = True,
) -> AsyncZarrBackend:
    """Zarr store for all batches of a run, written in background threads.

    Chunks hold one member, IC, lead time and variable, so ranks never write
    to the same chunk.

    Parameters
    ----------
    path : str
        Path of the store
    items : Sequence[WorkItem]
        All batches of the run
    lead_times : np.ndarray
        Lead times of the run
    shard_per_member : bool, optional
        Store all lead times of a member, IC and variable as one shard, i.e.
        one file, by default True. The shard is kept in host memory until the
        member is complete. Without, every chunk is its own file.

    Returns
    -------
    AsyncZarrBackend
        The store
    """
    return AsyncZarrBackend(
        file_name=path,
        parallel_coords=OrderedDict(**run_coords(items), lead_time=lead_times),
        blocking=False,
        shard_coords={"lead_time": len(lead_times)} if shard_per_member else {},
    )


class BatchNetCDFBackend:
    """IO backend writing one NetCDF file per batch of members.

    A pipeline writes each stream to one backend shared by all ranks, but a
    NetCDF file can only be written by one process. This backend therefore
    routes every field by its ``time`` and ``ensemble`` labels to the file of
    its batch, ``<prefix>_<ic>_mems<first>-<last>.nc``. A file is created on
    its batch's first write, replacing an existing one, and closed by
    :meth:`flush`, which the pipeline calls once the batch is done.

    Parameters
    ----------
    prefix : str
        Path and file name prefix of the files
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        # templates of all arrays, spanning all batches of the run
        self._templates: list[xr.DataArray] = []
        self._files: dict[tuple, NetCDF4Backend] = {}

    @property
    def write_blocks(self) -> dict[str, int]:
        """Each batch has its own file, so ranks can write any member and IC."""
        return {"time": 1, "ensemble": 1}

    def add_array(self, template: xr.DataArray) -> None:
        """Add an array to the files of all batches.

        Parameters
        ----------
        template : xr.DataArray
            Allocation-free template of the array for all batches of the run
        """
        self._templates.append(template)

    def write(self, x: xr.DataArray) -> None:
        """Write fields of one batch to its file.

        Parameters
        ----------
        x : xr.DataArray
            Fields with dimensions ``time``, holding one IC, and ``ensemble``,
            holding the members of the batch
        """
        if x.sizes.get("time") != 1 or "ensemble" not in x.dims:
            raise ValueError(
                "fields must have one time and a dimension ensemble to be "
                "written to the file of their batch"
            )
        time = x.coords["time"].values
        members = x.coords["ensemble"].values
        key = (time[0], tuple(members))
        if key not in self._files:
            self._files[key] = self._create(time, members)
        self._files[key].write(x)

    def _create(self, time: np.ndarray, members: np.ndarray) -> NetCDF4Backend:
        """Create the file of a batch with all arrays, replacing an existing one."""
        store = NetCDF4Backend(
            f"{self.prefix}_{batch_name(time[0], members)}.nc",
            backend_kwargs={"mode": "w", "diskless": False},
        )
        for template in self._templates:
            store.add_array(
                coord_array_like(template, {"time": time, "ensemble": members})
            )
        return store

    def flush(self) -> None:
        """Close the files of the batches written so far."""
        files, self._files = self._files, {}
        for store in files.values():
            store.close()

    def close(self) -> None:
        """Close all open files."""
        self.flush()


class SeedRecorder:
    """Record the seed of every member as array ``random_seed`` of a store.

    The array has dimensions ``(ensemble, time)``. A member whose attempt was
    unstable gets :data:`FAILED_SEED`, until a retry replaces it. As a
    :class:`~src.ensemble.runner.BatchObserver`, it writes the seeds of each
    attempt when the attempt is done, so they are flushed with its fields.

    Parameters
    ----------
    store : IOBackend
        Store of the fields
    items : Sequence[WorkItem]
        All batches of the run
    """

    def __init__(self, store: IOBackend, items: Sequence[WorkItem]) -> None:
        self.store = store
        self.items = items

    def add_array(self) -> None:
        """Add the ``random_seed`` array to the store.

        Not thread-safe for stores shared by several ranks: let one rank add
        it first, e.g. with :func:`~earth2studio.run.work.rank0_first`.
        """
        self.store.add_array(
            coord_array(
                ("ensemble", "time"),
                run_coords(self.items),
                dtype=np.int64,
                name="random_seed",
            )
        )

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """Write the seed of the batch's members.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the attempt, None if it was unstable
        attempts : int
            Number of attempts made
        """
        self.store.write(
            xr.DataArray(
                np.full(
                    (len(item.member_ids), 1), FAILED_SEED if seed is None else seed
                ),
                dims=("ensemble", "time"),
                coords={
                    "ensemble": list(item.member_ids),
                    "time": np.array([item.time], dtype="datetime64[ns]"),
                },
                name="random_seed",
            )
        )
