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

"""Output-store lifecycle for :class:`earth2studio.run.pipeline.Pipeline`.

:class:`OutputManager` is internal: it owns one IO backend per stream and
drives their collective lifecycle (``prepare``, ``write``, ``flush``,
``close``) as ``dev/spec/EXECUTION_CONTRACT_SPEC.md`` specifies.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import ArrayLike

from earth2studio.io.async_zarr import AsyncZarrBackend
from earth2studio.io.base import IOBackend
from earth2studio.io.kv import KVBackend
from earth2studio.io.utils import output_template
from earth2studio.io.xarray import XarrayBackend
from earth2studio.io.zarr import ZarrBackend
from earth2studio.run.work import WorkItem, barrier, rank0_first, rank_info
from earth2studio.utils.type import CoordinateSystem

# Dimensions work is distributed over: an item writes one `time` and its
# `member_ids` along `ensemble`. Every other dimension is written whole by a
# single item, so no two ranks ever share one of its chunks.
_DISTRIBUTED_DIMS = ("time", "ensemble")


class OutputManager:
    """Per-stream output stores, created and closed collectively.

    One IO backend is held per stream name. Streams without a backend are not
    written, so a caller may hand over a partial mapping.

    Parameters
    ----------
    backends : Mapping[str, IOBackend]
        Stream name to the backend its outputs are written to.
    """

    def __init__(self, backends: Mapping[str, IOBackend]) -> None:
        self.backends: dict[str, IOBackend] = dict(backends)
        self._closed = False
        self._warned: set[str] = set()

    def prepare(
        self, schemas: Mapping[str, CoordinateSystem], items: Sequence[WorkItem]
    ) -> None:
        """Create or validate every store, spanning all work items.

        Each store's template comes from its stream's planning signature, the
        sorted unique item times and, for ensembles, the sorted union of
        ``member_ids`` over every item. The stores therefore span the whole
        campaign, so a resume at any world size reuses them (``X8``).

        Rank 0 calls ``add_array`` first; after a barrier the other ranks call
        it too, which validates the store and otherwise does nothing (``I4``,
        ``X13``). With more than one rank, the layout of every store that can be
        checked is rejected if two items would share a chunk or shard (``X14``),
        and backends that cannot be checked warn once (``X15``).

        Parameters
        ----------
        schemas : Mapping[str, CoordinateSystem]
            Planning signature per stream, as a runner's ``output_coords``
            gives them: ``lead_time``, ``variable`` and the grid's dimensions
            are fixed, ``time`` and ``ensemble`` are not.
        items : Sequence[WorkItem]
            Every item of the run, including the items already done.

        Raises
        ------
        ValueError
            If a backend names a stream without a schema, ``items`` is empty, or
            a distributed store's layout lets two items share storage.
        """
        unknown = sorted(set(self.backends).difference(schemas))
        if unknown:
            raise ValueError(f"Output backends name unknown streams: {unknown}")
        if self.backends and not items:
            raise ValueError("Output stores cannot be prepared without work items")

        times = np.unique(
            np.array([item.time for item in items], dtype="datetime64[ns]")
        )
        members = np.array(
            sorted({int(member) for item in items for member in item.member_ids}),
            dtype=np.int64,
        )
        leading: dict[str, ArrayLike] = {}
        if members.size:
            leading["ensemble"] = members
        leading["time"] = times

        templates = {
            stream: output_template(schemas[stream], leading)
            for stream in self.backends
        }

        _, world_size = rank_info()
        if world_size > 1:
            for stream, backend in self.backends.items():
                self._check_ownership(stream, backend, templates[stream], items)

        def add_arrays() -> None:
            for stream, backend in self.backends.items():
                logger.debug(f"Adding arrays for stream '{stream}'")
                backend.add_array(templates[stream])

        rank0_first(add_arrays)

    def write(self, stream: str, x: xr.DataArray) -> None:
        """Write one field, skipping streams without a backend.

        Parameters
        ----------
        stream : str
            Stream the field belongs to.
        x : xr.DataArray
            Field to write, located in the store by its coordinate labels.
        """
        backend = self.backends.get(stream)
        if backend is None:
            return
        backend.write(x)

    def flush(self) -> None:
        """Flush every backend, making earlier writes visible to readers (``I9``)."""
        for backend in self.backends.values():
            backend.flush()

    def close(self) -> Exception | None:
        """Close every backend, then barrier. Collective and idempotent.

        Every rank closes its backends even if one of them fails, so a partial
        failure never leaves a rank out of the final barrier. Closed backends
        are kept, so they stay readable for callers that return them (``I11``).

        A failure is returned rather than raised: raising on one rank would
        skip the failure-flag reduction the other ranks then wait in, so the
        caller passes it to :func:`~earth2studio.run.work.raise_if_any_failed`
        instead.

        Returns
        -------
        Exception | None
            The first error a backend's ``close`` raised, or ``None``. Always
            ``None`` on a repeated call.
        """
        if self._closed:
            return None
        self._closed = True
        errors: list[Exception] = []
        for stream, backend in self.backends.items():
            try:
                backend.close()
            except Exception as error:  # noqa: BLE001 - every backend must close
                logger.error(f"Closing the store of stream '{stream}' failed: {error}")
                errors.append(error)
        barrier()
        return errors[0] if errors else None

    def _check_ownership(
        self,
        stream: str,
        backend: IOBackend,
        template: xr.DataArray,
        items: Sequence[WorkItem],
    ) -> None:
        """Reject or warn about a store whose writers would race (``X14``, ``X15``).

        Ranks race when one storage chunk or shard holds positions of items
        owned by different ranks, so only the distributed dimensions matter.
        Known backends are checked through their existing attributes; nothing in
        the IO protocol or in a backend changes for this.
        """
        sizes = {str(dim): int(size) for dim, size in template.sizes.items()}
        groups = _member_groups(items)
        members = (
            np.asarray(template.coords["ensemble"].values)
            if "ensemble" in sizes
            else np.empty(0)
        )

        if isinstance(backend, ZarrBackend):
            # Also covers IceChunkBackend, which subclasses it.
            blocks = _zarr_blocks(backend, sizes)
            for dim, (block, source) in blocks.items():
                problem = _shared_block(dim, block, members, groups, sizes[dim])
                if problem is not None:
                    raise ValueError(
                        f"Store of stream '{stream}' has {source} of {block} on "
                        f"'{dim}': {problem}. Work is distributed over "
                        f"{list(_DISTRIBUTED_DIMS)}, so two ranks would write the "
                        f"same {source} and silently lose data. Use "
                        f"{type(backend).__name__}(chunks={{'time': 1, "
                        "'ensemble': 1}}) or recreate the store."
                    )
            return

        if isinstance(backend, AsyncZarrBackend):
            parallel = {str(dim) for dim in backend.parallel_coords}
            missing = [
                dim for dim in _DISTRIBUTED_DIMS if dim in sizes and dim not in parallel
            ]
            if missing:
                raise ValueError(
                    f"Store of stream '{stream}' does not list {missing} in "
                    "parallel_coords. Work is distributed over those dimensions, so "
                    "they must be parallel coordinates for ranks to write them "
                    "independently."
                )
            for dim, (block, source) in _async_zarr_blocks(backend, sizes).items():
                problem = _shared_block(dim, block, members, groups, sizes[dim])
                if problem is not None:
                    raise ValueError(
                        f"Store of stream '{stream}' has {source} of {block} on "
                        f"'{dim}': {problem}. A shard is written as a whole, so two "
                        "ranks would each write it and silently lose data. Shard "
                        "'lead_time' instead."
                    )
            return

        if isinstance(backend, (XarrayBackend, KVBackend)):
            self._warn_once(
                backend,
                f"{type(backend).__name__} of stream '{stream}' is process-local: "
                "each rank holds only the items it ran, and nothing gathers them.",
            )
            return

        self._warn_once(
            backend,
            f"{type(backend).__name__} of stream '{stream}' cannot be checked for "
            "shard ownership: if its storage is shared between ranks, concurrent "
            "writes may corrupt it.",
        )

    def _warn_once(self, backend: IOBackend, message: str) -> None:
        """Log ``message`` the first time a backend of this type needs it."""
        key = type(backend).__name__
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message)


def _member_groups(items: Sequence[WorkItem]) -> list[set[tuple[int, ...]]]:
    """Group the member groups of ``items`` by reference time.

    Two items of different times never share a chunk once ``time`` is chunked
    at 1, so only the member groups of one time can collide.
    """
    grouped: dict[int, set[tuple[int, ...]]] = {}
    for item in items:
        if not item.member_ids:
            continue
        key = int(item.time.astype("datetime64[ns]").astype("int64"))
        grouped.setdefault(key, set()).add(
            tuple(sorted(int(member) for member in item.member_ids))
        )
    return list(grouped.values())


def _shared_block(
    dim: str,
    block: int,
    members: np.ndarray,
    groups: Sequence[set[tuple[int, ...]]],
    size: int,
) -> str | None:
    """Describe how a block of ``block`` positions along ``dim`` spans two items.

    Returns ``None`` when every block holds positions of at most one item. A
    block of one never does; along ``time`` any larger block does, since each
    item writes exactly one time. Along ``ensemble`` a block is safe while it
    stays inside one member group.
    """
    if block <= 1 or size <= 1:
        return None
    if dim == "time":
        return f"blocks {block} of the {size} item times together"
    index = {int(member): position for position, member in enumerate(members)}
    for per_time in groups:
        for start in range(0, len(members), block):
            stop = min(start + block, len(members))
            owners = sorted(
                group
                for group in per_time
                if any(start <= index[member] < stop for member in group)
            )
            if len(owners) > 1:
                return (
                    f"positions {start}-{stop - 1} hold the member groups "
                    f"{list(owners[0])} and {list(owners[1])}"
                )
    return None


def _zarr_blocks(
    backend: ZarrBackend, sizes: Mapping[str, int]
) -> dict[str, tuple[int, str]]:
    """Chunk and shard sizes of a Zarr store's distributed dimensions.

    ``ZarrBackend.chunks`` holds the requested chunk sizes, already updated from
    the arrays of a store that is being reopened, and a dimension without an
    entry is stored in a single chunk. Shards only come from a store written
    elsewhere, so they are read from the arrays when there are any.
    """
    chunks = {
        str(dim): int(size) for dim, size in getattr(backend, "chunks", {}).items()
    }
    blocks = {
        dim: (min(chunks.get(dim, sizes[dim]), sizes[dim]), "chunks")
        for dim in _DISTRIBUTED_DIMS
        if dim in sizes
    }
    for dim, shard in _stored_shards(backend).items():
        if dim in blocks and shard > blocks[dim][0]:
            blocks[dim] = (shard, "shards")
    return blocks


def _stored_shards(backend: ZarrBackend) -> dict[str, int]:
    """Shard size per dimension of the arrays already in a Zarr store."""
    shards: dict[str, int] = {}
    try:
        names = [name for name in backend.root.array_keys()]
        for name in names:
            array: Any = backend[name]
            dims = [str(dim) for dim in (array.metadata.dimension_names or ())]
            if array.shards is None or len(dims) != len(array.shards):
                continue
            for dim, shard in zip(dims, array.shards):
                shards[dim] = max(shards.get(dim, 0), int(shard))
    except Exception as error:  # noqa: BLE001 - geometry is best effort
        logger.debug(f"Could not read the shards of a Zarr store: {error}")
    return shards


def _async_zarr_blocks(
    backend: IOBackend, sizes: Mapping[str, int]
) -> dict[str, tuple[int, str]]:
    """Chunk and shard sizes of an async Zarr store's distributed dimensions.

    The backend chunks a parallel coordinate at 1 and rejects any
    ``chunked_coords`` entry that would widen it, so ``chunked_coords`` is read
    only as a safeguard; ``shard_coords`` groups those chunks into one written
    shard, which is what a rank must not share.
    """
    chunked: Mapping[Hashable, int] = getattr(backend, "chunked_coords", {})
    sharded: Mapping[Hashable, int] = getattr(backend, "shard_coords", {})
    blocks: dict[str, tuple[int, str]] = {}
    for dim in _DISTRIBUTED_DIMS:
        if dim not in sizes:
            continue
        chunk = int(chunked.get(dim, 1))
        shard = int(sharded.get(dim, chunk))
        blocks[dim] = (chunk, "chunks") if chunk >= shard else (shard, "shards")
    return blocks
