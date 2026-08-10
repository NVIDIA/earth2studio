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

import asyncio
import concurrent
import concurrent.futures
import datetime
import threading
from collections.abc import Callable
from dataclasses import dataclass

# import threading
from typing import Any

import fsspec
import fsspec.asyn
import numpy as np
import torch
import zarr
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem
from loguru import logger
from zarr import AsyncGroup
from zarr.core.array import CompressorsLike

from earth2studio.utils.type import CoordSystem

# https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
torch_to_numpy_dtype_dict = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


@dataclass
class _ShardSpec:
    """Geometry of a sharded Zarr array, read back from the store metadata"""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...]
    dtype: np.dtype
    fill_value: Any


class _ShardBuffer:
    """Host memory accumulation buffer for a single Zarr shard

    A shard is a single storage object holding many chunks, so a write that only
    covers part of a shard forces Zarr into a read-modify-write of the whole file.
    Concurrent partial writes to one shard therefore lose data. This buffer collects
    the chunks of a shard in host memory so that the shard can be emitted with a
    single write.

    The buffer is initialized to the array fill value, which makes a flush of an
    incomplete shard byte-for-byte identical to what an unwritten chunk would read
    back as. Partial shards are therefore safe to flush.

    Parameters
    ----------
    region : tuple[slice, ...]
        Region of the array covered by this shard, clipped to the array shape
    chunks : tuple[int, ...]
        Inner chunk shape of the array
    dtype : np.dtype
        Numpy data type of the array
    fill_value : Any
        Fill value of the array, used to initialize the buffer
    preexisting : bool
        If the shard object was already present in the store before this buffer was
        created. If so the buffer must be merged into the shard rather than
        overwriting it
    """

    __slots__ = (
        "data",
        "region",
        "chunks",
        "mask",
        "remaining",
        "claimed",
        "preexisting",
        "pending_copies",
        "pending_flush",
    )

    def __init__(
        self,
        region: tuple[slice, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        fill_value: Any,
        preexisting: bool,
    ) -> None:
        self.region = region
        self.chunks = chunks
        self.preexisting = preexisting
        self.claimed = False
        # Number of numpy copies into this buffer that are currently in flight
        # (started but mark() not yet called). async_flush must not begin setitem
        # while this is > 0, since the copy and the read race on buffer.data.
        self.pending_copies = 0
        # Set by async_flush when it claims the buffer while pending_copies > 0.
        # The last completing copy sees this and triggers the write.
        self.pending_flush = False

        extent = tuple(s.stop - s.start for s in region)
        if fill_value is None:
            self.data = np.zeros(extent, dtype=dtype)
        else:
            self.data = np.full(extent, fill_value, dtype=dtype)

        # Boolean mask over the inner chunk grid of this shard, makes repeated writes
        # of the same chunk idempotent
        grid = tuple(-(-e // c) for e, c in zip(extent, chunks))
        self.mask = np.zeros(grid, dtype=bool)
        self.remaining = int(np.prod(grid))

    def _grid_index(self, slices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Converts a global slice tuple into inner chunk grid coordinates"""
        return tuple(
            slice((s.start - r.start) // c, -(-(s.stop - r.start) // c))
            for s, r, c in zip(slices, self.region, self.chunks)
        )

    def mark(self, slices: tuple[slice, ...]) -> None:
        """Marks the inner chunks covered by a global slice tuple as filled

        Parameters
        ----------
        slices : tuple[slice, ...]
            Global (array level) slices of the region that was just copied in
        """
        index = self._grid_index(slices)
        self.remaining -= int(np.count_nonzero(~self.mask[index]))
        self.mask[index] = True

    def filled_regions(self) -> list[tuple[tuple[slice, ...], np.ndarray]]:
        """Global slices and data views covering only the chunks that were filled

        Used on the merge path, where writing the untouched parts of the buffer would
        clobber data already present in the shard. Emits a single region when the
        filled chunks form a dense box (the common case), otherwise falls back to one
        region per chunk.

        Returns
        -------
        list[tuple[tuple[slice, ...], np.ndarray]]
            Global slices and the buffer views to write there
        """
        if self.remaining == 0:
            return [(self.region, self.data)]

        filled = np.argwhere(self.mask)
        if filled.size == 0:
            return []

        low = filled.min(axis=0)
        high = filled.max(axis=0) + 1
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        if self.mask[box].all():
            cells = [tuple((int(a), int(b)) for a, b in zip(low, high))]
        else:
            cells = [tuple((int(i), int(i) + 1) for i in cell) for cell in filled]

        regions = []
        for cell in cells:
            local = tuple(
                slice(a * c, min(b * c, e))
                for (a, b), c, e in zip(cell, self.chunks, self.data.shape)
            )
            glob = tuple(
                slice(r.start + s.start, r.start + s.stop)
                for r, s in zip(self.region, local)
            )
            regions.append((glob, self.data[local]))
        return regions


class AsyncZarrBackend:
    """Async Zarr v3 IO Backend

    Warning
    -------
    This IO backend presently does not support overwritting existing Zarr stores. Only
    creation of new arrays or writing to existing.

    Warning
    -------
    Enabling sharding via `shard_coords` buffers chunks in host memory until a shard is
    complete, trading host memory for a smaller file count. Budget roughly

        max_inflight_shards * 4 * prod(shard_shape) * itemsize + pool_size * write_bytes

    per process, since a flushing shard costs several times its own size once Zarr's
    encoded copy is counted. Sharded writes are also slower than unsharded ones, which
    is hidden as long as the model takes longer to produce a step than the store takes
    to absorb it. This latency hiding only applies in non-blocking mode; with
    ``blocking=True`` each shard flush runs synchronously.

    Warning
    -------
    When sharding, a shard must not contain data owned by more than one process. This
    backend keeps every shard object to a single write by buffering its chunks, but that
    only holds within a process, separate ranks have separate buffers. If two ranks each
    hold part of the same shard they will both write it in full and the later write
    wins, silently discarding the other's data. Shard along a coordinate that each rank
    owns entirely (typically `lead_time`, since a rank runs a whole forecast), not along
    the coordinate the work is distributed over.

    Parameters
    ----------
    file_name : str
        Path location to place zarr store
    parallel_coords : CoordSystem
        Coordinates that enable parallel writes during inference. These coordinates
        specify which dimensions will be written in parallel via async operations,
        typically representing dimensions that are iteratively generated (such as time
        or lead_time). The chunk size for each of these dimensions will be set to 1.
        These coordinates should contain the complete set of values needed for the
        entire  inference pipeline. The remaining coordinates of a given array will be
        populated upon the first write to the respective array.
    fs_factory : Callable[..., fsspec.spec.AbstractFileSystem], optional
        FSSpec file system factory method. This is a callable object that should return
        an instance of the desired filesystem to use, by default LocalFileSystem
    blocking : bool, optional
        Blocking write calls in the synchronous API. When set to false, the IO backend
        will execute write calls in separate threads. Users should call the `close()`
        API to ensure all threads have finished / cleaned up, by default True
    pool_size : int, optional
        The thread / async loop pool used with the synchronous write API in non-blocking
        mode, by default 8
    async_timeout : int, optional
        Async operation timeout for a given write operation, by default 600. When
        sharding, the write that completes a shard carries the entire flush plus any
        wait for a concurrency slot, so this should be scaled with shard size and
        expected store throughput.
    zarr_kwargs : dict[str, Any], optional
        Additional keyword arguments to provide to the ` zarr.api.asynchronous.open`
        function, by default {"mode": "a"}
    zarr_codecs: CompressorsLike, optional
        Compression codec to use when creating any new arrays. If None, will use no
        compressor, by default None
    chunked_coords : dict[str, int], optional
        Chunk sizes for coordinates that are not in `parallel_coords`. By default any
        such coordinate is stored as a single chunk spanning its full length. Keys not
        present in a given array are ignored, by default {}
    shard_coords : dict[str, int], optional
        Number of elements per shard along the given coordinates, enabling Zarr v3
        sharding. Each value must be a multiple of that coordinate's chunk size, and any
        coordinate not listed uses a shard size equal to its chunk size. See the
        Sharding notes below. By default, {} (unsharded).
    max_inflight_shards : int, optional
        Maximum number of shard flushes allowed to run at once. Concurrent flushes are
        what keep sharded write throughput up, at the cost of holding that many shards
        in memory. Lower it if memory is tight, raise it if writes are the bottleneck
        and the store has bandwidth to spare, by default 4

    Raises
    ------
    ImportError
        If Zarr 2.0 is installed. This io backend only supports Zarr 3.0
    TypeError
        If fs_factory is not a callable, this should be a callable method not an object
    ValueError
        If a `shard_coords` value is not positive

    Notes
    -----
    Sharding

    Because every coordinate in `parallel_coords` is chunked with a size of 1, a large
    inference campaign can produce an enormous number of small files, which is a common
    way to exhaust an inode quota on a parallel filesystem. Sharding packs many chunks
    into a single storage object to avoid that. The chunk layout is unchanged, so
    readers still fetch one chunk at a time and only the file count changes.

    A shard is one object, so writing part of one would force Zarr to read, modify and
    rewrite all of it. To keep every shard to a single write, this backend accumulates
    a shard's chunks in host memory and writes it once complete, hence the memory and
    throughput tradeoffs in the warnings above.

    Shard sizes need not divide evenly into a coordinate. `close()` writes out any shard
    that never filled, using the array fill value where nothing was supplied, which
    reads back exactly as an unwritten chunk would. Writing into a shard already present
    in the store still works but falls back to a read-modify-write of the whole shard
    and logs a warning. That happens when `close()` or `flush()` is called mid run and
    the same shards are written again, or when restarting into a store left with
    incomplete shards, so aligning restart boundaries with the shard size keeps writes
    on the fast path.

    Sharding composes with `zarr_codecs`, which compresses the inner chunks within a
    shard, and with `chunked_coords`, which sets the chunk size of coordinates outside
    `parallel_coords`.
    """

    def __init__(
        self,
        file_name: str,
        parallel_coords: CoordSystem,
        fs_factory: Callable[..., fsspec.spec.AbstractFileSystem] = LocalFileSystem,
        blocking: bool = True,
        pool_size: int = 8,
        async_timeout: int = 600,
        zarr_kwargs: dict[str, Any] = {"mode": "a"},
        zarr_codecs: CompressorsLike = None,
        chunked_coords: dict[str, int] = {},
        shard_coords: dict[str, int] = {},
        max_inflight_shards: int = 4,
    ) -> None:
        # May need to trigger warning about this, needed to handle multi-threading!
        # But silent for now since people wont know what this means / get confused by an error message I think
        AsyncFileSystem.cachable = False

        if not callable(fs_factory):
            raise TypeError(
                "fs_factory must be a callable that returns a fsspec.spec.AbstractFileSystem"
            )

        self.overwrite = False  # Not formally supported
        self.parallel_coords = self._scrub_coordinates(parallel_coords.copy())
        # Parameter to also chunk some of the other dims if needed
        self.chunked_coords: dict[str, int] = dict(chunked_coords)
        self.shard_coords: dict[str, int] = dict(shard_coords)
        self.zarr_codecs = zarr_codecs

        for key, value in self.shard_coords.items():
            if value <= 0:
                raise ValueError(
                    f"Shard size for coordinate '{key}' must be positive but got {value}"
                )

        # Shard accumulation state, shared across every loop in the pool so all access
        # is guarded by a threading lock (the loops live in different threads, an
        # asyncio lock would provide no mutual exclusion between them)
        self._shard_lock = threading.Lock()
        self._shard_specs: dict[str, _ShardSpec | None] = {}
        self._shard_buffers: dict[tuple[str, tuple[int, ...]], _ShardBuffer] = {}
        self._flushed_shards: set[tuple[str, tuple[int, ...]]] = set()
        self._shard_flush_futures: dict[
            tuple[str, tuple[int, ...]], concurrent.futures.Future
        ] = {}
        self._inflight_flushes: list[concurrent.futures.Future] = []
        self._merge_warned: set[str] = set()
        self._live_warned = False

        if max_inflight_shards < 1:
            raise ValueError(
                f"max_inflight_shards must be at least 1 but got {max_inflight_shards}"
            )
        self.max_inflight_shards = max_inflight_shards

        # Async / multi-thread items
        self.blocking = blocking
        if blocking:
            pool_size = 1
        self.async_timeout = async_timeout
        self.io_futures: list[concurrent.futures._base.Future] = []
        self.pool_index = 0
        self.loop_pool = self._initialize_loop_pool(pool_size)
        self.fs_pool = []
        self.zarr_pool = []
        logger.debug(f"Setting up Zarr object pool of size {pool_size}, may take a bit")
        for loop in self.loop_pool:
            future = asyncio.run_coroutine_threadsafe(
                self._initialize_zarr_group(file_name, fs_factory, zarr_kwargs), loop
            )
            zs0, fs0 = future.result()
            self.zarr_pool.append(zs0)
            self.fs_pool.append(fs0)

        # Set up base zarr group file system on current thread loop
        # (good for blocking calls and direct async)
        loop = fsspec.asyn.get_loop()
        self.root, self.fs = fsspec.asyn.sync(
            loop, self._initialize_zarr_group, file_name, fs_factory, zarr_kwargs
        )
        fsspec.asyn.sync(loop, self._validate_parallel_coords)
        self.loop = loop

    def _initialize_loop_pool(
        self, max_pool_size: int
    ) -> list[asyncio.AbstractEventLoop]:
        """Initializes asyncio loop (thread) pool

        Parameters
        ----------
        max_pool_size : int
            Pool size

        Returns
        -------
        list[asyncio.AbstractEventLoop]
            List of asyncio event loops in seperate threads
        """
        loops = []
        for _ in range(max_pool_size):
            loops.append(asyncio.new_event_loop())
            threading.Thread(target=loops[-1].run_forever, daemon=True).start()
        return loops

    async def _validate_parallel_coords(self) -> None:
        """Runs a few checks on the parallel coords to make sure they are valid"""
        # Verify all index coordinate arrays have unique values
        for key, value in self.parallel_coords.items():
            if len(np.unique(value)) != len(value):
                raise ValueError(
                    f"Chunked coordinate array '{key}' contains duplicate values. "
                    + "All index coordinates must have unique values."
                )
            if await self.root.contains(key):
                # Check that all elements in value are in parallel_coords array
                data = await (await self.root.get(key)).getitem(slice(None))
                if not np.array_equal(data, value):
                    raise ValueError(
                        f"Parallel coordinate array '{key}' already present in Zarr store but has different values than provided array. "
                        + "This isn't allowed, either make them the same, create a new Zarr store (suggested) or modify the existing arrays manually."
                    )

    async def _initialize_zarr_group(
        self,
        root: str,
        fs_factory: Callable[..., fsspec.spec.AbstractFileSystem],
        zarr_kwargs: dict[str, Any] = {},
    ) -> tuple[AsyncGroup, fsspec.AbstractFileSystem]:
        """Initializes both the fsspec filesystem and zarr group, its critical this
        function is called inside the correct loop

        Parameters
        ----------
        root : str
            Root location of the zarr store
        fs_factory : Callable[..., fsspec.spec.AbstractFileSystem]
            fsspec factory method
        zarr_kwargs : dict[str, Any], optional
            Zarr open key word arguments, by default {}

        Returns
        -------
        tuple[zarr.AsyncGroup, fsspec.AbstractFileSystem]
            Initialzied zarr group and file system
        """
        fs = fs_factory()
        if "local" in fs.protocol:
            zstore = zarr.storage.LocalStore(root=root)
        elif "memory" in fs.protocol:
            # In in memory store we just reuse the same zarr object for the entire pool
            # async loop is not a concern here
            if len(self.zarr_pool) > 0:
                return self.zarr_pool[0], fs
            zstore = zarr.storage.MemoryStore()
        else:
            if not fs.asynchronous:
                raise TypeError(
                    f"Initialized file system {fs} needs to be asynchronous"
                )
            zstore = zarr.storage.FsspecStore(fs, path=root)

        # Zarr ≥3.1 reads a zarr.json consolidated-metadata snapshot
        # Any workflow calling zarr.consolidate_metadata on exit therefore makes arrays
        # created on the next run invisible to the backend's own lookups.
        if zarr_kwargs.get("use_consolidated"):
            logger.warning(
                "Ignoring use_consolidated, this backend requires live store"
                "membership to create and write arrays"
            )
        zarr_kwargs = {**zarr_kwargs, "use_consolidated": False}
        zs = await zarr.api.asynchronous.open(store=zstore, **zarr_kwargs)
        return zs, fs

    async def _initialize_arrays(
        self,
        coords: CoordSystem,
        array_names: list[str],
        dtypes: list[np.dtype],
    ) -> None:
        """Initializes arrays (data and coordinates)

        Parameters
        ----------
        coords : CoordSystem
            Coordinate system of arrays
        array_names : list[str]
            Array names
        dtypes : list[np.dtype]
            Numpy data type of array

        Raises
        ------
        ValueError
            If some coords are index coords and container new values no in self.parallel_coords
        """
        # ======
        # Coordinate arrays
        # ======
        for key, value in coords.items():
            # Check coordinate in index coords
            if key in self.parallel_coords:
                # Check that all elements in value are in parallel_coords array
                if not np.all(np.isin(value, self.parallel_coords[key])):
                    raise ValueError(
                        f"Coordinate array '{key}' contains values not present in parallel_coords"
                    )
                value = self.parallel_coords[key]

            # Skip if coordinate array exists
            if await self.root.contains(key) and not self.overwrite:
                continue

            logger.debug(f"Writing coordinate array {key} to zarr store")
            array = await self.root.create_array(
                name=key,
                shape=value.shape,
                chunks=value.shape,
                dtype=value.dtype,
                dimension_names=[key],
                overwrite=self.overwrite,
                compressors=self.zarr_codecs,
            )
            await array.setitem(Ellipsis, value)

        # ======
        # Data arrays
        # ======
        for name, dtype in zip(array_names, dtypes):
            # if self.root.contains(name) and not self.overwrite:
            if await self.root.contains(name) and not self.overwrite:
                continue
            array_coords = coords.copy()
            chunked: dict[str, int] = {
                key: value.shape[0] for key, value in array_coords.items()
            }
            for key, value in self.parallel_coords.items():
                if key in array_coords:
                    array_coords[key] = value
                    chunked[key] = 1
            chunked.update(
                {
                    key: value
                    for key, value in self.chunked_coords.items()
                    if key in array_coords
                }
            )

            shape: tuple[int] = tuple(value.shape[0] for value in array_coords.values())
            chunks = tuple(value for value in chunked.values())
            shards = self._compute_shards(name, array_coords, chunked, dtype)

            logger.debug(
                f"Initializing array {name} with shape {shape} with chunks {chunks} "
                + f"shards {shards} dtype {dtype}"
            )
            await self.root.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                shards=shards,
                dtype=dtype,
                dimension_names=list(coords.keys()),
                overwrite=self.overwrite,
                compressors=self.zarr_codecs,
            )

        self.overwrite = False

    def _compute_shards(
        self,
        name: str,
        array_coords: CoordSystem,
        chunked: dict[str, int],
        dtype: np.dtype,
    ) -> tuple[int, ...] | None:
        """Resolves the shard shape of an array from the user provided `shard_coords`

        Parameters
        ----------
        name : str
            Array name, used for error messages
        array_coords : CoordSystem
            Complete coordinate system of the array
        chunked : dict[str, int]
            Chunk size of each coordinate of the array
        dtype : np.dtype
            Numpy data type of the array, used for the memory estimate

        Returns
        -------
        tuple[int, ...] | None
            Shard shape, or None if this array should not be sharded

        Raises
        ------
        ValueError
            If a shard size is not a multiple of the corresponding chunk size
        """
        if not self.shard_coords:
            return None

        unmatched = set(self.shard_coords) - set(array_coords)
        if unmatched:
            logger.warning(
                f"shard_coords keys {sorted(unmatched)} do not appear in array '{name}' "
                + "and will be ignored. The array will be unsharded along those coordinates."
            )

        shards: dict[str, int] = {}
        for key in array_coords:
            chunk_size = chunked[key]
            if key not in self.shard_coords:
                shards[key] = (
                    chunk_size
                    if key in self.parallel_coords
                    else array_coords[key].shape[0]
                )
                continue

            shard_size = self.shard_coords[key]
            if shard_size % chunk_size != 0:
                raise ValueError(
                    f"Shard size {shard_size} for coordinate '{key}' of array '{name}' "
                    + f"must be a multiple of its chunk size {chunk_size}. "
                    + "Coordinates in `parallel_coords` have a chunk size of 1, others "
                    + "default to their full length unless set via `chunked_coords`."
                )
            dim_size = array_coords[key].shape[0]
            if key not in self.parallel_coords and shard_size < dim_size:
                raise ValueError(
                    f"Shard size {shard_size} for coordinate '{key}' of array '{name}' "
                    + f"is smaller than its length {dim_size}. Non-parallel coordinates "
                    + "are always written as a single slice so their shard size must "
                    + "cover the full dimension. Set the shard size to at least "
                    + f"{dim_size} or add '{key}' to `parallel_coords`."
                )
            if shard_size > dim_size:
                logger.warning(
                    f"Shard size {shard_size} for coordinate '{key}' of array '{name}' "
                    + f"exceeds its length {dim_size}, the shard will be truncated"
                )
            shards[key] = shard_size

        shard_shape = tuple(shards.values())
        # No dimension is actually grouped, skip the sharding codec entirely
        if shard_shape == tuple(chunked.values()):
            logger.debug(
                f"Shard shape of array {name} matches its chunk shape, not sharding"
            )
            return None

        buffer_bytes = int(np.prod(shard_shape)) * np.dtype(dtype).itemsize
        logger.info(
            f"Array {name} sharded with shape {shard_shape}, each in flight shard "
            + f"buffer holds {buffer_bytes / 1e9:.2f} GB of host memory"
        )
        return shard_shape

    def _scrub_coordinates(self, coords: CoordSystem) -> CoordSystem:
        """And cleaning / adjustment operations on coordinates, modifies in place

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        CoordSystem
            Scrubbed coordinate system
        """
        for key, value in coords.items():
            # Handle some datetime conversions for users
            if np.issubdtype(value.dtype, object):
                if isinstance(value[0], datetime.datetime):
                    coords[key] = value.astype("datetime64[ns]")
                elif isinstance(value[0], datetime.timedelta):
                    coords[key] = value.astype("timedelta64[ns]")

            if len(coords[key].shape) == 0:
                raise ValueError(
                    f"Coordinate {key} has a 0 shape, needs to be a 1D coordinate"
                )

        return coords

    async def prepare_inputs(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> tuple[dict[str, torch.Tensor], CoordSystem]:
        """Prepares input coordinates and tensors for writting

        This function is a blocking function that will run any needed input checks as
        well as handle the initialization of any arrays that are not present already
        inside the Zarr store. This function will ensure that writes of the input
        data / arrays at each index of an `index_coord` can be written in parallel.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Input tensors to write
        coords : CoordSystem
            Tensor coordinate system
        array_name : str | list[str]
            Array name(s) to write

        Returns
        -------
        tuple[dict[str, torch.Tensor], CoordSystem]
            Prepared tensor list, coordinate system and array names for writting
        """
        coords = coords.copy()

        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(array_name, str):
            array_name = [array_name]
        # Run input checks
        if not (len(x) == len(array_name)):
            raise ValueError(
                f"Input tensors and array names must same length but got {len(x)} and {len(array_name)}."
            )

        # If fsspec store has a aiohttp session, collect it so we can then close it
        # manually...
        # https://s3fs.readthedocs.io/en/latest/#async
        try:
            session = await self.fs.set_session(refresh=True)
        except AttributeError:
            session = None

        x = {array_name[i]: x[i] for i in range(len(x))}
        dtypes = [torch_to_numpy_dtype_dict[x0.dtype] for x0 in x.values()]

        coords = self._scrub_coordinates(coords.copy())
        # Initialize arrays (coords and data) if needed
        # Note that this is blocking, which is intentional so we avoid race conditions
        # upon array creation
        await self._initialize_arrays(coords, list(x.keys()), dtypes)
        await self._register_shard_specs(list(x.keys()))

        for key, value in coords.items():
            zarray = await self.root.get(key)
            if key in self.parallel_coords:
                z0 = np.where(np.isin(await zarray.getitem(slice(None)), value))[0]
                if len(z0) != value.shape[0]:
                    raise ValueError(
                        f"Could not find coordinate value {value} in zarr parallel coordinate array {key}. "
                        + "All index coordinates must be fully defined on construction of the IO object via `parallel_coords`."
                    )
            # Otherwise check that the coordinate system is the complete coordinate system
            # We do not support sliced writes of non-index coords... this is done for
            # thread safety reasons
            else:
                if not np.array_equal(value, await zarray.getitem(slice(None))):
                    raise ValueError(
                        f"Non-index coordinate {key} must match the complete coordinate system defined in zarr array. "
                        + "Sliced writes of non-index coordinates are not supported for thread safety reasons."
                    )

        if session:
            await session.close()

        return x, coords

    async def _register_shard_specs(self, array_names: list[str]) -> None:
        """Records the shard geometry of each array, read back from the Zarr metadata

        Reading the geometry back from the store (rather than recomputing it) keeps
        this correct when writing into an array that already exists, which may have
        been created with a different shard configuration.

        Parameters
        ----------
        array_names : list[str]
            Array names to register
        """
        for name in array_names:
            if name in self._shard_specs:
                continue
            zarray = await self.root.get(name)
            if zarray.shards is None:
                self._shard_specs[name] = None
                continue
            self._shard_specs[name] = _ShardSpec(
                shape=tuple(zarray.shape),
                chunks=tuple(zarray.chunks),
                shards=tuple(zarray.shards),
                dtype=zarray.dtype,
                fill_value=zarray.metadata.fill_value,
            )

    async def _shard_exists(
        self, name: str, zarray: Any, shard_index: tuple[int, ...]
    ) -> bool:
        """Checks if a shard object is already present in the store

        A shard that is already present cannot be overwritten wholesale, doing so would
        discard the chunks it already holds. This is what makes restarting into an
        existing store safe, the in memory bookkeeping alone cannot know what a previous
        process wrote.

        Parameters
        ----------
        name : str
            Array name
        zarray : zarr.AsyncArray
            Array to probe, must belong to the calling loop's Zarr group
        shard_index : tuple[int, ...]
            Index of the shard in the shard grid

        Returns
        -------
        bool
            True if the shard is present, or if the check could not be performed
        """
        with self._shard_lock:
            if (name, shard_index) in self._flushed_shards:
                return True
        try:
            chunk_key = zarray.metadata.chunk_key_encoding.encode_chunk_key(shard_index)
            store_path = zarray.store_path
            key = f"{store_path.path}/{chunk_key}" if store_path.path else chunk_key
            return await store_path.store.exists(key)
        except Exception as e:
            # Assume the shard exists, the merge path is slower but never destructive
            logger.debug(
                f"Could not probe shard {shard_index} of array {name} ({e}), "
                + "assuming it exists"
            )
            return True

    async def _stage_chunk(
        self,
        name: str,
        zarray: Any,
        spec: _ShardSpec,
        array_slice: tuple[slice, ...],
        data: np.ndarray,
    ) -> None:
        """Accumulates a single chunk into its shard buffer, flushing once complete

        Parameters
        ----------
        name : str
            Array name
        zarray : zarr.AsyncArray
            Array to write into, must belong to the calling loop's Zarr group
        spec : _ShardSpec
            Shard geometry of the array
        array_slice : tuple[slice, ...]
            Global slices of the array this chunk occupies
        data : np.ndarray
            Chunk data
        """
        # Concretize any full dimension slices so the shard math has real bounds
        array_slice = tuple(
            slice(0, size) if s.start is None else s
            for s, size in zip(array_slice, spec.shape)
        )
        shard_index = tuple(s.start // h for s, h in zip(array_slice, spec.shards))
        key = (name, shard_index)

        with self._shard_lock:
            buffer = self._shard_buffers.get(key)
            if buffer is not None:
                buffer.pending_copies += 1

        if buffer is None:
            # Probe outside the lock, a concurrent duplicate probe is harmless
            preexisting = await self._shard_exists(name, zarray, shard_index)
            region = tuple(
                slice(i * h, min((i + 1) * h, size))
                for i, h, size in zip(shard_index, spec.shards, spec.shape)
            )
            candidate = _ShardBuffer(
                region, spec.chunks, spec.dtype, spec.fill_value, preexisting
            )
            with self._shard_lock:
                buffer = self._shard_buffers.setdefault(key, candidate)
                buffer.pending_copies += 1
                live = len(self._shard_buffers)
            if live > len(self.loop_pool) and not self._live_warned:
                self._live_warned = True
                logger.warning(
                    f"{live} shard buffers are live, host memory use grows with this "
                    + "count. Consider a smaller shard size or writing coordinates in "
                    + "an order that completes shards sooner."
                )

        # Copy outside the lock — chunk regions are disjoint so concurrent _stage_chunk
        # calls on the same buffer do not conflict. pending_copies > 0 prevents
        # async_flush from starting setitem while this copy is in flight.
        local_slice = tuple(
            slice(s.start - r.start, s.stop - r.start)
            for s, r in zip(array_slice, buffer.region)
        )
        buffer.data[local_slice] = data

        # The lock is held only for bookkeeping, never across an await, so a large
        # flush on one thread cannot stall staging on another
        should_flush = False
        with self._shard_lock:
            buffer.pending_copies -= 1
            buffer.mark(array_slice)
            if buffer.claimed:
                # Claimed by async_flush while our copy was running; if we are the
                # last copy it left for us to flush
                if buffer.pending_copies == 0 and buffer.pending_flush:
                    should_flush = True
            elif buffer.remaining > 0:
                pass  # shard not yet full
            else:
                # Shard complete and unclaimed — we flush it
                buffer.claimed = True
                self._shard_buffers.pop(key, None)
                self._flushed_shards.add(key)
                should_flush = True

        if should_flush:
            await self._flush_buffer(name, zarray, key, buffer)

    async def _acquire_flush_slot(self, current: concurrent.futures.Future) -> None:
        """Waits until fewer than `max_inflight_shards` flushes are running

        Every concurrent flush holds its shard buffer plus whatever copies Zarr's
        sharding codec makes while encoding it, so unbounded flush concurrency is
        unbounded host memory. Waits on the oldest running flush rather than blocking
        the thread, since the flushes run across several event loops.

        Parameters
        ----------
        current : concurrent.futures.Future
            Future of the flush that is being admitted, completed by the caller
        """
        while True:
            with self._shard_lock:
                self._inflight_flushes = [
                    f for f in self._inflight_flushes if not f.done()
                ]
                if len(self._inflight_flushes) < self.max_inflight_shards:
                    self._inflight_flushes.append(current)
                    return
                oldest = self._inflight_flushes[0]
            try:
                await asyncio.wrap_future(oldest)
            except Exception as e:
                logger.debug(f"Flush waited on by the shard slot gate failed ({e})")

    async def _flush_buffer(
        self,
        name: str,
        zarray: Any,
        key: tuple[str, tuple[int, ...]],
        buffer: _ShardBuffer,
    ) -> None:
        """Writes a shard buffer out to the store

        Flushes of the same shard are chained so they cannot overlap, which matters on
        the merge path where each write is a read-modify-write of the whole shard.
        Flushes of different shards run concurrently up to `max_inflight_shards`.

        Parameters
        ----------
        name : str
            Array name
        zarray : zarr.AsyncArray
            Array to write into, must belong to the calling loop's Zarr group
        key : tuple[str, tuple[int, ...]]
            Buffer key, the array name and shard grid index
        buffer : _ShardBuffer
            Buffer to write, must already be claimed by the caller
        """
        with self._shard_lock:
            previous = self._shard_flush_futures.get(key)
            current: concurrent.futures.Future = concurrent.futures.Future()
            self._shard_flush_futures[key] = current

        try:
            if previous is not None and not previous.done():
                try:
                    await asyncio.wrap_future(previous)
                except Exception as e:
                    # Only ordering matters here, the failure itself is raised by
                    # whichever write originally scheduled that flush
                    logger.debug(
                        f"Preceding flush of shard {key[1]} of array {name} failed "
                        + f"({e}), continuing with this flush"
                    )

            # Admitted only after the same shard ordering wait above, so a flush
            # never holds a slot while waiting on another flush that needs one
            await self._acquire_flush_slot(current)

            if buffer.preexisting:
                if buffer.remaining > 0 and name not in self._merge_warned:
                    self._merge_warned.add(name)
                    logger.warning(
                        f"Writing into shards of array '{name}' that are already "
                        + "present in the store, these writes fall back to a "
                        + "read-modify-write of the entire shard. This happens when "
                        + "`close()` / `flush()` is called mid run or when restarting "
                        + "into an existing store with shards left incomplete."
                    )
                regions = buffer.filled_regions()
            else:
                # Fast path, the whole shard in one write with no read of prior state
                regions = [(buffer.region, buffer.data)]

            for region, values in regions:
                await zarray.setitem(region, values)
        finally:
            with self._shard_lock:
                if self._shard_flush_futures.get(key) is current:
                    del self._shard_flush_futures[key]
            current.set_result(None)

    async def async_flush(self) -> None:
        """Writes out every shard buffer that is still incomplete

        Shards that never filled are written with the array fill value in the positions
        that were never supplied, which reads back identically to an unwritten chunk.
        """
        with self._shard_lock:
            pending = list(self._shard_buffers.items())
            self._shard_buffers.clear()
            # Claim all buffers and add keys to _flushed_shards atomically so any
            # _stage_chunk that probes after this lock sees the key as flushed and
            # does not open a fresh preexisting=False buffer that would clobber the
            # write this flush is about to make.
            #
            # For buffers with copies still in flight (pending_copies > 0) we cannot
            # call setitem yet — the numpy copy is still writing into buffer.data.
            # Set pending_flush instead: the last completing copy will flush those.
            to_flush = []
            for key, buffer in pending:
                buffer.claimed = True
                self._flushed_shards.add(key)
                if buffer.pending_copies == 0:
                    to_flush.append((key, buffer))
                else:
                    buffer.pending_flush = True

        if not to_flush:
            return

        logger.debug(f"Flushing {len(to_flush)} incomplete shard buffers")
        array_names = {key[0] for key, _ in to_flush}
        zarrays = {name: await self.root.get(name) for name in array_names}
        await asyncio.gather(
            *[
                self._flush_buffer(key[0], zarrays[key[0]], key, buffer)
                for key, buffer in to_flush
            ]
        )

    def flush(self) -> None:
        """Drains in flight writes and writes out every incomplete shard buffer

        Draining first ensures no write is still accumulating into a buffer that is
        about to be flushed. All in flight writes are waited on even when one of
        them failed — a raise on the first error would abandon the rest mid write —
        and the first error is raised after the drain.
        """
        first_error: BaseException | None = None
        if self.io_futures:
            # Detach the list first so a cancelled future cannot leave drained
            # writes behind, and so the buffer flush below always runs
            io_futures, self.io_futures = self.io_futures, []
            concurrent.futures.wait(io_futures)
            for io_future in io_futures:
                try:
                    io_future.result()
                except BaseException as e:  # noqa: BLE001
                    if first_error is None:
                        first_error = e
        fsspec.asyn.sync(self.loop, self.async_flush)
        if first_error is not None:
            raise first_error

    def _limit_pool_size(self, max_pool_size: int) -> None:
        """Helper function to limit the number of parallel io processes

        Counts operations that are still running, not operations that have been
        submitted. This matters when sharding, where most writes only copy into a
        shard buffer and finish immediately while one in every shard's worth of
        writes performs the actual IO. Counting submissions would let a shard larger
        than the pool serialize every flush against the one before it.

        Parameters
        ----------
        max_pool_size : int
            Max number of in flight io futures allowed
        """
        pending = []
        for io_future in self.io_futures:
            if io_future.done():
                # Surfaces any error the write failed with, a completed future that
                # is never resulted swallows its exception
                io_future.result()
            else:
                pending.append(io_future)
        self.io_futures = pending

        while len(self.io_futures) > max_pool_size:
            logger.debug("In IO thread pool throttle, limiting ")
            # Waits for whichever write finishes first rather than the oldest one.
            # When sharding, writes that only fill a buffer finish quickly while the
            # write that flushes a shard takes far longer, so waiting on the head of
            # the queue would stall the caller on the slowest operation and serialize
            # every flush against the one before it
            done, not_done = concurrent.futures.wait(
                self.io_futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for io_future in done:
                io_future.result()
            self.io_futures = [f for f in self.io_futures if f in not_done]

    def add_array(
        self, coords: CoordSystem, array_name: str | list[str], **kwargs: dict[str, Any]
    ) -> None:
        """Pass through, arrays are initialized lazily in this io object"""
        # TODO: Warning?
        pass

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """Write data

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to zarr store.
        coords : OrderedDict
            Coordinates of the passed data.
        array_name : str | list[str]
            Name(s) of the array(s) that will be written to.
        """
        # Block this until complete, prevents race conditions when initialization
        x, coords = fsspec.asyn.sync(
            self.loop, self.prepare_inputs, x, coords, array_name
        )

        if not self.blocking:
            # prevents race conditions when the data is mutated in place before the
            # write is completed.
            x = {key: value.detach().to("cpu", copy=True) for key, value in x.items()}

        # Threads are cycled based on rotating index, pretty crude but works
        self._limit_pool_size(len(self.loop_pool) - 1)
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(
                self._write(
                    x,
                    coords,
                    self.zarr_pool[self.pool_index],
                    self.fs_pool[self.pool_index],
                ),
                timeout=self.async_timeout,
            ),
            self.loop_pool[self.pool_index],
        )

        if self.blocking:
            future.result()
        else:
            self.io_futures.append(future)
            self.pool_index = (self.pool_index + 1) % len(self.loop_pool)

    async def async_write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """Async write data

        Warning
        -------
        Unlike the non-blocking ``write``, no copy of the data is made here.
        The tensors must not be mutated or re-allocated until this coroutine
        completes — scheduling it as a task while stepping a state tensor in
        place will silently store the mutated values.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to zarr store.
        coords : OrderedDict
            Coordinates of the passed data.
        array_name : str | list[str]
            Name(s) of the array(s) that will be written to.
        """
        x, coords = await self.prepare_inputs(x, coords, array_name)
        await self._write(x, coords, self.root, self.fs)

    async def _write(
        self,
        x: dict[str, torch.Tensor],
        coords: CoordSystem,
        zs: AsyncGroup,
        fs: fsspec.AbstractFileSystem,
    ) -> None:
        """_summary_

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of tensor(s) to be written to zarr arrays.
        coords : CoordSystem
            Coordinates of the passed data.
        zs : zarr.AsyncGroup
            Zarr store to use
        fs : fsspec.AbstractFileSystem
            File system to use (relevant for session creation)
        """

        # Move data to CPU
        # TODO: could this be asynced?
        x = {key: value.detach().cpu().numpy() for key, value in x.items()}

        # If fsspec store has a aiohttp session, collect it so we can then close it
        # manually...
        # https://s3fs.readthedocs.io/en/latest/#async
        session = None
        try:
            session = await fs.set_session(refresh=True)
        except AttributeError:
            pass

        # Start with building a list of slices for every array and index that needs to
        # be written
        input_slices = []
        output_slices = []
        for i, key in enumerate(coords.keys()):
            in_slices = []
            out_slices = []
            if key in self.parallel_coords:
                for in_idx, out_idx in enumerate(
                    np.where(np.isin(self.parallel_coords[key], coords[key]))[0]
                ):
                    in_slices.append(slice(in_idx, in_idx + 1))
                    out_slices.append(slice(out_idx, out_idx + 1))
            else:
                in_slices.append(slice(None))
                out_slices.append(slice(None))
            output_slices.append(out_slices)
            input_slices.append(in_slices)

        # Mesh grid slices
        slice_mesh = np.meshgrid(*output_slices, indexing="ij")
        output_slice_arr = np.stack([mesh.flatten() for mesh in slice_mesh], axis=-1)

        slice_mesh = np.meshgrid(*input_slices, indexing="ij")
        input_slice_arr = np.stack([mesh.flatten() for mesh in slice_mesh], axis=-1)
        n_slices = output_slice_arr.shape[0]

        logger.debug(f"Writing {n_slices} chunks to {len(x)} Zarr arrays")
        writes = []
        for array in x.keys():
            zarray = await zs.get(array)
            spec = self._shard_specs.get(array)
            # Loop through each element of the index mesh (chunk to write)
            for i in range(n_slices):
                input_slice = tuple(input_slice_arr[i])
                array_slice = tuple(output_slice_arr[i])
                if spec is None:
                    # Unsharded, one chunk is one object so it is written directly
                    writes.append(
                        asyncio.create_task(
                            zarray.setitem(array_slice, x[array][input_slice])
                        )
                    )
                else:
                    # Sharded, accumulate in host memory and write once the shard is
                    # complete so each shard object is written exactly once
                    writes.append(
                        asyncio.create_task(
                            self._stage_chunk(
                                array, zarray, spec, array_slice, x[array][input_slice]
                            )
                        )
                    )
        # Every single chunk is written async...
        await asyncio.gather(*writes)
        if session:
            await session.close()

    def close(self) -> None:
        """Cleans up an remaining io processes that are currently running. Should be
        called explicitly at the end of an inference workflow to ensure all data has
        been written.

        When sharding is enabled this also writes out any shard that never filled. Note
        that calling this mid run and then writing into the same shards again forces a
        read-modify-write of those shards.
        """
        # Clean up process pool and write out any partially filled shards
        self.flush()

    def __del__(self) -> None:
        if not hasattr(self, "io_futures"):
            return
        if len(self.io_futures) > 0 or len(self._shard_buffers) > 0:
            logger.warning(
                f"IO object found {len(self.io_futures)} in flight processes and "
                + f"{len(self._shard_buffers)} buffered shards, cleaning up. "
                + "Call `close()` manually to avoid this warning"
            )
            self.close()
