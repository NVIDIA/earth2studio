# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import threading
from collections import OrderedDict
from collections.abc import Callable

# import threading
from importlib.metadata import version
from typing import Any

import fsspec
import nest_asyncio
import numpy as np
import torch
import zarr
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

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


class AsyncZarrBackend:
    """Async Zarr v3 IO Backend

    Warning
    -------
    This IO backend has a non-blocking mode which will execut IO writes in seperate
    threads. There is an assumption that the data being written will not be re-allocated
    during the writes.

    Parameters
    ----------
    root : str
        Root location to place zarr store
    index_coords : CoordSystem, optional
        Index coordinates. These are coordinates that will be used to define chunks in
        each zarr array. Each element in these coordinates will be written in parallel.
        Typically index coordinates reflect the dimensions that are being iteratively
        generated during inference, such as time or lead_time. The remaining coordinates
        of a given array will be populated upon the first write to the array, by default {}
    fs_factory : Callable[..., fsspec.spec.AbstractFileSystem], optional
        FSSpec file system factory method. This is a callable object that should return
        an instance of the desired filesystem to use, by default LocalFileSystem
    blocking : bool, optional
        Blocking write calls in the synchronous API. When set to fall, the IO backend
        will execute write calls in seperate threads. Users should call the `close()`
        API to ensure all threads have finished / cleaned up, by default True
    pool_size : int, optional
        The thread / async loop pool used with the synchronous write API in non-blocking
        mode, by default 4
    async_timeout : int, optional
        Async operation timeout for a given write operation, by default 600
    zarr_kwargs : _type_, optional
        Additional key work arguments to provide to the ` zarr.api.asynchronous.open`
        function, by default {"mode": "a"}
    """

    def __init__(
        self,
        root: str,
        index_coords: CoordSystem = OrderedDict({}),
        fs_factory: Callable[..., fsspec.spec.AbstractFileSystem] = LocalFileSystem,
        blocking: bool = True,
        pool_size: int = 4,
        async_timeout: int = 600,
        zarr_kwargs: dict[str, Any] = {"mode": "a"},
    ) -> None:
        # May need to trigger warning about this, needed to handle multi-threading!
        # But silent for now since 99% people wont know what this does / get confused
        AsyncFileSystem.cachable = False

        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            zarr_major_version = 2

        if zarr_major_version < 3:
            raise ImportError("This IO store only support Zarr 3.0 and above")

        if not callable(fs_factory):
            raise TypeError(
                "fs_factory must be a callable that returns a fsspec.spec.AbstractFileSystem"
            )

        self.overwrite = False  # Not formally supported
        self.index_coords = self._scrub_coordinates(
            index_coords
        )  # TODO: Need to validate these somewhere, should just make a 1:1 match requirement

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
        logger.warning(
            f"Setting up Zarr object pool of size {pool_size}, may take a bit"
        )
        for loop in self.loop_pool:
            future = asyncio.run_coroutine_threadsafe(
                self._initialize_zarr_group(root, fs_factory, zarr_kwargs), loop
            )
            zs0, fs0 = future.result()
            self.zarr_pool.append(zs0)
            self.fs_pool.append(fs0)

        # Set up base zarr group file system on current thread loop
        # (good for blocking calls and direct async)
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self.zs, self.fs = loop.run_until_complete(
            self._initialize_zarr_group(root, fs_factory, zarr_kwargs)
        )
        self.loop = loop

        # Verify all index coordinate arrays have unique values
        for key, value in self.index_coords.items():
            if len(np.unique(value)) != len(value):
                raise ValueError(
                    f"Index coordinate array '{key}' contains duplicate values. "
                    + "All index coordinates must have unique values."
                )
            # Not possible atm becaus async
            # if key in self.zs.array_keys():
            #     # Check that all elements in value are in index_coords array
            #     if not np.array_equal(self.zs[key], value):
            #         raise ValueError(
            #             f"Index coordinate array '{key}' already present in Zarr store and has different values than provided array"
            #         )

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

    async def _initialize_zarr_group(
        self,
        root: str,
        fs_factory: Callable[..., fsspec.spec.AbstractFileSystem],
        zarr_kwargs: dict[str, Any] = {},
    ) -> tuple[zarr.AsyncGroup, fsspec.AbstractFileSystem]:
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
            If some coords are index coords and container new values no in self.index_coords
        """
        # DOESNT WORK????
        # current_arrays = [key async for key in self.zs.array_keys()]
        # ======
        # Coordinate arrays
        # ======
        for key, value in coords.items():
            # Check coordinate in index coords
            if key in self.index_coords:
                # Check that all elements in value are in index_coords array
                if not np.all(np.isin(value, self.index_coords[key])):
                    raise ValueError(
                        f"Coordinate array '{key}' contains values not present in index_coords"
                    )
                value = self.index_coords[key]

            # Skip if coordinate array exists
            if await self.zs.contains(key) and not self.overwrite:
                continue

            logger.debug(f"Writing coordinate array {key} to zarr store")
            array = await self.zs.create_array(
                name=key,
                shape=value.shape,
                chunks=value.shape,
                dtype=value.dtype,
                dimension_names=[key],
                overwrite=False,
            )
            await array.setitem(Ellipsis, value)

        # ======
        # Data arrays
        # ======
        for name, dtype in zip(array_names, dtypes):
            # if self.zs.contains(name) and not self.overwrite:
            if await self.zs.contains(name) and not self.overwrite:
                continue
            array_coords = coords.copy()
            chunked: dict[str, int] = {
                key: value.shape[0] for key, value in array_coords.items()
            }
            for key, value in self.index_coords.items():
                if key in array_coords:
                    array_coords[key] = value
                    chunked[key] = 1

            shape: tuple[int] = tuple(value.shape[0] for value in array_coords.values())
            chunks = tuple(value for value in chunked.values())

            logger.debug(
                f"Initializing array {name} with shape {shape} with chunks {chunks} dtype {dtype}"
            )
            await self.zs.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                dimension_names=list(coords.keys()),
                overwrite=False,
            )

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
            # Dates types not supported in zarr 3.0 at the moment
            # https://github.com/zarr-developers/zarr-python/issues/2616
            # TODO: Remove once fixed
            if np.issubdtype(value.dtype, np.datetime64):
                logger.warning(
                    "Datetime64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                )
                coords[key] = value.astype("datetime64[ns]").astype("int64")

            if np.issubdtype(value.dtype, np.timedelta64):
                logger.warning(
                    "Timedelta64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                )
                coords[key] = value.astype("timedelta64[ns]").astype("int64")
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
        session = None
        try:
            session = await self.fs.set_session()
        except AttributeError:
            pass

        x = {array_name[i]: x[i] for i in range(len(x))}
        dtypes = [torch_to_numpy_dtype_dict[x0.dtype] for x0 in x.values()]

        coords = self._scrub_coordinates(coords)
        # Initialize arrays (coords and data) if needed
        # Note that this is blocking, which is intentional so we avoid race conditions
        # upon array creation
        await self._initialize_arrays(coords, list(x.keys()), dtypes)

        for key, value in coords.items():
            zarray = await self.zs.get(key)
            if key in self.index_coords:
                z0 = np.where(await zarray.getitem(slice(None)) == value)[
                    0
                ]  # Index of slice in zarr array
                if len(z0) == 0:
                    raise ValueError(
                        f"Could not find coordinate value {value} in zarr index coordinate array {key}. "
                        + "All index coordinates must be fully defined on construction of the IO object via `index_coords`."
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

    def _limit_pool_size(self, max_pool_size: int) -> None:
        """Helper function to limit the number of parallel io processes

        Parameters
        ----------
        max_pool_size : int
            Max number of io futures allowed to be queued
        """
        while len(self.io_futures) > max_pool_size:
            io_future = self.io_futures.pop(0)
            if not io_future.done():
                logger.debug("In IO thread pool throttle, limiting ")
                io_future.result()

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
        x, coords = self.loop.run_until_complete(
            self.prepare_inputs(x, coords, array_name)
        )

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
        await self._write(x, coords, self.zs, self.fs)

    async def _write(
        self,
        x: dict[str, torch.Tensor],
        coords: CoordSystem,
        zs: zarr.core.group.AsyncGroup,
        fs: fsspec.AbstractFileSystem,
    ) -> None:
        """_summary_

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of tensor(s) to be written to zarr arrays.
        coords : CoordSystem
            Coordinates of the passed data.
        zs : zarr.core.group.AsyncGroup
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
            session = await fs.set_session()
        except AttributeError:
            pass

        # Start with building a list of slices for every array and index that needs to
        # be written
        input_slices = []
        output_slices = []
        for i, key in enumerate(coords.keys()):
            in_slices = []
            out_slices = []
            if key in self.index_coords:
                for in_idx, out_idx in enumerate(
                    np.where(np.isin(self.index_coords[key], coords[key]))[0]
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
            # Loop through each element of the index mesh (chunk to write)
            for i in range(n_slices):
                # Finally set the selection in the array
                async def write(
                    name: str,
                    input_slice: list[slice],
                    array_slice: list[slice],
                ) -> None:
                    """Small helper function"""
                    zarray = await zs.get(name)
                    await zarray.setitem(tuple(array_slice), x[name][*input_slice])

                writes.append(
                    asyncio.create_task(
                        write(
                            array, list(input_slice_arr[i]), list(output_slice_arr[i])
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
        """
        # Clean up process pool
        self._limit_pool_size(0)

    def __del__(self) -> None:
        if len(self.io_futures) > 0:
            logger.warning(
                f"IO object found {len(self.io_futures)} in flight processes, cleaning up. Call `close()` manually to avoid this warning"
            )
            self.close()
