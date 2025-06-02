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
from asyncio.events import AbstractEventLoop
from importlib.metadata import version
from typing import Any

import fsspec
import nest_asyncio
import numpy as np
import torch
import zarr
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

from earth2studio.utils.type import CoordSystem

# https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
torch_to_numpy_dtype_dict = {
    torch.bool: np.bool,
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
    """Asynchrounous Zarr IO Backend
    This IO object provides built in non-blocking IO writes to Zarr stores, ideal for
    inference pipelines that have a large amount of write volume to local or remote
    storage systems.

    Note
    ----
    This IO backend has notably different functionality and intended APIs than the
    :py:class:`earth2studio.io.ZarrBackend`. While cross compatability is supported,
    users are encouraged to familiarize themselves with the other APIs this provides.

    Warning
    -------
    When running in non-blocking mode, users should be mindful to not modify

    Parameters
    ----------
    root : str
        Location for IO object
    index_coords : dict[str, np.array]
        Coordinates that will be indexed along between write calls. The arrays will be
        chunked with a size of 1 for these coordinates to ensure parallel writes do not
        conflict. Other "domain" cooridnates will be lazily determined on first write
        call, by default {}
    fs : fsspec.spec.AbstractFileSystem, optional
        Underlying fsspec filesystem for the Zarr store to use. If this store is remote
        the IO object will upload to this remote store on write call, by default
        LocalFileSystem()
    blocking : bool, optional
        Blocking sync write calls. If false IO writes will be added to a thread pool and
        not wait for completion before returning, by default True
    max_pool_size : int, optional
        Max workers in async io thread pool. Only applied when using sync call function.
        Should the number of needed IO threads exceed the pool size, sync write calls
        will become blocking, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600
    backend_kwargs : dict[str, Any], optional
        Key word arguments for zarr.Group root object, by default {"overwrite": False}

    Note
    ----
    For keyword argument options see: https://zarr.readthedocs.io/en/latest/api/zarr/api/asynchronous/index.html#zarr.api.asynchronous.open_group
    """

    def __init__(
        self,
        root: str,
        index_coords: dict[str, np.array] = {},
        fs: fsspec.spec.AbstractFileSystem = LocalFileSystem(),
        blocking: bool = True,
        max_pool_size: int = 4,
        loop: AbstractEventLoop | None = None,
        async_timeout: int = 600,
        backend_kwargs: dict[str, Any] = {},
    ) -> None:

        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            zarr_major_version = 2

        if zarr_major_version < 3:
            raise ImportError("This IO store only support Zarr 3.0 and above")

        self.index_coords = index_coords
        self.fs = fs

        # Async / multi-thread items
        self.blocking = blocking
        self.max_pool_size = max_pool_size
        self.async_timeout = async_timeout
        self.loop = loop
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_pool_size
        )
        self.init_sem = threading.Semaphore(1)
        self.io_futures: list[concurrent.futures._base.Future] = []

        try:
            self.fs.mkdir(root)
        except FileExistsError:
            pass

        if "local" in fs.protocol:
            zstore = zarr.storage.LocalStore(root=root)
        elif "memory" in fs.protocol:
            zstore = zarr.storage.MemoryStore()
        else:
            # Needs to be an sync fs atm
            zstore = zarr.storage.FsspecStore(fs, path=root)

        self.root = asyncio.run(
            zarr.api.asynchronous.open(store=zstore, mode="a", **backend_kwargs)
        )

        # Initialize index coords
        # Verify all index coordinate arrays have unique values
        for key, value in self.index_coords.items():
            if len(np.unique(value)) != len(value):
                raise ValueError(
                    f"Index coordinate array '{key}' contains duplicate values. All index coordinates must have unique values."
                )
        asyncio.run(self._initilize_coords(self.index_coords))

    async def _initilize_coords(
        self, coords: dict[str, np.array], overwrite_coords: bool = False
    ) -> None:
        """Initializes the provided coordinate dictionary as an array in the zarr store

        Parameters
        ----------
        coords : dict[str, np.array]
            Coordinates to write to arrays
        overwrite_coords : bool
            Overwrite any coordinate arrays that already exist inside the zarr store
            (assuming overwrite is enabled for the root object), by default false
        """

        for key, value in coords.items():

            if not overwrite_coords and await self.root.contains(key):
                continue

            logger.debug(f"Writing coordinate array {key} to zarr store")

            array = await self.root.create_array(
                name=key,
                shape=value.shape,
                chunks=value.shape,
                dtype=value.dtype,
                dimension_names=[key],
            )
            await array.setitem(Ellipsis, value)

    async def _initilize_array(
        self,
        name: str,
        slice_coords: dict[str, np.array],
        dtype: np.dtype = "float",
    ) -> None:
        """Initializes an array using the coordinate system of a slice and the index
        coordinates

        Parameters
        ----------
        name: str
            Name of the array in the Zarr store
        slice_coords : dict[str, np.array]
            Coordinates system of array slice that will be initally written
        dtype: np.DTypeLike = "float"
            Data type of the array, by default float
        """
        coords = slice_coords.copy()
        chunked: dict[str, int] = {key: value.shape[0] for key, value in coords.items()}
        for key, value in self.index_coords.items():
            if key in coords:
                coords[key] = value
                chunked[key] = 1

        shape: tuple[int] = tuple(value.shape[0] for value in coords.values())
        chunks = tuple(value for value in chunked.values())

        logger.debug(
            f"Initializing array {name} with shape {shape} with chunks {chunks}"
        )
        await self.root.create_array(
            name=name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            dimension_names=list(coords.keys()),
        )

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

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        **kwargs: Any,
    ) -> None:
        """Pass through function for compatability, does nothing

        To create an array, immediately call the write function with a input slice. The
        IO backend will then initialize the array based on the index_coords provided in
        the constructor and coordinates of the slice to write.
        """
        logger.warning("AsyncZarrBackend does not require add_array calls...")
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
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.blocking:
            loop.run_until_complete(
                asyncio.wait_for(
                    self.async_write(x, coords, array_name), timeout=self.async_timeout
                )
            )
        else:
            # First block if we have space in the pool
            self._limit_pool_size(self.max_pool_size - 1)
            # Launch async write
            io_future = self.thread_pool.submit(
                asyncio.run,
                asyncio.wait_for(
                    self.async_write(x, coords, array_name),
                    timeout=self.async_timeout,
                ),
            )

            self.io_futures.append(io_future)

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
        # Check inputs
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(array_name, str):
            array_name = [array_name]
        if not (len(x) == len(array_name)):
            raise ValueError(
                f"Input tensors and array names must same length but got {len(x)} and {len(array_name)}."
            )

        # Check to see whats intialized, only one process can do this at a time
        # Should be a one time process, so lock here doesnt matter too much for perf
        with self.init_sem:
            new_coords = {}
            for key, value in coords.items():
                if not await self.root.contains(key):
                    new_coords[key] = value
            await self._initilize_coords(new_coords)

            # Initialize arrays in parallel if they don't exist
            tasks = []
            for i, name in enumerate(array_name):
                # TODO: If does exist already, check that coords are consistent
                if not await self.root.contains(name):
                    x_dtype = torch_to_numpy_dtype_dict[x[i].dtype]
                    tasks.append(self._initilize_array(name, coords, x_dtype))
            if tasks:
                await asyncio.gather(*tasks)

        # Move data to CPU
        # TODO: could this be asynced?
        x = [x0.detach().cpu().numpy() for x0 in x]

        # Set up write tasks
        indexed_dims = {}
        input_tensor_indices = {}
        output_zarr_indices = {}
        for i, key in enumerate(coords.keys()):
            # TODO: this should also check the coord is of right size if not index coord
            if key in self.index_coords:
                indexed_dims[key] = i
                input_tensor_indices[key] = np.arange(coords[key].shape[0])
                # Convert input indices into zarr array indices
                # Probably cooler way to do this, but this is readable
                z_idx = []
                zarr_coord = await (await self.root.get(key)).getitem(
                    ...
                )  # Fetch coord array
                for i in range(coords[key].shape[0]):
                    z0 = np.where(zarr_coord == coords[key][i : i + 1])[
                        0
                    ]  # Index of slice in zarr array
                    if len(z0) == 0:
                        raise ValueError(
                            f"Could not find coordinate value {coords[key][i:i+1]} in zarr coordinate array {key}. "
                            + "All index coordinates must be fully defined on construction of the IO object via `index_coords`."
                        )
                    z_idx.append(z0[0])
                output_zarr_indices[key] = np.array(z_idx)

        # If no indexed coords just write and return entire array
        if len(input_tensor_indices) == 0:
            logger.debug("No indexed coordinates present, writing entire Zarr array")
            for i, array in enumerate(array_name):
                await (await self.root.get(name)).setitem(Ellipsis, x[i])

        # Mesh together all indices (i.e. all chunk indexes that need writing)
        # Basically we are getting a full array of index combinations
        index_mesh = np.meshgrid(*list(input_tensor_indices.values()), indexing="ij")
        input_tensor_indices = {
            key: index_mesh[i].flatten()
            for i, key in enumerate(input_tensor_indices.keys())
        }

        index_mesh = np.meshgrid(*list(output_zarr_indices.values()), indexing="ij")
        output_zarr_indices = {
            key: index_mesh[i].flatten()
            for i, key in enumerate(output_zarr_indices.keys())
        }

        n_writes = index_mesh[0].size
        logger.debug(f"Writing {n_writes} chunks to {len(array_name)} Zarr arrays")
        writes = []
        for i, array in enumerate(array_name):
            data = x[i]
            input_slice = [slice(None) for _ in data.shape]
            array_slice = [slice(None) for _ in data.shape]
            # Loop through each element of the index mesh (chunk to write)
            for i in range(n_writes):
                # Set the respective dim index in the slice
                for key, value in indexed_dims.items():
                    input_slice[value] = input_tensor_indices[key][i]
                    array_slice[value] = output_zarr_indices[key][i]

                # Finally set the selection in the array
                async def write(
                    name: str,
                    array_slice: list[Any],
                    input_slice: list[Any],
                ) -> None:
                    """Small helper function"""
                    zarray = await self.root.get(name)
                    await zarray.setitem(tuple(array_slice), data[*input_slice])

                writes.append(
                    asyncio.create_task(write(array, array_slice, input_slice))
                )
        # Every single chunk is written async...
        await asyncio.gather(*writes)

    def read(
        self, coords: CoordSystem, array_name: str, device: torch.device = "cpu"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current zarr group using the passed array_name.

        Parameters
        ----------
        coords : OrderedDict
            Coordinates of the data to be read.
        array_name : str | list[str]
            Name(s) of the array(s) to read from.
        device : torch.device
            device to place the read data from, by default 'cpu'
        """
        raise NotImplementedError("Not implemented yet")

    def close(self) -> None:
        """Cleans up an remaining io processes that are currently running. Should be
        called explicitly at the end of an inference workflow to ensure all data has
        been written.
        """
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Clean up process pool
        self._limit_pool_size(0)
        self.thread_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        if len(self.io_futures) > 0:
            logger.warning(
                f"IO object found {len(self.io_futures)} in flight processes, cleaning up. Call `close()` manually to avoid this warning"
            )
            self.close()
