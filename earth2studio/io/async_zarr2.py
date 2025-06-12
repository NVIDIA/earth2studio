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

# import threading
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
    """Async Zarr v3"""

    def __init__(
        self,
        root: str,
        index_coords: dict[str, np.array] = {},
        fs: fsspec.spec.AbstractFileSystem = LocalFileSystem(),
        blocking: bool = True,
        max_pool_size: int = 4,
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
        self.overwrite = False

        # Async / multi-thread items
        self.blocking = blocking
        self.max_pool_size = max_pool_size
        self.async_timeout = async_timeout
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_pool_size
        )
        self.io_futures: list[concurrent.futures._base.Future] = []

        try:
            self.fs.mkdir(root)
        except FileExistsError:
            pass

        # Verify all index coordinate arrays have unique values
        for key, value in self.index_coords.items():
            if len(np.unique(value)) != len(value):
                raise ValueError(
                    f"Index coordinate array '{key}' contains duplicate values. "
                    + "All index coordinates must have unique values."
                )

        if "local" in fs.protocol:
            self.zstore = zarr.storage.LocalStore(root=root)
        elif "memory" in fs.protocol:
            self.zstore = zarr.storage.MemoryStore()
        else:
            # Needs to be an sync fs atm
            self.zstore = zarr.storage.FsspecStore(fs, path=root)

        # Set up root file system
        # try:
        #     loop = asyncio.get_running_loop()
        #     # self.zs = loop.run_until_complete(zarr.api.asynchronous.open(store=self.zstore, mode="a"))
        # except RuntimeError:
        #     # If the get running loop fails we know we arent in an async initialization
        #     # so the root zarr store will be sync.
        #     pass

        self.zs = zarr.api.synchronous.open(
            store=self.zstore, mode="a", **backend_kwargs
        )

    def _initialize_arrays(
        self,
        coords: CoordSystem,
        array_name: list[str],
        dtype: list[np.dtype],
    ) -> None:
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
            if self.zs.contains(key) and not self.overwrite:
                continue

            logger.debug(f"Writing coordinate array {key} to zarr store")
            array = self.zs.create_array(
                name=key,
                shape=value.shape,
                chunks=value.shape,
                dtype=value.dtype,
                dimension_names=[key],
            )
            array.setitem(Ellipsis, value)

        # ======
        # Data arrays
        # ======
        for name in array_name:
            if self.zs.contains(name) and not self.overwrite:
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
                f"Initializing array {name} with shape {shape} with chunks {chunks}"
            )
            self.zs.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                dimension_names=list(coords.keys()),
                overwrite=True,
            )

    def _scrub_coordinates(self, coords: CoordSystem) -> CoordSystem:
        coords = coords.copy()
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

    async def _init_async_zs(self) -> zarr.core.group.AsyncGroup:
        return await zarr.api.asynchronous.open(store=self.zstore, mode="a")

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
        # Check inputs
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(array_name, str):
            array_name = [array_name]
        if not (len(x) == len(array_name)):
            raise ValueError(
                f"Input tensors and array names must same length but got {len(x)} and {len(array_name)}."
            )

        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Note that this is blocking, which is intentional so we avoid race conditions
        # upon array creation
        self._initialize_arrays(
            coords, array_name, dtype=[torch_to_numpy_dtype_dict[x0.dtype] for x0 in x]
        )

        if self.blocking:
            loop.run_until_complete(
                asyncio.wait_for(
                    self.async_write(x, coords, array_name), timeout=self.async_timeout
                )
            )
        else:
            # First block if we have space in the pool
            # self._limit_pool_size(self.max_pool_size - 1)
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
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(array_name, str):
            array_name = [array_name]
        # TODO: Duplicate as write function, I know, but need to execute this here if
        # someone calls the async API
        self._initialize_arrays(
            coords, array_name, dtype=[torch_to_numpy_dtype_dict[x0.dtype] for x0 in x]
        )
        # Initialize async zarr store
        # TODO: doing this every write call could be avoided but needs to be done with
        # non-block threads
        zs = await self._init_async_zs()
        await self._write(x, coords, array_name, zs)

    async def _write(
        self,
        x: list[torch.Tensor],
        coords: CoordSystem,
        array_name: list[str],
        zs: zarr.core.group.AsyncGroup,
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
                zarr_coord = await (await zs.get(key)).getitem(...)  # Fetch coord array
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
                await (await zs.get(array)).setitem(Ellipsis, x[i])

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
            input_slice = [slice(None) for _ in x[i].shape]
            array_slice = [slice(None) for _ in x[i].shape]
            # Loop through each element of the index mesh (chunk to write)
            for j in range(n_writes):
                # Set the respective dim index in the slice
                for key, value in indexed_dims.items():
                    input_slice[value] = input_tensor_indices[key][j]
                    array_slice[value] = output_zarr_indices[key][j]

                # Finally set the selection in the array
                async def write(
                    name: str,
                    index: int,
                    array_slice: list[Any],
                    input_slice: list[Any],
                ) -> None:
                    """Small helper function"""
                    zarray = await zs.get(name)
                    await zarray.setitem(tuple(array_slice), x[index][*input_slice])

                writes.append(
                    asyncio.create_task(write(array, i, array_slice, input_slice))
                )
        # Every single chunk is written async...
        await asyncio.gather(*writes)
