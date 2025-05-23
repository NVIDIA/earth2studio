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

from collections.abc import Iterator
import hashlib
import inspect
import os
from string import Template
from typing import Any

from loguru import logger
import nest_asyncio
import asyncio
import xarray as xr
import fsspec
from fsspec.implementations.local import LocalFileSystem
import torch

from earth2studio.data.utils import datasource_cache_root
from earth2studio.utils.type import CoordSystem


class NetCDF4Backend:

    _index: int = 0

    def __init__(
        self,
        root: str,
        fs: fsspec.spec.AbstractFileSystem = LocalFileSystem(),
        ft: Template = Template("earth2studio_${index}.nc"),
        split_dim: str = "variable",
        blocking: bool = False,
        pool_size: int = 4,
        async_timeout: int = 600,
    ):
        self.root = root
        self.fs = fs
        self.ft = ft
        self.split_dim = split_dim  # TODO: Support None
        self.engine = "netcdf4"
        try:
            self.fs.mkdir(self.root)
        except FileExistsError:
            pass

        # Async items
        self.blocking = blocking
        self.pool_size = pool_size
        self.async_timeout = async_timeout
        # Change to asyncio.Semaphore
        self._io_futures = []

        self._scratch_space = "netcdf"

    def write(
        self, x: torch.Tensor, coords: CoordSystem, ft_kwargs: dict[str, Any] = {}
    ) -> None:
        """Write data tensor and coords to file

        Parameters
        ----------
        x : torch.tensor
            Tensor to save to file
        coords : CoordSystem
            Coordinate system
        ft_kwargs : dict[str, Any], optional
            File template key word args which will be used to generate the output file
            name, by default {}
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
                    self.write_async(x, coords, ft_kwargs), timeout=self.async_timeout
                )
            )
        else:
            # Create future but don't wait for it
            # Should maybe use tasks

            # Maybe should have seperate thread pool for write and upload

            # Create task is better here
            future = asyncio.ensure_future(
                asyncio.wait_for(
                    self.write_async(x, coords, ft_kwargs), timeout=self.async_timeout
                )
            )
            self._io_futures.append(future)

            # Wait for pool if above capacity
            loop.run_until_complete(self.throttle_io_pool(self.pool_size))

    async def throttle_io_pool(self, pool_size: int):
        """Simple function to await for pooled io tasks if over capacity

        Parameters
        ----------
        pool_size : int
            Target pool size
        """
        while len(self._io_futures) > pool_size:
            out = self._io_futures.pop(0)
            if inspect.isawaitable(out):
                await out

    async def write_async(
        self, x: torch.tensor, coords: CoordSystem, ft_kwargs: dict[str, Any] = {}
    ) -> None:
        """Async write data tensor and coords to file

        Parameters
        ----------
        x : torch.tensor
            Tensor to save to file
        coords : CoordSystem
            Coordinate system
        ft_kwargs : dict[str, Any], optional
            File template key word args which will be used to generate the output file
            name, by default {}
        """
        da = xr.DataArray(x.cpu().numpy(), coords=coords)
        ds = da.to_dataset(dim=self.split_dim)

        file_name = os.path.join(
            self.root, self.ft.safe_substitute(index=self._index, **ft_kwargs)
        )

        if "local" in self.fs.protocol:
            with self.fs.open(file_name, "wb") as f:
                await asyncio.to_thread(ds.to_netcdf, f, engine=self.engine)
        elif "memory" in self.fs.protocol:
            with self.fs.open(file_name, "wb") as f:
                await asyncio.to_thread(ds.to_netcdf, f, engine=self.engine)
        else:
            temp_file = os.path.join(self.scratch, f"temp_{file_name}")
            # Possible to maybe use a in-memory pipe to avoid the use of a temp file
            with open(temp_file, "rb") as local_file:
                await asyncio.to_thread(ds.to_netcdf, local_file, engine=self.engine)
            # TODO: Check this works / is best...
            if self.fs.async_impl:
                # Maybe different thread pool here???
                await self.fs._put(temp_file, file_name)
            else:
                await asyncio.to_thread(self.fs.put, temp_file, file_name)

            os.remove(temp_file)

        # Bump cls write index
        # potential not thread safe, todo
        self._index += 1

    def consolidate(
        self,
        ft_iter: Iterator[dict[str, Any] | Iterator],
        concat_dims: list[str],
        file_name: str = "consolidated.nc",
        replace: bool = False,
    ):
        """Consolidates a set of io objects into a single netcdf

        Parameters
        ----------
        ft_iter : Iterator[dict[str, Any]  |  Iterator]
            File template iterator, this is a collection of template keyword
            dictionaries that will be templated into file names for merging. This is
            a n-deep interable of dictionaries with hashable values.
        concat_dims : list[str]
            List of dimensions to concatenate, positioning is based on order provided
            in ft_iter
        file_name : str, optional
            Output file name, by default "consolidated.nc"
        replace : bool, optional
            The intermediate templated files consolidated will be deleted, by default False

        Raises
        ------
        FileNotFoundError
            If any of the requirest files based on ft_iter do not exist
        TypeError
            Invalid object inside ft_iter
        """
        file_names = []

        def load_io_objects(nested_iter: Iterator[Any]) -> None:
            output = []
            for item in nested_iter:
                if isinstance(item, dict):
                    file_name = os.path.join(self.root, self.ft.safe_substitute(**item))
                    if not os.path.exists(file_name):
                        raise FileNotFoundError(f"File {file_name} not found")

                    file_names.append(file_name)
                    if "local" in self.fs.protocol:
                        output.append(xr.open_dataset(file_name, engine=self.engine))
                    else:
                        # ValueError: can only read bytes or file-like objects with engine='scipy' or 'h5netcdf'
                        file = self.fs.open(file_name)
                        output.append(xr.open_dataset(file, engine="scipy"))
                elif isinstance(item, Iterator):
                    output.append(load_io_objects(item))
                else:
                    raise TypeError("Invalid type")
            return output

        # The lift
        consolidated_ds = xr.combine_nested(load_io_objects(ft_iter), concat_dims)

        # Write ops
        file_name = os.path.join(self.root, file_name)
        if "local" in self.fs.protocol:
            consolidated_ds.to_netcdf(file_name, engine=self.engine)
        elif "memory" in self.fs.protocol:
            with self.fs.open(file_name, "wb") as f:
                consolidated_ds.to_netcdf(f, engine=self.engine)
        else:
            temp_file_name = f"{hashlib.sha256(file_name.encode())}.nc"
            consolidated_ds.to_netcdf(temp_file_name, engine=self.engine)
            # Move to remote file store (TODO async)
            with self.fs.open(file_name, "wb") as f:
                with open(temp_file_name, "rb") as temp:
                    f.write(temp.read())

            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

        if replace:
            for file_name in file_names:
                self.fs.delete(file_name)

    # IDK if this is needed
    async def consolidate_async(self, files: Iterator[dict[str, Any] | Iterator]):
        pass

    def close(self):
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
        loop.run_until_complete(asyncio.wait_for(self.throttle_io_pool(0), timeout=self.async_timeout))

    def __del__(self):
        if len(self._io_futures) > 0:
            logger.warning(
                f"IO object found {len(self._io_futures)} in flight processes, cleaning up. Call `close()` manually to avoid this warning"
            )
            self.close()

    @property
    def scratch(self) -> str:
        """Return appropriate scratch location."""
        # TODO: replace the data cache root
        cache_location = os.path.join(datasource_cache_root(), self._scratch_space)
        if not os.path.exists(cache_location):
            os.makedirs(cache_location)

        return cache_location
