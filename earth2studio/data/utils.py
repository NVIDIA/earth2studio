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

from __future__ import annotations

import inspect
import os
import tempfile
import time
import typing
import weakref
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from hashlib import sha256
from inspect import signature
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

import numpy as np
import torch
import xarray as xr
from fsspec import filesystem
from fsspec.asyn import AsyncFileSystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.utils import isfilelike
from loguru import logger

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.time import (
    leadtimearray_to_timedelta,
    timearray_to_datetime,
    to_time_array,
)
from earth2studio.utils.type import CoordSystem, LeadTimeArray, TimeArray, VariableArray

if TYPE_CHECKING:
    from fsspec.implementations.cache_mapper import AbstractCacheMapper


def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
    interp_to: CoordSystem = None,
    interp_method: str = "nearest",
) -> tuple[torch.Tensor, CoordSystem]:
    """Utility function to fetch data for models and load data on the target device.
    If desired, xarray interpolation/regridding in the spatial domain can be used
    by passing a target coordinate system via the optional `interp_to` argument.

    Parameters
    ----------
    source : DataSource
        The data source to fetch from
    time : TimeArray
        Timestamps to return data for (UTC).
    variable : VariableArray
        Strings or list of strings that refer to variables to return
    lead_time : LeadTimeArray, optional
        Lead times to fetch for each provided time, by default
        np.array(np.timedelta64(0, "h"))
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"
    interp_to : CoordSystem, optional
        If provided, the fetched data will be interpolated to the coordinates
        specified by lat/lon arrays in this CoordSystem
    interp_method : str
        Interpolation method to use with xarray (by default 'nearest')

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    sig = signature(source.__call__)

    if "lead_time" in sig.parameters:
        # Working with a Forecast Data Source
        da = source(time, lead_time, variable)  # type: ignore

    else:
        da = []
        for lead in lead_time:
            adjust_times = np.array([t + lead for t in time], dtype="datetime64[ns]")
            da0 = source(adjust_times, variable)  # type: ignore
            da0 = da0.expand_dims(dim={"lead_time": 1}, axis=1)
            da0 = da0.assign_coords(lead_time=np.array([lead], dtype="timedelta64[ns]"))
            da0 = da0.assign_coords(time=time)
            da.append(da0)

        da = xr.concat(da, "lead_time")

    return prep_data_array(
        da,
        device=device,
        interp_to=interp_to,
        interp_method=interp_method,
    )


def prep_data_array(
    da: xr.DataArray,
    device: torch.device = "cpu",
    interp_to: CoordSystem | None = None,
    interp_method: str = "nearest",
) -> tuple[torch.Tensor, CoordSystem]:
    """Prepares a data array from a data source for inference workflows by converting
    the data array to a torch tensor and the coordinate system to an OrderedDict.

    If desired, xarray interpolation/regridding in the spatial domain can be used
    by passing a target coordinate system via the optional `interp_to` argument.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"
    interp_to : CoordSystem, optional
        If provided, the fetched data will be interpolated to the coordinates
        specified by lat/lon arrays in this CoordSystem
    interp_method : str
        Interpolation method to use with xarray (by default 'nearest')

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    # Initialize the output CoordSystem
    out_coords = OrderedDict()
    for dim in da.coords.dims:
        if dim in da.coords:
            out_coords[dim] = np.array(da.coords[dim])

    # Fetch data and regrid if necessary
    if interp_to is not None:
        if len(interp_to["lat"].shape) != len(interp_to["lon"].shape):
            raise ValueError(
                "Discrepancy in interpolation coordinates: latitude has different number of dims than longitude"
            )

        if "lat" not in da.dims:
            # Data source uses curvilinear coordinates
            if interp_method != "linear":
                raise ValueError(
                    "fetch_data does not support interpolation methods other than linear when data source has a curvilinear grid"
                )
            interp = LatLonInterpolation(
                lat_in=da["lat"].values,
                lon_in=da["lon"].values,
                lat_out=interp_to["lat"],
                lon_out=interp_to["lon"],
            ).to(device)
            data = torch.Tensor(da.values).to(device)
            out = interp(data)

            # HARD CODE FOR STORMCAST
            # TODO: FIX THIS BY CORRECTING STORMCAST COORDINATES
            if "hrrr_y" in out_coords:
                del out_coords["hrrr_y"]
            if "hrrr_x" in out_coords:
                del out_coords["hrrr_x"]
        else:
            if len(interp_to["lat"].shape) > 1 or len(interp_to["lon"].shape) > 1:
                # Target grid uses curvilinear coordinates: define internal dims y, x
                target_lat = xr.DataArray(interp_to["lat"], dims=["y", "x"])
                target_lon = xr.DataArray(interp_to["lon"], dims=["y", "x"])
            else:
                target_lat = xr.DataArray(interp_to["lat"], dims=["lat"])
                target_lon = xr.DataArray(interp_to["lon"], dims=["lon"])

            da = da.interp(
                lat=target_lat,
                lon=target_lon,
                method=interp_method,
            )

            out = torch.Tensor(da.values).to(device)

        out_coords["lat"] = interp_to["lat"]
        out_coords["lon"] = interp_to["lon"]

    else:
        out = torch.Tensor(da.values).to(device)
        if "lat" in da.coords and "lat" not in da.coords.dims:
            # Curvilinear grid case: lat/lon coords are 2D arrays, not in dims
            out_coords["lat"] = da.coords["lat"].values
            out_coords["lon"] = da.coords["lon"].values
        else:
            for dim in da.coords.dims:
                if dim not in ["time", "lead_time", "variable"]:
                    out_coords[dim] = np.array(da.coords[dim])

    return out, out_coords


def prep_data_inputs(
    time: datetime | list[datetime] | TimeArray,
    variable: str | list[str] | VariableArray,
) -> tuple[list[datetime], list[str]]:
    """Simple method to pre-process data source inputs into a common form

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Datetime, list of datetimes or array of np.datetime64 to fetch
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables

    Returns
    -------
    tuple[list[datetime], list[str]]
        Time and variable lists
    """
    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    if isinstance(time, np.ndarray):  # np.datetime64 -> datetime
        time = timearray_to_datetime(time)

    return time, variable


def prep_forecast_inputs(
    time: datetime | list[datetime] | TimeArray,
    lead_time: timedelta | list[timedelta] | LeadTimeArray,
    variable: str | list[str] | VariableArray,
) -> tuple[list[datetime], list[timedelta], list[str]]:
    """Simple method to pre-process forecast source inputs into a common form

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Datetime, list of datetimes or array of np.datetime64 to fetch
    lead_time: timedelta | list[timedelta], LeadTimeArray
        Timedelta, list of timedeltas or array of np.timedelta to fetch
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables

    Returns
    -------
    tuple[list[datetime], list[timedelta], list[str]]
        Time, lead time, and variable lists
    """
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(lead_time, np.ndarray):  # np.timedelta64 -> timedelta
        lead_time = leadtimearray_to_timedelta(lead_time)

    time, variable = prep_data_inputs(time, variable)

    return time, lead_time, variable


def datasource_to_file(
    file_name: str,
    source: DataSource,
    time: list[str] | list[datetime] | TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    backend: Literal["netcdf", "zarr"] = "netcdf",
    chunks: dict[str, int] = {"variable": 1},
    dtype: np.dtype | None = None,
) -> None:
    """Utility function that can be used for building a local data store needed
    for an inference request. This file can then be used with the
    :py:class:`earth2studio.data.DataArrayFile` data source to load data from file.
    This is useful when multiple runs of the same input data is needed.

    Parameters
    ----------
    file_name : str
        File name of output NetCDF
    source : DataSource
        The original data source to fetch from
    time : list[str] | list[datetime] | list[np.datetime64]
        List of time strings, datetimes or np.datetime64 (UTC)
    variable : VariableArray
        Strings or list of strings that refer to variables to return
    lead_time : LeadTimeArray, optional
        Lead times to fetch for each provided time, by default
        np.array(np.timedelta64(0, "h"))
    backend : Literal["netcdf", "zarr"], optional
        Storage backend to save output file as, by default "netcdf"
    chunks : dict[str, int], optional
        Chunk sizes along each dimension, by default {"variable": 1}
    dtype : np.dtype, optional
        Data type for storing data
    """
    if isinstance(time, datetime):
        time = [time]

    time = to_time_array(time)

    # Spot check the write location is okay before pull
    testfile = tempfile.TemporaryFile(dir=Path(file_name).parent.resolve())
    testfile.close()

    # Compile all times
    for lead in lead_time:
        adjust_times = np.array([t + lead for t in time], dtype="datetime64[ns]")
        time = np.concatenate([time, adjust_times], axis=0)
    time = np.unique(time)

    # Fetch
    da = source(time, variable)
    da = da.assign_coords(time=time)
    da = da.chunk(chunks=chunks)

    if dtype is not None:
        da = da.astype(dtype=dtype)

    match backend:
        case "netcdf":
            da.to_netcdf(file_name)
        case "zarr":
            da.to_zarr(file_name)
        case _:
            raise ValueError(f"Unsupported backend {backend}")


def datasource_cache_root() -> str:
    """Returns the root directory for data sources"""
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")
    default_cache = os.environ.get("EARTH2STUDIO_CACHE", default_cache)

    try:
        os.makedirs(default_cache, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create cache folder {default_cache}, check permissions"
        )
        raise e

    return default_cache


T = TypeVar("T")


@typing.no_type_check
class AsyncCachingFileSystem(AsyncFileSystem):
    """Async locally caching filesystem, layer over any other FS for use with zarr 3.0.
    Presently extremely limited, much is just copied from the WholeFileCache

    Parameters
    ----------
    target_protocol: str (optional)
        Target filesystem protocol. Provide either this or ``fs``.
    cache_storage: str or list(str)
        Location to store files. If "TMP", this is a temporary directory,
        and will be cleaned up by the OS when this process ends (or later).
        If a list, each location will be tried in the order given, but
        only the last will be considered writable.
    cache_check: int
        Number of seconds between reload of cache metadata
    check_files: bool
        Whether to explicitly see if the UID of the remote file matches
        the stored one before using. Warning: some file systems such as
        HTTP cannot reliably give a unique hash of the contents of some
        path, so be sure to set this option to False.
    expiry_time: int
        The time in seconds after which a local copy is considered useless.
        Set to falsy to prevent expiry. The default is equivalent to one
        week.
    target_options: dict or None
        Passed to the instantiation of the FS, if fs is None.
    fs: filesystem instance
        The target filesystem to run against. Provide this or ``protocol``.
    same_names: bool (optional)
        By default, target URLs are hashed using a ``HashCacheMapper`` so
        that files from different backends with the same basename do not
        conflict. If this argument is ``true``, a ``BasenameCacheMapper``
        is used instead. Other cache mapper options are available by using
        the ``cache_mapper`` keyword argument. Only one of this and
        ``cache_mapper`` should be specified.
    compression: str (optional)
        To decompress on download. Can be 'infer' (guess from the URL name),
        one of the entries in ``fsspec.compression.compr``, or None for no
        decompression.
    cache_mapper: AbstractCacheMapper (optional)
        The object use to map from original filenames to cached filenames.
        Only one of this and ``same_names`` should be specified.

    TODO: Move out of package
    https://github.com/NickGeneva/asyncachefs
    """

    protocol: ClassVar[str | tuple[str, ...]] = ("blockcache", "cached")

    @typing.no_type_check
    def __init__(
        self,
        target_protocol=None,
        cache_storage="TMP",
        cache_check=10,
        check_files=False,
        expiry_time=604800,
        target_options=None,
        fs=None,
        same_names: bool | None = None,
        compression=None,
        cache_mapper: AbstractCacheMapper | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if fs is None and target_protocol is None:
            raise ValueError(
                "Please provide filesystem instance(fs) or target_protocol"
            )
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Both filesystems (fs) and target_protocol may not be both given."
            )
        if cache_storage == "TMP":
            tempdir = tempfile.mkdtemp()
            storage = [tempdir]
            weakref.finalize(self, self._remove_tempdir, tempdir)
        else:
            if isinstance(cache_storage, str):
                storage = [cache_storage]
            else:
                storage = cache_storage
        os.makedirs(storage[-1], exist_ok=True)
        self.storage = storage
        self.kwargs = target_options or {}
        self.cache_check = cache_check
        self.check_files = check_files
        self.expiry = expiry_time
        self.compression = compression

        # Size of cache in bytes. If None then the size is unknown and will be
        # recalculated the next time cache_size() is called. On writes to the
        # cache this is reset to None.
        self._cache_size = None

        if same_names is not None and cache_mapper is not None:
            raise ValueError(
                "Cannot specify both same_names and cache_mapper in "
                "CachingFileSystem.__init__"
            )
        if cache_mapper is not None:
            self._mapper = cache_mapper
        else:
            self._mapper = create_cache_mapper(
                same_names if same_names is not None else False
            )

        self.target_protocol = (
            target_protocol
            if isinstance(target_protocol, str)
            else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
        )
        self._metadata = CacheMetadata(self.storage)
        self.load_cache()
        self.fs = fs if fs is not None else filesystem(target_protocol, **self.kwargs)
        if not self.fs.async_impl:
            raise ValueError("Underlying filesystem needs to be async")

        def _strip_protocol(path):
            # acts as a method, since each instance has a difference target
            return self.fs._strip_protocol(type(self)._strip_protocol(path))

        self._strip_protocol: Callable = _strip_protocol

    @typing.no_type_check
    @staticmethod
    def _remove_tempdir(tempdir):
        try:
            rmtree(tempdir)
        except Exception:  # noqa: S110
            pass

    @typing.no_type_check
    def _mkcache(self):
        os.makedirs(self.storage[-1], exist_ok=True)

    @typing.no_type_check
    def cache_size(self):
        """Return size of cache in bytes.

        If more than one cache directory is in use, only the size of the last
        one (the writable cache directory) is returned.
        """
        if self._cache_size is None:
            cache_dir = self.storage[-1]
            self._cache_size = filesystem("file").du(cache_dir, withdirs=True)
        return self._cache_size

    @typing.no_type_check
    def load_cache(self):
        """Read set of stored blocks from file"""
        self._metadata.load()
        self._mkcache()
        self.last_cache = time.time()

    @typing.no_type_check
    def save_cache(self):
        """Save set of stored blocks from file"""
        self._mkcache()
        self._metadata.save()
        self.last_cache = time.time()
        self._cache_size = None

    @typing.no_type_check
    def _check_cache(self):
        """Reload caches if time elapsed or any disappeared"""
        self._mkcache()
        if not self.cache_check:
            # explicitly told not to bother checking
            return
        timecond = time.time() - self.last_cache > self.cache_check
        existcond = all(os.path.exists(storage) for storage in self.storage)
        if timecond or not existcond:
            self.load_cache()

    @typing.no_type_check
    def _check_file(self, path):
        """Is path in cache and still valid"""
        path = self._strip_protocol(path)
        self._check_cache()
        return self._metadata.check_file(path, self)

    @typing.no_type_check
    def clear_cache(self):
        """Remove all files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.
        """
        rmtree(self.storage[-1])
        self.load_cache()
        self._cache_size = None

    @typing.no_type_check
    def clear_expired_cache(self, expiry_time=None):
        """Remove all expired files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.

        Parameters
        ----------
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            If not defined the default is equivalent to the attribute from the
            file caching instantiation.
        """

        if not expiry_time:
            expiry_time = self.expiry

        self._check_cache()

        expired_files, writable_cache_empty = self._metadata.clear_expired(expiry_time)
        for fn in expired_files:
            if os.path.exists(fn):
                os.remove(fn)

        if writable_cache_empty:
            rmtree(self.storage[-1])
            self.load_cache()

        self._cache_size = None

    @typing.no_type_check
    def pop_from_cache(self, path):
        """Remove cached version of given file

        Deletes local copy of the given (remote) path. If it is found in a cache
        location which is not the last, it is assumed to be read-only, and
        raises PermissionError
        """
        path = self._strip_protocol(path)
        fn = self._metadata.pop_file(path)
        if fn is not None:
            os.remove(fn)
        self._cache_size = None

    @typing.no_type_check
    def _parent(self, path):
        return self.fs._parent(path)

    @typing.no_type_check
    async def _ukey(self, path):
        """Hash of file properties, to tell if it has changed"""
        return sha256(str(await self.fs._info(path)).encode()).hexdigest()

    @typing.no_type_check
    async def _make_local_details(self, path):
        """Create file detail dictionary for given file path"""
        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        detail = {
            "original": path,
            "fn": hash,
            "blocks": True,
            "time": time.time(),
            "uid": await self._ukey(path),
        }
        self._metadata.update_file(path, detail)
        logger.debug(f"Copying {path} to local cache")
        return fn

    @typing.no_type_check
    async def _cat_file(
        self,
        path,
        start=None,
        end=None,
        on_error="raise",
        callback=DEFAULT_CALLBACK,
        **kwargs,
    ):
        """Cat file, this is what is used inside Zarr 3.0 at the moment"""
        getpath = None
        try:
            # Check if file is in cache
            # This doesnt do anything fancy for chunked data, different chunk ranges
            # are stored in different files even if there is over lap
            path_chunked = path
            if start:
                path_chunked += f"_{start}"
            if end:
                path_chunked += f"_{end}"

            detail = self._check_file(path_chunked)
            if not detail:
                storepath = await self._make_local_details(path_chunked)
                getpath = path
            else:
                detail, storepath = (
                    detail if isinstance(detail, tuple) else (None, detail)
                )
        except Exception as e:
            if on_error == "raise":
                raise
            if on_error == "return":
                out = e

        # If file was not in cache, get it using the base file system
        if getpath:
            resp = await self.fs._cat_file(getpath, start=start, end=end)
            # Save to file
            if isfilelike(storepath):
                outfile = storepath
            else:
                outfile = open(storepath, "wb")  # noqa: ASYNC101, ASYNC230
            try:
                outfile.write(resp)
                # IDK yet
                # callback.relative_update(len(chunk))
            finally:
                if not isfilelike(storepath):
                    outfile.close()
            self.save_cache()

        # Call back is weird here, like the progress should be on the file fetch
        # but then how do we deal with the read? Maybe we times it by 2x?
        if start is None:
            start = 0

        callback.set_size(1)
        with open(storepath, "rb") as f:
            f.seek(start)
            if end is not None:
                out = f.read(end - start)
            else:
                out = f.read()
        callback.relative_update(1)
        return out

    @typing.no_type_check
    def __getattribute__(self, item):
        # TODO: Update
        if item in {
            "load_cache",
            # "_open",
            "save_cache",
            # "close_and_update",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_make_local_details",
            "_ukey",
            "open",
            "cat",
            "_cat_file",
            "cat_ranges",
            "get",
            "read_block",
            "tail",
            "head",
            # "info",
            # "ls",
            "exists",
            "isfile",
            "isdir",
            "_check_file",
            "_check_cache",
            "_mkcache",
            "clear_cache",
            "clear_expired_cache",
            "pop_from_cache",
            "local_file",
            "_paths_from_path",
            "get_mapper",
            "open_many",
            "commit_many",
            "hash_name",
            "__hash__",
            "__eq__",
            "to_json",
            "to_dict",
            "cache_size",
            "pipe_file",
            "pipe",
            "start_transaction",
            "end_transaction",
        }:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                *args, **kw
            )
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            # property
            return type(self).transaction.__get__(self)
        if item in ["_cache", "transaction_type"]:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)
