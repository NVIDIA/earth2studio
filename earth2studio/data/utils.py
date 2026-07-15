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

from __future__ import annotations

import asyncio
import os
import random
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from inspect import signature
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import fsspec.asyn
import numpy as np
import obstore as obs
import obstore.store
import pandas as pd
import torch
import xarray as xr
import zarr
import zarr.abc.store
import zarr.storage
from loguru import logger
from tqdm.asyncio import tqdm
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype

from earth2studio.data.base import (
    DataFrameSource,
    DataSource,
    ForecastFrameSource,
    ForecastSource,
)
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.time import (
    leadtimearray_to_timedelta,
    timearray_to_datetime,
    to_time_array,
)
from earth2studio.utils.type import (
    CoordSystem,
    FieldArray,
    LeadTimeArray,
    TimeArray,
    VariableArray,
)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import cudf
except ImportError:
    cudf = None


def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
    interp_to: CoordSystem | None = None,
    interp_method: str = "nearest",
    legacy: bool = True,
) -> tuple[torch.Tensor, CoordSystem] | xr.DataArray:
    """Utility function to fetch data arrays from particular sources and load data on
    the target device. If desired, xarray interpolation/regridding in the spatial
    domain can be used by passing a target coordinate system via the optional
    `interp_to` argument.

    Parameters
    ----------
    source : DataSource | ForecastSource
        The data source to fetch from
    time : TimeArray
        Timestamps to return data for (UTC).
    variable : VariableArray
        Strings or list of strings that refer to variables to return
    lead_time : LeadTimeArray, optional
        Lead times to fetch for each provided time, by default
        np.array(np.timedelta64(0, "h"))
    device : torch.device, optional
        Torch device to load data tensor to, by default "cpu"
    interp_to : CoordSystem, optional
        If provided, the fetched data will be interpolated to the coordinates
        specified by lat/lon arrays in this CoordSystem
    interp_method : str
        Interpolation method to use with xarray (by default 'nearest')
    legacy : bool, optional
        If True (default), returns tuple of (torch.Tensor, CoordSystem).
        If False, returns xr.DataArray with numpy arrays for CPU or cupy arrays for CUDA.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem] | xr.DataArray
        If legacy=True: Tuple containing output tensor and coordinate OrderedDict.
        If legacy=False: xr.DataArray with numpy arrays (CPU) or cupy arrays (CUDA).
    """
    sig = signature(source.__call__)
    device = torch.device(device)

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

    if legacy:
        return prep_data_array(
            da,
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )

    # Non-legacy path: return xr.DataArray
    else:
        if interp_to is not None:
            raise ValueError(
                "The interp_to argument is not supported when legacy is False. Set legacy=True to use interpolation."
            )
        # Convert to cupy arrays if CUDA device and cupy is available
        if device.type == "cuda":
            if cp is not None:
                with cp.cuda.Device(device.index or 0):
                    da = da.copy(data=cp.asarray(da.values))
            else:
                raise ImportError(
                    "cupy is required when using device='cuda' with legacy=False. "
                    "Install cupy or use legacy=True."
                )
        return da


def fetch_dataframe(
    source: DataFrameSource | ForecastFrameSource,
    time: TimeArray,
    variable: VariableArray,
    fields: FieldArray | None = None,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
) -> pd.DataFrame | cudf.DataFrame:
    """Utility function to fetch data frames from particular sources

    Parameters
    ----------
    source : DataFrameSource | ForecastFrameSource
        The data source to fetch from
    time : TimeArray
        Timestamps to return data for (UTC).
    variable : VariableArray
        Strings or list of strings that refer to variables to return
    fields : FieldArray | None
        Array of strings indicating which fields to return in data frame, if None all
        possible fields will be returned, by default None
    lead_time : LeadTimeArray, optional
        Lead times to fetch for each provided time, by default
        np.array(np.timedelta64(0, "h"))
    device : torch.device, optional
        Torch device to load data tensor to, by default "cpu"

    Returns
    -------
    pd.DataFrame | cudf.DataFrame
        Pandas dataframe if device is CPU, cudf DataFrame if device is CUDA
    """
    sig = signature(source.__call__)

    if "lead_time" in sig.parameters:
        # Working with a ForecastFrameSource
        df = source(time, lead_time, variable, fields=fields)  # type: ignore
    else:
        # Combine all adjusted times and get unique values using broadcasting
        all_times = (time[:, None] + lead_time).flatten()
        unique_times = np.unique(all_times)
        df = source(unique_times, variable, fields=fields)  # type: ignore
    # Add request meta-data
    df.attrs = {"request_time": time, "request_lead_time": lead_time}

    # Convert to appropriate format based on device
    device = torch.device(device)
    if device.type == "cuda":
        if cudf is None:
            raise ImportError(
                "cudf is required for CUDA device. Install with: pip install cudf"
            )
        with cp.cuda.Device(device.index):
            result = cudf.from_pandas(df)
        return result
    else:
        return df


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
        if len(interp_to["_lat"].shape) != len(interp_to["_lon"].shape):
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
                lat_in=da["_lat"].values,
                lon_in=da["_lon"].values,
                lat_out=interp_to["_lat"],
                lon_out=interp_to["_lon"],
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
            if len(interp_to["_lat"].shape) > 1 or len(interp_to["_lon"].shape) > 1:
                # Target grid uses curvilinear coordinates: define internal dims y, x
                target_lat = xr.DataArray(interp_to["_lat"], dims=["y", "x"])
                target_lon = xr.DataArray(interp_to["_lon"], dims=["y", "x"])
            else:
                target_lat = xr.DataArray(interp_to["_lat"], dims=["_lat"])
                target_lon = xr.DataArray(interp_to["_lon"], dims=["_lon"])

            da = da.interp(
                lat=target_lat,
                lon=target_lon,
                method=interp_method,
            )

            out = torch.Tensor(da.values).to(device)

        out_coords["_lat"] = interp_to["_lat"]
        out_coords["_lon"] = interp_to["_lon"]

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


def ensure_utc(time: datetime) -> datetime:
    """Normalize a datetime to naive UTC.

    If timezone-aware, convert to UTC then strip tzinfo.
    If naive, assume UTC and return unchanged.

    Parameters
    ----------
    time : datetime
        Input datetime (naive or tz-aware)

    Returns
    -------
    datetime
        Naive datetime in UTC
    """
    if time.tzinfo is not None:
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return time


def prep_data_inputs(
    time: datetime | list[datetime] | np.datetime64 | TimeArray,
    variable: str | list[str] | VariableArray,
) -> tuple[list[datetime], list[str]]:
    """Simple method to pre-process data source inputs into a common form

    Parameters
    ----------
    time : datetime | list[datetime] | np.datetime64 | TimeArray
        Datetime, list of datetimes or array of np.datetime64 to fetch (UTC)
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables

    Returns
    -------
    tuple[list[datetime], list[str]]
        Time and variable lists (times normalized to naive UTC)
    """

    def _to_datetime(t: datetime | np.datetime64 | pd.Timestamp) -> datetime:
        """Convert a single time value to a datetime object."""
        if isinstance(t, np.datetime64):
            return pd.Timestamp(t).to_pydatetime()
        if isinstance(t, pd.Timestamp):
            return t.to_pydatetime()
        return t

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(variable, np.ndarray):
        variable = variable.astype(str).tolist()

    if isinstance(time, (datetime, np.datetime64, pd.Timestamp)):
        time = [ensure_utc(_to_datetime(time))]

    elif isinstance(time, np.ndarray):  # np.datetime64 -> datetime
        time = [ensure_utc(t) for t in timearray_to_datetime(time)]

    elif isinstance(time, list):
        time = [ensure_utc(_to_datetime(t)) for t in time]

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
    backend_kwargs: dict[str, Any] = {},
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
    backend_kwargs : dict[str, Any], optional
        Dictionary of keyword arguments forwarded to the underlying
        ``xarray.DataArray.to_netcdf`` / ``xarray.DataArray.to_zarr``
        call depending on the selected backend.
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
            da.to_netcdf(file_name, **backend_kwargs)
        case "zarr":
            da.to_zarr(file_name, **backend_kwargs)
        case _:
            raise ValueError(f"Unsupported backend {backend}")


def datasource_cache_root() -> str:
    """Returns the root directory for data sources"""
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")
    default_cache = os.environ.get("EARTH2STUDIO_CACHE", default_cache)
    default_cache = os.environ.get("EARTH2STUDIO_DATA_CACHE", default_cache)

    try:
        os.makedirs(default_cache, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create cache folder {default_cache}, check permissions"
        )
        raise e

    return default_cache


# =============================================================================
# Async Utilities for Data Sources
# =============================================================================
# These utilities provide standardized patterns for async data fetching with
# proper error handling, concurrency control, and resource cleanup.
# IMPORTANT: Pure async operations are ALWAYS preferred over asyncio.to_thread.
# Only use to_thread as a last resort when no async alternative exists.
# =============================================================================


def _sync_async(
    coro: Any,
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run an async function synchronously using fsspec's background IO loop.

    This works from any calling context -- scripts, Jupyter notebooks with a
    running event loop, or existing async contexts -- because it dispatches
    to fsspec's dedicated background IO thread rather than nesting into the
    caller's loop.

    Parameters
    ----------
    coro : coroutine function or coroutine
        Async callable (or already-awaitable coroutine) to execute.
    *args : Any
        Positional arguments forwarded to coro (if callable).
    timeout : float | None, optional
        Timeout in seconds, by default None (no timeout).
    **kwargs : Any
        Keyword arguments forwarded to coro (if callable).

    Returns
    -------
    Any
        The return value of the coroutine.
    """
    loop = fsspec.asyn.get_loop()
    return fsspec.asyn.sync(loop, coro, *args, timeout=timeout, **kwargs)


async def async_retry(
    coro_func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    backoff: float = 1.0,
    task_timeout: float | None = None,
    exceptions: tuple[type[BaseException], ...] = (OSError, TimeoutError),
    **kwargs: Any,
) -> Any:
    """Retry an async callable with exponential backoff and jitter.

    Parameters
    ----------
    coro_func : Callable[..., Awaitable[T]]
        Async function to call
    *args : Any
        Positional arguments for coro_func
    retries : int, optional
        Maximum number of retry attempts, by default 3
    backoff : float, optional
        Base delay in seconds (doubled each retry), by default 1.0
    task_timeout : float | None, optional
        Timeout in seconds for each attempt, by default None (no timeout)
    exceptions : tuple, optional
        Exception types to catch and retry on. Should be scoped to transient
        I/O errors (OSError, IOError, TimeoutError, ConnectionError), not
        broad Exception which would mask programming errors.
    **kwargs : Any
        Keyword arguments for coro_func

    Returns
    -------
    T
        Return value of coro_func

    Raises
    ------
    Exception
        The last exception if all retries are exhausted
    asyncio.TimeoutError
        If task_timeout is exceeded on the final attempt
    """
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            coro = coro_func(*args, **kwargs)
            if task_timeout is not None:
                return await asyncio.wait_for(coro, timeout=task_timeout)
            return await coro
        except asyncio.TimeoutError:
            # Always re-raise TimeoutError - don't mask it
            if attempt >= retries:
                raise
            last_exc = asyncio.TimeoutError(
                f"Attempt {attempt + 1}/{retries + 1} timed out after {task_timeout}s"
            )
        except exceptions as e:
            last_exc = e
        if attempt < retries:
            delay = backoff * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                f"Retry {attempt + 1}/{retries} after error: {last_exc}. "
                f"Waiting {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


@asynccontextmanager
async def managed_session(fs: Any) -> Any:
    """Context manager for fsspec async sessions.

    Ensures the aiohttp session is properly closed even if an exception or
    timeout occurs during fetching. Use this instead of bare set_session/close
    to prevent session leaks.

    Parameters
    ----------
    fs : Any
        An fsspec filesystem instance (s3fs, gcsfs, etc.)

    Yields
    ------
    Any
        The aiohttp client session (or None for non-session fs)

    Example
    -------
    .. code-block:: python

        async with managed_session(self.fs) as session:
            # fetch data here - session will be closed even on error
            await gather_with_concurrency(coros, ...)
    """
    session = None
    try:
        if hasattr(fs, "set_session"):
            session = await fs.set_session()
        yield session
    finally:
        if session is not None:
            await session.close()
            # Reset fs._session so the next managed_session call creates a
            # fresh aiohttp client (fsspec ≥ 2026 removed the refresh= kwarg).
            if hasattr(fs, "_session"):
                fs._session = None


async def gather_with_concurrency(
    coros: list[Any],
    max_workers: int = 16,
    task_timeout: float | None = None,
    desc: str = "Fetching",
    verbose: bool = False,
) -> list[Any]:
    """Run coroutines with bounded concurrency and progress bar.

    This prevents resource exhaustion (connections, memory) when fetching
    hundreds of items by limiting concurrent tasks via an asyncio.Semaphore.

    Parameters
    ----------
    coros : list[Coroutine]
        Coroutines to execute
    max_workers : int, optional
        Maximum concurrent tasks (semaphore bound), by default 16.
        This limits how many coroutines run simultaneously, NOT thread pool size.
    task_timeout : float | None, optional
        Timeout in seconds for each individual task, by default None.
        If a task exceeds this timeout, it raises asyncio.TimeoutError.
    desc : str, optional
        Progress bar description
    verbose : bool, optional
        Disable the progress bar

    Returns
    -------
    list[Any]
        Results from all coroutines in the same order as input

    Raises
    ------
    asyncio.TimeoutError
        If any task exceeds task_timeout
    Exception
        Any exception raised by a coroutine is propagated
    """
    semaphore = asyncio.Semaphore(max_workers)

    async def _bounded(coro: Any) -> Any:
        async with semaphore:
            if task_timeout is not None:
                return await asyncio.wait_for(coro, timeout=task_timeout)
            return await coro

    bounded = [_bounded(c) for c in coros]
    return await tqdm.gather(*bounded, desc=desc, disable=verbose)


async def cancellable_to_thread(
    func: Callable[..., T],
    *args: Any,
    timeout: float = 30.0,
    **kwargs: Any,
) -> T:
    """Run a blocking function in a thread with timeout support.

    WARNING: This should be a LAST RESORT. Pure async operations are ALWAYS
    preferred because:
    1. Threads cannot be forcibly cancelled in Python - when timeout fires,
       the coroutine is abandoned but the thread continues running
    2. This causes issues with pytest --timeout where tests hang
    3. Thread pool exhaustion can occur with many concurrent tasks

    PREFER these async alternatives:
    - fs._cat_file() for byte-range reads from s3fs/gcsfs
    - async zarr for zarr stores
    - httpx.AsyncClient for HTTP requests
    - aiofiles for file I/O

    Only use this for unavoidably synchronous operations like:
    - pygrib GRIB parsing (no async alternative)
    - Legacy libraries without async support

    Parameters
    ----------
    func : Callable[..., T]
        Blocking function to run in thread
    *args : Any
        Positional arguments for func
    timeout : float, optional
        Timeout in seconds, by default 30.0. After timeout, the coroutine
        raises TimeoutError but the thread continues (cannot be killed).
    **kwargs : Any
        Keyword arguments for func

    Returns
    -------
    T
        Return value of func

    Raises
    ------
    asyncio.TimeoutError
        If the operation exceeds timeout. Note: the underlying thread
        continues running - it cannot be force-killed in Python.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Blocking operation {func.__name__} timed out after {timeout}s. "
            "Note: underlying thread may still be running."
        )
        raise


def resolve_async_workers(
    async_workers: int | None, n_tasks: int, cap: int = 64
) -> int:
    """Resolves the concurrent download worker count for a data source.

    When ``async_workers`` is None (the default for obstore-backed sources),
    concurrency autoscales to the number of pending download tasks, bounded by
    ``cap`` to keep request bursts against public endpoints reasonable. An
    explicit ``async_workers`` value is always honored as-is.

    Parameters
    ----------
    async_workers : int | None
        User-provided worker count, or None to autoscale
    n_tasks : int
        Number of pending download tasks
    cap : int, optional
        Upper bound applied when autoscaling, by default 64

    Returns
    -------
    int
        Number of concurrent workers to use (at least 1)
    """
    if async_workers is not None:
        return async_workers
    return max(1, min(n_tasks, cap))


def obstore_store_from_url(
    url: str,
    anonymous: bool = True,
    max_pool_connections: int = 24,
    **store_kwargs: Any,
) -> obstore.store.ObjectStore:
    """Creates an obstore ObjectStore from a URL for byte-range object reads.

    Serves as the shared store factory for GRIB `.idx` + byte-range data
    sources. Supports ``s3://`` (anonymous access injects
    ``skip_signature=True`` and a ``us-east-1`` region default), ``gs://``
    (anonymous access injects ``skip_signature=True``) and ``http(s)://``
    URLs via :func:`obstore.store.from_url`.

    Parameters
    ----------
    url : str
        Store URL, e.g. ``s3://noaa-gfs-bdp-pds``
    anonymous : bool, optional
        Use unsigned/anonymous requests for s3/gs stores, by default True
    max_pool_connections : int, optional
        Maximum idle connections kept per host, should match the fetch
        concurrency of the caller, by default 24
    **store_kwargs : Any
        Additional store configuration forwarded to
        :func:`obstore.store.from_url`, overriding the defaults above

    Returns
    -------
    obstore.store.ObjectStore
        Configured object store
    """
    kwargs: dict[str, Any] = {
        "client_options": {"pool_max_idle_per_host": str(max_pool_connections)},
    }
    if anonymous and url.startswith(("s3://", "gs://")):
        kwargs["skip_signature"] = True
    if url.startswith("s3://"):
        kwargs["region"] = "us-east-1"
    kwargs.update(store_kwargs)
    return obstore.store.from_url(url, **kwargs)


async def obstore_read_range(
    store: obstore.store.ObjectStore,
    key: str,
    byte_offset: int = 0,
    byte_length: int | None = None,
) -> bytes:
    """Reads a byte range (or the remainder / whole) of an object via obstore.

    Behavior by argument combination:

    - ``byte_length`` set: ranged read of exactly ``byte_length`` bytes
      starting at ``byte_offset``
    - ``byte_length`` is None and ``byte_offset`` is 0: whole-object read
      (e.g. GRIB ``.idx`` files)
    - ``byte_length`` is None and ``byte_offset`` > 0: read from offset to the
      end of the object (size resolved with a head request)

    Parameters
    ----------
    store : obstore.store.ObjectStore
        Object store to read from
    key : str
        Object key (bucket-relative path)
    byte_offset : int, optional
        Start byte, by default 0
    byte_length : int | None, optional
        Number of bytes to read, by default None (read to end)

    Returns
    -------
    bytes
        The requested bytes

    Raises
    ------
    FileNotFoundError
        If the object does not exist. obstore's ``NotFoundError`` subclasses
        plain ``Exception``, so it is translated here to keep callers'
        existing ``except FileNotFoundError`` handling working.
    """
    try:
        if byte_length is not None:
            data = await obs.get_range_async(
                store, key, start=byte_offset, length=byte_length
            )
        elif byte_offset == 0:
            resp = await obs.get_async(store, key)
            data = await resp.bytes_async()
        else:
            meta = await obs.head_async(store, key)
            data = await obs.get_range_async(
                store, key, start=byte_offset, end=int(meta["size"])
            )
    except (FileNotFoundError, obs.exceptions.NotFoundError):
        raise FileNotFoundError(f"Object {key} not found in store")
    return bytes(data)


async def obstore_fetch_to_cache(
    store: obstore.store.ObjectStore,
    key: str,
    cache_dir: str,
    byte_offset: int = 0,
    byte_length: int | None = None,
    cache_key: str | None = None,
) -> str:
    """Fetches a byte range of an object into a local cache file.

    Skips the fetch entirely when the cache file already exists. The cache
    file name defaults to ``sha256(key + str(byte_offset))``; callers
    migrating from fsspec-based fetching should pass an explicit ``cache_key``
    hashed from their historical (e.g. bucket-prefixed) path so existing warm
    caches remain valid.

    Parameters
    ----------
    store : obstore.store.ObjectStore
        Object store to read from
    key : str
        Object key (bucket-relative path)
    cache_dir : str
        Directory to place the cache file in
    byte_offset : int, optional
        Start byte, by default 0
    byte_length : int | None, optional
        Number of bytes to read, by default None (read to end)
    cache_key : str | None, optional
        Explicit cache file name, by default None (sha256 of key + offset)

    Returns
    -------
    str
        Path to the local cache file
    """
    if cache_key is None:
        cache_key = sha256((key + str(byte_offset)).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, cache_key)
    if Path(cache_path).is_file():
        return cache_path
    data = await obstore_read_range(
        store, key, byte_offset=byte_offset, byte_length=byte_length
    )
    await asyncio.to_thread(Path(cache_path).write_bytes, data)
    return cache_path


class LocalCachingStore(zarr.storage.WrapperStore):
    """Wraps a read-only zarr store with a local on-disk cache backed by a zarr
    LocalStore, intended for append-only immutable archive stores.

    Chunk data is cached indefinitely: existing chunks never change and newly
    appended data arrives under new keys (a cache miss). Metadata objects
    (``zarr.json`` / ``.zarray`` / ``.zgroup`` / ``.zattrs`` / ``.zmetadata``)
    encode the growing shape and attrs, so they are always refetched to keep a
    warm/persisted cache from going blind to newly appended data. Ranged reads
    (e.g. sharded stores) bypass the cache entirely — caching is a no-op for
    sharded stores.

    Concurrent reads of the same uncached key are de-duplicated with a per-key
    lock (double-checked against the cache), so a chunk is fetched and written
    at most once even under Zarr's concurrent chunk fan-out. Locks are shared
    process-wide and keyed by (cache directory, key), so separate store
    instances caching to the same directory de-duplicate against each other.

    Parameters
    ----------
    store : zarr.abc.store.Store
        Remote store to cache reads from
    cache_storage : str
        Local directory to store cached objects in. Must be unique per remote
        store, cached objects are keyed by their in-store path only.
    """

    # Metadata keys mutate as archives append (shape/attrs), so are never
    # cached. Covers both Zarr v3 (zarr.json) and v2 (.z*, incl. consolidated
    # .zmetadata) — ARCO/WB2 are v2 despite the "-v3" store name.
    _METADATA_SUFFIXES = ("zarr.json", ".zmetadata", ".zarray", ".zgroup", ".zattrs")

    # Class-level so every instance caching to the same directory shares the
    # same per-key locks. Held only across a cache miss, so distinct chunks
    # still fetch concurrently. Keyed by "{cache directory}::{key}".
    _fetch_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, store: zarr.abc.store.Store, cache_storage: str) -> None:
        super().__init__(store)
        self._cache_path = cache_storage
        self._cache = zarr.storage.LocalStore(cache_storage)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # Ranged reads (e.g. sharded stores) are not cached.
        if byte_range is not None:
            return await self._store.get(key, prototype, byte_range)

        # Metadata mutates as the archive appends; always refetch it so a warm
        # cache does not miss newly available data.
        if key.endswith(self._METADATA_SUFFIXES):
            return await self._store.get(key, prototype)

        # Fast path: cache hit needs no lock.
        cached = await self._cache.get(key, prototype)
        if cached is not None:
            return cached

        # Slow path: serialise concurrent fetchers of this same key. setdefault
        # is atomic here: asyncio runs single-threaded and there is no await
        # between lookup and insert.
        lock_key = f"{self._cache_path}::{key}"
        lock = self._fetch_locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            try:
                # Re-check under the lock: a prior holder may have filled the cache.
                cached = await self._cache.get(key, prototype)
                if cached is not None:
                    return cached
                value = await self._store.get(key, prototype)
                if value is not None:
                    # The cache is best-effort: a write failure (disk full,
                    # permissions, ...) must not fail a read that already
                    # succeeded against the remote store.
                    try:
                        await self._cache.set(key, value)
                    except Exception as e:
                        logger.warning(f"Failed to write {key} to local cache: {e}")
                return value
            finally:
                # Drop the entry while the lock is still held so the dict does
                # not grow with every key ever missed — and so no task can ever
                # remove a lock that belongs to someone else. Waiters already
                # holding a reference acquire the stale lock, re-check the
                # cache, and hit it.
                self._fetch_locks.pop(lock_key, None)


def obstore_zarr_store(
    url: str,
    cache_storage: str | None = None,
    credential_provider: Any | None = None,
    auth_token: str | None = None,
    store_kwargs: dict[str, Any] | None = None,
) -> zarr.abc.store.Store:
    """Creates a read-only zarr store backed by obstore from a store URL.

    Serves as the single integration point for obstore-backed zarr reads, use
    this over hand-rolled fsspec stores for cloud Zarr data sources.

    Parameters
    ----------
    url : str
        Store URL, e.g. "gs://bucket/path/to/store.zarr". Scheme dispatch
        (s3://, gs://, az://, file://, ...) is handled by
        :func:`obstore.store.from_url`.
    cache_storage : str | None, optional
        Local cache directory. If provided, whole-object reads are cached to
        a URL-specific sub-directory via :class:`LocalCachingStore`, by
        default None (no caching)
    credential_provider : Any | None, optional
        An obstore credential provider for authenticated access. Takes
        precedence over ``auth_token``, by default None
    auth_token : str | None, optional
        Bearer token sent as an ``Authorization`` header for authenticated
        access, by default None
    store_kwargs : dict[str, Any] | None, optional
        Additional configuration forwarded to :func:`obstore.store.from_url`,
        e.g. ``{"skip_signature": True}`` for anonymous access to public
        buckets, by default None

    Returns
    -------
    zarr.abc.store.Store
        Read-only zarr store
    """
    import obstore.store

    if store_kwargs is None:
        store_kwargs = {}

    # Store construction mirrors titiler-cmr: prefer a credential provider,
    # fall back to a bearer token, else anonymous / config-driven access.
    if credential_provider is not None:
        store = obstore.store.from_url(
            url, credential_provider=credential_provider, **store_kwargs
        )
    elif auth_token:
        client_options = {"default_headers": {"Authorization": f"Bearer {auth_token}"}}
        store = obstore.store.from_url(
            url, client_options=client_options, **store_kwargs
        )
    else:
        store = obstore.store.from_url(url, **store_kwargs)

    zstore: zarr.abc.store.Store = zarr.storage.ObjectStore(store, read_only=True)
    if cache_storage is not None:
        # URL-derived sub-directory to avoid key collisions between stores
        # sharing a cache root
        cache_storage = os.path.join(
            cache_storage, sha256(url.encode()).hexdigest()[:16]
        )
        zstore = LocalCachingStore(zstore, cache_storage)
    return zstore


T = TypeVar("T")


# Physical constants for radiance-to-brightness-temperature conversion
# -----------------------------------------------------------------------------
# First radiation constant C1 = 2hc² in mW/(m²·sr·cm⁻⁴)
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt
PLANCK_C1: float = 1.191042972e-5  # mW/(m²·sr·cm⁻⁴)   W m^2 sr^-1

# Second radiation constant C2 = hc/k in K·cm
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt
PLANCK_C2: float = 1.438776877  # K·cm


def radiance_to_bt(
    radiance: np.ndarray,
    wavenumber: np.ndarray | float,
    band_correction: tuple[float, float] | None = None,
    correction_formula: Literal["additive", "divisive"] = "additive",
) -> np.ndarray:
    """Convert spectral radiance to brightness temperature via inverse Planck function.

    Computes brightness temperature from calibrated spectral radiance using:

        T* = C2 * ν / ln(1 + C1 * ν³ / L)

    where C1 and C2 are the first and second radiation constants, ν is the
    wavenumber, and L is the spectral radiance.

    Optionally applies band correction for microwave/infrared sensors:

    - **Additive** (default): T = A + B * T* (AVHRR, MHS style)
    - **Divisive**: T = (T* - A) / B (AMSU-A style)

    Parameters
    ----------
    radiance : np.ndarray
        Spectral radiance in mW/(m²·sr·cm⁻¹). Can be 1D (n_obs,), 2D (n_obs, n_channels),
        or any shape. NaN and non-positive values are preserved as NaN in output.
    wavenumber : np.ndarray | float
        Central wavenumber(s) in cm⁻¹. If array, must broadcast with radiance
        (e.g., shape (n_channels,) for radiance shape (n_obs, n_channels)).
    band_correction : tuple[float, float] | None, optional
        Band correction coefficients (A, B). If None, pure Planck inversion is used.
        For per-channel corrections, call this function per-channel.
    correction_formula : {"additive", "divisive"}, optional
        How to apply band correction:
        - "additive": T = A + B * T* (default, used by AVHRR, MHS)
        - "divisive": T = (T* - A) / B (used by AMSU-A)

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin, same shape as radiance.
        Invalid radiance values (≤0 or NaN) yield NaN.

    Notes
    -----
    Uses NIST CODATA 2018 radiation constants:
    - C1 = 1.191042953e-5 mW/(m²·sr·cm⁻⁴)
    - C2 = 1.4387774 K·cm

    Examples
    --------
    Pure Planck inversion for hyperspectral sounder (IASI, CrIS):

    >>> wavenumbers = np.array([650.0, 700.0, 750.0])  # cm⁻¹
    >>> radiance = np.array([[10.0, 12.0, 15.0]])  # mW/(m²·sr·cm⁻¹)
    >>> bt = radiance_to_bt(radiance, wavenumbers)

    With band correction for microwave sounder (MHS):

    >>> radiance = np.array([5.0, 6.0, 7.0])  # single channel
    >>> bt = radiance_to_bt(radiance, 18.75, band_correction=(0.5, 0.998))
    """
    nu = np.asarray(wavenumber)
    nu3 = nu * nu * nu

    # Compute inverse Planck, suppressing warnings for invalid radiance
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = PLANCK_C2 * nu / np.log1p(PLANCK_C1 * nu3 / radiance)

    # Mask invalid radiance (≤0 or NaN) → NaN in output
    invalid = ~(radiance > 0)
    if np.any(invalid):
        t_star = np.where(invalid, np.nan, t_star)

    # Apply band correction if provided
    if band_correction is not None:
        a, b = band_correction
        if correction_formula == "additive":
            bt = a + b * t_star
        else:  # divisive
            bt = (t_star - a) / b
    else:
        bt = t_star

    return bt
