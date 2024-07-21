# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import os
import tempfile
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data.base import DataSource
from earth2studio.utils.time import (
    leadtimearray_to_timedelta,
    timearray_to_datetime,
    to_time_array,
)
from earth2studio.utils.type import CoordSystem, LeadTimeArray, TimeArray, VariableArray


def fetch_data(
    source: DataSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, CoordSystem]:
    """Utility function to fetch data for models and load data on the target device.

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

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    da = []
    for lead in lead_time:
        adjust_times = np.array([t + lead for t in time], dtype="datetime64[ns]")
        da0 = source(adjust_times, variable)
        da0 = da0.expand_dims(dim={"lead_time": 1}, axis=1)
        da0 = da0.assign_coords(lead_time=np.array([lead], dtype="timedelta64[ns]"))
        da0 = da0.assign_coords(time=time)
        da.append(da0)

    return prep_data_array(xr.concat(da, "lead_time"), device=device)


def prep_data_array(
    da: xr.DataArray,
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, CoordSystem]:
    """Prepares a data array from a data source for inference workflows by converting
    the data array to a torch tensor and the coordinate system to an OrderedDict.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    out = torch.Tensor(da.values).to(device)

    out_coords = OrderedDict()
    for dim in da.coords.dims:
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


async def unordered_generator(
    func_map: Iterator[Awaitable[Any]], limit: int
) -> AsyncGenerator[T, None]:
    """Creates an async unordered generator

    Parameters
    ----------
    func_map : Iterator[Awaitable[Any]]
        Function map to get exectured. Such as the output of map(func, args)
    limit : int
        Limit of concurrent async processes

    Yields
    ------
    AsyncGenerator[T, None]
        Unordered async generator
    """
    async for task in _limit_concurrency(func_map, limit):
        yield await task


async def _limit_concurrency(
    aws: Iterator[Awaitable[T]], limit: int
) -> AsyncGenerator[T, None]:
    """Limited concurrency async generator. Throttles number of async io processes

    Note
    ----
    Taken from: https://death.andgravity.com/limit-concurrency
    """
    try:
        aws = aiter(aws)  # type: ignore
        is_async = True
    except TypeError:
        aws = iter(aws)
        is_async = False

    aws_ended = False
    pending: set[asyncio.Future] = set()

    while pending or not aws_ended:
        while len(pending) < limit and not aws_ended:
            try:
                aw = await anext(aws) if is_async else next(aws)  # type: ignore
            except StopAsyncIteration if is_async else StopIteration:  # noqa
                aws_ended = True
            else:
                pending.add(asyncio.ensure_future(aw))

        if not pending:
            return

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        while done:
            yield done.pop()
