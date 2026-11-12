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

from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd
import pyarrow as pa
import xarray as xr

from earth2studio.utils.type import (
    LeadTimeArray,
    MemberArray,
    TimeArray,
    VariableArray,
)


# --8<-- [start:data-source-interface]
@runtime_checkable
class DataSource(Protocol):
    """Data source interface.

    Sources take a ``device`` constructor keyword, defaulting to ``"cpu"``, and
    expose it as a ``device`` attribute holding a ``torch.device``. They return
    NumPy-backed fields on CPU and CuPy-backed fields on a CUDA device. The keyword
    is not part of the protocol; sources without the attribute are CPU sources.

    Sources with ensemble members also follow :class:`EnsembleDataSource`.
    """

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....]. The coords
            should be provided. Time coordinate should be a datetime array holding
            the requested times, and the variable coordinate should be array of
            strings with Earth2Studio variable ids.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data. Async data sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....]. The coords
            should be provided. Time coordinate should be a datetime array holding
            the requested times, and the variable coordinate should be array of
            strings with Earth2Studio variable ids.
        """
        pass


# --8<-- [end:data-source-interface]


@runtime_checkable
class ForecastSource(Protocol):
    """Forecast source interface.

    Device placement follows the same optional ``device`` convention as
    :class:`DataSource`. Sources with ensemble members also follow
    :class:`EnsembleForecastSource`.
    """

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, lead_time, variable, ...].
            The coords should be provided. Time coordinate should be a TimeArray,
            lead time coordinate a LeadTimeArray and the variable coordinate should be
            an array of strings with Earth2Studio variable ids.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data. Async forecast sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, lead_time, variable, ...].
            The coords should be provided. Time coordinate should be a TimeArray,
            lead time coordinate a LeadTimeArray and the variable coordinate should be
            an array of strings with Earth2Studio variable ids.
        """
        pass


# --8<-- [start:ensemble-source-interface]
@runtime_checkable
class EnsembleDataSource(DataSource, Protocol):
    """Data source interface for sources with ensemble members.

    An ensemble source is a :class:`DataSource` whose ``__call__`` and ``fetch``
    also take an optional ``member`` keyword selecting members along a leading
    ``ensemble`` dimension, as ``time`` and ``variable`` select along theirs. Member
    labels are the source's own, such as ``0`` for a control forecast or
    ``"gep01"``. A scalar or sequence returns the members in request order on a
    leading ``ensemble`` dimension; omitting ``member`` returns the member the
    source was built with, without the dimension, so an ensemble source still
    serves any :class:`DataSource` caller.

    ``isinstance`` checks only that the methods exist, so it does not tell
    ensemble sources from other data sources. Pipelines detect them by the
    ``member`` keyword on ``__call__``; wrappers that forward requests declare
    it through an ``is_ensemble`` property instead.
    """

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        member: int | str | list[int] | list[str] | MemberArray | None = None,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        member : int | str | list[int] | list[str] | MemberArray, optional
            Member label, list of labels or array of labels to return, by default
            None, which returns the member the source was built with.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....], led by
            an ``ensemble`` dimension when ``member`` is given.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        member: int | str | list[int] | list[str] | MemberArray | None = None,
    ) -> xr.DataArray:
        """Async function to get data. Async data sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        member : int | str | list[int] | list[str] | MemberArray, optional
            Member label, list of labels or array of labels to return, by default
            None, which returns the member the source was built with.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....], led by
            an ``ensemble`` dimension when ``member`` is given.
        """
        pass


# --8<-- [end:ensemble-source-interface]


@runtime_checkable
class EnsembleForecastSource(ForecastSource, Protocol):
    """Forecast source interface for sources with ensemble members.

    Selects members with an optional ``member`` keyword as
    :class:`EnsembleDataSource` does.
    """

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
        member: int | str | list[int] | list[str] | MemberArray | None = None,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        member : int | str | list[int] | list[str] | MemberArray, optional
            Member label, list of labels or array of labels to return, by default
            None, which returns the member the source was built with.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, lead_time, variable, ...],
            led by an ``ensemble`` dimension when ``member`` is given.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
        member: int | str | list[int] | list[str] | MemberArray | None = None,
    ) -> xr.DataArray:
        """Async function to get data. Async forecast sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        member : int | str | list[int] | list[str] | MemberArray, optional
            Member label, list of labels or array of labels to return, by default
            None, which returns the member the source was built with.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, lead_time, variable, ...],
            led by an ``ensemble`` dimension when ``member`` is given.
        """
        pass


@runtime_checkable
class DataFrameSource(Protocol):
    """Data frame source interface. Commonly used for sparse sensor data

    Device placement follows the ``device`` convention of :class:`DataSource`,
    returning pandas frames on CPU and cuDF frames on a CUDA device.
    """

    SCHEMA: pa.Schema

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields / columns to return, must be subset of the data sources schema. Will
            return all possible columns if None, by default None

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame like object where the columns are the requested variable
            list and rows filtered for the time requested. Variables/columns in data
            frame sources may refer to geospatial fields/data (e.g. wind speed) or
            relevant metadata (e.g. time of observation).
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data. Async data frame sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields / columns to return, must be subset of the data sources schema. Will
            return all possible columns if None, by default None

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame like object where the columns are the requested variable
            list and rows filtered for the time requested. Variables/columns in data
            frame sources may refer to geospatial fields/data (e.g. wind speed) or
            relevant metadata (e.g. time of observation).
        """
        pass


@runtime_checkable
class ForecastFrameSource(Protocol):
    """Forecast data frame source interface. Commonly used for sparse sensor data

    Device placement follows the ``device`` convention of :class:`DataFrameSource`.
    """

    SCHEMA: pa.Schema

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields / columns to return, must be subset of the data sources schema. Will
            return all possible columns if None, by default None

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame like object where the columns are the requested variable
            list and rows filtered for the time requested. Variables/columns in data
            frame sources may refer to geospatial fields/data (e.g. wind speed) or
            relevant metadata (e.g. time of observation).
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data. Async data frame sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields / columns to return, must be subset of the data sources schema. Will
            return all possible columns if None, by default None

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame like object where the columns are the requested variable
            list and rows filtered for the time requested. Variables/columns in data
            frame sources may refer to geospatial fields/data (e.g. wind speed) or
            relevant metadata (e.g. time of observation).
        """
        pass
