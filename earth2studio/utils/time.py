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
from datetime import datetime, timedelta

import numpy as np

from earth2studio.utils.type import LeadTimeArray, TimeArray


def timearray_to_datetime(time: TimeArray) -> list[datetime]:
    """Simple converter from numpy datetime64 array into a list of datetimes.

    Parameters
    ----------
    time : TimeArray
        Numpy datetime64 array

    Returns
    -------
    list[datetime]
        List of datetime object
    """
    _unix = np.datetime64(0, "s")
    _ds = np.timedelta64(1, "s")
    time = [datetime.utcfromtimestamp((date - _unix) / _ds) for date in time]

    return time


def leadtimearray_to_timedelta(lead_time: LeadTimeArray) -> list[timedelta]:
    """Simple converter from numpy timedelta64 array into a list of timedeltas

    Parameters
    ----------
    lead_time : TimeArray
        Numpy timedelta64 array

    Returns
    -------
    list[timedelta]
        List of timedelta object
    """
    # microsecond is smallest unit python timedelta supports
    return [
        timedelta(microseconds=int(time.astype("timedelta64[us]").astype(int)))
        for time in lead_time
    ]


def to_time_array(time: list[str] | list[datetime] | TimeArray) -> TimeArray:
    """A general converter for various time iterables into a numpy datetime64 array

    Parameters
    ----------
    time : list[str] | list[datetime] | TimeArray
        Time object iterable

    Returns
    -------
    TimeArray
        Numpy array of datetimes

    Raises
    ------
    TypeError
        If element in iterable is not a value time object
    """
    output = []

    for ts in time:
        if isinstance(ts, datetime):
            output.append(np.datetime64(ts))
        elif isinstance(ts, str):
            output.append(np.datetime64(ts))
        elif isinstance(ts, np.datetime64):
            output.append(ts)
        else:
            raise TypeError(
                f"Invalid time data type provided {ts}, should be datetime, string or np.datetime64"
            )

    return np.array(output).astype("datetime64[ns]")
