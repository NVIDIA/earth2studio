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

from datetime import datetime

import numpy as np
import pandas as pd

try:
    import cudf
except ImportError:
    cudf = None


def validate_observation_fields(
    observation: pd.DataFrame | cudf.DataFrame, required_fields: list[str]
) -> None:
    """Validate that required fields are present as columns in the DataFrame.

    Parameters
    ----------
    observation : pd.DataFrame | cudf.DataFrame
        DataFrame observation to validate
    required_fields : list[str]
        List of required field/column names

    Raises
    ------
    ValueError
        If any required fields are missing from the DataFrame columns
    """
    missing_fields = [
        field for field in required_fields if field not in observation.columns
    ]
    if missing_fields:
        raise ValueError(
            f"DataFrame missing required fields: {missing_fields}. "
            f"Available columns: {list(observation.columns)}"
        )


def filter_time_range(
    df: pd.DataFrame | cudf.DataFrame,
    request_time: np.datetime64 | datetime | str,
    tolerance: tuple[np.timedelta64, np.timedelta64],
    time_column: str = "time",
) -> pd.DataFrame | cudf.DataFrame:
    """Filter DataFrame rows where time column is within the specified tolerance range.

    Filters the DataFrame to include only rows where the time column value is within
    [request_time + lower_bound, request_time + upper_bound]. Ensures the time column
    is of dtype datetime64[ns].

    Parameters
    ----------
    df : pd.DataFrame | cudf.DataFrame
        DataFrame to filter. Can be pandas or cudf DataFrame.
    request_time : np.datetime64 | datetime | str
        Reference time for filtering. Observations within the tolerance window
        around this time will be included.
    tolerance : tuple[np.timedelta64, np.timedelta64]
        Tuple of (lower_bound, upper_bound) time deltas defining the tolerance window.
    time_column : str, optional
        Name of the time column in the DataFrame, by default "time"

    Returns
    -------
    pd.DataFrame | cudf.DataFrame
        Filtered DataFrame containing only rows within the time tolerance range.
        Returns the same DataFrame type as the input (pandas or cudf).

    Raises
    ------
    KeyError
        If the time_column is not present in the DataFrame
    """
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    request_time_ns = np.datetime64(request_time, "ns")

    # Ensure time column is datetime64[ns]
    time_series = df[time_column]
    if time_series.dtype != "datetime64[ns]":
        df = df.copy()
        # Use cudf methods if it's a cudf DataFrame, otherwise use pandas
        if cudf is not None and isinstance(df, cudf.DataFrame):
            df[time_column] = cudf.to_datetime(time_series).astype("datetime64[ns]")
        else:
            df[time_column] = pd.to_datetime(time_series).astype("datetime64[ns]")

    # Calculate time bounds
    lower_bound, upper_bound = tolerance
    time_min = request_time_ns + lower_bound
    time_max = request_time_ns + upper_bound

    time_mask = (df[time_column] >= time_min) & (df[time_column] <= time_max)
    return df[time_mask]
