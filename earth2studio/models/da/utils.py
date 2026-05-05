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
from functools import reduce

import numpy as np
import pandas as pd
import torch
from loguru import logger

from earth2studio.utils.type import TimeArray

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
    request_time: np.datetime64 | datetime | str | TimeArray,
    tolerance: tuple[np.timedelta64, np.timedelta64],
    time_column: str = "time",
) -> pd.DataFrame | cudf.DataFrame:
    """Filter DataFrame rows where time column is within the specified tolerance range.

    Filters the DataFrame to include only rows where the time column value is within
    [request_time + lower_bound, request_time + upper_bound]. When *request_time* is a
    :class:`~numpy.ndarray` of datetime64 values, a row is kept if it falls within the
    tolerance window of **any** of the provided times.

    Parameters
    ----------
    df : pd.DataFrame | cudf.DataFrame
        DataFrame to filter. Can be pandas or cudf DataFrame.
    request_time : np.datetime64 | datetime | str | TimeArray
        Reference time(s) for filtering. Observations within the tolerance window
        around this time will be included. If a numpy array of datetime64 is provided,
        the union of all individual tolerance windows is used.
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

    # Ensure time column is datetime64[ns]
    time_series = df[time_column]
    if time_series.dtype != "datetime64[ns]":
        df = df.copy()
        # Use cudf methods if it's a cudf DataFrame, otherwise use pandas
        if cudf is not None and isinstance(df, cudf.DataFrame):
            df[time_column] = cudf.to_datetime(time_series).astype("datetime64[ns]")
        else:
            df[time_column] = pd.to_datetime(time_series).astype("datetime64[ns]")
    # Ensure request_time is datetime64[ns] array
    if isinstance(request_time, np.ndarray) and request_time.ndim >= 1:
        times_ns = request_time.astype("datetime64[ns]")
    else:
        times_ns = np.array([np.datetime64(request_time, "ns")])

    lower_bound, upper_bound = tolerance
    masks = [
        (df[time_column] >= t + lower_bound) & (df[time_column] <= t + upper_bound)
        for t in times_ns
    ]
    time_mask = reduce(lambda a, b: a | b, masks)
    return df[time_mask]


def dfseries_to_torch(
    series: pd.Series | cudf.Series,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a DataFrame series to a torch tensor with zero-copy for cudf.

    If the series is from a cudf DataFrame, uses dlpack for zero-copy transfer.
    If the series is from pandas but target device is GPU, transfers the data
    and warns that cudf is not being used.

    Parameters
    ----------
    series : pd.Series | cudf.Series
        Series to convert to torch tensor. Can be pandas or cudf Series.
    dtype : torch.dtype, optional
        Desired dtype for the tensor, by default torch.float32
    device : torch.device | str, optional
        Target device for the tensor, by default "cpu"

    Returns
    -------
    torch.Tensor
        Torch tensor with the series data on the specified device

    Raises
    ------
    ImportError
        If cudf is required but not available
    """
    device = torch.device(device)

    # Check if series is from cudf
    if cudf is not None and isinstance(series, cudf.Series):
        # Use dlpack for zero-copy transfer from cudf to torch
        return torch.from_dlpack(series.values).to(dtype=dtype, device=device)

    # Handle pandas Series
    if device.type == "cuda":
        # Warn that cudf is not being used for GPU transfer
        logger.warning(
            "Converting pandas Series to GPU tensor. Consider installing cudf "
            "for zero-copy transfer and better performance."
        )
        return torch.tensor(series.values, dtype=dtype, device=device)

    # CPU case - standard conversion
    return torch.tensor(series.values, dtype=dtype, device=device)
