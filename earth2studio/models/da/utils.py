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

from dataclasses import fields
from inspect import signature

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import xarray as xr

from earth2studio.data.base import DataFrameSource, ForecastFrameSource
from earth2studio.models.da.base import AssimilationInput
from earth2studio.utils.type import (
    CoordSystem,
    FieldArray,
    LeadTimeArray,
    TimeArray,
    VariableArray,
)

try:
    import cudf
except ImportError:
    cudf = None


def fetch_dataframe(
    source: DataFrameSource | ForecastFrameSource,
    time: TimeArray,
    variable: VariableArray,
    fields: FieldArray | None = None,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
) -> pa.Table | cudf.DataFrame:
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
    pa.Table | cudf.DataFrame
        PyArrow Table if device is CPU, cudf DataFrame if device is CUDA
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

    # Convert to appropriate format based on device
    if device.type == "cuda":
        if cudf is None:
            raise ImportError(
                "cudf is required for CUDA device. Install with: pip install cudf"
            )
        # Convert pandas DataFrame to cudf DataFrame
        return cudf.from_pandas(df)
    else:
        # Convert pandas DataFrame to PyArrow Table
        return pa.Table.from_pandas(df)


def validate_input(x: AssimilationInput, required_fields: list[str]) -> None:
    """Validate that required fields in AssimilationInput are not None and have correct types.

    Parameters
    ----------
    x : AssimilationInput
        Assimilation input to validate
    required_fields : list[str]
        List of required field names to validate (e.g., ["time", "lead_time"])

    Raises
    ------
    ValueError
        If a required field is None or empty
    TypeError
        If a required field has an incorrect data type
    """
    # Map type annotations to numpy dtype classes
    type_to_dtype = {
        TimeArray: np.datetime64,
        LeadTimeArray: np.timedelta64,
    }

    # Get all field names from the dataclass
    dataclass_fields = {f.name for f in fields(AssimilationInput)}
    field_dtypes = {}
    for field_obj in fields(AssimilationInput):
        field_type = field_obj.type
        if field_type in type_to_dtype:
            field_dtypes[field_obj.name] = type_to_dtype[field_type]

    for field in required_fields:
        if field not in dataclass_fields:
            raise ValueError(
                f"Unknown required field '{field}'. "
                f"Valid fields are: {sorted(dataclass_fields)}"
            )

        # Check if field is None
        value = getattr(x, field, None)
        if value is None:
            raise ValueError(f"Required field '{field}' in AssimilationInput is None")

        # Check if value is a numpy array
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Field '{field}' must be a numpy array, got {type(value)}")

        # Check dtype compatibility if we have a dtype mapping for this field
        if field in field_dtypes:
            expected_dtype = field_dtypes[field]
            if not np.issubdtype(value.dtype, expected_dtype):
                raise TypeError(
                    f"Field '{field}' must have dtype compatible with {expected_dtype.__name__}, "
                    f"got {value.dtype}"
                )


def validate_observation_type(
    observations: tuple[pd.DataFrame | xr.DataArray, ...],
    input_coords: tuple[CoordSystem, ...],
) -> None:
    """Validate that observations match the input coordinate systems in type and length.

    Parameters
    ----------
    observations : tuple[pd.DataFrame | xr.DataArray, ...]
        Tuple of observation data (DataFrames or DataArrays)
    input_coords : tuple[CoordSystem, ...]
        Tuple of input coordinate systems corresponding to each observation

    Raises
    ------
    ValueError
        If the length of observations doesn't match input_coords, or if the observation
        type doesn't match what's expected based on the coordinate system
    """
    if len(observations) != len(input_coords):
        raise ValueError(
            f"Number of observations ({len(observations)}) must match number of "
            f"input coordinate systems ({len(input_coords)})"
        )

    # DataFrame fields that indicate a CoordSystem expects DataFrame input
    dataframe_fields = {"observation", "lat", "lon", "time", "variable"}

    for i, (obs, coords) in enumerate(zip(observations, input_coords)):
        # Check if CoordSystem has DataFrame-like fields
        has_dataframe_fields = any(field in coords for field in dataframe_fields)

        if has_dataframe_fields:
            # CoordSystem indicates DataFrame input is expected
            if not isinstance(obs, pd.DataFrame):
                raise TypeError(
                    f"Observation {i} must be a pandas DataFrame based on input_coords, "
                    f"got {type(obs)}"
                )
        else:
            # CoordSystem indicates DataArray/tensor input is expected
            if not isinstance(obs, xr.DataArray):
                raise TypeError(
                    f"Observation {i} must be an xarray DataArray based on input_coords, "
                    f"got {type(obs)}"
                )


def validate_observation_fields(
    observation: pd.DataFrame, required_fields: list[str]
) -> None:
    """Validate that required fields are present as columns in the DataFrame.

    Parameters
    ----------
    observation : pd.DataFrame
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
