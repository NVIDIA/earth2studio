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

import pandas as pd
import torch
from loguru import logger

from earth2studio.utils.obs import filter_time_range  # noqa: F401

try:
    import cudf
    from cudf import DataFrame as cudf_DataFrame
except ImportError:
    cudf = None
    cudf_DataFrame = None


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
