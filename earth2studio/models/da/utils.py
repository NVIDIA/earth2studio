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

import numpy as np
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


# US Standard Atmosphere 1976 layer table for pressure -> height.
_USSA_H_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0])
_USSA_LAPSE_K_M = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
_USSA_T_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_USSA_P_PA = np.array(
    [
        101_325.0,
        22_632.06397346295,
        5_474.888669677785,
        868.0186847552303,
        110.9063055549665,
        66.93887311868764,
        3.9564204280407553,
    ]
)
_USSA_TOP_P_PA = 0.37338358997621796
_USSA_K = 8.31432 / (9.80665 * 0.0289644)


def pressure_to_height_m(pressure_pa: np.ndarray) -> np.ndarray:
    """USSA-1976 geopotential height (m) for a pressure (Pa), floored at 0.

    A climatological pressure altitude for observations that report only a
    pressure (e.g. satellite AMVs); NaN where the pressure is not finite, not
    positive, or above the 84.852 km table top.
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    height = np.full(p.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(p) & (p >= _USSA_TOP_P_PA)
    for index in range(_USSA_P_PA.size):
        last = index + 1 == _USSA_P_PA.size
        top_p = _USSA_TOP_P_PA if last else _USSA_P_PA[index + 1]
        below = p <= _USSA_P_PA[index] if index else np.ones(p.shape, dtype=bool)
        above_top = p >= top_p if last else p > top_p
        layer = usable & below & above_top
        if not layer.any():
            continue
        ratio = p[layer] / _USSA_P_PA[index]
        lapse = _USSA_LAPSE_K_M[index]
        if lapse:
            height[layer] = _USSA_H_M[index] + _USSA_T_K[index] / lapse * (
                np.power(ratio, -lapse * _USSA_K) - 1.0
            )
        else:
            height[layer] = _USSA_H_M[index] - _USSA_K * _USSA_T_K[index] * np.log(
                ratio
            )
    return np.maximum(height, 0.0).astype(np.float32)
