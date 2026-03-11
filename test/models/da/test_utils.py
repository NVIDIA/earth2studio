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

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch

try:
    import cudf
except ImportError:
    cudf = None

from earth2studio.models.da.utils import (
    dfseries_to_torch,
    filter_time_range,
    validate_observation_fields,
)


@pytest.fixture
def sample_pandas_df():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01T10:00:00",
                    "2024-01-01T11:00:00",
                    "2024-01-01T12:00:00",
                    "2024-01-01T13:00:00",
                ]
            ),
            "lat": [30.0, 31.0, 32.0, 33.0],
            "lon": [240.0, 241.0, 242.0, 243.0],
            "observation": [10.0, 20.0, 30.0, 40.0],
            "variable": ["t2m", "t2m", "u10m", "u10m"],
        }
    )


@pytest.fixture
def sample_cudf_df():
    """Create a sample cudf DataFrame for testing."""
    if cudf is None:
        pytest.skip("cudf not available")
    return cudf.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01T10:00:00",
                    "2024-01-01T11:00:00",
                    "2024-01-01T12:00:00",
                    "2024-01-01T13:00:00",
                ]
            ),
            "lat": [30.0, 31.0, 32.0, 33.0],
            "lon": [240.0, 241.0, 242.0, 243.0],
            "observation": [10.0, 20.0, 30.0, 40.0],
            "variable": ["t2m", "t2m", "u10m", "u10m"],
        }
    )


def test_validate_observation_fields(sample_pandas_df):
    validate_observation_fields(sample_pandas_df, ["time", "lat", "lon"])
    validate_observation_fields(
        sample_pandas_df, ["time", "lat", "lon", "observation", "variable"]
    )
    validate_observation_fields(sample_pandas_df, ["time"])

    with pytest.raises(ValueError, match="DataFrame missing required fields"):
        validate_observation_fields(sample_pandas_df, ["time", "missing_field"])

    with pytest.raises(ValueError, match="DataFrame missing required fields"):
        validate_observation_fields(
            sample_pandas_df, ["time", "missing_field1", "missing_field2"]
        )


class TestFilterTimeRange:
    @pytest.mark.parametrize(
        "request_time",
        [
            np.datetime64("2024-01-01T12:00:00"),
            datetime(2024, 1, 1, 12, 0, 0),
            "2024-01-01T12:00:00",
        ],
    )
    def test_filter_time_range_pandas(self, sample_pandas_df, request_time):
        tolerance = (np.timedelta64(-30, "m"), np.timedelta64(30, "m"))
        result = filter_time_range(sample_pandas_df, request_time, tolerance)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["time"] == pd.Timestamp(request_time)

    @pytest.mark.parametrize(
        "request_time",
        [
            np.datetime64("2024-01-01T12:00:00"),
            datetime(2024, 1, 1, 12, 0, 0),
            "2024-01-01T12:00:00",
        ],
    )
    def test_filter_time_range_cudf(self, sample_cudf_df, request_time):
        if cudf is None:
            pytest.skip("cudf not available")
        tolerance = (np.timedelta64(-30, "m"), np.timedelta64(30, "m"))
        result = filter_time_range(sample_cudf_df, request_time, tolerance)

        assert isinstance(result, cudf.DataFrame)
        assert len(result) == 1
        assert (result.iloc[0]["time"] == pd.Timestamp(request_time)).all()

    def test_filter_time_range_tolerance_window(self, sample_pandas_df):
        request_time = np.datetime64("2024-01-01T11:30:00")
        tolerance = (np.timedelta64(-30, "m"), np.timedelta64(30, "m"))
        result = filter_time_range(sample_pandas_df, request_time, tolerance)
        assert len(result) == 2

        tolerance = (np.timedelta64(-2, "h"), np.timedelta64(2, "h"))
        result = filter_time_range(sample_pandas_df, request_time, tolerance)
        assert len(result) >= 2

        request_time = np.datetime64("2024-01-01T20:00:00")
        tolerance = (np.timedelta64(-1, "h"), np.timedelta64(1, "h"))
        result = filter_time_range(sample_pandas_df, request_time, tolerance)
        assert len(result) == 0

    def test_filter_time_range_custom_column(self, sample_pandas_df):
        df = sample_pandas_df.rename(columns={"time": "timestamp"})
        request_time = np.datetime64("2024-01-01T12:00:00")
        tolerance = (np.timedelta64(-1, "h"), np.timedelta64(1, "h"))
        result = filter_time_range(df, request_time, tolerance, time_column="timestamp")
        assert len(result) == 3


class TestDfseriesToTorch:
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="cuda missing"
                ),
            ),
        ],
    )
    def test_dfseries_to_torch_pandas_cpu(self, sample_pandas_df, device):
        if device != "cpu" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        series = sample_pandas_df["lat"]
        tensor = dfseries_to_torch(series, dtype=torch.float32, device=device)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == device.split(":")[0] if ":" in device else device
        assert len(tensor) == len(series)
        assert torch.allclose(
            tensor, torch.tensor(series.values, dtype=torch.float32, device=device)
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="cuda missing"
                ),
            ),
        ],
    )
    def test_dfseries_to_torch_cudf(self, sample_cudf_df, device):
        if cudf is None:
            pytest.skip("cudf not available")
        if device != "cpu" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        series = sample_cudf_df["lat"]
        tensor = dfseries_to_torch(series, dtype=torch.float32, device=device)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == device.split(":")[0] if ":" in device else device
        assert len(tensor) == len(series)
