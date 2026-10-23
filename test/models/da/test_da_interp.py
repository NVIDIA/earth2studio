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
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

try:
    import cudf
except ImportError:
    cudf = None

try:
    import cupy as cp
except ImportError:
    cp = None

from earth2studio.models.da.interp import InterpEquirectangular


@pytest.fixture
def sample_observations_pandas():
    """Create sample pandas DataFrame observations for testing with multiple times and variables."""
    time1 = np.datetime64("2024-01-01T12:00:00")
    time2 = np.datetime64("2024-01-01T13:00:00")
    return pd.DataFrame(
        {
            "time": [
                time1,
                time1,
                time1,
                time1,
                time2,
                time2,
                time2,
                time2,
            ],
            "lat": [30.0, 30.0, 40.0, 40.0, 30.0, 30.0, 40.0, 40.0],
            "lon": [240.0, 250.0, 240.0, 250.0, 240.0, 250.0, 240.0, 250.0],
            "observation": [10.0, 20.0, 25.0, 35.0, 30.0, 40.0, 35.0, 45.0],
            "variable": ["t2m", "t2m", "u10m", "u10m", "t2m", "t2m", "u10m", "u10m"],
        }
    )


@pytest.fixture
def sample_observations_cudf():
    """Create sample cudf DataFrame observations for testing with multiple times and variables."""
    if cudf is None:
        pytest.skip("cudf not available")
    time1 = np.datetime64("2024-01-01T12:00:00")
    time2 = np.datetime64("2024-01-01T13:00:00")
    return cudf.DataFrame(
        {
            "time": [
                time1,
                time1,
                time1,
                time1,
                time2,
                time2,
                time2,
                time2,
            ],
            "lat": [30.0, 30.0, 40.0, 40.0, 30.0, 30.0, 40.0, 40.0],
            "lon": [240.0, 250.0, 240.0, 250.0, 240.0, 250.0, 240.0, 250.0],
            "observation": [10.0, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0, 60.0],
            "variable": ["t2m", "t2m", "u10m", "u10m", "t2m", "t2m", "u10m", "u10m"],
        }
    )


@pytest.fixture
def small_grid():
    """Create a small lat/lon grid for testing."""
    lat = np.linspace(25.0, 50.0, 11, dtype=np.float32)
    lon = np.linspace(235.0, 295.0, 13, dtype=np.float32)
    return lat, lon


@pytest.mark.parametrize(
    "interp_method",
    ["nearest", "smolyak"],
)
def test_interp_init(interp_method, small_grid):
    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method=interp_method)
    assert model.interp_method == interp_method
    assert np.array_equal(model._lat, lat)
    assert np.array_equal(model._lon, lon)
    assert model.input_coords() is None
    assert isinstance(model.forcing_coords(), dict)
    assert isinstance(model.output_coords(), xr.DataArray)

    with pytest.raises(ValueError, match="interp_method must be one of"):
        InterpEquirectangular(interp_method="invalid")


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
@pytest.mark.parametrize(
    "interp_method",
    ["nearest", "smolyak"],
)
def test_interp_call_pandas(
    sample_observations_pandas, small_grid, device, interp_method
):
    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method=interp_method).to(
        device
    )

    # Set up request metadata
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    sample_observations_pandas.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(sample_observations_pandas)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.shape == (1, 4, len(lat), len(lon))  # 4 times, 2 variables
    assert da.coords["time"].values[0] == request_time[0]
    assert len(da.coords["variable"]) == 4
    assert "t2m" in da.coords["variable"].values
    assert "u10m" in da.coords["variable"].values
    assert np.array_equal(da.coords["lat"].values, lat)
    assert np.array_equal(da.coords["lon"].values, lon)

    # Check device-specific return type
    if device == "cuda:0" and torch.cuda.is_available():
        if cp is not None:
            assert isinstance(da.data, cp.ndarray)
            assert not cp.all(cp.isnan(da.data))
        else:
            # If cupy not available, should fall back to numpy
            assert isinstance(da.data, np.ndarray)
    else:
        assert isinstance(da.data, np.ndarray)
        assert not np.all(np.isnan(da.values))


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "interp_method",
    ["nearest", "smolyak"],
)
def test_interp_call_cudf(sample_observations_cudf, small_grid, device, interp_method):
    if cudf is None:
        pytest.skip("cudf not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method=interp_method).to(
        device
    )

    # Set up request metadata
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    sample_observations_cudf.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(sample_observations_cudf)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.shape == (1, 4, len(lat), len(lon))  # 1 time, 4 variables
    assert da.coords["time"].values[0] == request_time[0]
    assert len(da.coords["variable"]) == 4
    assert "t2m" in da.coords["variable"].values
    assert "u10m" in da.coords["variable"].values
    assert np.array_equal(da.coords["lat"].values, lat)
    assert np.array_equal(da.coords["lon"].values, lon)

    # Check device-specific return type
    if cp is not None:
        assert isinstance(da.data, cp.ndarray)
        assert not cp.all(cp.isnan(da.data))
        t2m_data = da.sel(variable="t2m").data
        u10m_data = da.sel(variable="u10m").data
        assert not cp.any(cp.isnan(t2m_data))
        assert not cp.any(cp.isnan(u10m_data))
    else:
        assert isinstance(da.data, np.ndarray)
        assert not np.all(np.isnan(da.values))
        t2m_data = da.sel(variable="t2m").values
        u10m_data = da.sel(variable="u10m").values
        assert not np.any(np.isnan(t2m_data))
        assert not np.any(np.isnan(u10m_data))


def test_interp_multiple_times(sample_observations_pandas, small_grid):
    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method="nearest")
    model.VARIABLES = ["t2m"]

    time1 = np.datetime64("2024-01-01T12:00:00")
    time2 = np.datetime64("2024-01-01T13:00:00")
    request_time = np.array([time1, time2])
    sample_observations_pandas.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h"), np.timedelta64(0, "h")]),
    }

    da = model(sample_observations_pandas)

    assert da.shape == (2, 1, len(lat), len(lon))  # 2 times, 1 variable
    assert len(da.coords["time"]) == 2
    assert da.coords["time"].values[0] == time1
    assert da.coords["time"].values[1] == time2

    # Verify interpolated values for each time step
    # time1 t2m observations: (30.0, 240.0) = 10.0, (30.0, 250.0) = 20.0
    # time2 t2m observations: (30.0, 240.0) = 15.0, (30.0, 250.0) = 25.0
    time1_data = da.sel(time=time1).values
    time2_data = da.sel(time=time2).values

    assert not np.all(np.isnan(time1_data))
    assert not np.all(np.isnan(time2_data))

    # Values should be different between time steps
    assert not np.allclose(time1_data, time2_data, equal_nan=True)

    if len(time1_data) > 0:
        assert np.min(time1_data) >= 0  # Should be positive based on observations
        assert np.max(time1_data) <= 25  # Should be close to max observation (20.0)

    if len(time2_data) > 0:
        assert np.min(time2_data) >= 30  # Should be close to min observation (15.0)
        assert np.max(time2_data) <= 40  # Should be close to max observation (25.0)


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
def test_interp_tolerance(sample_observations_pandas, small_grid, device):
    lat, lon = small_grid
    # Use a larger tolerance to capture observations
    time_tolerance = np.timedelta64(2, "h")
    model = InterpEquirectangular(lat=lat, lon=lon, time_tolerance=time_tolerance).to(
        device
    )

    # Create observations with times spread out
    base_time = np.datetime64("2024-01-01T12:00:00")
    time1 = base_time - np.timedelta64(1, "h")
    time2 = base_time
    time3 = base_time + np.timedelta64(1, "h")
    time4 = base_time + np.timedelta64(3, "h")  # Outside tolerance

    tolerance_df = pd.DataFrame(
        {
            "time": [time1, time2, time3, time4],
            "lat": [30.0, 30.0, 40.0, 40.0],
            "lon": [240.0, 250.0, 240.0, 250.0],
            "observation": [10.0, 20.0, 30.0, 40.0],
            "variable": ["t2m", "t2m", "t2m", "t2m"],
        }
    )

    request_time = np.array([base_time])
    tolerance_df.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(tolerance_df)

    # Should interpolate using observations within tolerance (time1, time2, time3)
    assert da.shape == (1, 4, len(lat), len(lon))
    if cp is not None and isinstance(da.data, cp.ndarray):
        assert not cp.all(cp.isnan(da.data))
    else:
        assert not np.all(np.isnan(da.values))


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
@pytest.mark.parametrize("interp_method", ["nearest", "smolyak"])
def test_interp_independent_calls(
    sample_observations_pandas: pd.DataFrame,
    small_grid: tuple[np.ndarray, np.ndarray],
    device: str,
    interp_method: str,
) -> None:
    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method=interp_method).to(
        device
    )

    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    obs = sample_observations_pandas.iloc[:4].copy()
    obs["variable"] = "t2m"
    obs.attrs = {
        "request_time": request_time,
        "variable": np.array(["t2m"]),
    }
    original_obs = obs.copy(deep=True)

    first = model(obs)
    first_values = first.e2s.to_torch()[0].clone()
    assert torch.isfinite(first_values).all()

    next_obs = obs.copy(deep=True)
    next_obs["time"] += np.timedelta64(1, "D")
    next_obs["observation"] += 50.0
    next_obs["variable"] = "u10m"
    next_obs.attrs = {
        "request_time": request_time + np.timedelta64(1, "D"),
        "variable": np.array(["u10m"]),
    }
    second = model(next_obs)
    assert second.shape == (1, 1, len(lat), len(lon))
    np.testing.assert_array_equal(second.time, next_obs.attrs["request_time"])
    np.testing.assert_array_equal(second.coords["variable"], ["u10m"])
    torch.testing.assert_close(
        second.e2s.to_torch()[0], first_values + 50.0, atol=1e-4, rtol=1e-5
    )

    repeated = model(obs)
    np.testing.assert_array_equal(repeated.time, request_time)
    np.testing.assert_array_equal(repeated.coords["variable"], ["t2m"])
    torch.testing.assert_close(repeated.e2s.to_torch()[0], first_values)
    torch.testing.assert_close(first.e2s.to_torch()[0], first_values)
    assert repeated.e2s.to_torch()[0].device == torch.device(device)
    pd.testing.assert_frame_equal(obs, original_obs)
    for key in original_obs.attrs:
        np.testing.assert_array_equal(obs.attrs[key], original_obs.attrs[key])


def test_interp_call_validates_observations(
    sample_observations_pandas: pd.DataFrame,
    small_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    lat, lon = small_grid
    model = InterpEquirectangular(lat=lat, lon=lon, interp_method="nearest")
    with pytest.raises(ValueError, match="request_time"):
        model(sample_observations_pandas)

    sample_observations_pandas.attrs["request_time"] = np.array(
        [np.datetime64("2024-01-01T12:00:00")]
    )
    with pytest.raises(ValueError, match="missing required fields"):
        model(sample_observations_pandas.drop(columns="observation"))
