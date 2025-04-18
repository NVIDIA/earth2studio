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

import numpy as np
import pytest
import torch

from earth2studio.models.dx import (
    TCTrackerVitart,
    TCTrackerWuDuan,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_vorticity_calculation(device):
    nx, ny = 4, 4
    dx = dy = 1.0

    y_vals = torch.arange(ny, dtype=torch.float32, device=device)
    x_vals = torch.arange(nx, dtype=torch.float32, device=device)

    # u varies with y
    u = y_vals.view(-1, 1).repeat(1, nx)
    # v varies with x
    v = x_vals.view(1, -1).repeat(ny, 1)

    # Expected vorticity should be 2 since du/dy = 1 and dv/dx = 1
    expected_vorticity = torch.full((ny, nx), 2.0, device=device)
    calculated_vorticity = TCTrackerVitart.vorticity(u, v, dx=dx, dy=dy)

    torch.testing.assert_close(
        calculated_vorticity,
        expected_vorticity,
        rtol=1e-5,
        atol=1e-5,
    )

    # u varies with x
    u = y_vals.view(1, -1).repeat(ny, 1)
    # v varies with y
    v = x_vals.view(-1, 1).repeat(1, nx)

    # Expected vorticity should be 2 since du/dy = 0 and dv/dx = 0
    expected_vorticity = torch.full((ny, nx), 0.0, device=device)
    calculated_vorticity = TCTrackerVitart.vorticity(u, v, dx=dx, dy=dy)

    torch.testing.assert_close(
        calculated_vorticity,
        expected_vorticity,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_haversine_torch(device):
    # Define known points
    lat1 = torch.tensor(0, dtype=torch.float32, device=device)
    lon1 = torch.tensor(0, dtype=torch.float32, device=device)
    lat2 = torch.tensor(30, dtype=torch.float32, device=device)
    lon2 = torch.tensor(120, dtype=torch.float32, device=device)

    # Known expected distance in kilometers (approximate)
    expected_distance = torch.tensor(12860.0).to(device)
    dist = TCTrackerVitart.haversine_torch(lat1, lon1, lat2, lon2)

    torch.testing.assert_close(expected_distance, dist, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_latlon_to_equirectangular(device):
    lats = torch.tensor(
        [[0.0, 45.0, -45.0], [90.0, -90.0, 0.0]],  # Equator and mid-latitudes
        device=device,
    )  # Poles and equator
    lons = torch.tensor(
        [[0.0, 90.0, -90.0], [180.0, -180.0, 45.0]],  # Prime meridian and +/- 90°
        device=device,
    )  # Date line and 45°
    result = TCTrackerVitart.latlon_to_equirectangular(lats, lons)

    # Expected values (approximate)
    R = 6371.0
    expected_x = R * torch.deg2rad(lons)  # x = R * λ * cos(0°)
    expected_y = R * torch.deg2rad(lats)  # y = R * φ
    expected = torch.stack([expected_x, expected_y], dim=-1)

    assert result.shape == (2, 3, 2)
    assert torch.allclose(result, expected, rtol=1e-5)
    assert torch.allclose(result[0, 0, 0], torch.tensor(0.0), atol=1e-6)  # x at 0° lon
    assert torch.allclose(result[1, 0, 0], -result[1, 1, 0], atol=1e-6)  # 180° vs -180°
    assert torch.allclose(result[0, 0, 1], result[1, 2, 1], atol=1e-6)  # Both at 0° lat


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("num_maxima", [0, 1, 2])
def test_get_local_max(device, num_maxima):
    # Define test cases based on the number of maxima
    test_cases = {
        0: {
            "x": torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "expected": torch.empty((0, 2), dtype=torch.int64),
        },
        1: {
            "x": torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "expected": torch.tensor([[1, 1]], dtype=torch.int64),
        },
        2: {
            "x": torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 5],
                ],
                dtype=torch.float32,
            ),
            "expected": torch.tensor([[3, 3], [1, 1]], dtype=torch.int64),
        },
    }

    # Get the specific test case based on num_maxima
    test_case = test_cases[num_maxima]

    x = test_case["x"].to(device)
    expected_output = test_case["expected"].to(device)

    # Define parameters for get_local_max
    threshold_abs = 0.9
    min_distance = 1
    exclude_border = False

    # Call the method
    local_max = TCTrackerVitart.get_local_max(
        x, threshold_abs, min_distance, exclude_border
    )

    # Assertions
    assert torch.all(
        local_max == expected_output
    ), f"Expected {expected_output}, but got {local_max}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_multiple_tracks(device):
    # Create two tracks: one moving northeast, one moving northwest
    frames = [
        torch.tensor(
            [
                [[30.0, 60.0, 1.0, 1.0], [35.0, 70.0, 2.0, 2.0]],
                [[-20.0, 190.0, 1.0, 1.0], [-25.0, 210.0, 2.0, 2.0]],
            ],
            device=device,
        ),
        torch.tensor(
            [
                [[30.2, 60.2, 1.1, 1.1], [35.2, 69.8, 2.1, 2.1]],
                [[-20.3, 191.0, 1.0, 1.0], [-25.2, 209.0, 2.0, 2.0]],
            ],
            device=device,
        ),
    ]

    path_buffer = frames[0].unsqueeze(2)
    for frame in frames:
        path_buffer = TCTrackerVitart.append_paths(
            frame, path_buffer, path_search_distance=100.0
        )

    assert path_buffer.shape == (2, 3, 3, 4)
    assert torch.abs(path_buffer[0, 0, :, 0] - path_buffer[0, 1, :, 0]).mean() > 4.0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_track_with_gap(device):
    frames = [
        torch.tensor([[[30.0, 60.0, 1.0, 1.0]]], device=device),  # t=0
        torch.tensor(
            [[[TCTrackerVitart.PATH_FILL_VALUE] * 4]], device=device
        ),  # t=1 (gap)
        torch.tensor([[[30.4, 60.4, 1.2, 1.2]]], device=device),  # t=2
    ]

    path_buffer = torch.empty(0)
    for frame in frames:
        path_buffer = TCTrackerVitart.append_paths(
            frame, path_buffer, path_search_distance=50.0, path_search_window_size=2
        )

    # Assertions
    assert path_buffer.shape == (1, 1, 3, 4)
    assert torch.allclose(path_buffer[0, 0, 0], frames[0][0, 0])
    assert torch.allclose(path_buffer[0, 0, -1], frames[-1][0, 0])
    assert torch.all(path_buffer[0, 0, 1] == TCTrackerVitart.PATH_FILL_VALUE)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_empty_initialization(device):
    frame = torch.tensor([[[30.0, 60.0, 1.0, 1.0]]], device=device)
    path_buffer = torch.empty(0, device=device)

    result = TCTrackerVitart.append_paths(frame, path_buffer)

    assert result.shape == (1, 1, 1, 4)
    assert torch.allclose(result[0, 0, 0], frame[0, 0])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        TCTrackerVitart.append_paths(
            torch.ones((1, 1, 4)),
            torch.ones((2, 1, 1, 4)),  # Mismatched batch size
            path_search_distance=50.0,
        )

    with pytest.raises(ValueError):
        TCTrackerVitart.append_paths(
            torch.ones((1, 1, 4)),
            torch.ones((1, 1, 1, 4)),
            path_search_distance=-1.0,  # Invalid distance
        )

    with pytest.raises(ValueError):
        TCTrackerVitart.append_paths(
            torch.ones((1, 1, 4)),
            torch.ones((1, 1, 1, 4)),
            path_search_window_size=0,  # Invalid window size
        )


@pytest.mark.parametrize("num_timesteps", [1, 2])
@pytest.mark.parametrize("tc_included", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cyclone_tracking_wuduan(num_timesteps, tc_included, device):
    # Set dimensions
    num_vars, height, width = 5, 721, 1440

    # Initialize x with random values
    x = torch.zeros(num_timesteps, num_vars, height, width, device=device)

    # Define the initial center latitude and longitude in actual units
    initial_center_lat = 15
    initial_center_lon = 280.0

    # Define the movement of the center in actual latitude and longitude units
    lat_movement = [
        0.0,
        1.0,
        -1.5,
    ]  # Center moves north by 2 and then 4 degrees latitude
    lon_movement = [
        0.0,
        -1.0,
        -2.0,
    ]  # Center moves west by 1 and then 2 degrees longitude

    # Create latitude and longitude arrays
    lats = np.linspace(90, -90, height, endpoint=True)
    lons = np.linspace(0, 360, width, endpoint=False)

    # Initialize the CycloneTrackingVorticity model
    ct = TCTrackerWuDuan()

    for t in range(num_timesteps):
        current_center_lat = initial_center_lat + lat_movement[t]
        current_center_lon = initial_center_lon + lon_movement[t]

        if not tc_included:
            continue

        y_vals = (
            torch.from_numpy(lats)
            .to(dtype=torch.float32, device=device)
            .view(-1, 1)
            .repeat(1, width)
        )
        x_vals = (
            torch.from_numpy(lons)
            .to(dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(height, 1)
        )

        # Calculate distances in lat-lon space
        lat_dist = (y_vals - current_center_lat) * np.pi / 180.0
        lon_dist = (x_vals - current_center_lon) * np.pi / 180.0

        # Convert to radians
        R = 6371.0  # Earth radius in kilometers
        dlat = lat_dist
        dlon = lon_dist * torch.cos(y_vals * np.pi / 180.0)
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(y_vals * np.pi / 180.0)
            * torch.cos(torch.tensor(current_center_lat * np.pi / 180.0))
            * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = R * c  # Distance in kilometers

        radius_km = 100.0  # Radius of the cyclone center in kilometers

        # Increase the amplitude of u850 and v850 to ensure higher vorticity
        amplitude = 1e4  # Amplitude to increase the magnitude of wind components
        u850 = (
            amplitude * torch.sin(dlat) * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        v850 = (
            -amplitude
            * torch.sin(dlon)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )

        # Assign u850 and v850 to the appropriate index in x for each timestep
        x[t, 3] = u850
        x[t, 4] = v850
        # Create 10m wind components with similar structure to 850hPa winds but lower amplitude
        max_10m = 20
        amplitude_10m = 5e3  # Lower amplitude for surface winds
        u10m = (
            amplitude_10m
            * torch.sin(dlat)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        v10m = (
            amplitude_10m
            * torch.sin(dlon)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        u10m = torch.clamp(u10m, -max_10m, max_10m)
        v10m = torch.clamp(v10m, -max_10m, max_10m)

        # Create MSL pressure field centered on hurricane with lower pressure in center
        msl_ambient = 1013.0  # Ambient pressure in hPa
        msl_center = 980.0  # Center pressure in hPa
        msl = msl_ambient - (msl_ambient - msl_center) * torch.exp(
            -(distance**2) / (2 * radius_km**2)
        )

        x[t, 0] = u10m  # u10m
        x[t, 1] = v10m  # v10m
        x[t, 2] = 100 * msl  # msl

    # Set up mock coordinates dictionary
    coords = {
        "time": np.array([1]),
        "variable": ct.input_coords()["variable"],
        "lat": lats,
        "lon": lons,
    }
    # Forward pass through the model
    for t in range(x.shape[0]):
        y, c = ct(x[t : t + 1], coords)

    if not tc_included:
        assert torch.all(y.isnan())
        assert np.all(c["step"] == np.arange(num_timesteps))
        assert y.shape == (1, 1, num_timesteps, 4)
        return

    assert y.shape == (1, 1, num_timesteps, 4)
    assert np.all(c["step"] == np.arange(num_timesteps))
    # Check the tracking is close to expected
    for t in range(num_timesteps):
        current_center_lat = initial_center_lat + lat_movement[t]
        current_center_lon = initial_center_lon + lon_movement[t]
        assert np.allclose(
            y[0, 0, t, :2].cpu(),
            np.array([current_center_lat, current_center_lon]),
            rtol=1e-1,
        )
        assert np.allclose(y[0, 0, t, 2].cpu(), 100 * msl_center, rtol=1e-1)
        assert np.allclose(
            y[0, 0, t, 3].cpu(), np.sqrt([max_10m**2 + max_10m**2]), rtol=1e-1
        )
    assert y.device == torch.device(device)


@pytest.mark.parametrize("num_timesteps", [1, 2])
@pytest.mark.parametrize("tc_included", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cyclone_tracking_vitart(num_timesteps, tc_included, device):
    """Runs hurricane identification with synthetic data"""
    # Set dimensions
    num_vars, height, width = 13, 721, 1440

    # Initialize x with random values
    torch.manual_seed(0)
    x = torch.rand(num_timesteps, num_vars, height, width, device=device)
    x[:, 2] = 101300  # Bump up msl field to atmos value

    # Define the initial center in gulf of mexico
    initial_center_lat = 24
    initial_center_lon = 90.0

    # Define the movement of the center in actual latitude and longitude units
    lat_movement = [
        0.0,
        -2.0,
        -4.0,
    ]  # Center moves north by 2 and then 4 degrees latitude
    lon_movement = [
        0.0,
        -1.0,
        -2.0,
    ]  # Center moves west by 1 and then 2 degrees longitude

    # Create latitude and longitude arrays
    lats = np.linspace(90, -90, height, endpoint=True)
    lons = np.linspace(0, 360, width, endpoint=False)

    # Initialize the CycloneTracking model
    ct = TCTrackerVitart()

    for t in range(num_timesteps):
        current_center_lat = initial_center_lat + lat_movement[t]
        current_center_lon = initial_center_lon + lon_movement[t]

        if not tc_included:
            continue

        y_vals = (
            torch.from_numpy(lats)
            .to(dtype=torch.float32, device=device)
            .view(-1, 1)
            .repeat(1, width)
        )
        x_vals = (
            torch.from_numpy(lons)
            .to(dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(height, 1)
        )

        # Define the center of the cyclone for other variables
        center_lat_index = np.argmin(np.abs(lats - current_center_lat))
        center_lon_index = np.argmin(np.abs(lons - current_center_lon))

        # Calculate distances in lat-lon space
        lat_dist = (y_vals - current_center_lat) * np.pi / 180.0
        lon_dist = (x_vals - current_center_lon) * np.pi / 180.0

        # Convert to radians
        R = 6371.0  # Earth radius in kilometers
        dlat = lat_dist
        dlon = lon_dist * torch.cos(y_vals * np.pi / 180.0)
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(y_vals * np.pi / 180.0)
            * torch.cos(torch.tensor(current_center_lat * np.pi / 180.0))
            * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = R * c  # Distance in kilometers
        radius_km = 100.0  # Radius of the cyclone center in kilometers

        # Define the MSL (mean sea level pressure) with a local minimum at the center
        msl_ambient = 101300.0  # Ambient pressure in Pa
        msl_center = 98000.0  # Center pressure in Pa
        msl = msl_ambient - (msl_ambient - msl_center) * torch.exp(
            -(distance**2) / (2 * radius_km**2)
        )

        # Create 10m wind components with similar structure to 850hPa winds but lower amplitude
        max_10m = 20
        amplitude_10m = 5e3  # Lower amplitude for surface winds
        u10m = (
            amplitude_10m
            * torch.sin(dlat)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        v10m = (
            amplitude_10m
            * torch.sin(dlon)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        u10m = torch.clamp(u10m, -max_10m, max_10m)
        v10m = torch.clamp(v10m, -max_10m, max_10m)

        x[t, 0] = u10m
        x[t, 1] = v10m
        x[t, 2] = msl

        # Increase the amplitude of u850 and v850 to ensure higher vorticity
        amplitude = 1e4  # Amplitude to increase the magnitude of wind components
        u850 = (
            amplitude * torch.sin(dlat) * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        v850 = (
            -amplitude
            * torch.sin(dlon)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        x[t, 3] = u850
        x[t, 4] = v850

        # Define local maximum of average temperatures between 200hPa and 500hPa
        # Tempertures need to decrease strongly with distance to the center
        t_dummy = torch.ones_like(x[t, 0, :, :]) * 273
        t_dummy[
            center_lat_index - 45 : center_lat_index + 45,
            center_lon_index - 45 : center_lon_index + 45,
        ] = 300
        t_dummy[
            center_lat_index - 5 : center_lat_index + 5,
            center_lon_index - 5 : center_lon_index + 5,
        ] = 320
        t_dummy[center_lat_index, center_lon_index] = 340
        # Assign the same profile to all temperature variables
        for var_index in range(8, 13):
            x[t, var_index] = t_dummy

        # Define local local z200 - z850
        z200 = torch.ones_like(x[t, 0, :, :]) * 0
        z850 = torch.ones_like(x[t, 0, :, :]) * 0
        z200[
            center_lat_index - 5 : center_lat_index + 5,
            center_lon_index - 5 : center_lon_index + 5,
        ] = 1
        x[t, 6] = z850
        x[t, 7] = z200

    # Set up mock coordinates dictionary
    coords = {
        "time": np.array(list(range(0, num_timesteps))),
        "variable": ct.input_coords()["variable"],
        "lat": lats,
        "lon": lons,
    }

    # Forward pass through the model
    for t in range(x.shape[0]):
        y, c = ct(x[t : t + 1], coords)

    if not tc_included:
        assert torch.all(y.isnan())
        assert np.all(c["step"] == np.arange(num_timesteps))
        assert y.shape == (1, 1, num_timesteps, 4)
        return

    assert y.shape == (1, 1, num_timesteps, 4)
    assert np.all(c["step"] == np.arange(num_timesteps))
    # Check the tracking is close to expected
    for t in range(num_timesteps):
        current_center_lat = initial_center_lat + lat_movement[t]
        current_center_lon = initial_center_lon + lon_movement[t]
        assert np.allclose(
            y[0, 0, t, :2].cpu(),
            np.array([current_center_lat, current_center_lon]),
            rtol=1e-1,
        )
        assert np.allclose(y[0, 0, t, 2].cpu(), msl_center, rtol=1e-1)  # mls
        assert np.allclose(
            y[0, 0, t, 3].cpu(), np.sqrt([max_10m**2 + max_10m**2]), rtol=1e-1
        )  # z
    assert y.device == torch.device(device)
