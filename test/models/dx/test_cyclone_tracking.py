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

from earth2studio.models.dx import CycloneTracking, CycloneTrackingVorticity


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
    calculated_vorticity = CycloneTracking.vorticity(u, v, dx=dx, dy=dy)

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
    calculated_vorticity = CycloneTracking.vorticity(u, v, dx=dx, dy=dy)

    torch.testing.assert_close(
        calculated_vorticity,
        expected_vorticity,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_haversine_torch(device):
    # Define known points
    lat1 = torch.tensor(0, dtype=torch.float32)
    lon1 = torch.tensor(0, dtype=torch.float32)
    lat2 = torch.tensor(30, dtype=torch.float32)
    lon2 = torch.tensor(120, dtype=torch.float32)

    # Known expected distance in kilometers (approximate)
    expected_distance = torch.tensor(12860.0)

    # Calculate distance using haversine_torch
    dist = CycloneTracking.haversine_torch(lat1, lon1, lat2, lon2, meters=False)

    # Assert that the calculated distance is close to the expected distance
    torch.testing.assert_close(expected_distance, dist, rtol=1e-3, atol=1e-3)


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
    local_max = CycloneTracking.get_local_max(
        x, threshold_abs, min_distance, exclude_border
    )

    # Assertions
    assert torch.all(
        local_max == expected_output
    ), f"Expected {expected_output}, but got {local_max}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("num_timesteps", [0, 1, 2, 3])
def test_cyclone_tracking_vorticity(num_timesteps, device):
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

    # Initialize the CycloneTrackingVorticity model
    ct = CycloneTrackingVorticity()

    for t in range(num_timesteps):
        current_center_lat = initial_center_lat + lat_movement[t]
        current_center_lon = initial_center_lon + lon_movement[t]

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
        amplitude = 1e7  # Amplitude to increase the magnitude of wind components
        u850 = (
            amplitude * torch.sin(dlat) * torch.exp(-(distance**2) / (2 * radius_km**2))
        )
        v850 = (
            -amplitude
            * torch.sin(dlon)
            * torch.exp(-(distance**2) / (2 * radius_km**2))
        )

        # Calculate vorticity
        vorticity = ct.vorticity(u850, v850)

        # Debug: Print max vorticity to ensure it is sufficient
        print(f"Max vorticity at timestep {t}: {torch.max(vorticity).item()}")

        # Assign u850 and v850 to the appropriate index in x for each timestep
        x[t, 0] = u850
        x[t, 1] = v850

    # Set up mock coordinates dictionary
    coords = {
        "time": np.array(list(range(0, num_timesteps))),
        "variable": np.array(["u850", "v850", "u10m", "v10m", "msl"]),
        "lat": lats,
        "lon": lons,
    }

    # Forward pass through the model
    y, c = ct(x, coords)

    # Define the expected center positions
    expected_lat_values = np.array(
        [initial_center_lat + lat_movement[t] for t in range(num_timesteps)]
    )
    expected_lon_values = np.array(
        [initial_center_lon + lon_movement[t] for t in range(num_timesteps)]
    )

    # Check expected coordinates
    assert list(c["time"]) == list(range(0, num_timesteps))
    assert list(c["variable"]) == ["tc_lat", "tc_lon", "tc_msl", "tc_w10m"]

    # Check expected values for each timestep
    for t in range(num_timesteps):
        assert (
            pytest.approx(y[t, 0].item(), expected_lat_values[t], 0.1) == y[t, 0].item()
        ), f"Expected {expected_lat_values[t]}, but got {y[t, 0].item()}"
        assert (
            pytest.approx(y[t, 1].item(), expected_lon_values[t], 0.1) == y[t, 1].item()
        ), f"Expected {expected_lon_values[t]}, but got {y[t, 1].item()}"
        assert (
            pytest.approx(y[t, 2].item(), 0, 0.01) == y[t, 2].item()
        ), f"Expected 0, but got {y[t, 2].item()}"
        assert (
            pytest.approx(y[t, 3].item(), 0, 0.01) == y[t, 3].item()
        ), f"Expected 0, but got {y[t, 3].item()}"


# import numpy as np
# import pytest

# from earth2studio.models.dx import (
#     CycloneTracking,
#     CycloneTrackingVorticity,
# )
# from earth2studio.utils.coords import map_coords


# @pytest.mark.timeout(30)
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# @pytest.mark.parametrize("tracker", ["T1", "T2"])
# def test_tc_shapes(device, tracker):

#     if tracker == "T1":
#         CT = CycloneTracking()
#     elif tracker == "T2":
#         CT = CycloneTrackingVorticity()

#     time = np.array(
#         [
#             np.datetime64("2023-08-29T00:00:00"),
#             np.datetime64("2024-06-30T00:00:00"),
#             np.datetime64("2024-07-04T00:00:00"),
#         ]
#     )
#     variable = CT.input_coords()["variable"]

#     gfs = GFS()
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da, device=device)
#     y, c = CT(x, coords)

#     assert y.shape[0] == len(time)
#     if tracker == "T1":
#         assert y.shape[1] == 2
#         assert y.shape[2] == 2
#     elif tracker == "T2":
#         assert y.shape[1] == 4
#         assert y.shape[2] == 16

#     assert "point" in c
#     if tracker == "T1":
#         assert all(c["point"] == np.arange(2))
#     elif tracker == "T2":
#         assert all(c["point"] == np.arange(16))

#     if tracker == "T1":
#         assert "coord" in c
#         assert all(c["coord"] == np.array(["lat", "lon"]))
#     elif tracker == "T2":
#         assert "variable" in c
#         assert all(c["variable"] == np.array(["tc_lat", "tc_lon", "tc_msl", "tc_w10m"]))

#     assert "time" in c
#     assert all(c["time"] == coords["time"])


# @pytest.mark.timeout(30)
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# @pytest.mark.parametrize("tracker", ["T1", "T2"])
# def test_tc_shapes_ensemble(device, tracker):

#     if tracker == "T1":
#         CT = CycloneTracking()
#     elif tracker == "T2":
#         CT = CycloneTrackingVorticity()

#     time = np.array(
#         [
#             np.datetime64("2023-08-29T00:00:00"),
#             np.datetime64("2024-06-30T00:00:00"),
#             np.datetime64("2024-07-04T00:00:00"),
#         ]
#     )
#     variable = CT.input_coords()["variable"]

#     gfs = GFS()
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da, device=device)

#     x = x.unsqueeze(0)
#     coords = {"ensemble": np.arange(1)} | coords

#     y, c = CT(x, coords)

#     assert y.shape[0] == 1
#     assert y.shape[1] == len(time)
#     if tracker == "T1":
#         assert y.shape[2] == 2
#         assert y.shape[3] == 2
#     elif tracker == "T2":
#         assert y.shape[2] == 4
#         assert y.shape[3] == 16

#     assert "point" in c
#     if tracker == "T1":
#         assert all(c["point"] == np.arange(2))
#     elif tracker == "T2":
#         assert all(c["point"] == np.arange(16))

#     if tracker == "T1":
#         assert "coord" in c
#         assert all(c["coord"] == np.array(["lat", "lon"]))
#     elif tracker == "T2":
#         assert "variable" in c
#         assert all(c["variable"] == np.array(["tc_lat", "tc_lon", "tc_msl", "tc_w10m"]))

#     assert "time" in c
#     assert all(c["time"] == coords["time"])

#     assert "ensemble" in c
#     assert all(c["ensemble"] == coords["ensemble"])


# @pytest.mark.ci_cache
# @pytest.mark.timeout(30)
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# def test_tc_non_rectangular(device, model_cache_context):

#     with model_cache_context():
#         package = CorrDiffTaiwan.load_default_package()
#         dx = CorrDiffTaiwan.load_model(package).to(device)

#     CT = CycloneTracking()

#     time = np.array(
#         [
#             np.datetime64("2023-10-04T18:00:00"),
#         ]
#     )
#     variable = CT.input_coords()["variable"]

#     gfs = GFS()
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da, device=device)

#     # Map to non-rectangular lat, lon
#     output_coords = dx.input_coords()
#     output_coords.pop("variable")
#     x, coords = map_coords(x, coords, output_coords)
#     x = dx._interpolate(x)
#     coords["lat"] = dx.out_lat
#     coords["lon"] = dx.out_lon

#     y, c = CT(x, coords)

#     assert y.shape[0] == len(time)
#     assert y.shape[1] == 2
#     assert y.shape[2] == 1

#     assert "point" in c
#     assert all(c["point"] == np.arange(1))

#     assert "coord" in c
#     assert all(c["coord"] == np.array(["lat", "lon"]))

#     assert "time" in c
#     assert all(c["time"] == coords["time"])


# @pytest.mark.timeout(30)
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# @pytest.mark.parametrize("tracker", ["T1", "T2"])
# def test_time_information(device, tracker):

#     if tracker == "T1":
#         CT = CycloneTracking()
#     elif tracker == "T2":
#         CT = CycloneTrackingVorticity()

#     time = np.array(
#         [
#             np.datetime64("2024-09-26 12:00:00"),
#             np.datetime64("2024-09-26 18:00:00"),
#             np.datetime64("2024-09-27 00:00:00"),
#         ]
#     )
#     variable = CT.input_coords()["variable"]

#     gfs = GFS()
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da, device=device)

#     x = x.unsqueeze(0)
#     coords = {"ensemble": np.arange(1)} | coords

#     y, c = CT(x, coords)
#     assert np.all(c["time"] == time)


# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# @pytest.mark.parametrize("tracker", ["T1", "T2"])
# @pytest.mark.parametrize("case_name", ["mixed", "without"])
# def test_time_without_tc(device, tracker, case_name):
#     # Determine the appropriate tracker based on the parameters
#     if tracker == "T1":
#         CT = CycloneTracking()
#     elif tracker == "T2":
#         CT = CycloneTrackingVorticity()
#     else:
#         raise ValueError("Invalid tracker value. Must be 'T1' or 'T2'.")

#     # Create the test times based on the case_name
#     if tracker == "T2":
#         times_mixed = [
#             np.datetime64("2024-05-13 00:00:00"),
#             np.datetime64("2024-05-13 06:00:00"),
#         ]
#         if case_name == "mixed":
#             time = np.array(times_mixed)
#         elif case_name == "without":
#             time = np.array([times_mixed[1]])
#         else:
#             raise ValueError("Invalid case_name value. Must be 'mixed' or 'without'.")
#     elif tracker == "T1":
#         times_mixed = [
#             np.datetime64("2024-05-13 18:00:00"),
#             np.datetime64("2024-05-14 00:00:00"),
#         ]
#         if case_name == "mixed":
#             time = np.array(times_mixed)
#         elif case_name == "without":
#             time = np.array([times_mixed[0]])
#         else:
#             raise ValueError("Invalid case_name value. Must be 'mixed' or 'without'.")
#     # Get the input coordinates for the tracker and return the variable
#     variable = CT.input_coords()["variable"]
#     # get input data from GFS
#     gfs = GFS()
#     da = gfs(time, variable)
#     x, coords = prep_data_array(da, device=device)

#     x = x.unsqueeze(0)
#     coords = {"ensemble": np.arange(1)} | coords

#     # Run the tracker
#     y, c = CT(x, coords)
#     # Assert that the output shape matches the number of times
#     assert y.shape[1] == len(time)
#     # Assert that the coordinates match the input times
#     assert len(c["time"]) == len(time)
#     assert np.all(c["time"] == time)
