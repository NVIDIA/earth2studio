# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
import torch

from earth2studio.models.dx import CycloneTracking


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_vorticity_calculation(device):
    # Create a simple grid
    nx, ny = 4, 4
    dx = dy = 1.0  # 1km grid spacing

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
