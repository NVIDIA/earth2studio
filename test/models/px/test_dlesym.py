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

import earth2studio.models.px.dlesym as dlesym_src
from earth2studio.models.px import DLESyM, DLESyMLatLon
from earth2studio.utils import handshake_coords


class PhooAtmosModel(torch.nn.Module):
    """Mock atmosphere model for testing."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(dlesym_src._ATMOS_OUTPUT_TIMES)
        self.input_time_dim = len(dlesym_src._ATMOS_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


class PhooOceanModel(torch.nn.Module):
    """Mock ocean model for testing."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(dlesym_src._OCEAN_OUTPUT_TIMES)
        self.input_time_dim = len(dlesym_src._OCEAN_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


def build_dlesym_model(device, nside=64, type="hpx"):
    """Build a DLESyM prognostic model with mock atmos/ocean models.

    Parameters
    ----------
    device : torch.device
        The device to build the model on.
    nside : int
        The nside of the HEALPix grid.
    type : str
        The type of model to build. Set to "hpx" for HEALPix grid or "ll" for lat/lon grid.

    Returns
    -------
    model : DLESyM or DLESyMLatLon
        The DLESyM model.
    """
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, 9, 1, 1, 1))  # 9 variables total
    scale = np.ones((1, 1, 1, 9, 1, 1, 1))
    atmos_constants = np.random.randn(12, 2, nside, nside)
    ocean_constants = np.random.randn(12, 2, nside, nside)

    atmos_input_times = dlesym_src._ATMOS_INPUT_TIMES
    ocean_input_times = dlesym_src._OCEAN_INPUT_TIMES
    atmos_output_times = dlesym_src._ATMOS_OUTPUT_TIMES
    ocean_output_times = dlesym_src._OCEAN_OUTPUT_TIMES

    atmos_variables = dlesym_src._ATMOS_VARIABLES
    ocean_variables = dlesym_src._OCEAN_VARIABLES
    atmos_coupling_variables = dlesym_src._ATMOS_COUPLING_VARIABLES
    ocean_coupling_variables = dlesym_src._OCEAN_COUPLING_VARIABLES

    model_constructor = DLESyM if type == "hpx" else DLESyMLatLon

    model = model_constructor(
        atmos_model=PhooAtmosModel(),
        ocean_model=PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=atmos_input_times,
        ocean_input_times=ocean_input_times,
        atmos_output_times=atmos_output_times,
        ocean_output_times=ocean_output_times,
        atmos_variables=atmos_variables,
        ocean_variables=ocean_variables,
        atmos_coupling_variables=atmos_coupling_variables,
        ocean_coupling_variables=ocean_coupling_variables,
    ).to(device)

    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("grid_type", ["hpx", "ll"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_forward(device, grid_type, batch_size):
    """Test DLESyM forward pass with mock models."""

    if grid_type == "ll" and device == "cpu":
        pytest.skip("Lat/lon regridding is slow on CPU")

    nside = 64
    model = build_dlesym_model(device, type=grid_type)

    spatial_dims = (12, nside, nside) if grid_type == "hpx" else (721, 1440)

    # Create test input
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        *spatial_dims,
        device=device,
    )

    # Test forward pass
    in_coords = model.input_coords()
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time
    output, output_coords = model(x, in_coords)
    expected_coords = model.output_coords(in_coords)
    assert output.shape == (
        batch_size,
        len(time),
        len(dlesym_src._ATMOS_OUTPUT_TIMES),
        len(variable),
        *spatial_dims,
    )
    for key in output_coords:
        handshake_coords(output_coords, expected_coords, key)
    assert np.all(output_coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES)

    # Test retrieving valid outputs
    atmos_outputs, atmos_coords = model.retrieve_valid_atmos_outputs(
        output, output_coords
    )
    assert atmos_outputs.size() == (
        batch_size,
        len(time),
        len(dlesym_src._ATMOS_OUTPUT_TIMES),
        len(dlesym_src._ATMOS_VARIABLES),
        *spatial_dims,
    )
    assert np.all(atmos_coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES)

    ocean_outputs, ocean_coords = model.retrieve_valid_ocean_outputs(
        output, output_coords
    )
    assert ocean_outputs.shape == (
        batch_size,
        len(time),
        len(dlesym_src._OCEAN_OUTPUT_TIMES),
        len(dlesym_src._OCEAN_VARIABLES),
        *spatial_dims,
    )
    assert np.all(ocean_coords["lead_time"] == dlesym_src._OCEAN_OUTPUT_TIMES)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_latlon_regridding(device, batch_size):
    """Test DLESyMLatLon regridding functionality."""
    nside = 64
    model = build_dlesym_model(device, type="ll")

    # Test basic coordinate conversion
    ll_coords = model.input_coords()
    hpx_coords = model.coords_to_hpx(ll_coords)
    for coord in ["lat", "lon"]:
        assert coord not in hpx_coords
        assert coord in ll_coords
    for coord in ["face", "height", "width"]:
        assert coord in hpx_coords
        assert coord not in ll_coords

    # Test regridding
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x_ll = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        721,  # lat
        1440,  # lon
        device=device,
    )

    # Test round-trip regridding
    in_coords = model.input_coords()
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time
    x_hpx = model.to_hpx(x_ll)
    assert x_hpx.shape == (
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        12,
        nside,
        nside,
    )
    x_ll_roundtrip = model.to_ll(x_hpx)

    # Round-trip regridding error is more sensitive to fine-scale detail
    # so here we just check that the mean error is less than 1e-2
    diff = x_ll - x_ll_roundtrip
    assert diff.mean().item() < 1e-2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("grid_type", ["hpx", "ll"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_iterator(device, grid_type, batch_size):
    """Test DLESyM iterator functionality."""

    if grid_type == "ll" and device == "cpu":
        pytest.skip("Lat/lon regridding is slow on CPU")

    nside = 64
    model = build_dlesym_model(device, type=grid_type, nside=nside)
    spatial_dims = (12, nside, nside) if grid_type == "hpx" else (721, 1440)
    # Create test input
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        *spatial_dims,
        device=device,
    )

    # Test iterator
    in_coords = model.input_coords()
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time
    iterator = model.create_iterator(x, in_coords)

    # First yield should be initial condition
    initial_x, initial_coords = next(iterator)
    assert torch.allclose(initial_x, x)

    # Test a few steps
    coupler_step = dlesym_src._ATMOS_OUTPUT_TIMES[-1]
    for i in range(3):
        x, coords = next(iterator)
        assert x.shape == (
            batch_size,
            len(time),
            len(dlesym_src._ATMOS_OUTPUT_TIMES),
            len(variable),
            *spatial_dims,
        )
        assert np.all(
            coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES + coupler_step * i
        )


# TODO test_model_package
