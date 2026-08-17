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

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import AtlasCRPS
from earth2studio.utils import handshake_dim


class PhooAtlasCRPSModel(torch.nn.Module):
    """Dummy AtlasCRPS model for testing.

    This model simulates the CRPS latent transformer by adding a time delta to the
    current state to represent a prognostic step.
    """

    def __init__(self, delta_t: int = 6, n_vars: int = 75):
        super().__init__()
        self.delta_t = delta_t
        self.n_vars = n_vars

    def forward(self, x_1, x_2):
        """Simple forward that adds delta_t to the current state stream."""
        return x_2[:, : self.n_vars] + self.delta_t


class PhooAutoencoder(torch.nn.Module):
    """Dummy autoencoder for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, residual_latent):
        return x


class PhooNormalizer(torch.nn.Module):
    """Dummy normalizer for testing."""

    def __init__(self):
        super().__init__()

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class PhooProcessor(torch.nn.Module):
    """Dummy processor for testing."""

    def __init__(self):
        super().__init__()
        self.normalizer_in = PhooNormalizer()
        self.normalizer_out = PhooNormalizer()
        self.downsample_grid_shape = (181, 360)

    def forward(self, x):
        return x

    def preprocess_input(self, x, current_date):
        return x, x

    def intep(self, x, downsample_grid_shape):
        return x

    def postprocess(self, x, x_cur):
        return x


@pytest.fixture()
def atlas_crps_test_components():
    """Create dummy AtlasCRPS model components for testing."""
    n_vars = 75

    return {
        "model": PhooAtlasCRPSModel(delta_t=6, n_vars=n_vars),
        "model_processor": PhooProcessor(),
        "autoencoder": PhooAutoencoder(),
        "autoencoder_processor": PhooProcessor(),
    }


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_atlas_crps_call(time, device, batch_size, atlas_crps_test_components):
    """Test AtlasCRPS __call__ method with different times and devices."""
    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add batch dimension
    x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch_size)})
    coords.move_to_end("batch", last=False)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size(
        [
            batch_size,
            len(time),
            1,
            len(p.output_coords(p.input_coords())["variable"]),
            721,
            1440,
        ]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")

    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_crps_iter(ensemble, atlas_crps_test_components, device):
    """Test AtlasCRPS iterator for autoregressive predictions."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    out, out_coords = next(p_iter)  # Skip first which should return the input
    # First output should be the latest lead time from input
    assert torch.allclose(out, x[:, :, 1:])

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape[0] == ensemble
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "ensemble", 0)

        if i > 3:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_atlas_crps_exceptions(dc, atlas_crps_test_components, device):
    """Test that AtlasCRPS raises exceptions for invalid coordinates."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    # Initialize Data Source with invalid coordinates
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        p(x, coords)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_crps_prep_next_input(atlas_crps_test_components, batch_size, device):
    """Test AtlasCRPS prep_next_input method for autoregressive stepping.

    The prep_next_input method should:
    1. Take the prediction at t+6h and place it as the latest input
    2. Shift the previous latest input (t=0) to the earlier position (t-6h)
    3. Update lead times by +6h
    """
    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    # Create input data with 2 lead times
    time_steps = 2
    n_vars = 75
    lat = 721
    lon = 1440

    # Input state at t-6h and t=0
    x = torch.randn(batch_size, 1, time_steps, n_vars, lat, lon, device=device)
    coords = p.input_coords()
    coords["batch"] = np.arange(batch_size)
    coords["time"] = np.array([np.datetime64("2020-01-01T00:00")])

    # Prediction at t+6h (output has shape [batch, 1, n_vars, lat, lon])
    x_pred = torch.randn(batch_size, 1, 1, n_vars, lat, lon, device=device)
    coords_pred = p.output_coords(coords)
    coords_pred["batch"] = coords["batch"]
    coords_pred["time"] = coords["time"]

    # Call prep_next_input
    x_next, coords_next = p.prep_next_input(x_pred, coords_pred, x, coords)

    # Check that x_next has the correct shape
    assert x_next.shape == x.shape

    # Check that the latest lead time contains the prediction
    assert torch.allclose(x_next[:, :, -1:], x_pred[:, :, :1])

    # Check that the earlier lead time contains the previous latest
    assert torch.allclose(x_next[:, :, :-1], x[:, :, 1:])

    # Check that lead times are updated correctly
    expected_lead_time = coords["lead_time"] + p.DT
    assert np.array_equal(coords_next["lead_time"], expected_lead_time)

    # Check other coordinates remain unchanged
    assert np.array_equal(coords_next["batch"], coords["batch"])
    assert np.array_equal(coords_next["time"], coords["time"])
    assert np.array_equal(coords_next["variable"], coords["variable"])
    assert np.array_equal(coords_next["lat"], coords["lat"])
    assert np.array_equal(coords_next["lon"], coords["lon"])


def test_atlas_crps_input_coords(atlas_crps_test_components):
    """Test that input_coords returns expected coordinate system."""
    p = AtlasCRPS(**atlas_crps_test_components)
    coords = p.input_coords()

    # Check expected keys
    assert "batch" in coords
    assert "time" in coords
    assert "lead_time" in coords
    assert "variable" in coords
    assert "lat" in coords
    assert "lon" in coords

    # Check lead_time has two steps: -6h and 0h
    assert len(coords["lead_time"]) == 2
    assert coords["lead_time"][0] == np.timedelta64(-6, "h")
    assert coords["lead_time"][1] == np.timedelta64(0, "h")

    # Check variable count
    assert len(coords["variable"]) == 75

    # Check spatial dimensions
    assert len(coords["lat"]) == 721
    assert len(coords["lon"]) == 1440

    # Check spatial range
    assert coords["lat"][0] == pytest.approx(90.0, abs=1e-5)
    assert coords["lat"][-1] == pytest.approx(-90.0, abs=1e-5)
    assert coords["lon"][0] == pytest.approx(0.0, abs=1e-5)
    assert coords["lon"][-1] == pytest.approx(360.0 - (360.0 / 1440.0), abs=1e-5)


def test_atlas_crps_output_coords(atlas_crps_test_components):
    """Test that output_coords returns expected coordinate system."""
    p = AtlasCRPS(**atlas_crps_test_components)
    input_coords = p.input_coords()
    output_coords = p.output_coords(input_coords)

    # Check expected keys
    assert "batch" in output_coords
    assert "time" in output_coords
    assert "lead_time" in output_coords
    assert "variable" in output_coords
    assert "lat" in output_coords
    assert "lon" in output_coords

    # Check lead_time is single step at +6h
    assert len(output_coords["lead_time"]) == 1
    assert output_coords["lead_time"][0] == np.timedelta64(6, "h")

    # Check variable count matches input
    assert len(output_coords["variable"]) == len(input_coords["variable"])

    # Check spatial dimensions match input
    assert len(output_coords["lat"]) == len(input_coords["lat"])
    assert len(output_coords["lon"]) == len(input_coords["lon"])
